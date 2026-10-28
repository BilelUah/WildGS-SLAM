from contextlib import nullcontext
from pathlib import Path

import numpy as np
import lietorch
import torch

import src.geom.projective_ops as pops
from src.modules.droid_net import CorrBlock
from src.utils.mono_priors.metric_depth_estimators import (
    get_metric_depth_estimator,
    predict_metric_depth,
)
from src.utils.mono_priors.img_feature_extractors import (
    get_feature_extractor,
    predict_img_features,
)


class MotionFilter:
    """
    Filter incoming frames, estimate approximate motion, and decide whether
    each frame should become a WildGS keyframe.

    This class is mainly inherited from DROID-SLAM.
    """

    def __init__(
        self,
        net,
        video,
        cfg,
        thresh=2.5,
        device="cuda:0",
    ):
        self.cfg = cfg

        # The shared SimpleProfiler is attached by the Navigator wrapper
        # immediately after MotionFilter construction.
        self.profiler = None

        # Split the DROID network into its main modules.
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update

        self.video = video
        self.thresh = thresh
        self.device = device

        # Number of consecutive frames rejected by the motion filter.
        self.count = 0

        # Mean and standard deviation used for image normalization.
        self.MEAN = torch.as_tensor(
            [0.485, 0.456, 0.406],
            device=self.device,
        )[:, None, None]

        self.STDV = torch.as_tensor(
            [0.229, 0.224, 0.225],
            device=self.device,
        )[:, None, None]

        self.uncertainty_aware = cfg[
            "tracking"
        ][
            "uncertainty_params"
        ][
            "activate"
        ]

        self.save_dir = (
            cfg["data"]["output"]
            + "/"
            + cfg["scene"]
        )

        # Navigator adds this optional policy only for integrated runs.
        # Standalone original WildGS still defaults to Metric3D.
        navigator_config = cfg.get("navigator", {})
        self.use_sensor_depth = (
            navigator_config.get("depth_source", "metric") == "sensor"
        )
        if self.use_sensor_depth and not navigator_config.get("is_live", False):
            raise ValueError(
                "[WILDGS SENSOR] RealSense sensor depth is live-only. "
                "Offline WildGS must use Metric3D."
            )

        if self.use_sensor_depth:
            # Avoid initializing the Metric3D model in live sensor mode.
            self.metric_depth_estimator = None
        else:
            # ORIGINAL WILDGS: preserved as-is in both offline and live metric.
            self.metric_depth_estimator = (
                get_metric_depth_estimator(cfg)
            )

        print(
            "[WildGS MotionFilter] depth source:",
            "sensor" if self.use_sensor_depth else "metric",
        )

        if cfg["mapping"][
            "uncertainty_params"
        ][
            "activate"
        ]:
            # Mapping needs image features, so create the extractor.
            self.feat_extractor = (
                get_feature_extractor(cfg)
            )

    # =========================================================
    # 1. SimpleProfiler helpers
    # =========================================================

    def _profile_timer(self, name: str):
        """
        Return a real profiler timer when profiling is enabled.

        Otherwise, return a no-op context manager so the original WildGS
        code continues to work without profiling.
        """
        profiler = getattr(
            self,
            "profiler",
            None,
        )

        if profiler is None:
            return nullcontext()

        return profiler.timer(name)

    def _profile_count(
        self,
        name: str,
        amount: int = 1,
    ) -> None:
        """
        Add an event or quantity to the profiler counter dictionary.
        """
        profiler = getattr(
            self,
            "profiler",
            None,
        )

        if profiler is None:
            return

        profiler.count(
            name,
            amount,
        )

    def _profile_value(
        self,
        name: str,
        value,
    ) -> None:
        """
        Store the latest value for one profiler field.
        """
        profiler = getattr(
            self,
            "profiler",
            None,
        )

        if profiler is None:
            return

        profiler.value(
            name,
            value,
        )

    # =========================================================
    # 2. Neural encoders
    # =========================================================

    @torch.amp.autocast(
        "cuda",
        enabled=True,
    )
    def __context_encoder(self, image):
        """
        Extract context features used by the recurrent update network.
        """
        net, inp = self.cnet(image).split(
            [128, 128],
            dim=2,
        )

        return (
            net.tanh().squeeze(0),
            inp.relu().squeeze(0),
        )

    @torch.amp.autocast(
        "cuda",
        enabled=True,
    )
    def __feature_encoder(self, image):
        """
        Extract feature maps used to construct the correlation volume.
        """
        return self.fnet(image).squeeze(0)

    # =========================================================
    # 3. Image-feature prediction helper
    # =========================================================

    def _predict_and_store_image_features(
        self,
        tstamp,
        image,
    ):
        """
        Predict image features when tracking or mapping uncertainty needs them.

        Returns:
            Features stored directly in DepthVideo when tracking uncertainty
            is active, otherwise None.
        """
        if self.uncertainty_aware:
            with self._profile_timer(
                "wildgs.motion_filter.image_features.predict"
            ):
                dino_features = predict_img_features(
                    self.feat_extractor,
                    tstamp,
                    image,
                    self.cfg,
                    self.device,
                )

            self._profile_count(
                "wildgs.motion_filter.image_feature_predictions"
            )

            return dino_features

        dino_features = None

        if self.cfg["mapping"][
            "uncertainty_params"
        ][
            "activate"
        ]:
            # Mapping needs image features. They are predicted and stored on
            # disk by the existing WildGS helper.
            with self._profile_timer(
                "wildgs.motion_filter.image_features.predict_for_mapping"
            ):
                _ = predict_img_features(
                    self.feat_extractor,
                    tstamp,
                    image,
                    self.cfg,
                    self.device,
                )

            self._profile_count(
                "wildgs.motion_filter.image_feature_predictions_for_mapping"
            )

        return dino_features

    def _sensor_depth_for_keyframe(self, external_depth, image, tstamp):
        """Validate aligned RealSense depth (metres) for DepthVideo insertion.

        RGB/depth rotation and resizing are handled by the existing Navigator
        ROS2 dataset and tracking adapter. Do not re-scale or infer depth here.
        """
        if external_depth is None:
            raise ValueError(
                "[WILDGS SENSOR] Missing synchronized RealSense depth."
            )

        depth = torch.as_tensor(
            external_depth, dtype=torch.float32, device=self.device
        )
        expected_shape = tuple(int(s) for s in image.shape[-2:])
        if depth.ndim != 2 or tuple(depth.shape) != expected_shape:
            raise ValueError(
                "[WILDGS SENSOR] Aligned depth/RGB size mismatch: "
                f"depth={tuple(depth.shape)}, RGB={expected_shape}"
            )
        if not bool(torch.isfinite(depth).all()):
            raise ValueError("[WILDGS SENSOR] Depth contains NaN or Inf")
        if bool((depth < 0).any()):
            raise ValueError("[WILDGS SENSOR] Depth contains negative values")
        if not bool((depth > 0).any()):
            raise ValueError("[WILDGS SENSOR] Depth has no valid pixels")

        depth = depth.contiguous()

        # The existing WildGS Mapper reloads each accepted keyframe through
        # load_metric_depth(frame_idx, self.save_dir), even when metric_depth_reg
        # is enabled. Preserve its original on-disk depth contract in Sensor mode.
        # This is a RAW sensor depth map in metres, not a Metric3D prediction.
        depth_dir = Path(self.save_dir) / "mono_priors" / "depths"
        depth_dir.mkdir(parents=True, exist_ok=True)
        depth_path = depth_dir / f"{int(tstamp):05d}.npy"
        np.save(depth_path, depth.detach().cpu().numpy().astype(np.float32))

        return depth

    # =========================================================
    # 4. Main MotionFilter operation
    # =========================================================

    @torch.amp.autocast(
        "cuda",
        enabled=True,
    )
    @torch.no_grad()
    def track(
        self,
        tstamp,
        image,
        intrinsics=None,
        external_depth=None,
    ):
        """
        Run MotionFilter on one incoming frame.

        Main responsibilities:
        1. Move and normalize the image.
        2. Extract correlation features.
        3. Always insert the first frame.
        4. For later frames, estimate approximate optical flow.
        5. Accept or reject the frame as a keyframe.
        6. Predict metric depth and uncertainty features for accepted frames.
        7. Append accepted frames to DepthVideo.

        Profiling hierarchy:

        wildgs.motion_filter.total
        ├── setup_frame
        ├── normalize_image
        ├── feature_encoder
        ├── first_frame.total
        │   ├── context_encoder
        │   ├── metric_depth
        │   ├── image_features.*
        │   └── video_append
        └── motion_check.total
            ├── coordinates_grid
            ├── correlation
            ├── update_operator
            ├── motion_score
            ├── forced_keyframe_check
            └── accepted_keyframe.total
                ├── context_encoder
                ├── metric_depth
                ├── image_features.*
                └── video_append

        Parent timers include their children.
        """
        # A sensor frame must never silently switch to Metric3D (or reverse).
        if self.use_sensor_depth and external_depth is None:
            raise ValueError("[WILDGS SENSOR] Missing external_depth")
        if not self.use_sensor_depth and external_depth is not None:
            raise ValueError(
                "[WILDGS METRIC] Unexpected external_depth in Metric3D mode"
            )

        self._profile_count(
            "wildgs.motion_filter.frames.received"
        )
        self._profile_value(
            "wildgs.motion_filter.last_frame_id",
            int(tstamp),
        )
        self._profile_value(
            "wildgs.motion_filter.motion_threshold",
            float(self.thresh),
        )

        with self._profile_timer(
            "wildgs.motion_filter.total"
        ):
            # -----------------------------------------------------
            # A. Basic frame geometry
            # -----------------------------------------------------
            with self._profile_timer(
                "wildgs.motion_filter.setup_frame"
            ):
                Id = (
                    lietorch.SE3
                    .Identity(1)
                    .data
                    .squeeze()
                )

                ht = (
                    image.shape[-2]
                    // self.video.down_scale
                )
                wd = (
                    image.shape[-1]
                    // self.video.down_scale
                )

            # -----------------------------------------------------
            # B. Move image to GPU and normalize it
            # -----------------------------------------------------
            with self._profile_timer(
                "wildgs.motion_filter.normalize_image"
            ):
                inputs = image[
                    None, :, :
                ].to(
                    self.device
                )

                inputs = (
                    inputs
                    .sub_(self.MEAN)
                    .div_(self.STDV)
                )

            # -----------------------------------------------------
            # C. Extract features used for correlation
            # -----------------------------------------------------
            with self._profile_timer(
                "wildgs.motion_filter.feature_encoder"
            ):
                gmap = self.__feature_encoder(
                    inputs
                )

            force_to_add_keyframe = False

            # =====================================================
            # First frame: always insert into DepthVideo
            # =====================================================
            if self.video.counter.value == 0:
                self._profile_count(
                    "wildgs.motion_filter.first_frames"
                )

                with self._profile_timer(
                    "wildgs.motion_filter.first_frame.total"
                ):
                    with self._profile_timer(
                        "wildgs.motion_filter.first_frame.context_encoder"
                    ):
                        net, inp = (
                            self.__context_encoder(
                                inputs[:, [0]]
                            )
                        )

                    self.net = net
                    self.inp = inp
                    self.fmap = gmap

                    with self._profile_timer(
                        (
                            "wildgs.motion_filter.first_frame.sensor_depth"
                            if self.use_sensor_depth
                            else "wildgs.motion_filter.first_frame.metric_depth"
                        )
                    ):
                        if self.use_sensor_depth:
                            mono_depth = self._sensor_depth_for_keyframe(
                                external_depth, image, tstamp
                            )
                        else:
                            mono_depth = (
                                predict_metric_depth(
                                    self.metric_depth_estimator,
                                    tstamp,
                                    image,
                                    self.cfg,
                                    self.device,
                                )
                            )

                    self._profile_count(
                        "wildgs.motion_filter.sensor_depth_keyframes"
                        if self.use_sensor_depth
                        else "wildgs.motion_filter.metric_depth_predictions"
                    )

                    dino_features = (
                        self._predict_and_store_image_features(
                            tstamp,
                            image,
                        )
                    )

                    with self._profile_timer(
                        "wildgs.motion_filter.first_frame.video_append"
                    ):
                        self.video.append(
                            tstamp,
                            image[0],
                            Id,
                            1.0,
                            mono_depth,
                            intrinsics
                            / float(
                                self.video.down_scale
                            ),
                            gmap,
                            net[0, 0],
                            inp[0, 0],
                            dino_features,
                        )

                self._profile_count(
                    "wildgs.motion_filter.video_appends"
                )
                self._profile_count(
                    "wildgs.motion_filter.keyframes.accepted"
                )

                self._profile_value(
                    "wildgs.motion_filter.last_motion_score",
                    None,
                )
                self._profile_value(
                    "wildgs.motion_filter.last_force_keyframe",
                    False,
                )
                self._profile_value(
                    "wildgs.motion_filter.last_keyframe_decision",
                    True,
                )

            # =====================================================
            # Later frames: estimate motion and decide
            # =====================================================
            else:
                with self._profile_timer(
                    "wildgs.motion_filter.motion_check.total"
                ):
                    # ---------------------------------------------
                    # D. Create pixel-coordinate grid
                    # ---------------------------------------------
                    with self._profile_timer(
                        "wildgs.motion_filter.motion_check.coordinates_grid"
                    ):
                        coords0 = pops.coords_grid(
                            ht,
                            wd,
                            device=self.device,
                        )[None, None]

                    # ---------------------------------------------
                    # E. Build and index correlation volume
                    # ---------------------------------------------
                    with self._profile_timer(
                        "wildgs.motion_filter.motion_check.correlation"
                    ):
                        corr = CorrBlock(
                            self.fmap[
                                None, [0]
                            ],
                            gmap[
                                None, [0]
                            ],
                        )(coords0)

                    # ---------------------------------------------
                    # F. One update iteration estimates flow delta
                    # ---------------------------------------------
                    with self._profile_timer(
                        "wildgs.motion_filter.motion_check.update_operator"
                    ):
                        _, delta, weight = self.update(
                            self.net[None],
                            self.inp[None],
                            corr,
                        )

                    del weight

                    # ---------------------------------------------
                    # G. Convert predicted flow to one motion score
                    # ---------------------------------------------
                    with self._profile_timer(
                        "wildgs.motion_filter.motion_check.motion_score"
                    ):
                        motion_score = (
                            delta
                            .norm(dim=-1)
                            .mean()
                            .item()
                        )

                    # ---------------------------------------------
                    # H. Optional forced-keyframe cadence
                    # ---------------------------------------------
                    with self._profile_timer(
                        "wildgs.motion_filter.motion_check.forced_keyframe_check"
                    ):
                        force_every_n = int(
                            self.cfg[
                                "tracking"
                            ][
                                "force_keyframe_every_n_frames"
                            ]
                        )

                        if force_every_n > 0:
                            last_tstamp = (
                                self.video.timestamp[
                                    self.video.counter.value - 1
                                ]
                            )

                            force_to_add_keyframe = bool(
                                (
                                    tstamp
                                    - last_tstamp
                                )
                                >= force_every_n
                            )

                    accepted_by_motion = bool(
                        motion_score > self.thresh
                    )

                    should_add_keyframe = bool(
                        accepted_by_motion
                        or force_to_add_keyframe
                    )

                    self._profile_value(
                        "wildgs.motion_filter.last_motion_score",
                        float(motion_score),
                    )
                    self._profile_value(
                        "wildgs.motion_filter.last_force_keyframe",
                        bool(force_to_add_keyframe),
                    )
                    self._profile_value(
                        "wildgs.motion_filter.last_keyframe_decision",
                        bool(should_add_keyframe),
                    )

                    # =============================================
                    # Accepted frame: compute full keyframe data
                    # =============================================
                    if should_add_keyframe:
                        self.count = 0

                        with self._profile_timer(
                            "wildgs.motion_filter.accepted_keyframe.total"
                        ):
                            with self._profile_timer(
                                "wildgs.motion_filter.accepted_keyframe.context_encoder"
                            ):
                                net, inp = (
                                    self.__context_encoder(
                                        inputs[:, [0]]
                                    )
                                )

                            self.net = net
                            self.inp = inp
                            self.fmap = gmap

                            with self._profile_timer(
                                (
                                    "wildgs.motion_filter.accepted_keyframe.sensor_depth"
                                    if self.use_sensor_depth
                                    else "wildgs.motion_filter.accepted_keyframe.metric_depth"
                                )
                            ):
                                if self.use_sensor_depth:
                                    mono_depth = self._sensor_depth_for_keyframe(
                                        external_depth, image, tstamp
                                    )
                                else:
                                    mono_depth = (
                                        predict_metric_depth(
                                            self.metric_depth_estimator,
                                            tstamp,
                                            image,
                                            self.cfg,
                                            self.device,
                                        )
                                    )

                            self._profile_count(
                                "wildgs.motion_filter.sensor_depth_keyframes"
                                if self.use_sensor_depth
                                else "wildgs.motion_filter.metric_depth_predictions"
                            )

                            dino_features = (
                                self._predict_and_store_image_features(
                                    tstamp,
                                    image,
                                )
                            )

                            with self._profile_timer(
                                "wildgs.motion_filter.accepted_keyframe.video_append"
                            ):
                                self.video.append(
                                    tstamp,
                                    image[0],
                                    None,
                                    None,
                                    mono_depth,
                                    intrinsics
                                    / float(
                                        self.video.down_scale
                                    ),
                                    gmap,
                                    net[0],
                                    inp[0],
                                    dino_features,
                                )

                        self._profile_count(
                            "wildgs.motion_filter.video_appends"
                        )
                        self._profile_count(
                            "wildgs.motion_filter.keyframes.accepted"
                        )

                        if accepted_by_motion:
                            self._profile_count(
                                "wildgs.motion_filter.keyframes.accepted_by_motion"
                            )

                        if force_to_add_keyframe:
                            self._profile_count(
                                "wildgs.motion_filter.keyframes.forced"
                            )

                    # =============================================
                    # Rejected frame: no DepthVideo insertion
                    # =============================================
                    else:
                        self.count += 1

                        self._profile_count(
                            "wildgs.motion_filter.frames.rejected"
                        )

                self._profile_value(
                    "wildgs.motion_filter.current_rejection_streak",
                    int(self.count),
                )

        return force_to_add_keyframe

    # =========================================================
    # 5. Public image-feature helper
    # =========================================================

    @torch.no_grad()
    def get_img_feature(
        self,
        tstamp,
        image,
        suffix="",
    ):
        """
        Predict image features outside the main MotionFilter.track() call.
        """
        with self._profile_timer(
            "wildgs.motion_filter.get_img_feature.total"
        ):
            dino_features = predict_img_features(
                self.feat_extractor,
                tstamp,
                image,
                self.cfg,
                self.device,
                suffix=suffix,
            )

        self._profile_count(
            "wildgs.motion_filter.get_img_feature.calls"
        )

        return dino_features