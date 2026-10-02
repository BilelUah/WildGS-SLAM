from contextlib import nullcontext

import torch

from src.factor_graph import FactorGraph
from src.backend import Backend as LoopClosing


class Frontend:
    """
    WildGS tracking frontend.

    Main responsibilities:
    - initialize the local factor graph
    - add and remove graph factors
    - run local graph optimization
    - decide whether the newest candidate keyframe should be kept or removed
    - optionally run loop-closing bundle adjustment
    - propagate the latest pose and disparity to the next video slot

    This implementation is mainly inherited from GO-SLAM.
    """

    def __init__(self, net, video, cfg):
        self.cfg = cfg
        self.video = video
        self.update_op = net.update

        # The Navigator wrapper attaches the shared SimpleProfiler
        # immediately after constructing Frontend.
        self.profiler = None

        # Local optimization window.
        self.t1 = 0

        # Frontend lifecycle state.
        self.is_initialized = False

        self.max_age = cfg["tracking"]["max_age"]
        self.iters1 = 4 * 2
        self.iters2 = 2 * 2

        self.warmup = cfg["tracking"]["warmup"]
        self.beta = cfg["tracking"]["beta"]

        frontend_cfg = cfg["tracking"]["frontend"]

        self.frontend_nms = frontend_cfg["nms"]
        self.keyframe_thresh = frontend_cfg["keyframe_thresh"]
        self.frontend_window = frontend_cfg["window"]
        self.frontend_thresh = frontend_cfg["thresh"]
        self.frontend_radius = frontend_cfg["radius"]
        self.frontend_max_factors = frontend_cfg["max_factors"]

        self.enable_loop = frontend_cfg["enable_loop"]

        self.loop_closing = LoopClosing(
            net,
            video,
            cfg,
        )

        self.graph = FactorGraph(
            video,
            net.update,
            device=cfg["device"],
            corr_impl="volume",
            max_factors=self.frontend_max_factors,
        )

        # Avoid too many consecutive candidate keyframes caused by large
        # moving objects but little real camera motion.
        self.max_consecutive_drop_of_keyframes = (
            cfg["tracking"]["max_age"]
            / self.iters1
        ) // 3

        self.num_keyframes_dropped = 0

    # =========================================================
    # 1. SimpleProfiler helpers
    # =========================================================

    def _profile_timer(self, name: str):
        """
        Return a real timer when the shared profiler is available.

        Otherwise, return a no-op context manager so original WildGS
        behavior remains unchanged.
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
        Add an event or quantity to a profiler counter.
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
        Store the latest value for a profiler field.
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

    def _record_frontend_state(self) -> None:
        """
        Store a small set of readable state values for the final report.
        """
        self._profile_value(
            "wildgs.frontend.state.video_counter",
            int(self.video.counter.value),
        )
        self._profile_value(
            "wildgs.frontend.state.t1",
            int(self.t1),
        )
        self._profile_value(
            "wildgs.frontend.state.is_initialized",
            bool(self.is_initialized),
        )
        self._profile_value(
            "wildgs.frontend.state.consecutive_dropped_keyframes",
            int(self.num_keyframes_dropped),
        )

        graph_ii = getattr(
            self.graph,
            "ii",
            None,
        )

        if graph_ii is not None:
            self._profile_value(
                "wildgs.frontend.state.active_factors",
                int(graph_ii.numel()),
            )

    # =========================================================
    # 2. Regular frontend update after initialization
    # =========================================================

    def __update(self, force_to_add_keyframe):
        """
        Add factors, optimize the local graph, and decide whether to keep
        or remove the newest candidate keyframe.

        Profiling hierarchy:

        wildgs.frontend.update.total
        ├── advance_window
        ├── remove_old_factors
        ├── add_proximity_factors
        ├── first_optimization_pass
        ├── metric_depth_error_filter
        ├── keyframe_distance
        ├── dropped_keyframe
        ├── loop_closing
        ├── loop_fallback_optimization
        ├── second_optimization_pass
        ├── propagate_pose_and_disparity
        ├── mark_visualization_dirty
        └── empty_cuda_cache

        Parent timers include their children.
        """
        self._profile_count(
            "wildgs.frontend.updates"
        )
        self._profile_value(
            "wildgs.frontend.last_force_keyframe",
            bool(force_to_add_keyframe),
        )

        with self._profile_timer(
            "wildgs.frontend.update.total"
        ):
            # -----------------------------------------------------
            # A. Advance local frontend time index
            # -----------------------------------------------------
            with self._profile_timer(
                "wildgs.frontend.update.advance_window"
            ):
                self.t1 += 1

            # -----------------------------------------------------
            # B. Remove factors that became too old
            # -----------------------------------------------------
            if self.graph.corr is not None:
                with self._profile_timer(
                    "wildgs.frontend.update.remove_old_factors"
                ):
                    self.graph.rm_factors(
                        self.graph.age > self.max_age,
                        store=True,
                    )

            # -----------------------------------------------------
            # C. Add factors connecting nearby keyframes
            # -----------------------------------------------------
            with self._profile_timer(
                "wildgs.frontend.update.add_proximity_factors"
            ):
                self.graph.add_proximity_factors(
                    self.t1 - 5,
                    max(
                        self.t1 - self.frontend_window,
                        0,
                    ),
                    rad=self.frontend_radius,
                    nms=self.frontend_nms,
                    thresh=self.frontend_thresh,
                    beta=self.beta,
                    remove=True,
                )

            # -----------------------------------------------------
            # D. First local optimization pass
            # -----------------------------------------------------
            with self._profile_timer(
                "wildgs.frontend.update.first_optimization_pass"
            ):
                for itr in range(self.iters1):
                    self.graph.update(
                        None,
                        None,
                        use_inactive=True,
                    )

                    self._profile_count(
                        "wildgs.frontend.graph_update_iterations"
                    )

                    if not self.cfg["fast_mode"]:
                        should_filter_depth = (
                            itr == 1
                            and self.video.metric_depth_reg
                            and self.cfg["tracking"][
                                "uncertainty_params"
                            ]["activate"]
                        )

                        if should_filter_depth:
                            with self._profile_timer(
                                "wildgs.frontend.update.metric_depth_error_filter"
                            ):
                                self.video.filter_high_err_mono_depth(
                                    self.t1 - 1,
                                    self.graph.ii,
                                    self.graph.jj,
                                )

                            self._profile_count(
                                "wildgs.frontend.metric_depth_error_filters"
                            )

            # -----------------------------------------------------
            # E. Measure camera-motion distance between last keyframes
            # -----------------------------------------------------
            with self._profile_timer(
                "wildgs.frontend.update.keyframe_distance"
            ):
                d = self.video.distance(
                    [self.t1 - 2],
                    [self.t1 - 1],
                    beta=self.beta,
                    bidirectional=True,
                )

                keyframe_distance = float(
                    d.item()
                )

            self._profile_value(
                "wildgs.frontend.last_keyframe_distance",
                keyframe_distance,
            )
            self._profile_value(
                "wildgs.frontend.keyframe_threshold",
                float(self.keyframe_thresh),
            )

            # -----------------------------------------------------
            # F. Decide whether newest candidate keyframe is kept
            # -----------------------------------------------------
            should_drop_keyframe = bool(
                (
                    d.item()
                    < self.keyframe_thresh
                )
                & (
                    self.num_keyframes_dropped
                    < self.max_consecutive_drop_of_keyframes
                )
                & (
                    not force_to_add_keyframe
                )
            )

            self._profile_value(
                "wildgs.frontend.last_keyframe_was_dropped",
                should_drop_keyframe,
            )

            if should_drop_keyframe:
                with self._profile_timer(
                    "wildgs.frontend.update.dropped_keyframe"
                ):
                    self.graph.rm_keyframe(
                        self.t1 - 1
                    )

                    self.num_keyframes_dropped += 1

                    with self.video.get_lock():
                        self.video.counter.value -= 1
                        self.t1 -= 1

                self._profile_count(
                    "wildgs.frontend.keyframes.dropped"
                )

            else:
                self._profile_count(
                    "wildgs.frontend.keyframes.kept"
                )

                cur_t = int(
                    self.video.counter.value
                )

                self.num_keyframes_dropped = 0

                # -------------------------------------------------
                # G. Optional loop-closing bundle adjustment
                # -------------------------------------------------
                if (
                    self.enable_loop
                    and cur_t > self.frontend_window
                ):
                    with self._profile_timer(
                        "wildgs.frontend.update.loop_closing"
                    ):
                        n_kf, n_edge = (
                            self.loop_closing.loop_ba(
                                t_start=0,
                                t_end=cur_t,
                                steps=self.iters2,
                                motion_only=False,
                                local_graph=self.graph,
                                enable_wq=True,
                            )
                        )

                    self._profile_count(
                        "wildgs.frontend.loop_closing.runs"
                    )
                    self._profile_value(
                        "wildgs.frontend.loop_closing.last_keyframes",
                        int(n_kf),
                    )
                    self._profile_value(
                        "wildgs.frontend.loop_closing.last_edges",
                        int(n_edge),
                    )

                    if n_edge == 0:
                        with self._profile_timer(
                            "wildgs.frontend.update.loop_fallback_optimization"
                        ):
                            for _ in range(self.iters2):
                                self.graph.update(
                                    t0=None,
                                    t1=None,
                                    use_inactive=True,
                                )

                                self._profile_count(
                                    "wildgs.frontend.graph_update_iterations"
                                )

                        self._profile_count(
                            "wildgs.frontend.loop_closing.zero_edge_fallbacks"
                        )

                    self.last_loop_t = cur_t

                # -------------------------------------------------
                # H. Normal second optimization pass
                # -------------------------------------------------
                else:
                    with self._profile_timer(
                        "wildgs.frontend.update.second_optimization_pass"
                    ):
                        for _ in range(self.iters2):
                            self.graph.update(
                                t0=None,
                                t1=None,
                                use_inactive=True,
                            )

                            self._profile_count(
                                "wildgs.frontend.graph_update_iterations"
                            )

            # -----------------------------------------------------
            # I. Initialize pose and disparity for the next slot
            # -----------------------------------------------------
            with self._profile_timer(
                "wildgs.frontend.update.propagate_pose_and_disparity"
            ):
                self.video.poses[
                    self.t1
                ] = self.video.poses[
                    self.t1 - 1
                ]

                self.video.disps[
                    self.t1
                ] = self.video.disps[
                    self.t1 - 1
                ].mean()

            # -----------------------------------------------------
            # J. Mark updated range for visualization
            # -----------------------------------------------------
            with self._profile_timer(
                "wildgs.frontend.update.mark_visualization_dirty"
            ):
                self.video.set_dirty(
                    self.graph.ii.min(),
                    self.t1,
                )

            # -----------------------------------------------------
            # K. Release unused cached CUDA memory
            # -----------------------------------------------------
            with self._profile_timer(
                "wildgs.frontend.update.empty_cuda_cache"
            ):
                torch.cuda.empty_cache()

        self._record_frontend_state()

    # =========================================================
    # 3. First-stage frontend initialization
    # =========================================================

    def __initialize(self):
        """
        Bootstrap the frontend factor graph after enough warmup keyframes.

        Profiling hierarchy:

        wildgs.frontend.initialization.total
        ├── set_window_end
        ├── add_neighborhood_factors
        ├── first_optimization_pass
        ├── add_proximity_factors
        ├── second_optimization_pass
        ├── initialize_next_pose_and_disparity
        ├── save_initial_state
        ├── mark_visualization_dirty
        └── remove_old_initialization_factors
        """
        self._profile_count(
            "wildgs.frontend.initializations"
        )

        with self._profile_timer(
            "wildgs.frontend.initialization.total"
        ):
            with self._profile_timer(
                "wildgs.frontend.initialization.set_window_end"
            ):
                self.t1 = int(
                    self.video.counter.value
                )

            with self._profile_timer(
                "wildgs.frontend.initialization.add_neighborhood_factors"
            ):
                self.graph.add_neighborhood_factors(
                    0,
                    self.t1,
                    r=3,
                )

            with self._profile_timer(
                "wildgs.frontend.initialization.first_optimization_pass"
            ):
                for _ in range(8):
                    self.graph.update(
                        1,
                        use_inactive=True,
                    )

                    self._profile_count(
                        "wildgs.frontend.graph_update_iterations"
                    )

            with self._profile_timer(
                "wildgs.frontend.initialization.add_proximity_factors"
            ):
                self.graph.add_proximity_factors(
                    0,
                    0,
                    rad=2,
                    nms=2,
                    thresh=self.frontend_thresh,
                    remove=False,
                )

            with self._profile_timer(
                "wildgs.frontend.initialization.second_optimization_pass"
            ):
                for _ in range(8):
                    self.graph.update(
                        1,
                        use_inactive=True,
                    )

                    self._profile_count(
                        "wildgs.frontend.graph_update_iterations"
                    )

            with self._profile_timer(
                "wildgs.frontend.initialization.initialize_next_pose_and_disparity"
            ):
                self.video.poses[
                    self.t1
                ] = self.video.poses[
                    self.t1 - 1
                ].clone()

                self.video.disps[
                    self.t1
                ] = self.video.disps[
                    self.t1 - 4:self.t1
                ].mean()

            with self._profile_timer(
                "wildgs.frontend.initialization.save_initial_state"
            ):
                self.is_initialized = True

                self.last_pose = (
                    self.video.poses[
                        self.t1 - 1
                    ].clone()
                )

                self.last_disp = (
                    self.video.disps[
                        self.t1 - 1
                    ].clone()
                )

                self.last_time = (
                    self.video.timestamp[
                        self.t1 - 1
                    ].clone()
                )

            with self._profile_timer(
                "wildgs.frontend.initialization.mark_visualization_dirty"
            ):
                with self.video.get_lock():
                    self.video.set_dirty(
                        0,
                        self.t1,
                    )

            with self._profile_timer(
                "wildgs.frontend.initialization.remove_old_initialization_factors"
            ):
                self.graph.rm_factors(
                    self.graph.ii
                    < self.warmup - 4,
                    store=True,
                )

        self._record_frontend_state()

    # =========================================================
    # 4. Second-stage initialization
    # =========================================================

    def initialize_second_stage(self):
        """
        Run the second frontend initialization stage after mapping provides
        reliable uncertainty masks.

        Profiling hierarchy:

        wildgs.frontend.second_stage.total
        ├── set_window_end
        ├── update_uncertainty_mask
        ├── add_proximity_factors
        ├── optimization_pass
        ├── reset_factor_ages
        ├── initialize_next_pose_and_disparity
        ├── save_initial_state
        ├── mark_visualization_dirty
        └── remove_old_initialization_factors
        """
        self._profile_count(
            "wildgs.frontend.second_stage.initializations"
        )

        with self._profile_timer(
            "wildgs.frontend.second_stage.total"
        ):
            with self._profile_timer(
                "wildgs.frontend.second_stage.set_window_end"
            ):
                self.t1 = int(
                    self.video.counter.value
                )

            if self.cfg["tracking"][
                "uncertainty_params"
            ]["activate"]:
                with self._profile_timer(
                    "wildgs.frontend.second_stage.update_uncertainty_mask"
                ):
                    self.video.update_all_uncertainty_mask()

                self._profile_count(
                    "wildgs.frontend.uncertainty_mask_updates"
                )

            with self._profile_timer(
                "wildgs.frontend.second_stage.add_proximity_factors"
            ):
                self.graph.add_proximity_factors(
                    0,
                    0,
                    rad=2,
                    nms=2,
                    thresh=self.frontend_thresh,
                    remove=False,
                )

            with self._profile_timer(
                "wildgs.frontend.second_stage.optimization_pass"
            ):
                for _ in range(8):
                    self.graph.update(
                        1,
                        use_inactive=True,
                    )

                    self._profile_count(
                        "wildgs.frontend.graph_update_iterations"
                    )

            with self._profile_timer(
                "wildgs.frontend.second_stage.reset_factor_ages"
            ):
                self.graph.age = torch.maximum(
                    self.graph.age - 8,
                    torch.tensor(
                        0,
                    ).to(
                        self.graph.age.device
                    ),
                )

            with self._profile_timer(
                "wildgs.frontend.second_stage.initialize_next_pose_and_disparity"
            ):
                self.video.poses[
                    self.t1
                ] = self.video.poses[
                    self.t1 - 1
                ].clone()

                self.video.disps[
                    self.t1
                ] = self.video.disps[
                    self.t1 - 4:self.t1
                ].mean()

            with self._profile_timer(
                "wildgs.frontend.second_stage.save_initial_state"
            ):
                self.is_initialized = True

                self.last_pose = (
                    self.video.poses[
                        self.t1 - 1
                    ].clone()
                )

                self.last_disp = (
                    self.video.disps[
                        self.t1 - 1
                    ].clone()
                )

                self.last_time = (
                    self.video.timestamp[
                        self.t1 - 1
                    ].clone()
                )

            with self._profile_timer(
                "wildgs.frontend.second_stage.mark_visualization_dirty"
            ):
                with self.video.get_lock():
                    self.video.set_dirty(
                        0,
                        self.t1,
                    )

            with self._profile_timer(
                "wildgs.frontend.second_stage.remove_old_initialization_factors"
            ):
                self.graph.rm_factors(
                    self.graph.ii
                    < self.warmup - 4,
                    store=True,
                )

        self._record_frontend_state()

    # =========================================================
    # 5. Public frontend entry point
    # =========================================================

    def __call__(
        self,
        force_to_add_keyframe,
    ):
        """
        Main Frontend entry point called after every MotionFilter execution.

        Possible outcomes:
        - initialize the frontend when warmup is reached
        - update uncertainty and run one normal frontend update
        - do nothing when no new keyframe is available

        Profiling hierarchy:

        wildgs.frontend.call.total
        ├── initialization.total
        ├── update_uncertainty_mask
        ├── update.total
        └── update_valid_depth_mask
        """
        self._profile_count(
            "wildgs.frontend.calls"
        )
        self._profile_value(
            "wildgs.frontend.last_force_keyframe",
            bool(force_to_add_keyframe),
        )

        with self._profile_timer(
            "wildgs.frontend.call.total"
        ):
            did_work = False

            # -----------------------------------------------------
            # A. First-stage initialization
            # -----------------------------------------------------
            if (
                not self.is_initialized
                and self.video.counter.value
                == self.warmup
            ):
                self.__initialize()

                with self._profile_timer(
                    "wildgs.frontend.call.update_valid_depth_mask"
                ):
                    self.video.update_valid_depth_mask()

                self._profile_count(
                    "wildgs.frontend.valid_depth_mask_updates"
                )

                did_work = True

            # -----------------------------------------------------
            # B. Normal update for a newly inserted keyframe
            # -----------------------------------------------------
            elif (
                self.is_initialized
                and self.t1
                < self.video.counter.value
            ):
                if self.cfg["tracking"][
                    "uncertainty_params"
                ]["activate"]:
                    with self._profile_timer(
                        "wildgs.frontend.call.update_uncertainty_mask"
                    ):
                        self.video.update_all_uncertainty_mask()

                    self._profile_count(
                        "wildgs.frontend.uncertainty_mask_updates"
                    )

                self.__update(
                    force_to_add_keyframe
                )

                with self._profile_timer(
                    "wildgs.frontend.call.update_valid_depth_mask"
                ):
                    self.video.update_valid_depth_mask()

                self._profile_count(
                    "wildgs.frontend.valid_depth_mask_updates"
                )

                did_work = True

            # -----------------------------------------------------
            # C. No new keyframe, so Frontend remains idle
            # -----------------------------------------------------
            if not did_work:
                self._profile_count(
                    "wildgs.frontend.idle_calls"
                )

        self._record_frontend_state()