from contextlib import nullcontext

import torch
import numpy as np

from src.modules.droid_net import CorrBlock, AltCorrBlock
import src.geom.projective_ops as pops
from copy import deepcopy


class FactorGraph:
    # mainly inherited from GO-SLAM
    def __init__(self, video, update_op, device="cuda:0", corr_impl="volume", max_factors=-1):
        self.video = video
        self.update_op = update_op
        self.device = device
        self.max_factors = max_factors
        self.corr_impl = corr_impl

        # The Navigator wrapper attaches the shared SimpleProfiler.
        self.profiler = None

        # operator at 1/8 resolution
        self.ht = ht = video.ht // self.video.down_scale
        self.wd = wd = video.wd // self.video.down_scale

        self.coords0 = pops.coords_grid(ht, wd, device=device)
        self.ii = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj = torch.as_tensor([], dtype=torch.long, device=device)
        self.age = torch.as_tensor([], dtype=torch.long, device=device)

        self.corr, self.net, self.inp = None, None, None
        self.damping = 1e-6 * torch.ones_like(self.video.disps)

        self.target = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)

        # inactive factors
        self.ii_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.ii_bad = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj_bad = torch.as_tensor([], dtype=torch.long, device=device)

        self.target_inac = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight_inac = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)

    # =========================================================
    # SimpleProfiler helpers
    # =========================================================

    def _profile_timer(self, name: str):
        """
        Return a profiler timer when profiling is available.

        Without a profiler, nullcontext() does nothing and the original
        WildGS execution continues normally.
        """
        profiler = getattr(self, "profiler", None)

        if profiler is None:
            return nullcontext()

        return profiler.timer(name)

    def _profile_count(
            self,
            name: str,
            amount: int = 1,
    ) -> None:
        """Add an event or quantity to a profiler counter."""
        profiler = getattr(self, "profiler", None)

        if profiler is None:
            return

        profiler.count(name, amount)

    def _profile_value(
            self,
            name: str,
            value,
    ) -> None:
        """Store the latest value for one profiler field."""
        profiler = getattr(self, "profiler", None)

        if profiler is None:
            return

        profiler.value(name, value)

    def _record_graph_state(self) -> None:
        """Store the latest readable state of the factor graph."""
        active_factors = (
            0
            if self.ii is None
            else int(self.ii.numel())
        )
        inactive_factors = (
            0
            if self.ii_inac is None
            else int(self.ii_inac.numel())
        )
        bad_factors = (
            0
            if self.ii_bad is None
            else int(self.ii_bad.numel())
        )

        self._profile_value(
            "wildgs.factor_graph.state.active_factors",
            active_factors,
        )
        self._profile_value(
            "wildgs.factor_graph.state.inactive_factors",
            inactive_factors,
        )
        self._profile_value(
            "wildgs.factor_graph.state.bad_factors",
            bad_factors,
        )

    def __filter_repeated_edges(self, ii, jj):
        """ remove duplicate edges """

        keep = torch.zeros(ii.shape[0], dtype=torch.bool, device=ii.device)
        eset = set(
            [(i.item(), j.item()) for i, j in zip(self.ii, self.jj)] +
            [(i.item(), j.item()) for i, j in zip(self.ii_inac, self.jj_inac)])

        for k, (i, j) in enumerate(zip(ii, jj)):
            keep[k] = (i.item(), j.item()) not in eset

        return ii[keep], jj[keep]

    def print_edges(self):
        ii = self.ii.cpu().numpy()
        jj = self.jj.cpu().numpy()

        ix = np.argsort(ii)
        ii = ii[ix]
        jj = jj[ix]

        w = torch.mean(self.weight, dim=[0, 2, 3, 4]).cpu().numpy()
        w = w[ix]
        for e in zip(ii, jj, w):
            print(e)
        print()

    def filter_edges(self):
        """Remove low-confidence long-range edges."""
        with self._profile_timer(
                "wildgs.factor_graph.filter_edges.total"
        ):
            confidence = torch.mean(
                self.weight,
                dim=[0, 2, 3, 4],
            )

            mask = (
                    (torch.abs(self.ii - self.jj) > 2)
                    & (confidence < 0.001)
            )

            removed_edges = int(mask.sum().item())

            self.ii_bad = torch.cat(
                [
                    self.ii_bad,
                    self.ii[mask],
                ]
            )
            self.jj_bad = torch.cat(
                [
                    self.jj_bad,
                    self.jj[mask],
                ]
            )

            self.rm_factors(
                mask,
                store=False,
            )

        if removed_edges > 0:
            self._profile_count(
                "wildgs.factor_graph.edges.bad_removed",
                removed_edges,
            )

        self._record_graph_state()

    def clear_edges(self):
        """Release all graph tensors and correlation state."""
        with self._profile_timer(
                "wildgs.factor_graph.clear_edges.total"
        ):
            self.ii = None
            self.jj = None
            self.age = None
            self.corr = None
            self.damping = None
            self.net = None
            self.inp = None
            self.target = None
            self.weight = None
            self.ii_inac = None
            self.jj_inac = None
            self.ii_bad = None
            self.jj_bad = None
            self.target_inac = None
            self.weight_inac = None

    @torch.amp.autocast(
        "cuda",
        enabled=True,
    )
    @torch.no_grad()
    def add_factors(
            self,
            ii,
            jj,
            remove=False,
    ):
        """
        Add directed edges to the factor graph.

        The outer timer measures the complete method. The nested timers
        explain which internal operation consumed the time.
        """
        with self._profile_timer(
                "wildgs.factor_graph.add_factors.total"
        ):
            with self._profile_timer(
                    "wildgs.factor_graph.add_factors.prepare_indices"
            ):
                if not isinstance(ii, torch.Tensor):
                    ii = torch.as_tensor(
                        ii,
                        dtype=torch.long,
                        device=self.device,
                    )

                if not isinstance(jj, torch.Tensor):
                    jj = torch.as_tensor(
                        jj,
                        dtype=torch.long,
                        device=self.device,
                    )

                requested_edges = int(ii.shape[0])

            with self._profile_timer(
                    "wildgs.factor_graph.add_factors.filter_repeated"
            ):
                ii, jj = self.__filter_repeated_edges(
                    ii,
                    jj,
                )

            added_edges = int(ii.shape[0])
            duplicate_edges = max(
                0,
                requested_edges - added_edges,
            )

            if duplicate_edges > 0:
                self._profile_count(
                    "wildgs.factor_graph.edges.duplicates_filtered",
                    duplicate_edges,
                )

            if added_edges == 0:
                self._profile_count(
                    "wildgs.factor_graph.add_factors.empty_after_filter"
                )
                return

            if (
                    self.max_factors > 0
                    and self.ii.shape[0] + ii.shape[0] > self.max_factors
                    and self.corr is not None
                    and remove
            ):
                with self._profile_timer(
                        "wildgs.factor_graph.add_factors.enforce_factor_limit"
                ):
                    index = torch.arange(
                        len(self.age)
                    )[
                        torch.argsort(self.age).cpu()
                    ]

                    self.rm_factors(
                        index >= self.max_factors - ii.shape[0],
                        store=True,
                    )

            with self._profile_timer(
                    "wildgs.factor_graph.add_factors.load_network_state"
            ):
                net = (
                    self.video.nets[ii]
                    .to(self.device)
                    .unsqueeze(0)
                )

            if self.corr_impl == "volume":
                with self._profile_timer(
                        "wildgs.factor_graph.add_factors.build_correlation_volume"
                ):
                    camera_index = (ii == jj).long()

                    fmap1 = (
                        self.video.fmaps[ii, 0]
                        .to(self.device)
                        .unsqueeze(0)
                    )
                    fmap2 = (
                        self.video.fmaps[jj, camera_index]
                        .to(self.device)
                        .unsqueeze(0)
                    )

                    corr = CorrBlock(
                        fmap1,
                        fmap2,
                    )

                    self.corr = (
                        corr
                        if self.corr is None
                        else self.corr.cat(corr)
                    )

                    inp = (
                        self.video.inps[ii]
                        .to(self.device)
                        .unsqueeze(0)
                    )

                    self.inp = (
                        inp
                        if self.inp is None
                        else torch.cat(
                            [
                                self.inp,
                                inp,
                            ],
                            1,
                        )
                    )

            with self._profile_timer(
                    "wildgs.factor_graph.add_factors.reproject"
            ):
                with torch.amp.autocast(
                        "cuda",
                        enabled=False,
                ):
                    target, _ = self.video.reproject(
                        ii,
                        jj,
                    )
                    weight = torch.zeros_like(target)

            with self._profile_timer(
                    "wildgs.factor_graph.add_factors.append_graph_state"
            ):
                self.ii = torch.cat(
                    [
                        self.ii,
                        ii,
                    ],
                    0,
                )
                self.jj = torch.cat(
                    [
                        self.jj,
                        jj,
                    ],
                    0,
                )
                self.age = torch.cat(
                    [
                        self.age,
                        torch.zeros_like(ii),
                    ],
                    0,
                )

                self.net = (
                    net
                    if self.net is None
                    else torch.cat(
                        [
                            self.net,
                            net,
                        ],
                        1,
                    )
                )

                self.target = torch.cat(
                    [
                        self.target,
                        target,
                    ],
                    1,
                )
                self.weight = torch.cat(
                    [
                        self.weight,
                        weight,
                    ],
                    1,
                )

        self._profile_count(
            "wildgs.factor_graph.edges.added",
            added_edges,
        )
        self._profile_value(
            "wildgs.factor_graph.last_edges_requested",
            requested_edges,
        )
        self._profile_value(
            "wildgs.factor_graph.last_edges_added",
            added_edges,
        )
        self._record_graph_state()

    @torch.amp.autocast(
        "cuda",
        enabled=True,
    )
    def rm_factors(
            self,
            mask,
            store=False,
    ):
        """Remove selected edges from the active factor graph."""
        removed_edges = int(mask.sum().item())

        with self._profile_timer(
                "wildgs.factor_graph.remove_factors.total"
        ):
            if store:
                with self._profile_timer(
                        "wildgs.factor_graph.remove_factors.store_inactive"
                ):
                    self.ii_inac = torch.cat(
                        [
                            self.ii_inac,
                            self.ii[mask],
                        ],
                        0,
                    )
                    self.jj_inac = torch.cat(
                        [
                            self.jj_inac,
                            self.jj[mask],
                        ],
                        0,
                    )
                    self.target_inac = torch.cat(
                        [
                            self.target_inac,
                            self.target[:, mask],
                        ],
                        1,
                    )
                    self.weight_inac = torch.cat(
                        [
                            self.weight_inac,
                            self.weight[:, mask],
                        ],
                        1,
                    )

            with self._profile_timer(
                    "wildgs.factor_graph.remove_factors.apply_mask"
            ):
                self.ii = self.ii[~mask]
                self.jj = self.jj[~mask]
                self.age = self.age[~mask]

                if self.corr_impl == "volume":
                    self.corr = self.corr[~mask]

                if self.net is not None:
                    self.net = self.net[:, ~mask]

                if self.inp is not None:
                    self.inp = self.inp[:, ~mask]

                self.target = self.target[:, ~mask]
                self.weight = self.weight[:, ~mask]

        if removed_edges > 0:
            self._profile_count(
                "wildgs.factor_graph.edges.removed",
                removed_edges,
            )

            if store:
                self._profile_count(
                    "wildgs.factor_graph.edges.moved_to_inactive",
                    removed_edges,
                )

        self._profile_value(
            "wildgs.factor_graph.last_edges_removed",
            removed_edges,
        )
        self._record_graph_state()

    @torch.amp.autocast(
        "cuda",
        enabled=True,
    )
    def rm_keyframe(self, ix):
        """Remove one keyframe and update its graph edges."""
        with self._profile_timer(
                "wildgs.factor_graph.remove_keyframe.total"
        ):
            with self._profile_timer(
                    "wildgs.factor_graph.remove_keyframe.shift_video_state"
            ):
                with self.video.get_lock():
                    self.video.timestamp[ix] = self.video.timestamp[ix + 1]
                    self.video.images[ix] = self.video.images[ix + 1]
                    self.video.dirty[ix] = self.video.dirty[ix + 1]
                    self.video.npc_dirty[ix] = self.video.npc_dirty[ix + 1]
                    self.video.poses[ix] = self.video.poses[ix + 1]
                    self.video.disps[ix] = self.video.disps[ix + 1]
                    self.video.disps_up[ix] = self.video.disps_up[ix + 1]
                    self.video.intrinsics[ix] = self.video.intrinsics[ix + 1]
                    self.video.depth_scale[ix] = self.video.depth_scale[ix + 1]
                    self.video.depth_shift[ix] = self.video.depth_shift[ix + 1]
                    self.video.mono_disps[ix] = self.video.mono_disps[ix + 1]
                    self.video.mono_disps_up[ix] = self.video.mono_disps_up[ix + 1]
                    self.video.mono_disps_mask_up[ix] = (
                        self.video.mono_disps_mask_up[ix + 1]
                    )
                    self.video.valid_depth_mask[ix] = (
                        self.video.valid_depth_mask[ix + 1]
                    )
                    self.video.valid_depth_mask_small[ix] = (
                        self.video.valid_depth_mask_small[ix + 1]
                    )

                    self.video.nets[ix] = self.video.nets[ix + 1]
                    self.video.inps[ix] = self.video.inps[ix + 1]
                    self.video.fmaps[ix] = self.video.fmaps[ix + 1]

                    if self.video.uncertainty_aware:
                        self.video.dino_feats[ix] = (
                            self.video.dino_feats[ix + 1]
                        )
                        self.video.uncertainties_inv[ix] = (
                            self.video.uncertainties_inv[ix + 1]
                        )

            with self._profile_timer(
                    "wildgs.factor_graph.remove_keyframe.update_inactive_edges"
            ):
                inactive_mask = (
                        (self.ii_inac == ix)
                        | (self.jj_inac == ix)
                )

                self.ii_inac[self.ii_inac >= ix] -= 1
                self.jj_inac[self.jj_inac >= ix] -= 1

                if torch.any(inactive_mask):
                    self.ii_inac = self.ii_inac[~inactive_mask]
                    self.jj_inac = self.jj_inac[~inactive_mask]
                    self.target_inac = self.target_inac[:, ~inactive_mask]
                    self.weight_inac = self.weight_inac[:, ~inactive_mask]

            with self._profile_timer(
                    "wildgs.factor_graph.remove_keyframe.update_active_edges"
            ):
                active_mask = (
                        (self.ii == ix)
                        | (self.jj == ix)
                )

                self.ii[self.ii >= ix] -= 1
                self.jj[self.jj >= ix] -= 1

                self.rm_factors(
                    active_mask,
                    store=False,
                )

        self._profile_count(
            "wildgs.factor_graph.keyframes.removed"
        )
        self._profile_value(
            "wildgs.factor_graph.last_removed_keyframe",
            int(ix),
        )
        self._record_graph_state()

    @torch.amp.autocast(
        "cuda",
        enabled=True,
    )
    @torch.no_grad()
    def update(
            self,
            t0=None,
            t1=None,
            itrs=2,
            use_inactive=False,
            EP=1e-7,
            motion_only=False,
    ):
        """Run the neural graph update and bundle adjustment."""
        with self._profile_timer(
                "wildgs.factor_graph.update.total"
        ):
            with self._profile_timer(
                    "wildgs.factor_graph.update.motion_features"
            ):
                with torch.amp.autocast(
                        "cuda",
                        enabled=False,
                ):
                    coords1, mask = self.video.reproject(
                        self.ii,
                        self.jj,
                    )

                    motn = torch.cat(
                        [
                            coords1 - self.coords0,
                            self.target - coords1,
                        ],
                        dim=-1,
                    )

                    motn = (
                        motn
                        .permute(0, 1, 4, 2, 3)
                        .clamp(-64.0, 64.0)
                    )

                del mask

            with self._profile_timer(
                    "wildgs.factor_graph.update.correlation_lookup"
            ):
                corr = self.corr(coords1)

            with self._profile_timer(
                    "wildgs.factor_graph.update.neural_update_operator"
            ):
                (
                    self.net,
                    delta,
                    weight,
                    damping,
                    upmask,
                ) = self.update_op(
                    self.net,
                    self.inp,
                    corr,
                    motn,
                    self.ii,
                    self.jj,
                )

            with self._profile_timer(
                    "wildgs.factor_graph.update.resolve_optimization_window"
            ):
                if t0 is None:
                    t0 = max(
                        1,
                        self.ii.min().item() + 1,
                    )

            with torch.amp.autocast(
                    "cuda",
                    enabled=False,
            ):
                with self._profile_timer(
                        "wildgs.factor_graph.update.prepare_ba_inputs"
                ):
                    self.target = (
                            coords1
                            + delta.to(dtype=torch.float)
                    )
                    self.weight = weight.to(
                        dtype=torch.float
                    )

                    self.damping[
                        torch.unique(self.ii)
                    ] = damping

                    if use_inactive:
                        inactive_mask = (
                                (self.ii_inac >= t0 - 3)
                                & (self.jj_inac >= t0 - 3)
                        )

                        ii = torch.cat(
                            [
                                self.ii_inac[inactive_mask],
                                self.ii,
                            ],
                            0,
                        )
                        jj = torch.cat(
                            [
                                self.jj_inac[inactive_mask],
                                self.jj,
                            ],
                            0,
                        )
                        target = torch.cat(
                            [
                                self.target_inac[:, inactive_mask],
                                self.target,
                            ],
                            1,
                        )
                        weight = torch.cat(
                            [
                                self.weight_inac[:, inactive_mask],
                                self.weight,
                            ],
                            1,
                        )

                    else:
                        ii = self.ii
                        jj = self.jj
                        target = self.target
                        weight = self.weight

                    damping = (
                            0.2
                            * self.damping[
                                torch.unique(ii)
                            ].contiguous()
                            + EP
                    )

                with self._profile_timer(
                        "wildgs.factor_graph.update.bundle_adjustment"
                ):
                    self.video.ba(
                        target,
                        weight,
                        damping,
                        ii,
                        jj,
                        t0,
                        t1,
                        iters=itrs,
                        lm=1e-4,
                        ep=0.1,
                        motion_only=motion_only,
                    )

                with self._profile_timer(
                        "wildgs.factor_graph.update.upsample"
                ):
                    self.video.upsample(
                        torch.unique(self.ii),
                        upmask,
                    )

            with self._profile_timer(
                    "wildgs.factor_graph.update.increment_factor_age"
            ):
                self.age += 1

        self._profile_count(
            "wildgs.factor_graph.ba_iterations_requested",
            int(itrs),
        )
        self._profile_value(
            "wildgs.factor_graph.update.last_t0",
            int(t0),
        )
        self._profile_value(
            "wildgs.factor_graph.update.last_t1",
            None if t1 is None else int(t1),
        )
        self._profile_value(
            "wildgs.factor_graph.update.last_factors_used",
            int(ii.numel()),
        )
        self._profile_value(
            "wildgs.factor_graph.update.last_use_inactive",
            bool(use_inactive),
        )
        self._profile_value(
            "wildgs.factor_graph.update.last_motion_only",
            bool(motion_only),
        )
        self._record_graph_state()

    @torch.amp.autocast(
        "cuda",
        enabled=False,
    )
    @torch.no_grad()
    def update_lowmem(
            self,
            t0=None,
            t1=None,
            itrs=2,
            use_inactive=False,
            EP=1e-7,
            steps=8,
            enable_wq=True,
    ):
        """Run the reduced-memory factor-graph implementation."""
        with self._profile_timer(
                "wildgs.factor_graph.update_lowmem.total"
        ):
            t = self.video.counter.value

            with self._profile_timer(
                    "wildgs.factor_graph.update_lowmem.build_alt_correlation"
            ):
                (
                    num,
                    rig,
                    channels,
                    height,
                    width,
                ) = self.video.fmaps.shape

                corr_op = AltCorrBlock(
                    self.video.fmaps.view(
                        1,
                        num * rig,
                        channels,
                        height,
                        width,
                    )
                )

            for _ in range(steps):
                self._profile_count(
                    "wildgs.factor_graph.update_lowmem.steps"
                )

                with self._profile_timer(
                        "wildgs.factor_graph.update_lowmem.motion_features"
                ):
                    with torch.amp.autocast(
                            "cuda",
                            enabled=False,
                    ):
                        coords1, mask = self.video.reproject(
                            self.ii,
                            self.jj,
                        )
                        motn = torch.cat(
                            [
                                coords1 - self.coords0,
                                self.target - coords1,
                            ],
                            dim=-1,
                        )
                        motn = (
                            motn
                            .permute(0, 1, 4, 2, 3)
                            .clamp(-64.0, 64.0)
                        )

                    del mask

                chunk_size = 8

                for index in range(
                        0,
                        self.jj.max() + 1,
                        chunk_size,
                ):
                    selected = (
                            (self.ii >= index)
                            & (self.ii < index + chunk_size)
                    )

                    if selected.sum() < 1:
                        continue

                    self._profile_count(
                        "wildgs.factor_graph.update_lowmem.chunks"
                    )

                    iis = self.ii[selected]
                    jjs = self.jj[selected]

                    height, width = self.coords0.shape[0:2]

                    with self._profile_timer(
                            "wildgs.factor_graph.update_lowmem.correlation_lookup"
                    ):
                        corr1 = corr_op(
                            coords1[:, selected],
                            rig * iis,
                            rig * jjs + (iis == jjs).long(),
                        )

                    with torch.amp.autocast(
                            "cuda",
                            enabled=True,
                    ):
                        with self._profile_timer(
                                "wildgs.factor_graph.update_lowmem.neural_update_operator"
                        ):
                            (
                                net,
                                delta,
                                weight,
                                damping,
                                upmask,
                            ) = self.update_op(
                                self.net[:, selected],
                                self.video.inps[None, iis],
                                corr1,
                                motn[:, selected],
                                iis,
                                jjs,
                            )

                        with self._profile_timer(
                                "wildgs.factor_graph.update_lowmem.upsample"
                        ):
                            self.video.upsample(
                                torch.unique(iis),
                                upmask,
                            )

                    with self._profile_timer(
                            "wildgs.factor_graph.update_lowmem.write_back_state"
                    ):
                        self.net[:, selected] = net
                        self.target[:, selected] = (
                                coords1[:, selected]
                                + delta.float()
                        )
                        self.weight[:, selected] = weight.float()
                        self.damping[
                            torch.unique(iis)
                        ] = damping

                damping = (
                        0.2
                        * self.damping[
                            torch.unique(self.ii)
                        ].contiguous()
                        + EP
                )

                target = self.target
                weight = self.weight

                with self._profile_timer(
                        "wildgs.factor_graph.update_lowmem.dense_bundle_adjustment"
                ):
                    self.video.ba(
                        target,
                        weight,
                        damping,
                        self.ii,
                        self.jj,
                        t0,
                        t1,
                        iters=itrs,
                        lm=1e-5,
                        ep=1e-2,
                        motion_only=False,
                    )

        self._profile_count(
            "wildgs.factor_graph.update_lowmem.ba_iterations_requested",
            int(itrs) * int(steps),
        )
        self._profile_value(
            "wildgs.factor_graph.update_lowmem.last_steps",
            int(steps),
        )
        self._profile_value(
            "wildgs.factor_graph.update_lowmem.last_video_counter",
            int(t),
        )
        self._record_graph_state()

    def add_neighborhood_factors(
            self,
            t0,
            t1,
            r=3,
    ):
        """Add edges between neighboring frames within radius r."""
        with self._profile_timer(
                "wildgs.factor_graph.add_neighborhood_factors.total"
        ):
            ii, jj = torch.meshgrid(
                torch.arange(t0, t1),
                torch.arange(t0, t1),
                indexing="ij",
            )

            ii = (
                ii.reshape(-1)
                .to(
                    dtype=torch.long,
                    device=self.device,
                )
            )
            jj = (
                jj.reshape(-1)
                .to(
                    dtype=torch.long,
                    device=self.device,
                )
            )

            keep = (
                    ((ii - jj).abs() > 0)
                    & ((ii - jj).abs() <= r)
            )

            requested_edges = int(
                keep.sum().item()
            )

            self.add_factors(
                ii[keep],
                jj[keep],
            )

        self._profile_count(
            "wildgs.factor_graph.neighborhood_edges_requested",
            requested_edges,
        )
        self._profile_value(
            "wildgs.factor_graph.last_neighborhood_radius",
            int(r),
        )

    def add_proximity_factors(
            self,
            t0=0,
            t1=0,
            rad=2,
            nms=2,
            beta=0.25,
            thresh=16.0,
            remove=False,
    ):
        """Add edges to the factor graph based on frame distance."""
        with self._profile_timer(
                "wildgs.factor_graph.add_proximity_factors.total"
        ):
            t = self.video.counter.value

            with self._profile_timer(
                    "wildgs.factor_graph.add_proximity_factors.build_candidates"
            ):
                ix = torch.arange(t0, t)
                jx = torch.arange(t1, t)

                ii, jj = torch.meshgrid(
                    ix,
                    jx,
                    indexing="ij",
                )
                ii = ii.reshape(-1)
                jj = jj.reshape(-1)

            with self._profile_timer(
                    "wildgs.factor_graph.add_proximity_factors.distance_matrix"
            ):
                distance = self.video.distance(
                    ii,
                    jj,
                    beta=beta,
                )

                distance[ii - rad < jj] = np.inf
                distance[distance > 100] = np.inf

            with self._profile_timer(
                    "wildgs.factor_graph.add_proximity_factors.suppress_existing_edges"
            ):
                existing_ii = torch.cat(
                    [
                        self.ii,
                        self.ii_bad,
                        self.ii_inac,
                    ],
                    0,
                )
                existing_jj = torch.cat(
                    [
                        self.jj,
                        self.jj_bad,
                        self.jj_inac,
                    ],
                    0,
                )

                for i, j in zip(
                        existing_ii.cpu().numpy(),
                        existing_jj.cpu().numpy(),
                ):
                    for di in range(-nms, nms + 1):
                        for dj in range(-nms, nms + 1):
                            if (
                                    abs(di) + abs(dj)
                                    <= max(
                                min(abs(i - j) - 2, nms),
                                0,
                            )
                            ):
                                i1 = i + di
                                j1 = j + dj

                                if (
                                        t0 <= i1 < t
                                        and t1 <= j1 < t
                                ):
                                    distance[
                                        (i1 - t0) * (t - t1)
                                        + (j1 - t1)
                                        ] = np.inf

            edges = []

            with self._profile_timer(
                    "wildgs.factor_graph.add_proximity_factors.add_local_edges"
            ):
                for i in range(t0, t):
                    for j in range(
                            max(i - rad - 1, 0),
                            i,
                    ):
                        edges.append((i, j))
                        edges.append((j, i))

                        distance[
                            (i - t0) * (t - t1)
                            + (j - t1)
                            ] = np.inf

            with self._profile_timer(
                    "wildgs.factor_graph.add_proximity_factors.select_proximity_edges"
            ):
                sorted_indices = torch.argsort(
                    distance
                )

                for candidate_index in sorted_indices:
                    if (
                            distance[candidate_index].item()
                            > thresh
                    ):
                        continue

                    if len(edges) > self.max_factors:
                        break

                    i = ii[candidate_index]
                    j = jj[candidate_index]

                    edges.append((i, j))
                    edges.append((j, i))

                    for di in range(-nms, nms + 1):
                        for dj in range(-nms, nms + 1):
                            if (
                                    abs(di) + abs(dj)
                                    <= max(
                                min(abs(i - j) - 2, nms),
                                0,
                            )
                            ):
                                i1 = i + di
                                j1 = j + dj

                                if (
                                        t0 <= i1 < t
                                        and t1 <= j1 < t
                                ):
                                    distance[
                                        (i1 - t0) * (t - t1)
                                        + (j1 - t1)
                                        ] = np.inf

            requested_edges = int(len(edges))

            with self._profile_timer(
                    "wildgs.factor_graph.add_proximity_factors.add_factors"
            ):
                ii, jj = torch.as_tensor(
                    edges,
                    device=self.device,
                ).unbind(dim=-1)

                self.add_factors(
                    ii,
                    jj,
                    remove,
                )

        self._profile_count(
            "wildgs.factor_graph.proximity_edges_requested",
            requested_edges,
        )
        self._profile_value(
            "wildgs.factor_graph.last_proximity_radius",
            int(rad),
        )
        self._profile_value(
            "wildgs.factor_graph.last_proximity_threshold",
            float(thresh),
        )

    def add_backend_proximity_factors(
            self,
            t_start,
            t_end,
            nms,
            radius,
            thresh,
            max_factors,
            beta,
            t_start_loop=None,
            loop=False,
    ):
        """Add backend or loop-closing proximity factors."""
        with self._profile_timer(
                "wildgs.factor_graph.add_backend_proximity_factors.total"
        ):
            if t_start_loop is None or not loop:
                t_start_loop = t_start

            assert t_start_loop >= t_start, (
                f"short: {t_start_loop}, "
                f"long: {t_start}."
            )

            i_length = t_end - t_start_loop
            j_length = t_end - t_start

            with self._profile_timer(
                    "wildgs.factor_graph.add_backend_proximity_factors.build_candidates"
            ):
                ix = torch.arange(
                    t_start_loop,
                    t_end,
                )
                jx = torch.arange(
                    t_start,
                    t_end,
                )

                ii, jj = torch.meshgrid(
                    ix,
                    jx,
                    indexing="ij",
                )
                ii = ii.reshape(-1)
                jj = jj.reshape(-1)

            with self._profile_timer(
                    "wildgs.factor_graph.add_backend_proximity_factors.distance_matrix"
            ):
                distance = self.video.distance(
                    ii,
                    jj,
                    beta=beta,
                )

                raw_distance = deepcopy(
                    distance
                ).reshape(
                    i_length,
                    j_length,
                )

                distance[ii - radius < jj] = np.inf
                distance[distance > thresh] = np.inf
                distance = distance.reshape(
                    i_length,
                    j_length,
                )

            edges = []

            with self._profile_timer(
                    "wildgs.factor_graph.add_backend_proximity_factors.add_local_edges"
            ):
                for i in range(
                        t_start_loop,
                        t_end,
                ):
                    for j in range(
                            max(i - radius - 1, 0),
                            i,
                    ):
                        edges.append((i, j))
                        edges.append((j, i))

                        di = i - t_start_loop
                        dj = j - t_start

                        distance[di, dj] = np.inf

            with self._profile_timer(
                    "wildgs.factor_graph.add_backend_proximity_factors.select_edges"
            ):
                values, sorted_indices = torch.sort(
                    distance.reshape(-1),
                    descending=False,
                )

                sorted_indices = sorted_indices[
                    values <= thresh
                    ].tolist()

                loop_edges = 0
                neighboring_radius = 1

                for candidate_index in sorted_indices:
                    di = candidate_index // j_length
                    dj = candidate_index % j_length

                    if distance[di, dj].item() > thresh:
                        continue

                    if len(edges) > max_factors:
                        break

                    i = ii[candidate_index]
                    j = jj[candidate_index]

                    if loop:
                        sub_edges = []
                        num_loop = 0

                        for source_id in range(
                                max(
                                    i - neighboring_radius,
                                    t_start_loop,
                                ),
                                min(
                                    i + neighboring_radius + 1,
                                    t_end,
                                ),
                        ):
                            for target_id in range(
                                    max(
                                        j - neighboring_radius,
                                        t_start,
                                    ),
                                    min(
                                        j + neighboring_radius + 1,
                                        t_end,
                                    ),
                            ):
                                if (
                                        raw_distance[
                                            source_id - t_start_loop,
                                            target_id - t_start,
                                        ]
                                        <= thresh
                                ):
                                    num_loop += 1

                                    if (
                                            source_id != target_id
                                            and source_id - target_id > 20
                                    ):
                                        sub_edges += [
                                            (
                                                source_id,
                                                target_id,
                                            )
                                        ]

                        edges += sub_edges
                        loop_edges += len(sub_edges)

                    else:
                        edges += [(i, j)]
                        edges += [(j, i)]

                    distance[
                        max(0, di - nms):
                        min(i_length, di + nms + 1),
                        max(0, dj - nms):
                        min(j_length, dj + nms + 1),
                    ] = np.inf

            if (
                    len(edges) < 3
                    or (loop and loop_edges == 0)
            ):
                self._profile_count(
                    "wildgs.factor_graph.backend_edge_builds_without_result"
                )
                return 0

            requested_edges = int(len(edges))

            with self._profile_timer(
                    "wildgs.factor_graph.add_backend_proximity_factors.add_factors"
            ):
                ii, jj = torch.tensor(
                    edges,
                    device=self.device,
                ).unbind(dim=-1)

                self.add_factors(
                    ii,
                    jj,
                    remove=True,
                )

                edge_num = len(self.ii)

        self._profile_count(
            "wildgs.factor_graph.backend_edges_requested",
            requested_edges,
        )

        if loop_edges > 0:
            self._profile_count(
                "wildgs.factor_graph.loop_edges_requested",
                int(loop_edges),
            )

        self._profile_value(
            "wildgs.factor_graph.last_backend_edge_count",
            int(edge_num),
        )
        self._profile_value(
            "wildgs.factor_graph.last_backend_loop_mode",
            bool(loop),
        )

        return edge_num