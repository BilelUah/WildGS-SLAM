# Copyright 2024 The GlORIE-SLAM Authors.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import nullcontext

import torch
from src.factor_graph import FactorGraph
from copy import deepcopy


class Backend:
    def __init__(self, net, video, cfg):
        self.cfg = cfg

        # The Navigator wrapper attaches the shared SimpleProfiler
        # after creating this original WildGS Backend object.
        self.profiler = getattr(
            video,
            "profiler",
            None,
        )
        self.video = video
        self.update_op = net.update
        self.device = cfg['device']
        # global optimization window
        self.t0 = 0
        self.t1 = 0

        self.beta = cfg['tracking']['beta']
        self.backend_thresh = cfg['tracking']['backend']['thresh']
        self.backend_radius = cfg['tracking']['backend']['radius']
        self.backend_nms = cfg['tracking']['backend']['nms']
        self.backend_normalize = cfg['tracking']['backend']['normalize']
        self.output = f"{cfg['data']['output']}/{cfg['scene']}"

        self.backend_loop_window = cfg['tracking']['backend']['loop_window']
        self.backend_loop_thresh = cfg['tracking']['backend']['loop_thresh']
        self.backend_loop_radius = cfg['tracking']['backend']['loop_radius']
        self.backend_loop_nms = cfg['tracking']['backend']['loop_nms']

    # =========================================================
    # SimpleProfiler helpers
    # =========================================================

    def _profile_timer(self, name: str):
        """
        Return a real profiler timer when one is attached.

        Otherwise, nullcontext() does nothing and the original Backend
        operation runs normally.
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
        """Accumulate an event or quantity."""
        profiler = getattr(
            self,
            "profiler",
            None,
        )

        if profiler is not None:
            profiler.count(
                name,
                amount,
            )

    def _profile_value(
            self,
            name: str,
            value,
    ) -> None:
        """Store the latest value for one profiler field."""
        profiler = getattr(
            self,
            "profiler",
            None,
        )

        if profiler is not None:
            profiler.value(
                name,
                value,
            )

    def _attach_profiler_to_graph(self, graph) -> None:
        """
        Give a newly created FactorGraph the same shared profiler.

        This only attaches a reference. It does not change graph logic.
        """
        if graph is not None:
            graph.profiler = self.profiler

    @torch.no_grad()
    def backend_ba(self, t_start, t_end, steps, graph, nms, radius, thresh, max_factors, t_start_loop=None, loop=False,
                   motion_only=False, enable_wq=True):
        """ main update """
        with self._profile_timer(
                "wildgs.backend.backend_ba.total"
        ):
            self._attach_profiler_to_graph(graph)

            if self.cfg['tracking']["uncertainty_params"]['activate']:
                with self._profile_timer(
                        "wildgs.backend.backend_ba.update_uncertainty_masks"
                ):
                    self.video.update_all_uncertainty_mask()

            if t_start_loop is None or not loop:
                t_start_loop = t_start
            assert t_start_loop >= t_start, f'short: {t_start_loop}, long: {t_start}.'
            with self._profile_timer(
                    "wildgs.backend.backend_ba.add_backend_proximity_factors"
            ):
                edge_num = graph.add_backend_proximity_factors(t_start, t_end, nms, radius, thresh, max_factors,
                                                               self.beta, t_start_loop, loop)
            if edge_num == 0:
                with self._profile_timer(
                        "wildgs.backend.backend_ba.clear_empty_graph"
                ):
                    graph.clear_edges()

                self._profile_count(
                    "wildgs.backend.backend_ba.empty_runs"
                )
                self._profile_value(
                    "wildgs.backend.backend_ba.last_edge_count",
                    0,
                )
                return 0

            with self._profile_timer(
                    "wildgs.backend.backend_ba.graph_update_lowmem"
            ):
                graph.update_lowmem(
                    t0=t_start_loop + 1,
                    # fix the start point to avoid drift, be sure to use t_start_loop rather than t_start here.
                    t1=t_end,
                    itrs=2,
                    use_inactive=False,
                    steps=steps,
                    enable_wq=enable_wq
                )

            with self._profile_timer(
                    "wildgs.backend.backend_ba.clear_graph"
            ):
                graph.clear_edges()

            self._profile_count(
                "wildgs.backend.backend_ba.edges_added",
                int(edge_num),
            )
            self._profile_value(
                "wildgs.backend.backend_ba.last_edge_count",
                int(edge_num),
            )
            self._profile_value(
                "wildgs.backend.backend_ba.last_loop_mode",
                bool(loop),
            )
            self._profile_value(
                "wildgs.backend.backend_ba.last_t_start",
                int(t_start),
            )
            self._profile_value(
                "wildgs.backend.backend_ba.last_t_end",
                int(t_end),
            )

            return edge_num

    @torch.no_grad()
    def dense_ba(self, steps=6, enable_wq=True):
        with self._profile_timer(
                "wildgs.backend.dense_ba.total"
        ):
            t_start = 0
            t_end = self.video.counter.value
            nms = self.backend_nms
            radius = self.backend_radius
            thresh = self.backend_thresh
            n = t_end - t_start
            max_factors = ((radius + 2) * 2) * n
            if self.backend_normalize:
                with self._profile_timer(
                        "wildgs.backend.dense_ba.normalize_video"
                ):
                    self.video.normalize()
            with self._profile_timer(
                    "wildgs.backend.dense_ba.create_factor_graph"
            ):
                graph = FactorGraph(self.video, self.update_op, device=self.device,
                                    corr_impl='alt', max_factors=max_factors)

            self._attach_profiler_to_graph(graph)
            with self._profile_timer(
                    "wildgs.backend.dense_ba.backend_ba"
            ):
                n_edges = self.backend_ba(t_start, t_end, steps, graph, nms, radius,
                                          thresh, max_factors, motion_only=False, enable_wq=enable_wq)

            with self._profile_timer(
                    "wildgs.backend.dense_ba.cleanup_graph"
            ):
                del graph
                torch.cuda.empty_cache()

            with self._profile_timer(
                    "wildgs.backend.dense_ba.mark_video_dirty"
            ):
                self.video.set_dirty(t_start, t_end)

            with self._profile_timer(
                    "wildgs.backend.dense_ba.update_valid_depth_mask"
            ):
                self.video.update_valid_depth_mask()

            self._profile_value(
                "wildgs.backend.dense_ba.last_frame_count",
                int(n),
            )
            self._profile_value(
                "wildgs.backend.dense_ba.last_edge_count",
                int(n_edges),
            )

            return n, n_edges

    @torch.no_grad()
    def loop_ba(self, t_start, t_end, steps=6, motion_only=False, local_graph=None, enable_wq=True):
        ''' loop closure, add edges with high-covisiablity'''
        with self._profile_timer(
                "wildgs.backend.loop_ba.total"
        ):
            radius = self.backend_loop_radius
            window = self.backend_loop_window
            max_factors = 8 * window
            nms = self.backend_loop_nms
            thresh = self.backend_loop_thresh
            t_start_loop = max(0, t_end - window)

            with self._profile_timer(
                    "wildgs.backend.loop_ba.create_factor_graph"
            ):
                graph = FactorGraph(self.video, self.update_op, device=self.device, corr_impl='alt',
                                    max_factors=max_factors)

            self._attach_profiler_to_graph(graph)
            if local_graph is not None:
                with self._profile_timer(
                        "wildgs.backend.loop_ba.copy_local_graph"
                ):
                    copy_attr = ['ii', 'jj', 'age', 'net', 'target', 'weight']
                    for key in copy_attr:
                        val = getattr(local_graph, key)
                        if val is not None:
                            setattr(graph, key, deepcopy(val))

            left_factors = max_factors - len(graph.ii)
            with self._profile_timer(
                    "wildgs.backend.loop_ba.backend_ba"
            ):
                n_edges = self.backend_ba(t_start, t_end, steps, graph, nms, radius, thresh,
                                          left_factors, t_start_loop=t_start_loop, loop=True,
                                          motion_only=motion_only, enable_wq=enable_wq)

            with self._profile_timer(
                    "wildgs.backend.loop_ba.cleanup_graph"
            ):
                del graph
                torch.cuda.empty_cache()

            loop_window_size = t_end - t_start_loop

            self._profile_value(
                "wildgs.backend.loop_ba.last_window_size",
                int(loop_window_size),
            )
            self._profile_value(
                "wildgs.backend.loop_ba.last_edge_count",
                int(n_edges),
            )

            return loop_window_size, n_edges
