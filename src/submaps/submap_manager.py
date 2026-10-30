"""Navigator / WildGS-SLAM: Phase 1, Step 02A submap lifecycle planner.

This module deliberately performs NO Gaussian operations. It manages *metadata*
for the next integration step: identity-safe keyframe grouping, split scheduling,
and transactional bookkeeping once the caller has completed an actual map switch.

Nothing here changes WildGS tracking, BA, Mapper math, or ROS2 input modes.
"""

from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Optional


def _nonnegative_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


@dataclass(frozen=True)
class KeyframeRef:
    """Both fields are needed: Frontend may reuse a video slot after culling."""

    video_idx: int
    frame_id: int


@dataclass
class SubmapRecord:
    """Metadata only; deliberately does not retain a CUDA GaussianModel."""

    submap_id: int
    overlap: tuple[KeyframeRef, ...] = ()
    owned: list[KeyframeRef] = field(default_factory=list)
    status: str = "active"
    saved_ply: Optional[str] = None

    @property
    def keyframes(self) -> tuple[KeyframeRef, ...]:
        # Overlap KFs are present in *both* neighboring maps by design.
        return self.overlap + tuple(self.owned)

    @property
    def keyframe_count(self) -> int:
        # Our 30-KF limit INCLUDES the overlap from the preceding map.
        return len(self.keyframes)


@dataclass(frozen=True)
class SplitPlan:
    """A proposal, NOT a completed Gaussian switch."""

    previous_submap_id: int
    new_submap_id: int
    overlap: tuple[KeyframeRef, ...]
    next_keyframe: KeyframeRef


class SubmapManager:
    """Track submap boundaries while leaving the global Tracker/Frontend alone.

    IMPORTANT: Do not wire this into the Navigator mapping loop until Step 02B
    implements saving the old Gaussian map and constructing the new map.
    """

    def __init__(self, max_keyframes: int = 30, overlap_keyframes: int = 3):
        max_keyframes = _nonnegative_int(max_keyframes, "max_keyframes")
        overlap_keyframes = _nonnegative_int(overlap_keyframes, "overlap_keyframes")
        if max_keyframes < 2:
            raise ValueError("max_keyframes must be >= 2")
        if overlap_keyframes >= max_keyframes:
            raise ValueError("overlap_keyframes must be < max_keyframes")

        self.max_keyframes = max_keyframes
        self.overlap_keyframes = overlap_keyframes
        self.active = SubmapRecord(submap_id=0)
        self.closed: list[SubmapRecord] = []
        self.pending_split = False

        # Stable identity across the whole session. Video indices alone may be
        # recycled after Frontend drops its most recent candidate.
        self._seen_frames: dict[int, int] = {}
        self._prepared_plan: Optional[SplitPlan] = None

    def register_mapped_keyframe(self, video_idx: int, frame_id: int) -> bool:
        """Call AFTER successful mapping. Returns True at the split threshold.

        A pending split blocks new unique frames: they must first be processed
        by a new GaussianModel, once Step 02B provides the physical switch.
        """
        video_idx = _nonnegative_int(video_idx, "video_idx")
        frame_id = _nonnegative_int(frame_id, "frame_id")

        previous_video_idx = self._seen_frames.get(frame_id)
        if previous_video_idx is not None:
            if previous_video_idx != video_idx:
                raise RuntimeError(
                    f"Frame {frame_id} was previously mapped with video_idx "
                    f"{previous_video_idx}, not {video_idx}"
                )
            return False

        if self.pending_split:
            raise RuntimeError(
                "A submap split is pending. Prepare and commit the map switch "
                "before mapping the next unique keyframe."
            )

        if any(kf.video_idx == video_idx for kf in self.active.keyframes):
            raise RuntimeError(
                f"video_idx={video_idx} was reused within active Submap "
                f"{self.active.submap_id}; resolve this before mapping"
            )

        self.active.owned.append(KeyframeRef(video_idx, frame_id))
        self._seen_frames[frame_id] = video_idx

        if self.active.keyframe_count >= self.max_keyframes:
            self.pending_split = True
            return True
        return False

    def prepare_split(self, next_video_idx: int, next_frame_id: int) -> SplitPlan:
        """Produce an overlap plan for the NEXT valid, not-yet-mapped KF.

        Does NOT mutate state or allocate a GaussianModel. The caller must
        validate the next viewpoint BEFORE this point.
        """
        if not self.pending_split:
            raise RuntimeError("No split has been requested")
        next_video_idx = _nonnegative_int(next_video_idx, "next_video_idx")
        next_frame_id = _nonnegative_int(next_frame_id, "next_frame_id")
        if next_frame_id in self._seen_frames:
            raise ValueError("The next split keyframe has already been mapped")

        overlap = self.active.keyframes[-self.overlap_keyframes:] if self.overlap_keyframes else ()
        plan = SplitPlan(
            previous_submap_id=self.active.submap_id,
            new_submap_id=self.active.submap_id + 1,
            overlap=tuple(overlap),
            next_keyframe=KeyframeRef(next_video_idx, next_frame_id),
        )
        if self._prepared_plan is not None and plan != self._prepared_plan:
            raise RuntimeError("A different split plan is already prepared")
        self._prepared_plan = plan
        return plan

    def commit_split(
        self,
        plan: SplitPlan,
        *,
        archived_ply: str,
        new_model_ready: bool = False,
    ) -> None:
        """Update records ONLY AFTER an actual, successful Gaussian map switch.

        The orchestrator (Step 02B) must already have:
          1. Saved the previous model to archived_ply.
          2. Created and initialized the new GaussianModel and optimizer.
          3. Transferred/reseeded overlap KFs, adjusted mapping state.
          4. Reinstalled Navigator Gaussian identity hooks.
        """
        if not self.pending_split or self._prepared_plan != plan:
            raise RuntimeError("Cannot commit an unknown or stale split plan")
        if not new_model_ready:
            raise RuntimeError("New GaussianModel/optimizer/identity hooks not confirmed ready")
        saved_path = Path(archived_ply)
        if not saved_path.is_file() or saved_path.stat().st_size == 0:
            raise RuntimeError(f"Archived submap PLY missing or empty: {saved_path}")

        self.active.status = "closed"
        self.active.saved_ply = str(saved_path.resolve())
        self.closed.append(self.active)
        self.active = SubmapRecord(submap_id=plan.new_submap_id, overlap=plan.overlap)
        self.pending_split = False
        self._prepared_plan = None

    def summary(self) -> dict:
        """Safe, metadata-only snapshot for logging/Profiler diagnostics."""
        return {
            "active_submap_id": self.active.submap_id,
            "active_keyframes": self.active.keyframe_count,
            "active_owned_keyframes": len(self.active.owned),
            "active_overlap_keyframes": len(self.active.overlap),
            "max_keyframes": self.max_keyframes,
            "pending_split": self.pending_split,
            "closed_submaps": len(self.closed),
            "unique_mapped_frames": len(self._seen_frames),
        }