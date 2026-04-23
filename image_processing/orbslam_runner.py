from __future__ import annotations

import logging
import math
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pyorbslam3_custom

from media_sources.config import ImageEnhancementConfig, OrbSlamConfig
from media_sources.contracts import FramePacket

from .image_enhancement import LowLightEnhancer


@dataclass(slots=True)
class RunSummary:
    processed_frames: int
    elapsed_s: float
    average_input_fps: float
    last_tracking_state: int
    first_tracking_ok_frame: int | None = None
    tracking_ok_frame_count: int = 0
    tracking_recently_lost_frame_count: int = 0
    tracking_lost_frame_count: int = 0
    tracking_state_counts: dict[int, int] | None = None
    initialization_succeeded: bool = False
    camera_pose_count: int = 0
    map_point_count: int = 0
    keyframe_trajectory_point_count: int = 0
    depth_inference_frame_count: int = 0
    depth_guidance_frame_count: int = 0
    depth_filter_applied_frame_count: int = 0
    depth_fallback_frame_count: int = 0
    output_dir: str | None = None
    trajectory_tum_path: str | None = None
    keyframes_tum_path: str | None = None
    map_points_path: str | None = None
    all_map_points_path: str | None = None
    atlas_map_summary_path: str | None = None
    atlas_maps_dir: str | None = None
    trajectory_points_path: str | None = None
    camera_trajectory_points_path: str | None = None
    trajectory_plot_path: str | None = None
    map_plot_path: str | None = None
    atlas_map_plot_path: str | None = None
    topdown_plot_path: str | None = None
    depth_stats_path: str | None = None
    depth_preview_last_path: str | None = None
    feature_filter_log_path: str | None = None
    trajectory_metrics_path: str | None = None
    ate_rmse: float | None = None
    ate_mean: float | None = None
    rpe_rmse: float | None = None
    rpe_mean: float | None = None
    map_points: np.ndarray | None = None
    all_map_points: np.ndarray | None = None
    trajectory_points: np.ndarray | None = None
    camera_trajectory_points: np.ndarray | None = None
    atlas_map_groups: list["AtlasMapGroup"] | None = None


@dataclass(slots=True)
class AtlasMapGroup:
    map_id: int
    is_current: bool
    points: np.ndarray

    @property
    def point_count(self) -> int:
        return int(self.points.shape[0]) if self.points.ndim == 2 else 0


class OrbSlamVideoRunner:
    def __init__(
        self,
        config: OrbSlamConfig,
        image_enhancement_config: ImageEnhancementConfig | None = None,
        log_mode: str = "full",
        logger: logging.Logger | None = None,
    ) -> None:
        self._config = config
        if log_mode not in {"full", "fps_adaptive"}:
            raise ValueError("log_mode must be 'full' or 'fps_adaptive'")
        self._log_mode = log_mode
        self._logger = logger or logging.getLogger(__name__)
        self._system = self._create_system(config)
        self._enhancer = LowLightEnhancer(image_enhancement_config)
        self._started_at: float | None = None
        self._processed_frames = 0
        self._last_tracking_state = 0
        self._is_shutdown = False
        self._trajectory_records: list[tuple[float, np.ndarray]] = []
        self._tracking_state_history: list[tuple[int, int]] = []

    def process_packet(self, packet: FramePacket) -> tuple[int, float]:
        if self._started_at is None:
            self._started_at = time.monotonic()

        if packet.is_grayscale:
            enhanced = self._enhancer.enhance_for_mono_pipeline(packet.frame_mono)
            if self._enhancer.enabled:
                packet.meta["display_frame"] = enhanced.frame_rgb
            self._system.process_mono(enhanced.frame_gray, packet.timestamp_s)
        else:
            enhanced = self._enhancer.enhance_for_packet(None, packet.frame_rgb)
            if self._enhancer.enabled:
                packet.meta["display_frame"] = enhanced.frame_rgb
            self._system.process_mono_rgb(enhanced.frame_rgb, packet.timestamp_s)

        self._processed_frames += 1
        self._last_tracking_state = self._system.get_tracking_state()
        self._tracking_state_history.append((packet.frame_index, self._last_tracking_state))
        self._record_pose(packet.timestamp_s, self._last_tracking_state)

        elapsed_s = max(time.monotonic() - self._started_at, 1e-6)
        input_fps = self._processed_frames / elapsed_s
        if self._should_log_frame(input_fps):
            self._logger.info(
                "frame=%d source=%s timestamp=%.3f input_fps=%.2f tracking_state=%d mode=%s log_mode=%s",
                packet.frame_index,
                packet.source_kind,
                packet.timestamp_s,
                input_fps,
                self._last_tracking_state,
                "mono" if packet.is_grayscale else "rgb",
                self._log_mode,
            )
        return self._last_tracking_state, input_fps

    def run(
        self,
        packets: Iterable[FramePacket],
        max_frames: int | None = None,
        output_dir: str | Path | None = None,
        inter_frame_delay_s: float = 0.0,
        on_packet: Callable[[FramePacket, int, float], None] | None = None,
    ) -> RunSummary:
        try:
            for packet in packets:
                if self._processed_frames > 0 and inter_frame_delay_s > 0.0:
                    time.sleep(inter_frame_delay_s)
                tracking_state, input_fps = self.process_packet(packet)
                if on_packet is not None:
                    on_packet(packet, tracking_state, input_fps)
                if max_frames and self._processed_frames >= max_frames:
                    self._logger.info("Reached runtime.max_frames=%d", max_frames)
                    break
        except KeyboardInterrupt:
            self._logger.info("Interrupted by user")

        summary = self._build_summary()
        if output_dir is not None:
            summary = self._persist_run_outputs(output_dir, summary)

        self.shutdown()
        summary.elapsed_s = self._compute_elapsed_s()
        summary.average_input_fps = (
            self._processed_frames / summary.elapsed_s if summary.elapsed_s > 0 else 0.0
        )
        return summary

    def shutdown(self) -> None:
        if self._is_shutdown:
            return
        self._system.shutdown()
        self._is_shutdown = True

    def _build_summary(self) -> RunSummary:
        elapsed_s = self._compute_elapsed_s()
        average_input_fps = self._processed_frames / elapsed_s if elapsed_s > 0 else 0.0
        tracking_metrics = _build_tracking_metrics(self._tracking_state_history)
        return RunSummary(
            processed_frames=self._processed_frames,
            elapsed_s=elapsed_s,
            average_input_fps=average_input_fps,
            last_tracking_state=self._last_tracking_state,
            first_tracking_ok_frame=tracking_metrics["first_tracking_ok_frame"],
            tracking_ok_frame_count=tracking_metrics["tracking_ok_frame_count"],
            tracking_recently_lost_frame_count=tracking_metrics["tracking_recently_lost_frame_count"],
            tracking_lost_frame_count=tracking_metrics["tracking_lost_frame_count"],
            tracking_state_counts=tracking_metrics["tracking_state_counts"],
            initialization_succeeded=tracking_metrics["initialization_succeeded"],
            camera_pose_count=len(self._trajectory_records),
        )

    def _persist_run_outputs(self, output_dir: str | Path, summary: RunSummary) -> RunSummary:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        trajectory_tum_path = output_path / "trajectory.tum"
        keyframes_tum_path = output_path / "keyframes.tum"
        map_points_path = output_path / "map_points.npy"
        all_map_points_path = output_path / "all_map_points.npy"
        atlas_maps_dir = output_path / "atlas_maps"
        atlas_map_summary_path = output_path / "atlas_maps_summary.json"
        trajectory_points_path = output_path / "trajectory_points.npy"
        camera_trajectory_points_path = output_path / "camera_trajectory_points.npy"

        self._system.save_keyframe_trajectory_tum(str(keyframes_tum_path))
        self._write_tum_trajectory(trajectory_tum_path)

        map_points = _get_point_array(self._system, "get_map_points")
        all_map_points = _get_point_array(
            self._system,
            "get_all_map_points",
            fallback=map_points,
        )
        atlas_map_groups = _get_atlas_map_groups(
            self._system,
            fallback_points=all_map_points,
        )
        trajectory_points = np.asarray(
            self._system.get_current_map_keyframe_trajectory(),
            dtype=np.float32,
        )
        camera_trajectory_points = _trajectory_records_to_points(self._trajectory_records)
        np.save(map_points_path, map_points)
        np.save(all_map_points_path, all_map_points)
        np.save(trajectory_points_path, trajectory_points)
        np.save(camera_trajectory_points_path, camera_trajectory_points)
        _write_atlas_map_groups(atlas_maps_dir, atlas_map_summary_path, atlas_map_groups)

        summary.output_dir = str(output_path)
        summary.trajectory_tum_path = str(trajectory_tum_path)
        summary.keyframes_tum_path = str(keyframes_tum_path)
        summary.map_points_path = str(map_points_path)
        summary.all_map_points_path = str(all_map_points_path)
        summary.atlas_maps_dir = str(atlas_maps_dir)
        summary.atlas_map_summary_path = str(atlas_map_summary_path)
        summary.trajectory_points_path = str(trajectory_points_path)
        summary.camera_trajectory_points_path = str(camera_trajectory_points_path)
        summary.map_points = map_points
        summary.all_map_points = all_map_points
        summary.trajectory_points = trajectory_points
        summary.camera_trajectory_points = camera_trajectory_points
        summary.atlas_map_groups = atlas_map_groups
        summary.map_point_count = int(map_points.shape[0]) if map_points.ndim >= 2 else 0
        summary.keyframe_trajectory_point_count = (
            int(trajectory_points.shape[0]) if trajectory_points.ndim >= 2 else 0
        )
        summary.camera_pose_count = (
            int(camera_trajectory_points.shape[0]) if camera_trajectory_points.ndim >= 2 else 0
        )
        return summary

    def _compute_elapsed_s(self) -> float:
        if self._started_at is None:
            return 0.0
        return max(time.monotonic() - self._started_at, 0.0)

    def _should_log_frame(self, input_fps: float) -> bool:
        if self._log_mode == "full":
            return True
        if self._processed_frames <= 1:
            return True
        if not math.isfinite(input_fps):
            return True
        interval = max(1, int(round(input_fps)))
        return self._processed_frames % interval == 0

    def _create_system(self, config: OrbSlamConfig):
        sensor = getattr(pyorbslam3_custom.Sensor, config.sensor)
        return pyorbslam3_custom.System(
            config.vocab_path,
            config.settings_path,
            sensor,
            config.use_viewer,
        )

    def _record_pose(self, timestamp_s: float, tracking_state: int) -> None:
        if tracking_state not in {2, 5}:
            return
        pose = self._system.get_pose()
        if pose is None:
            return
        pose_matrix = np.asarray(pose, dtype=np.float32)
        if pose_matrix.shape != (4, 4):
            return
        self._trajectory_records.append((timestamp_s, pose_matrix))

    def _write_tum_trajectory(self, output_path: Path) -> None:
        with output_path.open("w", encoding="utf-8") as handle:
            for timestamp_s, pose_matrix in self._trajectory_records:
                tx = float(pose_matrix[0, 3])
                ty = float(pose_matrix[1, 3])
                tz = float(pose_matrix[2, 3])
                qx, qy, qz, qw = _rotation_matrix_to_quaternion(pose_matrix[:3, :3])
                handle.write(
                    f"{timestamp_s:.9f} {tx:.9f} {ty:.9f} {tz:.9f} "
                    f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n"
                )


def _rotation_matrix_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(rotation[0, 0] + rotation[1, 1] + rotation[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        qw = (rotation[2, 1] - rotation[1, 2]) / s
        qx = 0.25 * s
        qy = (rotation[0, 1] + rotation[1, 0]) / s
        qz = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        qw = (rotation[0, 2] - rotation[2, 0]) / s
        qx = (rotation[0, 1] + rotation[1, 0]) / s
        qy = 0.25 * s
        qz = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        qw = (rotation[1, 0] - rotation[0, 1]) / s
        qx = (rotation[0, 2] + rotation[2, 0]) / s
        qy = (rotation[1, 2] + rotation[2, 1]) / s
        qz = 0.25 * s
    return qx, qy, qz, qw


def _trajectory_records_to_points(
    trajectory_records: list[tuple[float, np.ndarray]],
) -> np.ndarray:
    if not trajectory_records:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(
        [
            [float(pose_matrix[0, 3]), float(pose_matrix[1, 3]), float(pose_matrix[2, 3])]
            for _, pose_matrix in trajectory_records
        ],
        dtype=np.float32,
    )


def _normalize_points(points: np.ndarray | list[list[float]] | None) -> np.ndarray:
    if points is None:
        return np.empty((0, 3), dtype=np.float32)
    array = np.asarray(points, dtype=np.float32)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return array.reshape((-1, 3))


def _get_point_array(system, method_name: str, fallback: np.ndarray | None = None) -> np.ndarray:
    method = getattr(system, method_name, None)
    if callable(method):
        return _normalize_points(method())
    if fallback is not None:
        return np.asarray(fallback, dtype=np.float32)
    return np.empty((0, 3), dtype=np.float32)


def _get_atlas_map_groups(
    system,
    fallback_points: np.ndarray | None = None,
) -> list[AtlasMapGroup]:
    method = getattr(system, "get_atlas_map_point_groups", None)
    if callable(method):
        raw_groups = method()
        groups: list[AtlasMapGroup] = []
        for raw in raw_groups:
            if raw is None:
                continue
            map_id = int(raw.get("map_id", len(groups)))
            is_current = bool(raw.get("is_current", False))
            points = _normalize_points(raw.get("points"))
            groups.append(AtlasMapGroup(map_id=map_id, is_current=is_current, points=points))
        if groups:
            return groups

    if fallback_points is None:
        return []
    return [
        AtlasMapGroup(
            map_id=0,
            is_current=True,
            points=_normalize_points(fallback_points),
        )
    ]


def _write_atlas_map_groups(
    atlas_maps_dir: Path,
    atlas_map_summary_path: Path,
    atlas_map_groups: list[AtlasMapGroup],
) -> None:
    atlas_maps_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, int | bool | str]] = []
    for group in atlas_map_groups:
        filename = f"map_{group.map_id:04d}.npy"
        np.save(atlas_maps_dir / filename, group.points)
        summary_rows.append(
            {
                "map_id": group.map_id,
                "is_current": group.is_current,
                "point_count": group.point_count,
                "points_path": filename,
            }
        )
    atlas_map_summary_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")


# ORB-SLAM3 tracking states:
#   0 = NO_IMAGES_YET, 1 = NOT_INITIALIZED
#   2 = OK, 3 = RECENTLY_LOST, 4 = LOST, 5 = OK (KLT variant)
# Only states 2 and 5 represent genuine stable tracking with a recorded pose.
# State 3 (RECENTLY_LOST) is tracked separately — it inflates tracking_ok_ratio
# if counted as OK, since no pose is recorded in that state.
_TRACKING_OK_STATES = frozenset({2, 5})
_TRACKING_RECENTLY_LOST_STATES = frozenset({3})
_TRACKING_LOST_STATES = frozenset({4})


def _build_tracking_metrics(
    tracking_state_history: list[tuple[int, int]],
) -> dict[str, int | bool | dict[int, int] | None]:
    counts: dict[int, int] = {}
    first_ok_frame: int | None = None
    ok_frame_count = 0
    recently_lost_frame_count = 0
    lost_frame_count = 0

    for frame_index, state in tracking_state_history:
        counts[state] = counts.get(state, 0) + 1
        if state in _TRACKING_OK_STATES:
            ok_frame_count += 1
            if first_ok_frame is None:
                first_ok_frame = frame_index
        if state in _TRACKING_RECENTLY_LOST_STATES:
            recently_lost_frame_count += 1
        if state in _TRACKING_LOST_STATES:
            lost_frame_count += 1

    return {
        "first_tracking_ok_frame": first_ok_frame,
        "tracking_ok_frame_count": ok_frame_count,
        "tracking_recently_lost_frame_count": recently_lost_frame_count,
        "tracking_lost_frame_count": lost_frame_count,
        "tracking_state_counts": counts,
        "initialization_succeeded": first_ok_frame is not None,
    }
