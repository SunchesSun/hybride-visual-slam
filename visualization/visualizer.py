from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from image_processing.orbslam_runner import RunSummary
from media_sources.contracts import FramePacket


@dataclass(slots=True)
class VisualizationState:
    is_active: bool = False
    last_frame_index: int | None = None
    last_metrics: dict[str, Any] = field(default_factory=dict)


class VisualizationModule:
    def __init__(
        self,
        logger: logging.Logger | None = None,
        window_name: str = "ORB-SLAM3 Mono Preview",
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._window_name = window_name
        self._state = VisualizationState()

    @property
    def state(self) -> VisualizationState:
        return self._state

    def ensure_plot_backend(self) -> None:
        self._load_plotly_graph_objects()
        self._load_matplotlib_pyplot()

    def start_live_view(self) -> None:
        if self._state.is_active:
            return
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        self._state.is_active = True
        self._logger.debug("Live visualization started")

    def update_live_view(
        self,
        packet: FramePacket,
        tracking_state: int,
        input_fps: float,
    ) -> None:
        if not self._state.is_active:
            return

        display_frame = self._to_display_frame(packet)
        depth_preview = packet.meta.get("depth_preview")
        if isinstance(depth_preview, np.ndarray) and depth_preview.ndim == 2 and depth_preview.size > 0:
            depth_bgr = cv2.applyColorMap(depth_preview, cv2.COLORMAP_MAGMA)
            depth_bgr = cv2.resize(
                depth_bgr,
                (display_frame.shape[1], display_frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            display_frame = np.hstack((display_frame, depth_bgr))
        overlay_lines = [
            f"frame: {packet.frame_index}",
            f"timestamp: {packet.timestamp_s:.3f}",
            f"tracking_state: {tracking_state}",
            f"input_fps: {input_fps:.2f}",
            f"mode: {'mono' if packet.is_grayscale else 'rgb'}",
        ]
        for idx, line in enumerate(overlay_lines):
            cv2.putText(
                display_frame,
                line,
                (10, 24 + idx * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(self._window_name, display_frame)
        cv2.waitKey(1)
        self._state.last_frame_index = packet.frame_index

    def close_live_view(self) -> None:
        if not self._state.is_active:
            return
        cv2.destroyWindow(self._window_name)
        self._state.is_active = False
        self._logger.debug("Live visualization closed")

    def render_results(
        self,
        summary: RunSummary,
        output_dir: str | Path,
        ground_truth_tum_path: str | Path | None = None,
    ) -> RunSummary:
        output_path = Path(output_dir)
        trajectory_points = self._select_trajectory_points(summary)
        # keyframes.tum is loop-closure corrected — better for GT alignment
        alignment_tum = summary.keyframes_tum_path or summary.trajectory_tum_path
        ground_truth_points = self._load_aligned_ground_truth_points(
            alignment_tum,
            ground_truth_tum_path,
            reference_timestamps=self._extract_tum_timestamps(alignment_tum),
        )
        current_map_points = self._select_current_map_points(summary)
        trajectory_plot_path = self.render_trajectory_plot(
            trajectory_points,
            output_path / "trajectory_only.html",
        )
        map_plot_path = self.render_map_plot(
            current_map_points,
            trajectory_points,
            output_path / "map_and_trajectory.html",
        )
        atlas_map_plot_path = self.render_atlas_map_plot(
            self._select_atlas_map_groups(summary),
            trajectory_points,
            output_path / "atlas_maps_and_trajectory.html",
        )
        topdown_plot_path = self.render_topdown_trajectory_plot(
            trajectory_points,
            output_path / "trajectory_topdown.png",
            ground_truth_points=ground_truth_points,
        )
        summary.trajectory_plot_path = str(trajectory_plot_path)
        summary.map_plot_path = str(map_plot_path)
        summary.atlas_map_plot_path = str(atlas_map_plot_path)
        summary.topdown_plot_path = str(topdown_plot_path)
        return summary

    def save_depth_preview(self, depth_preview: np.ndarray, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        if depth_preview.ndim != 2:
            raise ValueError("depth_preview must be a single-channel image")
        if depth_preview.dtype != np.uint8:
            depth_preview = np.asarray(depth_preview, dtype=np.uint8)
        if not cv2.imwrite(str(output_path), depth_preview):
            raise RuntimeError(f"Failed to save depth preview to {output_path}")
        return output_path

    def render_trajectory_plot(
        self,
        trajectory_points: np.ndarray,
        output_path: str | Path,
    ) -> Path:
        go = self._load_plotly_graph_objects()
        output_path = Path(output_path)
        figure = go.Figure()
        if trajectory_points.size > 0:
            line_points = _with_trajectory_breaks(trajectory_points)
            figure.add_trace(
                go.Scatter3d(
                    x=line_points[:, 0],
                    y=line_points[:, 1],
                    z=line_points[:, 2],
                    mode="lines+markers",
                    name="trajectory",
                    line={"color": "#ff6b00", "width": 4},
                    marker={"size": 3},
                )
            )
        else:
            figure.add_annotation(
                text="No trajectory points available",
                showarrow=False,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
            )
        figure.update_layout(
            title="ORB-SLAM3 Trajectory",
            scene={
                "xaxis_title": "X",
                "yaxis_title": "Y",
                "zaxis_title": "Z",
            },
        )
        figure.write_html(str(output_path), include_plotlyjs="cdn")
        return output_path

    def render_map_plot(
        self,
        map_points: np.ndarray,
        trajectory_points: np.ndarray,
        output_path: str | Path,
    ) -> Path:
        go = self._load_plotly_graph_objects()
        output_path = Path(output_path)
        figure = go.Figure()
        filtered_map = _filter_map_points_to_trajectory_region(map_points, trajectory_points)
        if filtered_map.size > 0:
            figure.add_trace(
                go.Scatter3d(
                    x=filtered_map[:, 0],
                    y=filtered_map[:, 1],
                    z=filtered_map[:, 2],
                    mode="markers",
                    name="map points",
                    marker={"size": 2, "color": "#1f77b4", "opacity": 0.65},
                )
            )
        if trajectory_points.size > 0:
            line_points = _with_trajectory_breaks(trajectory_points)
            figure.add_trace(
                go.Scatter3d(
                    x=line_points[:, 0],
                    y=line_points[:, 1],
                    z=line_points[:, 2],
                    mode="lines+markers",
                    name="trajectory",
                    line={"color": "#ff6b00", "width": 4},
                    marker={"size": 3},
                )
            )
        if filtered_map.size == 0 and trajectory_points.size == 0:
            figure.add_annotation(
                text="No map data available",
                showarrow=False,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
            )
        figure.update_layout(
            title="ORB-SLAM3 Map and Trajectory",
            scene={
                "xaxis_title": "X",
                "yaxis_title": "Y",
                "zaxis_title": "Z",
            },
        )
        figure.write_html(str(output_path), include_plotlyjs="cdn")
        return output_path

    def render_topdown_trajectory_plot(
        self,
        trajectory_points: np.ndarray,
        output_path: str | Path,
        ground_truth_points: np.ndarray | None = None,
    ) -> Path:
        plt = self._load_matplotlib_pyplot()
        output_path = Path(output_path)

        figure, axis = plt.subplots(figsize=(8, 6))
        gt_points = np.asarray(ground_truth_points, dtype=np.float32) if ground_truth_points is not None else np.empty((0, 3), dtype=np.float32)
        if gt_points.size > 0:
            gt_line_points = _with_trajectory_breaks(gt_points)
            axis.plot(
                gt_line_points[:, 0],
                gt_line_points[:, 2],
                color="#1f77b4",
                linewidth=1.8,
                linestyle="--",
                label="ground truth",
            )
        if trajectory_points.size > 0:
            line_points = _with_trajectory_breaks(trajectory_points)
            axis.plot(
                line_points[:, 0],
                line_points[:, 2],
                color="#ff6b00",
                linewidth=2.0,
                label="trajectory",
            )
            axis.scatter(
                trajectory_points[0, 0],
                trajectory_points[0, 2],
                color="#2ca02c",
                s=50,
                label="start",
                zorder=3,
            )
            axis.scatter(
                trajectory_points[-1, 0],
                trajectory_points[-1, 2],
                color="#d62728",
                s=50,
                label="end",
                zorder=3,
            )
            axis.legend()
        else:
            axis.text(
                0.5,
                0.5,
                "No trajectory data available",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )

        axis.set_title("ORB-SLAM3 Top-Down Trajectory")
        axis.set_xlabel("X")
        axis.set_ylabel("Z")
        axis.axis("equal")
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        return output_path

    def render_atlas_map_plot(
        self,
        atlas_map_groups: list[Any],
        trajectory_points: np.ndarray,
        output_path: str | Path,
    ) -> Path:
        go = self._load_plotly_graph_objects()
        output_path = Path(output_path)
        figure = go.Figure()

        has_map_points = False
        palette = [
            "#1f77b4",
            "#2ca02c",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#17becf",
            "#bcbd22",
            "#7f7f7f",
            "#d62728",
            "#ff7f0e",
        ]
        for idx, group in enumerate(atlas_map_groups):
            raw_points = np.asarray(self._atlas_group_value(group, "points", np.empty((0, 3), dtype=np.float32)), dtype=np.float32)
            points = _filter_map_points_to_trajectory_region(raw_points, trajectory_points)
            if points.size == 0:
                continue
            has_map_points = True
            is_current = bool(self._atlas_group_value(group, "is_current", False))
            map_id = int(self._atlas_group_value(group, "map_id", idx))
            color = "#ff4d4f" if is_current else palette[idx % len(palette)]
            opacity = 0.85 if is_current else 0.45
            size = 2.8 if is_current else 1.8
            label = f"map {map_id}"
            if is_current:
                label += " (current)"
            figure.add_trace(
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode="markers",
                    name=label,
                    marker={"size": size, "color": color, "opacity": opacity},
                )
            )

        if trajectory_points.size > 0:
            line_points = _with_trajectory_breaks(trajectory_points)
            figure.add_trace(
                go.Scatter3d(
                    x=line_points[:, 0],
                    y=line_points[:, 1],
                    z=line_points[:, 2],
                    mode="lines",
                    name="camera trajectory",
                    line={"color": "#ff6b00", "width": 5},
                )
            )

        if not has_map_points and trajectory_points.size == 0:
            figure.add_annotation(
                text="No atlas map data available",
                showarrow=False,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
            )

        figure.update_layout(
            title="ORB-SLAM3 Atlas Maps and Trajectory",
            scene={
                "xaxis_title": "X",
                "yaxis_title": "Y",
                "zaxis_title": "Z",
            },
        )
        figure.write_html(str(output_path), include_plotlyjs="cdn")
        return output_path

    def _to_display_frame(self, packet: FramePacket) -> np.ndarray:
        display_frame = packet.meta.get("display_frame")
        if isinstance(display_frame, np.ndarray):
            if display_frame.ndim == 2:
                return cv2.cvtColor(np.asarray(display_frame, dtype=np.uint8), cv2.COLOR_GRAY2BGR)
            if display_frame.ndim == 3 and display_frame.shape[2] == 3:
                return cv2.cvtColor(np.asarray(display_frame, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        if packet.is_grayscale:
            return cv2.cvtColor(packet.frame_mono, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(packet.frame_rgb, cv2.COLOR_RGB2BGR)

    def _select_trajectory_points(self, summary: RunSummary) -> np.ndarray:
        # Keyframe trajectory is loop-closure corrected and lives in the same
        # coordinate frame as map points — prefer it for all visualizations.
        if summary.trajectory_points is not None and summary.trajectory_points.size > 0:
            return summary.trajectory_points
        if summary.camera_trajectory_points is not None and summary.camera_trajectory_points.size > 0:
            return summary.camera_trajectory_points
        return np.empty((0, 3), dtype=np.float32)

    def _select_current_map_points(self, summary: RunSummary) -> np.ndarray:
        if summary.map_points is not None:
            return summary.map_points
        return np.empty((0, 3), dtype=np.float32)

    def _select_atlas_map_groups(self, summary: RunSummary) -> list[Any]:
        if summary.atlas_map_groups:
            return summary.atlas_map_groups
        if summary.all_map_points is not None and summary.all_map_points.size > 0:
            return [{"map_id": 0, "is_current": True, "points": summary.all_map_points}]
        if summary.map_points is not None and summary.map_points.size > 0:
            return [{"map_id": 0, "is_current": True, "points": summary.map_points}]
        return []

    def _atlas_group_value(self, group: Any, key: str, default: Any) -> Any:
        if isinstance(group, dict):
            return group.get(key, default)
        return getattr(group, key, default)

    def _load_plotly_graph_objects(self):
        try:
            return importlib.import_module("plotly.graph_objects")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Plotly is required for result visualization. Install it with `python3 -m pip install plotly`."
            ) from exc

    def _load_matplotlib_pyplot(self):
        try:
            matplotlib = importlib.import_module("matplotlib")
            matplotlib.use("Agg")
            return importlib.import_module("matplotlib.pyplot")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Matplotlib is required for 2D top-down visualization. Install it with `python3 -m pip install matplotlib`."
            ) from exc

    def _load_aligned_ground_truth_points(
        self,
        estimated_tum_path: str | None,
        ground_truth_tum_path: str | Path | None,
        reference_timestamps: np.ndarray | None = None,
    ) -> np.ndarray:
        if estimated_tum_path is None or ground_truth_tum_path is None:
            return np.empty((0, 3), dtype=np.float32)

        est_path = Path(estimated_tum_path)
        gt_path = Path(ground_truth_tum_path)
        if not est_path.is_file() or not gt_path.is_file():
            return np.empty((0, 3), dtype=np.float32)

        try:
            est_timestamps, est_points = _read_tum_positions(est_path)
            gt_timestamps, gt_points = _read_tum_positions(gt_path)
            matched_gt_points, matched_est_points = _associate_positions(
                gt_timestamps,
                gt_points,
                est_timestamps,
                est_points,
            )
            if matched_gt_points.shape[0] < 3:
                return np.empty((0, 3), dtype=np.float32)
            scale, rotation, translation = _umeyama_alignment(
                matched_gt_points,
                matched_est_points,
            )
            aligned_gt_points = (scale * (rotation @ gt_points.T)).T + translation

            # Trim GT to the time range covered by SLAM to hide the
            # pre-initialization gap where no poses were recorded.
            ref_ts = reference_timestamps if reference_timestamps is not None else est_timestamps
            if ref_ts.size > 0:
                t_start = float(ref_ts.min()) - 0.5
                t_end = float(ref_ts.max()) + 0.5
                time_mask = (gt_timestamps >= t_start) & (gt_timestamps <= t_end)
                aligned_gt_points = aligned_gt_points[time_mask]

            return np.asarray(aligned_gt_points, dtype=np.float32)
        except Exception as exc:
            self._logger.debug("Ground-truth overlay skipped: %s", exc)
            return np.empty((0, 3), dtype=np.float32)

    def _extract_tum_timestamps(self, tum_path: str | None) -> np.ndarray | None:
        if tum_path is None:
            return None
        path = Path(tum_path)
        if not path.is_file():
            return None
        try:
            timestamps, _ = _read_tum_positions(path)
            return timestamps if timestamps.size > 0 else None
        except Exception:
            return None


VisualizationModuleStub = VisualizationModule


def _with_trajectory_breaks(
    trajectory_points: np.ndarray,
    jump_factor: float = 8.0,
    min_jump_distance: float = 0.25,
) -> np.ndarray:
    points = np.asarray(trajectory_points, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 3:
        return points

    planar_steps = np.linalg.norm(np.diff(points[:, [0, 2]], axis=0), axis=1)
    finite_steps = planar_steps[np.isfinite(planar_steps)]
    if finite_steps.size == 0:
        return points

    median_step = float(np.median(finite_steps))
    jump_threshold = max(min_jump_distance, median_step * jump_factor)
    if np.all(planar_steps <= jump_threshold):
        return points

    broken: list[np.ndarray] = [points[0]]
    for idx in range(1, points.shape[0]):
        if planar_steps[idx - 1] > jump_threshold:
            broken.append(np.full((3,), np.nan, dtype=np.float32))
        broken.append(points[idx])
    return np.vstack(broken)


def _read_tum_positions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    timestamps: list[float] = []
    points: list[list[float]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        timestamps.append(float(parts[0]))
        points.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(timestamps, dtype=np.float64), np.asarray(points, dtype=np.float64)


def _associate_positions(
    ref_timestamps: np.ndarray,
    ref_points: np.ndarray,
    est_timestamps: np.ndarray,
    est_points: np.ndarray,
    max_diff_s: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    matched_ref: list[np.ndarray] = []
    matched_est: list[np.ndarray] = []
    if ref_timestamps.size == 0 or est_timestamps.size == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    est_idx = 0
    for ref_idx, ref_ts in enumerate(ref_timestamps):
        while est_idx + 1 < est_timestamps.size and est_timestamps[est_idx] < ref_ts:
            est_idx += 1
        candidate_indices = {est_idx}
        if est_idx > 0:
            candidate_indices.add(est_idx - 1)
        if est_idx + 1 < est_timestamps.size:
            candidate_indices.add(est_idx + 1)
        best_idx = None
        best_diff = None
        for candidate_idx in candidate_indices:
            diff = abs(float(est_timestamps[candidate_idx] - ref_ts))
            if diff > max_diff_s:
                continue
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_idx = candidate_idx
        if best_idx is None:
            continue
        matched_ref.append(ref_points[ref_idx])
        matched_est.append(est_points[best_idx])

    if not matched_ref:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    return np.asarray(matched_ref, dtype=np.float64), np.asarray(matched_est, dtype=np.float64)


def _filter_map_points_to_trajectory_region(
    map_points: np.ndarray,
    trajectory_points: np.ndarray,
    margin_factor: float = 2.0,
) -> np.ndarray:
    if map_points.size == 0:
        return map_points
    if trajectory_points.size == 0:
        return map_points
    traj = np.asarray(trajectory_points, dtype=np.float32)
    traj_min = traj.min(axis=0)
    traj_max = traj.max(axis=0)
    margin = (traj_max - traj_min) * margin_factor
    lo = traj_min - margin
    hi = traj_max + margin
    pts = np.asarray(map_points, dtype=np.float32)
    mask = np.all((pts >= lo) & (pts <= hi), axis=1)
    return pts[mask]


def _umeyama_alignment(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    if source_points.shape != target_points.shape or source_points.shape[0] < 3:
        raise ValueError("Need at least 3 matched 3D points for alignment")

    src_mean = source_points.mean(axis=0)
    tgt_mean = target_points.mean(axis=0)
    src_centered = source_points - src_mean
    tgt_centered = target_points - tgt_mean

    covariance = (tgt_centered.T @ src_centered) / float(source_points.shape[0])
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt

    src_variance = np.mean(np.sum(src_centered ** 2, axis=1))
    if src_variance <= 1e-12:
        raise ValueError("Degenerate source trajectory")
    scale = float(np.sum(singular_values * np.diag(correction)) / src_variance)
    translation = tgt_mean - scale * (rotation @ src_mean)
    return scale, rotation, translation
