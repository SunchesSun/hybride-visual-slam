from __future__ import annotations

import csv
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
import pyorbslam3_custom

from media_sources.config import OrbSlamConfig, ProcessingConfig
from media_sources.contracts import FramePacket

from .depth_estimation import DepthEstimator, DepthPrediction
from .image_enhancement import LowLightEnhancer
from .orbslam_runner import (
    RunSummary,
    _build_tracking_metrics,
    _get_atlas_map_groups,
    _get_point_array,
    _rotation_matrix_to_quaternion,
    _trajectory_records_to_points,
    _write_atlas_map_groups,
)


@dataclass(slots=True)
class FeatureFilterLogEntry:
    frame_index: int
    timestamp_s: float
    raw_features: int
    valid_depth_features: int
    enhanced_features: int
    final_features: int
    far_points_removed: int
    invalid_depth_ratio: float
    filter_applied: bool
    fallback_to_raw: bool
    guidance_active: bool
    tracking_state: int
    inference_ms: float
    depth_score: float = 0.0
    effective_depth_threshold_m: float = 0.0


class AdeMonoDepthRunner:
    def __init__(
        self,
        config: OrbSlamConfig,
        processing: ProcessingConfig,
        depth_estimator: DepthEstimator,
        log_mode: str = "full",
        logger: logging.Logger | None = None,
    ) -> None:
        if processing.pipeline != "ade_mono_depth":
            raise ValueError("AdeMonoDepthRunner requires processing.pipeline=ade_mono_depth")
        if processing.depth_model is None:
            raise ValueError("AdeMonoDepthRunner requires processing.depth_model")
        self._config = config
        self._processing = processing
        self._depth_estimator = depth_estimator
        if log_mode not in {"full", "fps_adaptive"}:
            raise ValueError("log_mode must be 'full' or 'fps_adaptive'")
        self._log_mode = log_mode
        self._logger = logger or logging.getLogger(__name__)
        self._system = self._create_system(config)
        self._enhancer = LowLightEnhancer(processing.image_enhancement)
        self._started_at: float | None = None
        self._processed_frames = 0
        self._last_tracking_state = 0
        self._is_shutdown = False
        self._trajectory_records: list[tuple[float, np.ndarray]] = []
        self._tracking_state_history: list[tuple[int, int]] = []
        self._depth_frame_stats: list[dict[str, float]] = []
        self._feature_filter_rows: list[FeatureFilterLogEntry] = []
        self._last_depth_preview: np.ndarray | None = None
        self._last_depth_prediction: DepthPrediction | None = None
        self._stable_ok_frames = 0
        self._guidance_cooldown_remaining = 0
        self._last_depth_score: float = 0.0

    def run(
        self,
        packets: Iterable[FramePacket],
        max_frames: int | None = None,
        output_dir: str | Path | None = None,
        inter_frame_delay_s: float = 0.0,
        on_packet: Callable[[FramePacket, int, float], None] | None = None,
    ) -> RunSummary:
        output_path = Path(output_dir) if output_dir is not None else None
        runtime_dirs = self._prepare_output_dirs(output_path)
        try:
            for packet in packets:
                if self._processed_frames > 0 and inter_frame_delay_s > 0.0:
                    time.sleep(inter_frame_delay_s)
                tracking_state, input_fps = self.process_packet(packet, runtime_dirs)
                if on_packet is not None:
                    on_packet(packet, tracking_state, input_fps)
                if max_frames and self._processed_frames >= max_frames:
                    self._logger.info("Reached runtime.max_frames=%d", max_frames)
                    break
        except KeyboardInterrupt:
            self._logger.info("Interrupted by user")

        summary = self._build_summary()
        if output_path is not None:
            summary = self._persist_run_outputs(output_path, runtime_dirs, summary)

        self.shutdown()
        summary.elapsed_s = self._compute_elapsed_s()
        summary.average_input_fps = (
            self._processed_frames / summary.elapsed_s if summary.elapsed_s > 0 else 0.0
        )
        return summary

    def process_packet(
        self,
        packet: FramePacket,
        runtime_dirs: dict[str, Path | None] | None = None,
    ) -> tuple[int, float]:
        if self._started_at is None:
            self._started_at = time.monotonic()

        frame_gray = packet.frame_mono if packet.is_grayscale else None
        frame_rgb = packet.frame_rgb if packet.is_rgb else None
        enhanced = self._enhancer.enhance_for_packet(frame_gray, frame_rgb)
        frame_gray = enhanced.frame_gray
        frame_rgb = enhanced.frame_rgb
        if self._enhancer.enabled:
            packet.meta["display_frame"] = frame_rgb
        depth_prediction = self._predict_depth(packet, frame_rgb, runtime_dirs or {})
        depth_map = np.asarray(depth_prediction.depth_m, dtype=np.float32)
        depth_preview = _normalize_depth_preview(depth_map)
        packet.meta["depth_preview"] = depth_preview

        depth_score = _compute_depth_score(depth_map)
        self._last_depth_score = depth_score
        guidance_active = self._should_apply_guidance(depth_score)
        guidance_cfg = self._build_guidance_config(depth_map) if guidance_active else None
        effective_threshold = (
            guidance_cfg.depth_threshold_m if guidance_cfg is not None
            else self._processing.depth_guidance.depth_threshold_m
        )
        if guidance_active:
            self._system.process_mono_depth_guided(
                frame_gray,
                depth_map,
                packet.timestamp_s,
                guidance_cfg,
                packet.frame_index,
            )
        else:
            self._system.process_mono(frame_gray, packet.timestamp_s)
        self._processed_frames += 1
        self._last_tracking_state = self._system.get_tracking_state()
        self._tracking_state_history.append((packet.frame_index, self._last_tracking_state))
        self._update_guidance_recovery_state()
        self._record_pose(packet.timestamp_s, self._last_tracking_state)
        self._record_depth_stats(packet, depth_prediction, depth_map)
        self._record_feature_filter_stats(
            packet, depth_prediction, guidance_active, depth_score, effective_threshold
        )
        self._last_depth_preview = depth_preview
        self._last_depth_prediction = depth_prediction

        elapsed_s = max(time.monotonic() - self._started_at, 1e-6)
        input_fps = self._processed_frames / elapsed_s
        if self._should_log_frame(input_fps):
            self._logger.info(
                "frame=%d source=%s timestamp=%.3f input_fps=%.2f tracking_state=%d mode=ade_mono_depth depth_ms=%.2f depth_score=%.3f guidance=%s log_mode=%s",
                packet.frame_index,
                packet.source_kind,
                packet.timestamp_s,
                input_fps,
                self._last_tracking_state,
                depth_prediction.inference_ms,
                depth_score,
                "on" if guidance_active else self._guidance_status_label(),
                self._log_mode,
            )
        return self._last_tracking_state, input_fps

    def shutdown(self) -> None:
        if self._is_shutdown:
            return
        self._system.shutdown()
        self._is_shutdown = True

    def _predict_depth(
        self,
        packet: FramePacket,
        frame_rgb: np.ndarray,
        runtime_dirs: dict[str, Path | None],
    ) -> DepthPrediction:
        depth_model = self._processing.depth_model
        if depth_model is None:
            raise RuntimeError("Depth model config is missing")

        if (
            depth_model.inference_stride > 1
            and self._processed_frames > 0
            and self._processed_frames % depth_model.inference_stride != 0
            and self._last_depth_prediction is not None
        ):
            return DepthPrediction(
                depth_m=np.asarray(self._last_depth_prediction.depth_m, dtype=np.float32),
                confidence=self._last_depth_prediction.confidence,
                backend=self._last_depth_prediction.backend,
                inference_ms=0.0,
            )

        cache_dir = runtime_dirs.get("depth_cache")
        cache_path = None
        if cache_dir is not None:
            cache_path = cache_dir / f"frame_{packet.frame_index:06d}.npy"
            if cache_path.is_file():
                depth_map = np.load(cache_path)
                return DepthPrediction(
                    depth_m=np.asarray(depth_map, dtype=np.float32),
                    confidence=None,
                    backend=self._processing.depth_model.backend,
                    inference_ms=0.0,
                )

        prediction = self._depth_estimator.predict(frame_rgb)
        if cache_path is not None:
            np.save(cache_path, prediction.depth_m)

        depth_npy_dir = runtime_dirs.get("depth_npy")
        if depth_npy_dir is not None:
            np.save(depth_npy_dir / f"frame_{packet.frame_index:06d}.npy", prediction.depth_m)

        depth_preview_dir = runtime_dirs.get("depth_preview")
        if depth_preview_dir is not None:
            cv2.imwrite(
                str(depth_preview_dir / f"frame_{packet.frame_index:06d}.png"),
                _normalize_depth_preview(prediction.depth_m),
            )
        return prediction

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
            depth_inference_frame_count=len(self._depth_frame_stats),
            depth_guidance_frame_count=sum(1 for row in self._feature_filter_rows if row.guidance_active),
            depth_filter_applied_frame_count=sum(
                1 for row in self._feature_filter_rows if row.filter_applied
            ),
            depth_fallback_frame_count=sum(1 for row in self._feature_filter_rows if row.fallback_to_raw),
        )

    def _persist_run_outputs(
        self,
        output_dir: Path,
        runtime_dirs: dict[str, Path | None],
        summary: RunSummary,
    ) -> RunSummary:
        output_dir.mkdir(parents=True, exist_ok=True)

        trajectory_tum_path = output_dir / "trajectory.tum"
        keyframes_tum_path = output_dir / "keyframes.tum"
        map_points_path = output_dir / "map_points.npy"
        all_map_points_path = output_dir / "all_map_points.npy"
        atlas_maps_dir = output_dir / "atlas_maps"
        atlas_map_summary_path = output_dir / "atlas_maps_summary.json"
        trajectory_points_path = output_dir / "trajectory_points.npy"
        camera_trajectory_points_path = output_dir / "camera_trajectory_points.npy"
        depth_stats_path = output_dir / "depth_stats.json"
        feature_filter_log_path = output_dir / "feature_filter_log.csv"
        depth_preview_last_path = output_dir / "depth_preview_last.png"

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

        if self._last_depth_preview is not None:
            cv2.imwrite(str(depth_preview_last_path), self._last_depth_preview)
        self._write_depth_stats(depth_stats_path)
        self._write_feature_filter_log(feature_filter_log_path)

        summary.output_dir = str(output_dir)
        summary.trajectory_tum_path = str(trajectory_tum_path)
        summary.keyframes_tum_path = str(keyframes_tum_path)
        summary.map_points_path = str(map_points_path)
        summary.all_map_points_path = str(all_map_points_path)
        summary.atlas_maps_dir = str(atlas_maps_dir)
        summary.atlas_map_summary_path = str(atlas_map_summary_path)
        summary.trajectory_points_path = str(trajectory_points_path)
        summary.camera_trajectory_points_path = str(camera_trajectory_points_path)
        summary.depth_stats_path = str(depth_stats_path)
        summary.depth_preview_last_path = str(depth_preview_last_path)
        summary.feature_filter_log_path = str(feature_filter_log_path)
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

    def _prepare_output_dirs(self, output_dir: Path | None) -> dict[str, Path | None]:
        if output_dir is None:
            return {
                "depth_cache": None,
                "depth_npy": None,
                "depth_preview": None,
            }

        output_dir.mkdir(parents=True, exist_ok=True)
        depth_cache_dir = None
        if self._processing.depth_guidance.cache_depth:
            depth_cache_dir = output_dir / "depth_cache"
            depth_cache_dir.mkdir(parents=True, exist_ok=True)

        depth_npy_dir = None
        depth_preview_dir = None
        if self._processing.depth_guidance.save_depth_debug:
            depth_npy_dir = output_dir / "depth_npy"
            depth_preview_dir = output_dir / "depth_preview"
            depth_npy_dir.mkdir(parents=True, exist_ok=True)
            depth_preview_dir.mkdir(parents=True, exist_ok=True)

        return {
            "depth_cache": depth_cache_dir,
            "depth_npy": depth_npy_dir,
            "depth_preview": depth_preview_dir,
        }

    def _record_depth_stats(
        self,
        packet: FramePacket,
        prediction: DepthPrediction,
        depth_map: np.ndarray,
    ) -> None:
        self._depth_frame_stats.append(
            {
                "frame_index": float(packet.frame_index),
                "timestamp_s": float(packet.timestamp_s),
                "depth_min_m": float(np.nanmin(depth_map)),
                "depth_max_m": float(np.nanmax(depth_map)),
                "depth_mean_m": float(np.nanmean(depth_map)),
                "inference_ms": float(prediction.inference_ms),
            }
        )

    def _record_feature_filter_stats(
        self,
        packet: FramePacket,
        prediction: DepthPrediction,
        guidance_active: bool,
        depth_score: float = 0.0,
        effective_depth_threshold_m: float = 0.0,
    ) -> None:
        stats = self._system.get_last_depth_guidance_stats() if guidance_active else {}
        self._feature_filter_rows.append(
            FeatureFilterLogEntry(
                frame_index=packet.frame_index,
                timestamp_s=packet.timestamp_s,
                raw_features=int(stats.get("raw_features", 0)),
                valid_depth_features=int(stats.get("valid_depth_features", 0)),
                enhanced_features=int(stats.get("enhanced_features", 0)),
                final_features=int(stats.get("final_features", 0)),
                far_points_removed=int(stats.get("far_points_removed", 0)),
                invalid_depth_ratio=float(stats.get("invalid_depth_ratio", 0.0)),
                filter_applied=bool(stats.get("filter_applied", False)),
                fallback_to_raw=bool(stats.get("fallback_to_raw", False)),
                guidance_active=guidance_active,
                tracking_state=self._last_tracking_state,
                inference_ms=float(prediction.inference_ms),
                depth_score=depth_score,
                effective_depth_threshold_m=effective_depth_threshold_m,
            )
        )

    def _write_depth_stats(self, output_path: Path) -> None:
        if not self._depth_frame_stats:
            payload = {
                "backend": self._processing.depth_model.backend,
                "frames": 0,
                "avg_inference_ms": 0.0,
            }
        else:
            payload = {
                "backend": self._processing.depth_model.backend,
                "frames": len(self._depth_frame_stats),
                "avg_inference_ms": float(np.mean([row["inference_ms"] for row in self._depth_frame_stats])),
                "depth_min_m": float(np.min([row["depth_min_m"] for row in self._depth_frame_stats])),
                "depth_max_m": float(np.max([row["depth_max_m"] for row in self._depth_frame_stats])),
                "depth_mean_m": float(np.mean([row["depth_mean_m"] for row in self._depth_frame_stats])),
                "last_frame_index": int(self._depth_frame_stats[-1]["frame_index"]),
            }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_feature_filter_log(self, output_path: Path) -> None:
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "frame_index",
                    "timestamp_s",
                    "raw_features",
                    "valid_depth_features",
                    "enhanced_features",
                    "final_features",
                    "far_points_removed",
                    "invalid_depth_ratio",
                    "filter_applied",
                    "fallback_to_raw",
                    "guidance_active",
                    "tracking_state",
                    "inference_ms",
                    "depth_score",
                    "effective_depth_threshold_m",
                ]
            )
            for row in self._feature_filter_rows:
                writer.writerow(
                    [
                        row.frame_index,
                        f"{row.timestamp_s:.9f}",
                        row.raw_features,
                        row.valid_depth_features,
                        row.enhanced_features,
                        row.final_features,
                        row.far_points_removed,
                        f"{row.invalid_depth_ratio:.6f}",
                        int(row.filter_applied),
                        int(row.fallback_to_raw),
                        int(row.guidance_active),
                        row.tracking_state,
                        f"{row.inference_ms:.3f}",
                        f"{row.depth_score:.4f}",
                        f"{row.effective_depth_threshold_m:.3f}",
                    ]
                )

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

    def _build_guidance_config(self, depth_map: np.ndarray | None = None):
        guidance = pyorbslam3_custom.DepthGuidanceConfig()
        cfg = self._processing.depth_guidance
        if cfg.adaptive_depth_threshold and depth_map is not None:
            valid = depth_map[np.isfinite(depth_map) & (depth_map > 0)]
            if valid.size >= 10:
                guidance.depth_threshold_m = float(np.percentile(valid, cfg.adaptive_depth_percentile))
            else:
                guidance.depth_threshold_m = cfg.depth_threshold_m
        else:
            guidance.depth_threshold_m = cfg.depth_threshold_m
        guidance.far_point_drop_probability = self._processing.depth_guidance.far_point_drop_probability
        guidance.recursive_keep_ratio = self._processing.depth_guidance.recursive_keep_ratio
        guidance.max_recursion_depth = self._processing.depth_guidance.max_recursion_depth
        guidance.min_block_points = self._processing.depth_guidance.min_block_points
        guidance.min_feature_retention_ratio = self._processing.depth_guidance.min_feature_retention_ratio
        guidance.min_feature_retention_count = self._processing.depth_guidance.min_feature_retention_count
        guidance.max_invalid_depth_ratio = self._processing.depth_guidance.max_invalid_depth_ratio
        return guidance

    def _should_apply_guidance(self, depth_score: float = 0.0) -> bool:
        if self._guidance_cooldown_remaining > 0:
            return False
        cfg = self._processing.depth_guidance
        min_score = cfg.min_depth_score_for_guidance
        if min_score > 0.0 and depth_score < min_score:
            return False
        if self._last_tracking_state == 1:  # NOT_INITIALIZED
            return cfg.guidance_on_init
        if self._last_tracking_state == 3:  # RECENTLY_LOST
            return cfg.guidance_on_recovery
        if self._stable_ok_frames < cfg.activation_min_ok_frames:
            return False
        return self._last_tracking_state in {2, 5}

    def _update_guidance_recovery_state(self) -> None:
        if self._guidance_cooldown_remaining > 0:
            self._guidance_cooldown_remaining -= 1

        if self._last_tracking_state in {2, 5}:
            self._stable_ok_frames += 1
            return

        self._stable_ok_frames = 0
        cfg = self._processing.depth_guidance
        if self._last_tracking_state == 4:  # LOST — full cooldown always
            self._guidance_cooldown_remaining = cfg.recovery_cooldown_frames
        elif self._last_tracking_state == 3 and not cfg.guidance_on_recovery:
            # RECENTLY_LOST without recovery guidance — cooldown as before
            self._guidance_cooldown_remaining = cfg.recovery_cooldown_frames

    def _guidance_status_label(self) -> str:
        if self._guidance_cooldown_remaining > 0:
            return "cooldown"
        cfg = self._processing.depth_guidance
        if cfg.min_depth_score_for_guidance > 0.0 and self._last_depth_score < cfg.min_depth_score_for_guidance:
            return "low_score"
        if self._stable_ok_frames < cfg.activation_min_ok_frames:
            return "warmup"
        return "off"

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


def _compute_depth_score(depth_map: np.ndarray) -> float:
    valid = depth_map[np.isfinite(depth_map) & (depth_map > 0)]
    if valid.size < 10:
        return 0.0
    mean = float(np.mean(valid))
    if mean < 1e-6:
        return 0.0
    return float(np.std(valid) / mean)


def _normalize_depth_preview(depth_map: np.ndarray) -> np.ndarray:
    finite_mask = np.isfinite(depth_map)
    if not np.any(finite_mask):
        return np.zeros(depth_map.shape, dtype=np.uint8)
    finite_values = depth_map[finite_mask]
    min_value = float(np.percentile(finite_values, 2.0))
    max_value = float(np.percentile(finite_values, 98.0))
    if max_value <= min_value:
        max_value = min_value + 1e-6
    sanitized = np.where(finite_mask, depth_map, min_value)
    normalized = np.clip((sanitized - min_value) / (max_value - min_value), 0.0, 1.0)
    return np.asarray(normalized * 255.0, dtype=np.uint8)
