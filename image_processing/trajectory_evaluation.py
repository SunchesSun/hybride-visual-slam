from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class TrajectoryMetrics:
    ate_rmse: float
    ate_mean: float
    ate_median: float
    ate_std: float
    ate_min: float
    ate_max: float
    rpe_rmse: float
    rpe_mean: float
    rpe_median: float
    rpe_std: float
    n_frames_matched: int
    gt_path: str
    est_path: str


def compute_trajectory_metrics(
    est_tum_path: Path,
    gt_tum_path: Path,
    output_dir: Path | None = None,
    logger: logging.Logger | None = None,
) -> TrajectoryMetrics | None:
    log = logger or logging.getLogger(__name__)

    if not est_tum_path.is_file():
        log.warning("Estimated trajectory not found: %s", est_tum_path)
        return None
    if not gt_tum_path.is_file():
        log.warning("Ground truth trajectory not found: %s", gt_tum_path)
        return None

    try:
        from evo.tools import file_interface
        from evo.core import metrics, sync
        from evo.core.metrics import PoseRelation
    except ImportError:
        log.warning("evo package not available — skipping trajectory evaluation (pip install evo)")
        return None

    try:
        traj_ref = file_interface.read_tum_trajectory_file(str(gt_tum_path))
        traj_est = file_interface.read_tum_trajectory_file(str(est_tum_path))
    except Exception as exc:
        log.warning("Failed to read trajectory files for evaluation: %s", exc)
        return None

    if len(traj_est.timestamps) < 2:
        log.warning("Estimated trajectory has fewer than 2 poses — skipping evaluation")
        return None

    try:
        traj_ref_sync, traj_est_sync = sync.associate_trajectories(
            traj_ref, traj_est, max_diff=0.05
        )
    except Exception as exc:
        log.warning("Trajectory association failed: %s", exc)
        return None

    n_matched = len(traj_est_sync.timestamps)
    if n_matched < 2:
        log.warning("Only %d poses matched between GT and estimated trajectory — skipping", n_matched)
        return None

    try:
        traj_est_aligned = _align_trajectory(traj_ref_sync, traj_est_sync)

        ape_metric = metrics.APE(PoseRelation.translation_part)
        ape_metric.process_data((traj_ref_sync, traj_est_aligned))
        ate_stats = ape_metric.get_all_statistics()

        rpe_metric = metrics.RPE(
            PoseRelation.translation_part,
            delta=1,
            delta_unit=metrics.Unit.frames,
            all_pairs=False,
        )
        rpe_metric.process_data((traj_ref_sync, traj_est_aligned))
        rpe_stats = rpe_metric.get_all_statistics()
    except Exception as exc:
        log.warning("evo metric computation failed: %s", exc)
        return None

    result = TrajectoryMetrics(
        ate_rmse=float(ate_stats["rmse"]),
        ate_mean=float(ate_stats["mean"]),
        ate_median=float(ate_stats["median"]),
        ate_std=float(ate_stats["std"]),
        ate_min=float(ate_stats["min"]),
        ate_max=float(ate_stats["max"]),
        rpe_rmse=float(rpe_stats["rmse"]),
        rpe_mean=float(rpe_stats["mean"]),
        rpe_median=float(rpe_stats["median"]),
        rpe_std=float(rpe_stats["std"]),
        n_frames_matched=n_matched,
        gt_path=str(gt_tum_path),
        est_path=str(est_tum_path),
    )

    log.info(
        "Trajectory evaluation: matched=%d ATE rmse=%.4f m mean=%.4f m | RPE rmse=%.4f m mean=%.4f m",
        n_matched,
        result.ate_rmse,
        result.ate_mean,
        result.rpe_rmse,
        result.rpe_mean,
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_frames_matched": n_matched,
            "gt_path": str(gt_tum_path),
            "est_path": str(est_tum_path),
            "ate": {
                "rmse_m": result.ate_rmse,
                "mean_m": result.ate_mean,
                "median_m": result.ate_median,
                "std_m": result.ate_std,
                "min_m": result.ate_min,
                "max_m": result.ate_max,
            },
            "rpe": {
                "rmse_m": result.rpe_rmse,
                "mean_m": result.rpe_mean,
                "median_m": result.rpe_median,
                "std_m": result.rpe_std,
            },
        }
        (output_dir / "trajectory_metrics.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    return result


def _align_trajectory(traj_ref, traj_est):
    from evo.core import trajectory as evo_traj

    traj_est_copy = evo_traj.PoseTrajectory3D(
        poses_se3=traj_est.poses_se3.copy(),
        timestamps=traj_est.timestamps.copy(),
    )
    traj_est_copy.align(traj_ref, correct_scale=True)
    return traj_est_copy
