from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Sequence

from image_processing import AdeMonoDepthRunner, OrbSlamVideoRunner, RunSummary, build_depth_estimator
from image_processing.trajectory_evaluation import TrajectoryMetrics, compute_trajectory_metrics
from media_sources import TimestampManifestSourceConfig, load_app_config, open_source
from visualization import VisualizationModule


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = load_app_config(args.config)
    runner_log_mode = args.runner_log_mode or config.runtime.log_mode
    if args.compare_against is not None:
        baseline_config = load_app_config(args.compare_against)
        compare_output_dir = _resolve_output_dir(config)
        _run_comparison(config, baseline_config, runner_log_mode, compare_output_dir)
        return 0

    output_dir = _resolve_output_dir(config)
    summary = _run_single_config(config, runner_log_mode, output_dir)
    _log_run_summary(summary)
    return 0


def _resolve_output_dir(config) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if config.runtime.output_dir:
        output_dir = config.runtime.output_dir.replace("{timestamp}", timestamp)
        return Path(output_dir).expanduser()
    run_name = _infer_run_name(config.source)
    return Path("runs") / run_name / timestamp


def _build_runner(config, runner_log_mode: str):
    logger = logging.getLogger("orbslam_runner")
    if config.processing.pipeline == "ade_mono_depth":
        depth_estimator = build_depth_estimator(
            config.processing.depth_model,
            logger=logging.getLogger("depth_estimator"),
        )
        prepare = getattr(depth_estimator, "prepare", None)
        if callable(prepare):
            prepare()
        return AdeMonoDepthRunner(
            config.orbslam,
            config.processing,
            depth_estimator=depth_estimator,
            log_mode=runner_log_mode,
            logger=logger,
        )
    return OrbSlamVideoRunner(
        config.orbslam,
        image_enhancement_config=config.processing.image_enhancement,
        log_mode=runner_log_mode,
        logger=logger,
    )


def _run_single_config(
    config,
    runner_log_mode: str,
    output_dir: Path,
    live_view_override: bool | None = None,
    enforce_strict_depth: bool = True,
) -> RunSummary:
    visualization = VisualizationModule(logger=logging.getLogger("visualization"))
    visualization.ensure_plot_backend()
    live_view_enabled = (
        config.runtime.live_view if live_view_override is None else live_view_override
    )
    if live_view_enabled:
        visualization.start_live_view()

    runner = _build_runner(config, runner_log_mode)
    summary = None
    try:
        with open_source(config.source) as source:
            summary = runner.run(
                source,
                max_frames=config.runtime.max_frames,
                output_dir=output_dir,
                inter_frame_delay_s=config.runtime.inter_frame_delay_s,
                on_packet=visualization.update_live_view if live_view_enabled else None,
            )
        summary = visualization.render_results(
            summary,
            output_dir,
            ground_truth_tum_path=config.runtime.ground_truth_path,
        )
    finally:
        visualization.close_live_view()
        runner.shutdown()
    if enforce_strict_depth:
        _validate_strict_depth(config, summary)
    summary = _evaluate_trajectory(config, summary, output_dir)
    return summary


def _run_comparison(
    candidate_config,
    baseline_config,
    runner_log_mode: str,
    compare_output_dir: Path,
) -> dict:
    logger = logging.getLogger("comparison")
    if candidate_config.source != baseline_config.source:
        logger.warning(
            "Comparison configs use different sources; metrics may not be directly comparable"
        )
    if candidate_config.processing.pipeline != "ade_mono_depth":
        logger.warning(
            "Primary config pipeline is '%s', not 'ade_mono_depth'; depth comparison verdict may be uninformative",
            candidate_config.processing.pipeline,
        )

    compare_output_dir.mkdir(parents=True, exist_ok=True)
    baseline_output_dir = compare_output_dir / "baseline"
    candidate_output_dir = compare_output_dir / "candidate"
    logger.info("Running baseline mono pipeline for comparison")
    baseline_summary = _run_single_config(
        baseline_config,
        runner_log_mode,
        baseline_output_dir,
        live_view_override=False,
        enforce_strict_depth=False,
    )
    _log_run_summary(baseline_summary, label="baseline")

    logger.info("Running candidate pipeline for comparison")
    candidate_summary = _run_single_config(
        candidate_config,
        runner_log_mode,
        candidate_output_dir,
        live_view_override=False,
        enforce_strict_depth=False,
    )
    _log_run_summary(candidate_summary, label="candidate")

    report = _build_comparison_report(baseline_summary, candidate_summary)
    report_path = compare_output_dir / "depth_comparison.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved comparison report: %s", report_path)
    logger.info(
        "Depth comparison verdict: overall=%s depth_effect=%s depth_guidance_frames=%d filter_applied_frames=%d baseline_first_ok=%s candidate_first_ok=%s baseline_ok_ratio=%.3f candidate_ok_ratio=%.3f",
        report["overall_verdict"],
        report["depth_effect_status"],
        report["candidate"]["depth_guidance_frame_count"],
        report["candidate"]["depth_filter_applied_frame_count"],
        report["baseline"]["first_tracking_ok_frame"],
        report["candidate"]["first_tracking_ok_frame"],
        report["baseline"]["tracking_ok_ratio"],
        report["candidate"]["tracking_ok_ratio"],
    )
    _validate_strict_depth(candidate_config, candidate_summary)
    return report


def _infer_run_name(source_config) -> str:
    if isinstance(source_config, TimestampManifestSourceConfig):
        return Path(source_config.dataset_root).name
    return getattr(source_config, "kind", "run")


def _log_run_summary(summary: RunSummary, label: str = "run") -> None:
    logging.info(
        "Finished %s: frames=%d elapsed_s=%.3f avg_input_fps=%.2f last_tracking_state=%d "
        "first_ok_frame=%s ok_frames=%d recently_lost_frames=%d lost_frames=%d output_dir=%s",
        label,
        summary.processed_frames,
        summary.elapsed_s,
        summary.average_input_fps,
        summary.last_tracking_state,
        summary.first_tracking_ok_frame,
        summary.tracking_ok_frame_count,
        summary.tracking_recently_lost_frame_count,
        summary.tracking_lost_frame_count,
        summary.output_dir,
    )
    if summary.ate_rmse is not None:
        logging.info(
            "Trajectory evaluation: ATE rmse=%.4f m mean=%.4f m | RPE rmse=%.4f m mean=%.4f m",
            summary.ate_rmse,
            summary.ate_mean,
            summary.rpe_rmse,
            summary.rpe_mean,
        )
    if summary.trajectory_tum_path and summary.keyframes_tum_path:
        logging.info(
            "Saved TUM outputs: trajectory=%s keyframes=%s",
            summary.trajectory_tum_path,
            summary.keyframes_tum_path,
        )
    if summary.trajectory_plot_path and summary.map_plot_path:
        logging.info(
            "Saved visualization outputs: trajectory_plot=%s map_plot=%s atlas_map_plot=%s topdown_plot=%s",
            summary.trajectory_plot_path,
            summary.map_plot_path,
            summary.atlas_map_plot_path,
            summary.topdown_plot_path,
        )
    if summary.depth_inference_frame_count > 0:
        logging.info(
            "Depth activity: inference_frames=%d guidance_frames=%d filter_applied_frames=%d fallback_frames=%d",
            summary.depth_inference_frame_count,
            summary.depth_guidance_frame_count,
            summary.depth_filter_applied_frame_count,
            summary.depth_fallback_frame_count,
        )


def _build_comparison_report(
    baseline_summary: RunSummary,
    candidate_summary: RunSummary,
) -> dict:
    baseline = _summary_to_report_metrics(baseline_summary)
    candidate = _summary_to_report_metrics(candidate_summary)

    initialization_verdict = _compare_initialization(
        baseline_summary.first_tracking_ok_frame,
        candidate_summary.first_tracking_ok_frame,
    )
    tracking_verdict = _compare_tracking_quality(baseline_summary, candidate_summary)

    depth_was_estimated = candidate_summary.depth_inference_frame_count > 0
    depth_was_used = candidate_summary.depth_guidance_frame_count > 0
    depth_filter_applied = candidate_summary.depth_filter_applied_frame_count > 0

    if not depth_was_estimated:
        overall_verdict = "no_depth_pipeline"
        depth_effect_status = "depth_not_estimated"
    elif not depth_was_used:
        overall_verdict = "depth_not_engaged"
        depth_effect_status = "depth_did_not_intervene"
    elif not depth_filter_applied:
        overall_verdict = "depth_never_changed_features"
        depth_effect_status = "depth_did_not_intervene"
    elif initialization_verdict["status"] == "better" or tracking_verdict["status"] == "better":
        overall_verdict = "depth_improved"
        depth_effect_status = "depth_intervened_and_improved"
    elif initialization_verdict["status"] == "worse" or tracking_verdict["status"] == "worse":
        overall_verdict = "depth_worse"
        depth_effect_status = "depth_intervened_and_worsened"
    else:
        overall_verdict = "no_clear_gain"
        depth_effect_status = "depth_intervened_no_clear_gain"

    ate_verdict = _compare_ate(baseline_summary, candidate_summary)

    return {
        "overall_verdict": overall_verdict,
        "depth_effect_status": depth_effect_status,
        "depth_was_estimated": depth_was_estimated,
        "depth_was_used": depth_was_used,
        "depth_filter_applied": depth_filter_applied,
        "baseline": baseline,
        "candidate": candidate,
        "initialization": initialization_verdict,
        "tracking": tracking_verdict,
        "ate": ate_verdict,
    }


def _summary_to_report_metrics(summary: RunSummary) -> dict:
    tracking_state_counts = summary.tracking_state_counts or {}
    processed_frames = max(summary.processed_frames, 1)
    return {
        "processed_frames": summary.processed_frames,
        "elapsed_s": summary.elapsed_s,
        "average_input_fps": summary.average_input_fps,
        "last_tracking_state": summary.last_tracking_state,
        "first_tracking_ok_frame": summary.first_tracking_ok_frame,
        "tracking_ok_frame_count": summary.tracking_ok_frame_count,
        "tracking_ok_ratio": summary.tracking_ok_frame_count / processed_frames,
        "tracking_recently_lost_frame_count": summary.tracking_recently_lost_frame_count,
        "tracking_recently_lost_ratio": summary.tracking_recently_lost_frame_count / processed_frames,
        "tracking_lost_frame_count": summary.tracking_lost_frame_count,
        "initialization_succeeded": summary.initialization_succeeded,
        "camera_pose_count": summary.camera_pose_count,
        "map_point_count": summary.map_point_count,
        "keyframe_trajectory_point_count": summary.keyframe_trajectory_point_count,
        "depth_inference_frame_count": summary.depth_inference_frame_count,
        "depth_guidance_frame_count": summary.depth_guidance_frame_count,
        "depth_filter_applied_frame_count": summary.depth_filter_applied_frame_count,
        "depth_fallback_frame_count": summary.depth_fallback_frame_count,
        "ate_rmse_m": summary.ate_rmse,
        "ate_mean_m": summary.ate_mean,
        "rpe_rmse_m": summary.rpe_rmse,
        "rpe_mean_m": summary.rpe_mean,
        "output_dir": summary.output_dir,
        "trajectory_tum_path": summary.trajectory_tum_path,
        "keyframes_tum_path": summary.keyframes_tum_path,
        "trajectory_metrics_path": summary.trajectory_metrics_path,
        "tracking_state_counts": {str(key): value for key, value in tracking_state_counts.items()},
    }


def _compare_initialization(
    baseline_first_ok: int | None,
    candidate_first_ok: int | None,
) -> dict:
    if baseline_first_ok is None and candidate_first_ok is None:
        return {
            "status": "same",
            "message": "Neither pipeline reached a stable tracking state",
        }
    if baseline_first_ok is None:
        return {
            "status": "better",
            "message": "Candidate pipeline initialized, baseline did not",
        }
    if candidate_first_ok is None:
        return {
            "status": "worse",
            "message": "Baseline initialized, candidate did not",
        }
    if candidate_first_ok < baseline_first_ok:
        return {
            "status": "better",
            "message": f"Candidate initialized earlier: frame {candidate_first_ok} vs {baseline_first_ok}",
        }
    if candidate_first_ok > baseline_first_ok:
        return {
            "status": "worse",
            "message": f"Candidate initialized later: frame {candidate_first_ok} vs {baseline_first_ok}",
        }
    return {
        "status": "same",
        "message": f"Both pipelines initialized on frame {candidate_first_ok}",
    }


def _compare_tracking_quality(
    baseline_summary: RunSummary,
    candidate_summary: RunSummary,
) -> dict:
    baseline_ratio = baseline_summary.tracking_ok_frame_count / max(baseline_summary.processed_frames, 1)
    candidate_ratio = candidate_summary.tracking_ok_frame_count / max(candidate_summary.processed_frames, 1)

    if candidate_ratio > baseline_ratio:
        return {
            "status": "better",
            "message": f"Candidate kept stable tracking longer: ok_ratio {candidate_ratio:.3f} vs {baseline_ratio:.3f}",
        }
    if candidate_ratio < baseline_ratio:
        return {
            "status": "worse",
            "message": f"Candidate kept stable tracking for fewer frames: ok_ratio {candidate_ratio:.3f} vs {baseline_ratio:.3f}",
        }

    if candidate_summary.tracking_lost_frame_count < baseline_summary.tracking_lost_frame_count:
        return {
            "status": "better",
            "message": f"Candidate lost tracking less often: {candidate_summary.tracking_lost_frame_count} vs {baseline_summary.tracking_lost_frame_count}",
        }
    if candidate_summary.tracking_lost_frame_count > baseline_summary.tracking_lost_frame_count:
        return {
            "status": "worse",
            "message": f"Candidate lost tracking more often: {candidate_summary.tracking_lost_frame_count} vs {baseline_summary.tracking_lost_frame_count}",
        }

    return {
        "status": "same",
        "message": f"Tracking quality matched baseline: ok_ratio {candidate_ratio:.3f}",
    }


def _evaluate_trajectory(config, summary: RunSummary, output_dir: Path) -> RunSummary:
    gt_path = config.runtime.ground_truth_path
    if gt_path is None or summary is None:
        return summary
    if summary.trajectory_tum_path is None:
        return summary

    logger = logging.getLogger("trajectory_eval")
    metrics = compute_trajectory_metrics(
        est_tum_path=Path(summary.trajectory_tum_path),
        gt_tum_path=Path(gt_path),
        output_dir=output_dir,
        logger=logger,
    )
    if metrics is None:
        return summary

    summary.ate_rmse = metrics.ate_rmse
    summary.ate_mean = metrics.ate_mean
    summary.rpe_rmse = metrics.rpe_rmse
    summary.rpe_mean = metrics.rpe_mean
    summary.trajectory_metrics_path = str(output_dir / "trajectory_metrics.json")
    return summary


def _compare_ate(baseline_summary: RunSummary, candidate_summary: RunSummary) -> dict:
    b_ate = baseline_summary.ate_rmse
    c_ate = candidate_summary.ate_rmse
    if b_ate is None and c_ate is None:
        return {"status": "unavailable", "message": "ATE not computed for either run (no ground_truth_path?)"}
    if b_ate is None:
        return {"status": "unavailable", "message": "ATE not computed for baseline"}
    if c_ate is None:
        return {"status": "unavailable", "message": "ATE not computed for candidate"}
    improvement_pct = (b_ate - c_ate) / b_ate * 100.0
    if c_ate < b_ate:
        status = "better"
        message = f"Candidate ATE rmse improved by {improvement_pct:.1f}%: {c_ate:.4f} m vs {b_ate:.4f} m"
    elif c_ate > b_ate:
        status = "worse"
        message = f"Candidate ATE rmse worsened by {-improvement_pct:.1f}%: {c_ate:.4f} m vs {b_ate:.4f} m"
    else:
        status = "same"
        message = f"ATE rmse unchanged: {c_ate:.4f} m"
    return {
        "status": status,
        "message": message,
        "baseline_ate_rmse_m": b_ate,
        "candidate_ate_rmse_m": c_ate,
        "baseline_rpe_rmse_m": baseline_summary.rpe_rmse,
        "candidate_rpe_rmse_m": candidate_summary.rpe_rmse,
        "improvement_pct": improvement_pct,
    }


def _validate_strict_depth(config, summary: RunSummary) -> None:
    processing = getattr(config, "processing", None)
    if processing is None or processing.pipeline != "ade_mono_depth":
        return
    guidance = getattr(processing, "depth_guidance", None)
    if guidance is None or not guidance.strict_mode:
        return
    if summary.depth_inference_frame_count <= 0:
        raise RuntimeError(
            "Strict depth mode failed: depth estimation did not run on any frame"
        )
    if summary.depth_guidance_frame_count <= 0:
        raise RuntimeError(
            "Strict depth mode failed: depth-guided path never engaged after initialization"
        )
    if summary.depth_filter_applied_frame_count <= 0:
        raise RuntimeError(
            "Strict depth mode failed: depth was estimated, but it never changed ORB features"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ORB-SLAM3 on a configured video stream or image sequence"
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to YAML or JSON config file",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level, for example INFO or DEBUG",
    )
    parser.add_argument(
        "--runner-log-mode",
        choices=("full", "fps_adaptive"),
        help="Per-frame logging strategy for the ORB-SLAM runner",
    )
    parser.add_argument(
        "--compare-against",
        type=Path,
        help="Optional baseline config. If provided, runs baseline and candidate sequentially and saves a comparison report.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
