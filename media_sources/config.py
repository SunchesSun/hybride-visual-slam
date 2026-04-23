from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from .errors import ConfigError


@dataclass(frozen=True, slots=True)
class UdpSourceConfig:
    kind: str
    uri: str
    reconnect: bool = True
    reconnect_delay_s: float = 1.0


@dataclass(frozen=True, slots=True)
class VideoFileSourceConfig:
    kind: str
    path: str
    fps: float | None = None


@dataclass(frozen=True, slots=True)
class ImageSequenceSourceConfig:
    kind: str
    glob: str
    fps: float


@dataclass(frozen=True, slots=True)
class TimestampManifestSourceConfig:
    kind: str
    manifest_path: str
    dataset_root: str
    replay_realtime: bool = True


SourceConfig = (
    UdpSourceConfig
    | VideoFileSourceConfig
    | ImageSequenceSourceConfig
    | TimestampManifestSourceConfig
)


@dataclass(frozen=True, slots=True)
class OrbSlamConfig:
    vocab_path: str
    settings_path: str
    sensor: str = "MONOCULAR"
    use_viewer: bool = False


@dataclass(frozen=True, slots=True)
class DepthModelConfig:
    backend: str
    weights_path: str | None = None
    device: str = "cpu"
    input_size: int | None = None
    pad_input: bool = True
    with_flip_aug: bool = False
    use_half: bool = True
    inference_stride: int = 1
    model_variant: str = "N"


@dataclass(frozen=True, slots=True)
class DepthGuidanceConfig:
    depth_threshold_m: float = 30.0
    far_point_drop_probability: float = 0.5
    recursive_keep_ratio: float = 0.75
    max_recursion_depth: int = 3
    min_block_points: int = 12
    min_feature_retention_ratio: float = 0.9
    min_feature_retention_count: int = 400
    max_invalid_depth_ratio: float = 0.2
    activation_min_ok_frames: int = 15
    recovery_cooldown_frames: int = 45
    guidance_on_init: bool = False
    guidance_on_recovery: bool = False
    adaptive_depth_threshold: bool = False
    adaptive_depth_percentile: float = 85.0
    min_depth_score_for_guidance: float = 0.0
    strict_mode: bool = False
    save_depth_debug: bool = False
    cache_depth: bool = False


@dataclass(frozen=True, slots=True)
class ImageEnhancementConfig:
    enabled: bool = False
    clahe_clip_limit: float = 2.5
    clahe_tile_grid_size: int = 8
    gamma: float = 0.85
    denoise_strength: float = 0.0
    sharpen_amount: float = 0.15


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    depth_input_size: int | None = None
    depth_inference_stride: int | None = None
    low_light_denoise_strength: float | None = None


@dataclass(frozen=True, slots=True)
class ProcessingConfig:
    pipeline: str = "orbslam"
    depth_model: DepthModelConfig | None = None
    depth_guidance: DepthGuidanceConfig = field(default_factory=DepthGuidanceConfig)
    image_enhancement: ImageEnhancementConfig = field(default_factory=ImageEnhancementConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    max_frames: int | None = None
    output_dir: str | None = None
    live_view: bool = True
    log_mode: str = "full"
    ground_truth_path: str | None = None
    inter_frame_delay_s: float = 0.0


@dataclass(frozen=True, slots=True)
class AppConfig:
    source: SourceConfig
    orbslam: OrbSlamConfig
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def load_app_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    raw = _load_mapping(config_path)
    return parse_app_config(raw, base_dir=config_path.parent)


def parse_app_config(data: Mapping[str, Any], base_dir: str | Path = ".") -> AppConfig:
    if not isinstance(data, Mapping):
        raise ConfigError("Top-level config must be a mapping")

    source = parse_source_config(_require_mapping(data, "source"), base_dir=base_dir)
    orbslam = _parse_orbslam_config(_require_mapping(data, "orbslam"), base_dir=base_dir)
    processing = _parse_processing_config(data.get("processing"), base_dir=base_dir)
    runtime = _parse_runtime_config(data.get("runtime"), base_dir=base_dir)
    if processing.pipeline == "ade_mono_depth" and orbslam.sensor != "MONOCULAR":
        raise ConfigError("processing.pipeline=ade_mono_depth requires orbslam.sensor=MONOCULAR")
    return AppConfig(source=source, orbslam=orbslam, processing=processing, runtime=runtime)


def parse_source_config(data: Mapping[str, Any], base_dir: str | Path = ".") -> SourceConfig:
    if not isinstance(data, Mapping):
        raise ConfigError("source must be a mapping")

    kind = _require_non_empty_string(data.get("kind"), "source.kind")
    if kind == "udp":
        uri = _require_non_empty_string(data.get("uri"), "source.uri")
        reconnect = _require_bool(data.get("reconnect", True), "source.reconnect")
        reconnect_delay_s = _require_positive_float(
            data.get("reconnect_delay_s", 1.0),
            "source.reconnect_delay_s",
            allow_zero=False,
        )
        return UdpSourceConfig(
            kind=kind,
            uri=uri,
            reconnect=reconnect,
            reconnect_delay_s=reconnect_delay_s,
        )

    if kind == "video_file":
        raw_path = _require_non_empty_string(data.get("path"), "source.path")
        resolved_path = _resolve_path(raw_path, base_dir)
        if not resolved_path.is_file():
            raise ConfigError(f"source.path does not exist: {resolved_path}")
        fps = data.get("fps")
        return VideoFileSourceConfig(
            kind=kind,
            path=str(resolved_path),
            fps=None if fps is None else _require_positive_float(fps, "source.fps"),
        )

    if kind == "image_sequence":
        glob_pattern = _require_non_empty_string(data.get("glob"), "source.glob")
        fps = _require_positive_float(data.get("fps"), "source.fps")
        return ImageSequenceSourceConfig(
            kind=kind,
            glob=_resolve_glob_pattern(glob_pattern, base_dir),
            fps=fps,
        )

    if kind == "timestamp_manifest":
        raw_manifest_path = _require_non_empty_string(
            data.get("manifest_path"),
            "source.manifest_path",
        )
        raw_dataset_root = _require_non_empty_string(
            data.get("dataset_root"),
            "source.dataset_root",
        )
        manifest_path = _resolve_path(raw_manifest_path, base_dir)
        dataset_root = _resolve_path(raw_dataset_root, base_dir)
        if not manifest_path.is_file():
            raise ConfigError(f"source.manifest_path does not exist: {manifest_path}")
        if not dataset_root.is_dir():
            raise ConfigError(f"source.dataset_root does not exist: {dataset_root}")
        replay_realtime = _require_bool(
            data.get("replay_realtime", True),
            "source.replay_realtime",
        )
        return TimestampManifestSourceConfig(
            kind=kind,
            manifest_path=str(manifest_path),
            dataset_root=str(dataset_root),
            replay_realtime=replay_realtime,
        )

    raise ConfigError(f"Unsupported source.kind: {kind}")


def _parse_orbslam_config(data: Mapping[str, Any], base_dir: str | Path) -> OrbSlamConfig:
    vocab_path = _require_non_empty_string(data.get("vocab_path"), "orbslam.vocab_path")
    settings_path = _require_non_empty_string(
        data.get("settings_path"),
        "orbslam.settings_path",
    )
    sensor = _require_non_empty_string(data.get("sensor", "MONOCULAR"), "orbslam.sensor")
    if sensor != "MONOCULAR":
        raise ConfigError("orbslam.sensor must be MONOCULAR")
    use_viewer = _require_bool(data.get("use_viewer", False), "orbslam.use_viewer")
    return OrbSlamConfig(
        vocab_path=str(_resolve_path(vocab_path, base_dir)),
        settings_path=str(_resolve_path(settings_path, base_dir)),
        sensor=sensor,
        use_viewer=use_viewer,
    )


def _parse_runtime_config(data: Any, base_dir: str | Path) -> RuntimeConfig:
    if data is None:
        return RuntimeConfig()
    if not isinstance(data, Mapping):
        raise ConfigError("runtime must be a mapping")

    raw_max_frames = data.get("max_frames")
    output_dir = data.get("output_dir")
    live_view = _require_bool(data.get("live_view", True), "runtime.live_view")
    log_mode = _require_log_mode(data.get("log_mode", "full"), "runtime.log_mode")
    inter_frame_delay_s = _require_positive_float(
        data.get("inter_frame_delay_s", 0.0),
        "runtime.inter_frame_delay_s",
        allow_zero=True,
    )
    resolved_output_dir = None
    if output_dir is not None:
        resolved_output_dir = str(
            _resolve_path(_require_non_empty_string(output_dir, "runtime.output_dir"), base_dir)
        )
    raw_gt_path = data.get("ground_truth_path")
    resolved_gt_path = None
    if raw_gt_path is not None:
        resolved_gt_path = str(
            _resolve_path(_require_non_empty_string(raw_gt_path, "runtime.ground_truth_path"), base_dir)
        )
    if raw_max_frames is None:
        return RuntimeConfig(
            output_dir=resolved_output_dir,
            live_view=live_view,
            log_mode=log_mode,
            ground_truth_path=resolved_gt_path,
            inter_frame_delay_s=inter_frame_delay_s,
        )
    if isinstance(raw_max_frames, bool) or not isinstance(raw_max_frames, int) or raw_max_frames <= 0:
        raise ConfigError("runtime.max_frames must be a positive integer")
    return RuntimeConfig(
        max_frames=raw_max_frames,
        output_dir=resolved_output_dir,
        live_view=live_view,
        log_mode=log_mode,
        ground_truth_path=resolved_gt_path,
        inter_frame_delay_s=inter_frame_delay_s,
    )


def _parse_processing_config(data: Any, base_dir: str | Path) -> ProcessingConfig:
    if data is None:
        return ProcessingConfig()
    if not isinstance(data, Mapping):
        raise ConfigError("processing must be a mapping")

    pipeline = _require_pipeline(data.get("pipeline", "orbslam"), "processing.pipeline")
    depth_model_data = data.get("depth_model")
    depth_guidance_data = data.get("depth_guidance")

    if pipeline != "ade_mono_depth":
        return ProcessingConfig(
            pipeline=pipeline,
            image_enhancement=_parse_image_enhancement_config(data.get("image_enhancement")),
            optimization=_parse_optimization_config(data.get("optimization")),
        )

    if not isinstance(depth_model_data, Mapping):
        raise ConfigError("processing.depth_model must be a mapping for ade_mono_depth pipeline")

    depth_model = _parse_depth_model_config(depth_model_data, base_dir=base_dir)
    depth_guidance = _parse_depth_guidance_config(depth_guidance_data)
    image_enhancement = _parse_image_enhancement_config(data.get("image_enhancement"))
    optimization = _parse_optimization_config(data.get("optimization"))
    depth_model, image_enhancement = _apply_optimization_overrides(
        depth_model,
        image_enhancement,
        optimization,
    )
    return ProcessingConfig(
        pipeline=pipeline,
        depth_model=depth_model,
        depth_guidance=depth_guidance,
        image_enhancement=image_enhancement,
        optimization=optimization,
    )


def _parse_depth_model_config(data: Mapping[str, Any], base_dir: str | Path) -> DepthModelConfig:
    backend = _require_depth_backend(data.get("backend", "zoedepth"), "processing.depth_model.backend")
    raw_weights_path = data.get("weights_path")
    weights_path: str | None = None
    if raw_weights_path is not None:
        normalized_weights_spec = _require_non_empty_string(
            raw_weights_path,
            "processing.depth_model.weights_path",
        )
        if normalized_weights_spec.lower() not in {"auto", "official", "pretrained"}:
            resolved_weights_path = _resolve_path(normalized_weights_spec, base_dir)
            if not resolved_weights_path.is_file():
                raise ConfigError(
                    f"processing.depth_model.weights_path does not exist: {resolved_weights_path}"
                )
            weights_path = str(resolved_weights_path)
    device = _require_device(data.get("device", "cpu"), "processing.depth_model.device")
    raw_input_size = data.get("input_size")
    input_size = None
    if raw_input_size is not None:
        input_size = _require_positive_int(raw_input_size, "processing.depth_model.input_size")
    pad_input = _require_bool(
        data.get("pad_input", True),
        "processing.depth_model.pad_input",
    )
    with_flip_aug = _require_bool(
        data.get("with_flip_aug", False),
        "processing.depth_model.with_flip_aug",
    )
    use_half = _require_bool(
        data.get("use_half", True),
        "processing.depth_model.use_half",
    )
    inference_stride = _require_positive_int(
        data.get("inference_stride", 1),
        "processing.depth_model.inference_stride",
    )
    raw_variant = data.get("model_variant", "N")
    model_variant = _require_non_empty_string(raw_variant, "processing.depth_model.model_variant").upper()
    if model_variant not in {"N", "K", "NK"}:
        raise ConfigError(
            f"processing.depth_model.model_variant must be one of N, K, NK — got '{model_variant}'"
        )
    return DepthModelConfig(
        backend=backend,
        weights_path=weights_path,
        device=device,
        input_size=input_size,
        pad_input=pad_input,
        with_flip_aug=with_flip_aug,
        use_half=use_half,
        inference_stride=inference_stride,
        model_variant=model_variant,
    )


def _parse_depth_guidance_config(data: Any) -> DepthGuidanceConfig:
    if data is None:
        return DepthGuidanceConfig()
    if not isinstance(data, Mapping):
        raise ConfigError("processing.depth_guidance must be a mapping")

    return DepthGuidanceConfig(
        depth_threshold_m=_require_positive_float(
            data.get("depth_threshold_m", 30.0),
            "processing.depth_guidance.depth_threshold_m",
        ),
        far_point_drop_probability=_require_probability(
            data.get("far_point_drop_probability", 0.5),
            "processing.depth_guidance.far_point_drop_probability",
        ),
        recursive_keep_ratio=_require_probability(
            data.get("recursive_keep_ratio", 0.75),
            "processing.depth_guidance.recursive_keep_ratio",
            min_inclusive=False,
        ),
        max_recursion_depth=_require_non_negative_int(
            data.get("max_recursion_depth", 3),
            "processing.depth_guidance.max_recursion_depth",
        ),
        min_block_points=_require_positive_int(
            data.get("min_block_points", 12),
            "processing.depth_guidance.min_block_points",
        ),
        min_feature_retention_ratio=_require_probability(
            data.get("min_feature_retention_ratio", 0.9),
            "processing.depth_guidance.min_feature_retention_ratio",
            min_inclusive=False,
        ),
        min_feature_retention_count=_require_non_negative_int(
            data.get("min_feature_retention_count", 400),
            "processing.depth_guidance.min_feature_retention_count",
        ),
        max_invalid_depth_ratio=_require_probability(
            data.get("max_invalid_depth_ratio", 0.2),
            "processing.depth_guidance.max_invalid_depth_ratio",
        ),
        activation_min_ok_frames=_require_non_negative_int(
            data.get("activation_min_ok_frames", 15),
            "processing.depth_guidance.activation_min_ok_frames",
        ),
        recovery_cooldown_frames=_require_non_negative_int(
            data.get("recovery_cooldown_frames", 45),
            "processing.depth_guidance.recovery_cooldown_frames",
        ),
        guidance_on_init=_require_bool(
            data.get("guidance_on_init", False),
            "processing.depth_guidance.guidance_on_init",
        ),
        guidance_on_recovery=_require_bool(
            data.get("guidance_on_recovery", False),
            "processing.depth_guidance.guidance_on_recovery",
        ),
        adaptive_depth_threshold=_require_bool(
            data.get("adaptive_depth_threshold", False),
            "processing.depth_guidance.adaptive_depth_threshold",
        ),
        adaptive_depth_percentile=_require_positive_float(
            data.get("adaptive_depth_percentile", 85.0),
            "processing.depth_guidance.adaptive_depth_percentile",
        ),
        min_depth_score_for_guidance=_require_non_negative_float(
            data.get("min_depth_score_for_guidance", 0.0),
            "processing.depth_guidance.min_depth_score_for_guidance",
        ),
        strict_mode=_require_bool(
            data.get("strict_mode", False),
            "processing.depth_guidance.strict_mode",
        ),
        save_depth_debug=_require_bool(
            data.get("save_depth_debug", False),
            "processing.depth_guidance.save_depth_debug",
        ),
        cache_depth=_require_bool(
            data.get("cache_depth", False),
            "processing.depth_guidance.cache_depth",
        ),
    )


def _parse_image_enhancement_config(data: Any) -> ImageEnhancementConfig:
    if data is None:
        return ImageEnhancementConfig()
    if not isinstance(data, Mapping):
        raise ConfigError("processing.image_enhancement must be a mapping")

    return ImageEnhancementConfig(
        enabled=_require_bool(
            data.get("enabled", False),
            "processing.image_enhancement.enabled",
        ),
        clahe_clip_limit=_require_positive_float(
            data.get("clahe_clip_limit", 2.5),
            "processing.image_enhancement.clahe_clip_limit",
        ),
        clahe_tile_grid_size=_require_positive_int(
            data.get("clahe_tile_grid_size", 8),
            "processing.image_enhancement.clahe_tile_grid_size",
        ),
        gamma=_require_positive_float(
            data.get("gamma", 0.85),
            "processing.image_enhancement.gamma",
        ),
        denoise_strength=_require_non_negative_float(
            data.get("denoise_strength", 0.0),
            "processing.image_enhancement.denoise_strength",
        ),
        sharpen_amount=_require_non_negative_float(
            data.get("sharpen_amount", 0.15),
            "processing.image_enhancement.sharpen_amount",
        ),
    )


def _parse_optimization_config(data: Any) -> OptimizationConfig:
    if data is None:
        return OptimizationConfig()
    if not isinstance(data, Mapping):
        raise ConfigError("processing.optimization must be a mapping")

    raw_depth_input_size = data.get("depth_input_size")
    raw_depth_inference_stride = data.get("depth_inference_stride")
    raw_low_light_denoise_strength = data.get("low_light_denoise_strength")

    return OptimizationConfig(
        depth_input_size=(
            None
            if raw_depth_input_size is None
            else _require_positive_int(
                raw_depth_input_size,
                "processing.optimization.depth_input_size",
            )
        ),
        depth_inference_stride=(
            None
            if raw_depth_inference_stride is None
            else _require_positive_int(
                raw_depth_inference_stride,
                "processing.optimization.depth_inference_stride",
            )
        ),
        low_light_denoise_strength=(
            None
            if raw_low_light_denoise_strength is None
            else _require_non_negative_float(
                raw_low_light_denoise_strength,
                "processing.optimization.low_light_denoise_strength",
            )
        ),
    )


def _apply_optimization_overrides(
    depth_model: DepthModelConfig,
    image_enhancement: ImageEnhancementConfig,
    optimization: OptimizationConfig,
) -> tuple[DepthModelConfig, ImageEnhancementConfig]:
    if optimization.depth_input_size is not None:
        depth_model = replace(depth_model, input_size=optimization.depth_input_size)
    if optimization.depth_inference_stride is not None:
        depth_model = replace(depth_model, inference_stride=optimization.depth_inference_stride)
    if optimization.low_light_denoise_strength is not None:
        image_enhancement = replace(
            image_enhancement,
            denoise_strength=optimization.low_light_denoise_strength,
        )
    return depth_model, image_enhancement


def _load_mapping(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file does not exist: {path}")

    suffix = path.suffix.lower()
    try:
        with path.open("r", encoding="utf-8") as handle:
            if suffix == ".json":
                data = json.load(handle)
            elif suffix in {".yaml", ".yml"}:
                data = yaml.safe_load(handle)
            else:
                raise ConfigError("Config file must use .yaml, .yml, or .json extension")
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"Failed to load config file {path}: {exc}") from exc

    if not isinstance(data, Mapping):
        raise ConfigError("Config file must contain a mapping at the top level")
    return data


def _resolve_path(value: str, base_dir: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    return Path(base_dir).expanduser() / candidate


def _resolve_glob_pattern(value: str, base_dir: str | Path) -> str:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return value
    return str(Path(base_dir).expanduser() / candidate)


def _require_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{key} must be a mapping")
    return value


def _require_non_empty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty string")
    return value.strip()


def _require_non_negative_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a non-negative float")
    parsed = float(value)
    if parsed < 0.0:
        raise ConfigError(f"{name} must be a non-negative float")
    return parsed


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be a boolean")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return value


def _require_non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"{name} must be a non-negative integer")
    return value


def _require_probability(
    value: Any,
    name: str,
    min_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number between 0.0 and 1.0")
    value = float(value)
    lower_ok = value >= 0.0 if min_inclusive else value > 0.0
    if not lower_ok or value > 1.0:
        raise ConfigError(f"{name} must be a number between 0.0 and 1.0")
    return value


def _require_device(value: Any, name: str) -> str:
    value = _require_non_empty_string(value, name).lower()
    if value not in {"cpu", "cuda"}:
        raise ConfigError(f"{name} must be either 'cpu' or 'cuda'")
    return value


def _require_pipeline(value: Any, name: str) -> str:
    value = _require_non_empty_string(value, name)
    if value not in {"orbslam", "ade_mono_depth"}:
        raise ConfigError(f"{name} must be one of: orbslam, ade_mono_depth")
    return value


def _require_depth_backend(value: Any, name: str) -> str:
    value = _require_non_empty_string(value, name).lower()
    if value != "zoedepth":
        raise ConfigError(f"{name} must be 'zoedepth' in v1")
    return value


def _require_log_mode(value: Any, name: str) -> str:
    normalized = _require_non_empty_string(value, name)
    if normalized not in {"full", "fps_adaptive"}:
        raise ConfigError(f"{name} must be one of: full, fps_adaptive")
    return normalized


def _require_positive_float(value: Any, name: str, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number")
    converted = float(value)
    if allow_zero:
        if converted < 0:
            raise ConfigError(f"{name} must be >= 0")
    elif converted <= 0:
        raise ConfigError(f"{name} must be > 0")
    return converted
