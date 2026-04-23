from __future__ import annotations

import importlib
import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from PIL import Image

from media_sources.config import DepthModelConfig


@dataclass(slots=True)
class DepthPrediction:
    depth_m: np.ndarray
    confidence: np.ndarray | None
    backend: str
    inference_ms: float

    def __post_init__(self) -> None:
        if self.depth_m.ndim != 2:
            raise ValueError("depth_m must be a HxW map")
        if self.depth_m.dtype != np.float32:
            raise ValueError("depth_m must use float32 dtype")
        if self.depth_m.size == 0:
            raise ValueError("depth_m must not be empty")


class DepthEstimator(Protocol):
    def prepare(self) -> None:
        raise NotImplementedError

    def predict(self, frame_rgb: np.ndarray) -> DepthPrediction:
        raise NotImplementedError


class ZoeDepthEstimator:
    def __init__(
        self,
        config: DepthModelConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        self._config = config
        self._logger = logger or logging.getLogger(__name__)
        self._weights_path = None
        if config.weights_path:
            self._weights_path = Path(config.weights_path).expanduser()
            if not self._weights_path.is_file():
                raise FileNotFoundError(f"Depth weights were not found: {self._weights_path}")
        self._device = config.device
        self._model = None
        self._torch = None
        self._is_warmed_up = False

    def prepare(self) -> None:
        model = self._load_model()
        torch = self._torch
        if self._is_warmed_up or torch is None:
            return

        side = max(64, int(self._config.input_size or 512))
        dummy_height = max(64, side // 2)
        dummy_width = max(64, side)
        image = Image.fromarray(
            np.zeros((dummy_height, dummy_width, 3), dtype=np.uint8),
            mode="RGB",
        )
        with torch.inference_mode():
            with self._autocast_context(torch):
                model.infer_pil(
                    image,
                    pad_input=self._config.pad_input,
                    with_flip_aug=self._config.with_flip_aug,
                )
        self._is_warmed_up = True

    def predict(self, frame_rgb: np.ndarray) -> DepthPrediction:
        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError("ZoeDepthEstimator expects an HxWx3 RGB frame")
        if frame_rgb.dtype != np.uint8:
            raise ValueError("ZoeDepthEstimator expects uint8 RGB input")

        model = self._load_model()
        torch = self._torch
        if torch is None:
            raise RuntimeError("Torch backend was not initialized")

        original_height, original_width = frame_rgb.shape[:2]
        inference_rgb = frame_rgb
        if self._config.input_size:
            inference_rgb = _resize_with_aspect_ratio(frame_rgb, self._config.input_size)

        image = Image.fromarray(inference_rgb, mode="RGB")
        started_at = time.perf_counter()
        with torch.inference_mode():
            with self._autocast_context(torch):
                depth = model.infer_pil(
                    image,
                    pad_input=self._config.pad_input,
                    with_flip_aug=self._config.with_flip_aug,
                )
        inference_ms = (time.perf_counter() - started_at) * 1000.0

        depth_map = _to_depth_array(depth)
        if depth_map.shape != (original_height, original_width):
            depth_map = cv2.resize(
                depth_map,
                (original_width, original_height),
                interpolation=cv2.INTER_LINEAR,
            )
        return DepthPrediction(
            depth_m=np.asarray(depth_map, dtype=np.float32),
            confidence=None,
            backend=self._config.backend,
            inference_ms=inference_ms,
        )

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            torch = importlib.import_module("torch")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PyTorch is required for ZoeDepth inference. Install it in the active environment first."
            ) from exc

        self._torch = torch
        hub_dir = Path(torch.hub.get_dir())
        local_repo = hub_dir / "isl-org_ZoeDepth_main"
        load_started_at = time.perf_counter()
        variant = f"ZoeD_{getattr(self._config, 'model_variant', 'N')}"
        try:
            hub_load_kwargs = {"trust_repo": True}
            if self._weights_path is not None:
                weights_resource = f"local::{self._weights_path}"
                if local_repo.is_dir():
                    model = torch.hub.load(
                        str(local_repo),
                        variant,
                        source="local",
                        pretrained=False,
                        config_mode="infer",
                        pretrained_resource=weights_resource,
                    )
                else:
                    model = torch.hub.load(
                        "isl-org/ZoeDepth",
                        variant,
                        pretrained=False,
                        config_mode="infer",
                        pretrained_resource=weights_resource,
                        **hub_load_kwargs,
                    )
            else:
                model = self._load_official_pretrained_model(torch, local_repo, hub_load_kwargs, variant)
        except Exception as exc:  # pragma: no cover - exercised via mocked tests
            raise RuntimeError(
                "Failed to load ZoeDepth. Make sure the ZoeDepth hub dependencies are available "
                "and the weights file matches the selected backend."
            ) from exc

        model = model.to(self._device)
        model.eval()
        if self._device == "cuda":
            try:
                torch.backends.cudnn.benchmark = True
            except AttributeError:
                pass
        self._model = model
        if self._weights_path is None:
            source_description = "official pretrained ZoeDepth weights"
        else:
            source_description = str(self._weights_path)
        self._logger.info(
            "Loaded ZoeDepth backend from %s in %.2f ms",
            source_description,
            (time.perf_counter() - load_started_at) * 1000.0,
        )
        return self._model

    def _load_official_pretrained_model(self, torch, local_repo: Path, hub_load_kwargs: dict[str, object], variant: str = "ZoeD_N"):
        try:
            if local_repo.is_dir():
                return torch.hub.load(
                    str(local_repo),
                    variant,
                    source="local",
                    pretrained=True,
                )
            return torch.hub.load(
                "isl-org/ZoeDepth",
                variant,
                pretrained=True,
                **hub_load_kwargs,
            )
        except RuntimeError as exc:
            if "relative_position_index" not in str(exc):
                raise
            if not local_repo.is_dir():
                raise

            if self._patch_cached_zoedepth_repo(local_repo):
                self._logger.warning(
                    "Patched cached ZoeDepth repo for torch/timm compatibility and retrying official pretrained load"
                )
                return torch.hub.load(
                    str(local_repo),
                    variant,
                    source="local",
                    pretrained=True,
                )
            raise

    def _patch_cached_zoedepth_repo(self, local_repo: Path) -> bool:
        model_io_path = local_repo / "zoedepth" / "models" / "model_io.py"
        if not model_io_path.is_file():
            return False

        original_text = model_io_path.read_text(encoding="utf-8")
        if "relative_position_index" in original_text and "block.drop_path = torch.nn.Identity()" in original_text:
            return True

        old_block = """        state[k] = v\n\n    model.load_state_dict(state)\n    print(\"Loaded successfully\")\n    return model\n"""
        intermediate_block = """        state[k] = v\n\n    incompatible = model.load_state_dict(state, strict=False)\n    try:\n        blocks = model.core.core.pretrained.model.blocks\n    except AttributeError:\n        blocks = []\n    for block in blocks:\n        if hasattr(block, 'drop_path'):\n            block.drop_path = torch.nn.Identity()\n    missing_keys = getattr(incompatible, 'missing_keys', [])\n    unexpected_keys = getattr(incompatible, 'unexpected_keys', [])\n    if missing_keys or unexpected_keys:\n        print(f\"Loaded with non-strict state_dict. missing={len(missing_keys)} unexpected={len(unexpected_keys)}\")\n    else:\n        print(\"Loaded successfully\")\n    return model\n"""
        new_block = """        if k.endswith('relative_position_index'):\n            continue\n        state[k] = v\n\n    incompatible = model.load_state_dict(state, strict=False)\n    try:\n        blocks = model.core.core.pretrained.model.blocks\n    except AttributeError:\n        blocks = []\n    for block in blocks:\n        block.drop_path = torch.nn.Identity()\n    missing_keys = getattr(incompatible, 'missing_keys', [])\n    unexpected_keys = getattr(incompatible, 'unexpected_keys', [])\n    if missing_keys or unexpected_keys:\n        print(f\"Loaded with non-strict state_dict. missing={len(missing_keys)} unexpected={len(unexpected_keys)}\")\n    else:\n        print(\"Loaded successfully\")\n    return model\n"""
        if old_block in original_text:
            patched_text = original_text.replace(old_block, new_block)
        elif intermediate_block in original_text:
            patched_text = original_text.replace(intermediate_block, new_block)
        else:
            return False

        model_io_path.write_text(patched_text, encoding="utf-8")
        return True

    def _autocast_context(self, torch):
        if self._device != "cuda" or not self._config.use_half:
            return nullcontext()

        try:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        except (AttributeError, TypeError):
            return nullcontext()


def build_depth_estimator(
    config: DepthModelConfig,
    logger: logging.Logger | None = None,
) -> DepthEstimator:
    if config.backend != "zoedepth":
        raise ValueError(f"Unsupported depth backend: {config.backend}")
    return ZoeDepthEstimator(config, logger=logger)


def _resize_with_aspect_ratio(frame_rgb: np.ndarray, target_size: int) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    long_side = max(height, width)
    if long_side <= target_size:
        return frame_rgb
    scale = target_size / float(long_side)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return cv2.resize(frame_rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA)


def _to_depth_array(depth: object) -> np.ndarray:
    if isinstance(depth, np.ndarray):
        depth_map = depth
    else:
        try:
            torch = importlib.import_module("torch")
        except ModuleNotFoundError:
            raise RuntimeError("Depth output is not a numpy array and torch is unavailable for conversion")
        if isinstance(depth, torch.Tensor):
            depth_map = depth.detach().cpu().numpy()
        else:
            depth_map = np.asarray(depth)

    if depth_map.ndim == 3:
        depth_map = np.squeeze(depth_map, axis=0)
    if depth_map.ndim != 2:
        raise RuntimeError(f"Unexpected depth output shape: {depth_map.shape!r}")
    return np.asarray(depth_map, dtype=np.float32)
