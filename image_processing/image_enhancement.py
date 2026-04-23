from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from media_sources.config import ImageEnhancementConfig


@dataclass(frozen=True, slots=True)
class EnhancedFrame:
    frame_gray: np.ndarray
    frame_rgb: np.ndarray


class LowLightEnhancer:
    def __init__(self, config: ImageEnhancementConfig | None = None) -> None:
        self._config = config or ImageEnhancementConfig()
        self._clahe = cv2.createCLAHE(
            clipLimit=self._config.clahe_clip_limit,
            tileGridSize=(
                self._config.clahe_tile_grid_size,
                self._config.clahe_tile_grid_size,
            ),
        )
        self._gamma_lut = self._build_gamma_lut(self._config.gamma)

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def enhance_gray(self, frame_gray: np.ndarray) -> np.ndarray:
        gray = np.asarray(frame_gray, dtype=np.uint8)
        if not self.enabled:
            return gray.copy()

        enhanced = gray.copy()
        if self._config.denoise_strength > 0.0:
            enhanced = cv2.fastNlMeansDenoising(
                enhanced,
                None,
                h=float(self._config.denoise_strength),
                templateWindowSize=7,
                searchWindowSize=21,
            )

        enhanced = self._clahe.apply(enhanced)

        if self._config.gamma != 1.0:
            enhanced = cv2.LUT(enhanced, self._gamma_lut)

        if self._config.sharpen_amount > 0.0:
            blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.2, sigmaY=1.2)
            enhanced = cv2.addWeighted(
                enhanced,
                1.0 + float(self._config.sharpen_amount),
                blur,
                -float(self._config.sharpen_amount),
                0.0,
            )
        return np.asarray(enhanced, dtype=np.uint8)

    def enhance_rgb(self, frame_rgb: np.ndarray) -> np.ndarray:
        rgb = np.asarray(frame_rgb, dtype=np.uint8)
        if not self.enabled:
            return rgb.copy()

        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        enhanced_l = self.enhance_gray(l_channel)
        enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    def enhance_for_mono_pipeline(self, frame_gray: np.ndarray) -> EnhancedFrame:
        enhanced_gray = self.enhance_gray(frame_gray)
        enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
        return EnhancedFrame(frame_gray=enhanced_gray, frame_rgb=enhanced_rgb)

    def enhance_for_packet(
        self,
        frame_gray: np.ndarray | None,
        frame_rgb: np.ndarray | None,
    ) -> EnhancedFrame:
        if frame_gray is not None:
            return self.enhance_for_mono_pipeline(frame_gray)

        if frame_rgb is None:
            raise ValueError("Either frame_gray or frame_rgb must be provided")

        enhanced_rgb = self.enhance_rgb(frame_rgb)
        enhanced_gray = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2GRAY)
        return EnhancedFrame(frame_gray=enhanced_gray, frame_rgb=enhanced_rgb)

    @staticmethod
    def _build_gamma_lut(gamma: float) -> np.ndarray:
        gamma = max(float(gamma), 1e-6)
        values = np.arange(256, dtype=np.float32) / 255.0
        corrected = np.power(values, gamma) * 255.0
        return np.clip(corrected, 0.0, 255.0).astype(np.uint8)
