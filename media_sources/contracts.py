from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FramePacket:
    frame: np.ndarray
    timestamp_s: float
    frame_index: int
    source_kind: str
    source_id: str = ""
    width: int = 0
    height: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def is_grayscale(self) -> bool:
        return self.frame.ndim == 2

    @property
    def is_rgb(self) -> bool:
        return self.frame.ndim == 3

    @property
    def frame_mono(self) -> np.ndarray:
        return self.frame

    @property
    def frame_rgb(self) -> np.ndarray:
        return self.frame
