from __future__ import annotations

import logging

from .contracts import FramePacket
from .sources import open_source
from .config import SourceConfig


class DemoRunner:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger(__name__)

    def run(self, config: SourceConfig, max_frames: int | None = None) -> int:
        processed = 0
        with open_source(config) as source:
            for packet in source:
                self._logger.info(
                    "frame=%d timestamp=%.3f source=%s",
                    packet.frame_index,
                    packet.timestamp_s,
                    packet.source_kind,
                )
                processed += 1
                if max_frames is not None and processed >= max_frames:
                    break
        return processed
