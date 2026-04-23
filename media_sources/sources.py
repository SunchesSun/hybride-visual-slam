from __future__ import annotations

import glob as glob_module
import re
import time
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Iterator

import cv2
import numpy as np

from .config import (
    ImageSequenceSourceConfig,
    SourceConfig,
    TimestampManifestSourceConfig,
    UdpSourceConfig,
    VideoFileSourceConfig,
)
from .contracts import FramePacket
from .errors import SourceOpenError, SourceReadError


def open_source(config: SourceConfig) -> _BaseSource:
    if isinstance(config, UdpSourceConfig):
        return _UdpVideoSource(config)
    if isinstance(config, VideoFileSourceConfig):
        return _VideoFileSource(config)
    if isinstance(config, ImageSequenceSourceConfig):
        return _ImageSequenceSource(config)
    if isinstance(config, TimestampManifestSourceConfig):
        return _TimestampManifestSource(config)
    raise ValueError(f"Unknown source config type: {type(config)}")


class _BaseSource:
    def __iter__(self) -> Iterator[FramePacket]:
        return self

    def __next__(self) -> FramePacket:
        raise NotImplementedError

    def __enter__(self) -> _BaseSource:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        pass


class _VideoFileSource(_BaseSource):
    def __init__(self, config: VideoFileSourceConfig) -> None:
        self._config = config
        self._cap = cv2.VideoCapture(config.path)
        if not self._cap.isOpened():
            raise SourceOpenError(f"Failed to open video file: {config.path}")
        self._fps = config.fps or self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._frame_index = 0

    def __next__(self) -> FramePacket:
        ok, frame_bgr = self._cap.read()
        if not ok or frame_bgr is None:
            raise StopIteration
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        timestamp_s = self._frame_index / self._fps
        h, w = frame_rgb.shape[:2]
        packet = FramePacket(
            frame=frame_rgb,
            frame_index=self._frame_index,
            timestamp_s=timestamp_s,
            source_kind="video_file",
            source_id=self._config.path,
            width=w,
            height=h,
        )
        self._frame_index += 1
        return packet

    def close(self) -> None:
        if self._cap.isOpened():
            self._cap.release()


class _ImageSequenceSource(_BaseSource):
    def __init__(self, config: ImageSequenceSourceConfig) -> None:
        self._config = config
        paths = sorted(
            glob_module.glob(config.glob),
            key=_natural_sort_key,
        )
        if not paths:
            raise SourceOpenError(f"No images matched glob pattern: {config.glob}")
        self._paths = paths
        self._fps = config.fps
        self._frame_index = 0

    def __next__(self) -> FramePacket:
        if self._frame_index >= len(self._paths):
            raise StopIteration
        path = self._paths[self._frame_index]
        frame_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise SourceReadError(f"Failed to read image: {path}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        timestamp_s = self._frame_index / self._fps
        h, w = frame_rgb.shape[:2]
        packet = FramePacket(
            frame=frame_rgb,
            frame_index=self._frame_index,
            timestamp_s=timestamp_s,
            source_kind="image_sequence",
            source_id=path,
            width=w,
            height=h,
            meta={"path": path},
        )
        self._frame_index += 1
        return packet


class _TimestampManifestSource(_BaseSource):
    def __init__(self, config: TimestampManifestSourceConfig) -> None:
        self._config = config
        self._entries = _parse_manifest(
            Path(config.manifest_path),
            Path(config.dataset_root),
        )
        self._frame_index = 0
        self._started_at: float | None = None
        self._first_timestamp: float | None = None

    def __next__(self) -> FramePacket:
        if self._frame_index >= len(self._entries):
            raise StopIteration

        timestamp_s, image_path = self._entries[self._frame_index]

        if self._config.replay_realtime:
            now = time.monotonic()
            if self._started_at is None:
                self._started_at = now
                self._first_timestamp = timestamp_s
            else:
                elapsed_wall = now - self._started_at
                elapsed_dataset = timestamp_s - self._first_timestamp
                sleep_needed = elapsed_dataset - elapsed_wall
                if sleep_needed > 0.0:
                    time.sleep(sleep_needed)

        frame = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise SourceReadError(f"Failed to read image: {image_path}")

        h, w = frame.shape[:2]
        packet = FramePacket(
            frame=frame,
            frame_index=self._frame_index,
            timestamp_s=timestamp_s,
            source_kind="timestamp_manifest",
            source_id=str(self._config.manifest_path),
            width=w,
            height=h,
        )
        self._frame_index += 1
        return packet


class _UdpVideoSource(_BaseSource):
    def __init__(self, config: UdpSourceConfig) -> None:
        self._config = config
        self._frame_index = 0
        self._lock = threading.Lock()
        self._latest_packet: FramePacket | None = None
        self._exhausted = False
        self._cap = self._open_capture(config.uri)
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _open_capture(self, uri: str) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(uri)
        if not cap.isOpened():
            raise SourceOpenError(f"Failed to open UDP stream: {uri}")
        return cap

    def _read_loop(self) -> None:
        while True:
            ok, frame_bgr = self._cap.read()
            if not ok or frame_bgr is None:
                if self._config.reconnect:
                    self._cap.release()
                    time.sleep(self._config.reconnect_delay_s)
                    try:
                        self._cap = self._open_capture(self._config.uri)
                    except SourceOpenError:
                        with self._lock:
                            self._exhausted = True
                        return
                    continue
                else:
                    with self._lock:
                        self._exhausted = True
                    return

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            with self._lock:
                idx = self._frame_index
                self._frame_index += 1
                self._latest_packet = FramePacket(
                    frame=frame_rgb,
                    frame_index=idx,
                    timestamp_s=time.monotonic(),
                    source_kind="udp",
                    source_id=self._config.uri,
                    width=w,
                    height=h,
                )

    def __next__(self) -> FramePacket:
        while True:
            with self._lock:
                packet = self._latest_packet
                exhausted = self._exhausted
                if packet is not None:
                    self._latest_packet = None
                    return packet
                if exhausted:
                    raise SourceReadError("UDP stream ended")
            time.sleep(0.001)

    def close(self) -> None:
        with self._lock:
            self._exhausted = True
        if self._cap.isOpened():
            self._cap.release()


def _parse_manifest(
    manifest_path: Path,
    dataset_root: Path,
) -> list[tuple[float, Path]]:
    entries: list[tuple[float, Path]] = []
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SourceOpenError(f"Cannot read manifest file: {manifest_path}") from exc

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            raise SourceOpenError(
                f"Invalid manifest line (expected 'timestamp path'): {line!r}"
            )
        try:
            timestamp_s = float(parts[0])
        except ValueError as exc:
            raise SourceOpenError(
                f"Invalid timestamp in manifest line: {line!r}"
            ) from exc
        rel_path = parts[1]
        image_path = dataset_root / rel_path
        if not image_path.is_file():
            raise SourceOpenError(f"Image referenced in manifest not found: {image_path}")
        entries.append((timestamp_s, image_path))

    if not entries:
        raise SourceOpenError(f"Manifest file is empty or has no valid entries: {manifest_path}")
    return entries


def _natural_sort_key(path: str) -> list[int | str]:
    parts = re.split(r"(\d+)", path)
    return [int(p) if p.isdigit() else p for p in parts]
