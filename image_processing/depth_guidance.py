from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from media_sources.config import DepthGuidanceConfig


@dataclass(slots=True)
class DepthGuidanceStats:
    raw_features: int
    valid_depth_features: int
    enhanced_features: int
    far_points_removed: int
    final_features: int
    invalid_depth_ratio: float
    filter_applied: bool
    fallback_to_raw: bool


def compute_depth_guided_keep_indices(
    keypoints_xy: np.ndarray,
    depth_m: np.ndarray,
    config: DepthGuidanceConfig,
    frame_seed: int,
) -> tuple[np.ndarray, DepthGuidanceStats]:
    if keypoints_xy.ndim != 2 or keypoints_xy.shape[1] != 2:
        raise ValueError("keypoints_xy must have shape Nx2")
    if depth_m.ndim != 2:
        raise ValueError("depth_m must have shape HxW")
    if keypoints_xy.size == 0:
        return np.empty((0,), dtype=np.int32), DepthGuidanceStats(0, 0, 0, 0, 0, 0.0, False, False)

    raw_feature_count = int(keypoints_xy.shape[0])
    depths = _sample_depths(keypoints_xy, depth_m)
    valid_indices = np.flatnonzero(np.isfinite(depths) & (depths > 0.0))
    invalid_depth_ratio = 1.0 - (float(valid_indices.size) / float(raw_feature_count))
    if valid_indices.size == 0:
        all_indices = np.arange(raw_feature_count, dtype=np.int32)
        return all_indices, DepthGuidanceStats(
            raw_feature_count,
            0,
            raw_feature_count,
            0,
            raw_feature_count,
            1.0,
            False,
            True,
        )
    if invalid_depth_ratio > config.max_invalid_depth_ratio:
        all_indices = np.arange(raw_feature_count, dtype=np.int32)
        return all_indices, DepthGuidanceStats(
            raw_feature_count,
            int(valid_indices.size),
            raw_feature_count,
            0,
            raw_feature_count,
            invalid_depth_ratio,
            False,
            True,
        )

    rect = (0.0, 0.0, float(depth_m.shape[1]), float(depth_m.shape[0]))
    enhanced_indices = _recursive_select(
        keypoints_xy=keypoints_xy,
        depths=depths,
        indices=valid_indices,
        rect=rect,
        config=config,
        depth_level=0,
    )
    enhanced_indices = np.asarray(sorted(set(int(idx) for idx in enhanced_indices)), dtype=np.int32)

    rng = np.random.default_rng(int(frame_seed))
    keep_mask = np.ones(enhanced_indices.shape[0], dtype=bool)
    far_removed = 0
    for position, index in enumerate(enhanced_indices):
        if depths[index] <= config.depth_threshold_m:
            continue
        if rng.random() < config.far_point_drop_probability:
            keep_mask[position] = False
            far_removed += 1
    final_indices = enhanced_indices[keep_mask]
    minimum_retained = min(
        raw_feature_count,
        max(
            int(config.min_feature_retention_count),
            int(math.ceil(raw_feature_count * config.min_feature_retention_ratio)),
        ),
    )
    fallback_to_raw = False
    if final_indices.size < minimum_retained:
        final_indices = np.arange(raw_feature_count, dtype=np.int32)
        far_removed = 0
        fallback_to_raw = True
    stats = DepthGuidanceStats(
        raw_features=raw_feature_count,
        valid_depth_features=int(valid_indices.size),
        enhanced_features=int(enhanced_indices.shape[0]),
        far_points_removed=far_removed,
        final_features=int(final_indices.shape[0]),
        invalid_depth_ratio=invalid_depth_ratio,
        filter_applied=not fallback_to_raw,
        fallback_to_raw=fallback_to_raw,
    )
    return final_indices, stats


def _sample_depths(keypoints_xy: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
    xs = np.clip(np.rint(keypoints_xy[:, 0]).astype(np.int32), 0, depth_m.shape[1] - 1)
    ys = np.clip(np.rint(keypoints_xy[:, 1]).astype(np.int32), 0, depth_m.shape[0] - 1)
    return np.asarray(depth_m[ys, xs], dtype=np.float32)


def _recursive_select(
    keypoints_xy: np.ndarray,
    depths: np.ndarray,
    indices: np.ndarray,
    rect: tuple[float, float, float, float],
    config: DepthGuidanceConfig,
    depth_level: int,
) -> list[int]:
    if indices.size == 0:
        return []
    if depth_level >= config.max_recursion_depth or indices.size <= config.min_block_points:
        return indices.astype(np.int32).tolist()

    selected = _greedy_block_selection(keypoints_xy, depths, indices, rect, config.recursive_keep_ratio)
    kept: list[int] = []
    for child_rect in _split_quadrants(rect):
        child_indices = _indices_in_rect(keypoints_xy, selected, child_rect)
        if child_indices.size == 0:
            continue
        kept.extend(
            _recursive_select(
                keypoints_xy=keypoints_xy,
                depths=depths,
                indices=child_indices,
                rect=child_rect,
                config=config,
                depth_level=depth_level + 1,
            )
        )
    return kept or selected.astype(np.int32).tolist()


def _greedy_block_selection(
    keypoints_xy: np.ndarray,
    depths: np.ndarray,
    indices: np.ndarray,
    rect: tuple[float, float, float, float],
    keep_ratio: float,
) -> np.ndarray:
    target_count = max(1, int(round(indices.size * keep_ratio)))
    target_count = min(target_count, int(indices.size))
    block_depths = depths[indices]
    start_position = int(np.argmin(block_depths))
    selected_positions = [start_position]
    remaining_positions = set(range(indices.size))
    remaining_positions.remove(start_position)

    x0, y0, x1, y1 = rect
    diagonal = max(math.hypot(x1 - x0, y1 - y0), 1e-6)
    depth_min = float(np.min(block_depths))
    depth_max = float(np.max(block_depths))
    depth_span = max(depth_max - depth_min, 1e-6)

    while len(selected_positions) < target_count and remaining_positions:
        best_position = None
        best_score = -float("inf")
        for position in remaining_positions:
            point = keypoints_xy[indices[position]]
            nearest_distance = min(
                np.linalg.norm(point - keypoints_xy[indices[selected_position]])
                for selected_position in selected_positions
            )
            spatial_score = float(nearest_distance / diagonal)
            depth_score = float((depth_max - block_depths[position]) / depth_span)
            score = spatial_score + depth_score
            if score > best_score:
                best_score = score
                best_position = position
        if best_position is None:
            break
        selected_positions.append(best_position)
        remaining_positions.remove(best_position)

    return np.asarray([int(indices[position]) for position in selected_positions], dtype=np.int32)


def _split_quadrants(rect: tuple[float, float, float, float]) -> list[tuple[float, float, float, float]]:
    x0, y0, x1, y1 = rect
    mid_x = (x0 + x1) * 0.5
    mid_y = (y0 + y1) * 0.5
    return [
        (x0, y0, mid_x, mid_y),
        (mid_x, y0, x1, mid_y),
        (x0, mid_y, mid_x, y1),
        (mid_x, mid_y, x1, y1),
    ]


def _indices_in_rect(
    keypoints_xy: np.ndarray,
    indices: np.ndarray,
    rect: tuple[float, float, float, float],
) -> np.ndarray:
    x0, y0, x1, y1 = rect
    points = keypoints_xy[indices]
    mask = (
        (points[:, 0] >= x0)
        & (points[:, 0] < x1)
        & (points[:, 1] >= y0)
        & (points[:, 1] < y1)
    )
    return np.asarray(indices[mask], dtype=np.int32)
