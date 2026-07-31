from __future__ import annotations

from typing import Any

import torch

from easycom_saal.utils.bbox_utils import bbox_center, bbox_size, scale_bbox


def make_gaussian_heatmap(
    height: int,
    width: int,
    cx: float,
    cy: float,
    sigma: float,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    sigma = max(float(sigma), 1e-6)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    heatmap = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2))
    max_value = heatmap.max()
    if max_value > 0:
        heatmap = heatmap / max_value
    return heatmap


def make_heatmap_from_bbox(
    bbox: Any,
    video_size: tuple[int, int],
    heatmap_size: tuple[int, int],
    sigma_mode: str = "bbox_relative",
    sigma_ratio: float = 0.25,
    fixed_sigma: float = 5.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    heatmap_h, heatmap_w = heatmap_size
    scaled_bbox = scale_bbox(bbox, video_size, heatmap_size)
    cx, cy = bbox_center(scaled_bbox)
    bbox_w, bbox_h = bbox_size(scaled_bbox)

    if sigma_mode == "bbox_relative":
        sigma = sigma_ratio * max(bbox_w, bbox_h)
    elif sigma_mode == "fixed":
        sigma = fixed_sigma
    else:
        raise ValueError(f"Unsupported sigma_mode: {sigma_mode}")

    return make_gaussian_heatmap(
        heatmap_h,
        heatmap_w,
        cx=cx,
        cy=cy,
        sigma=sigma,
        device=device,
    )


def bbox_center_on_heatmap(
    bbox: Any,
    video_size: tuple[int, int],
    heatmap_size: tuple[int, int],
) -> tuple[float, float]:
    scaled_bbox = scale_bbox(bbox, video_size, heatmap_size)
    return bbox_center(scaled_bbox)

