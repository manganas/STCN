import numpy as np
from numpy.typing import NDArray


def calculate_bbox_center(bbox_dims: tuple[int], start_end: bool = True) -> tuple[int]:
    if not start_end:
        bbox_dims = convert_bbox_width_to_dims(bbox_dims)
    rmin, cmin, rmax, cmax = bbox_dims

    rmid = np.abs(rmax - rmin) // 2
    cmid = np.abs(cmax - cmin) // 2

    return rmid, cmid


def get_bbox_start_end(mask: NDArray) -> tuple[int]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, cmin, rmax, cmax


def get_bbox_box_width(mask: NDArray) -> tuple[int]:
    bbox_start_end = get_bbox_start_end(mask)

    return convert_bbox_dims_to_width(bbox_start_end)


def convert_bbox_dims_to_width(bbox_dims: tuple[int]) -> tuple[int]:
    rmin, cmin, rmax, cmax = bbox_dims
    width = np.abs(cmax - cmin)
    height = np.abs(rmax - rmin)
    return rmin, cmin, height, width


def convert_bbox_width_to_dims(bbox_dims_width: tuple[int]) -> tuple[int]:
    rmin, cmin, height, width = bbox_dims_width
    rmax = rmin + height
    cmax = cmin + width
    return rmin, cmin, rmax, cmax
