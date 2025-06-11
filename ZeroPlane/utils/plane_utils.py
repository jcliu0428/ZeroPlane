import cv2
import numpy as np

import torch


def make_pixel_normal_map(masks, params, vis=False):
    masks = np.asarray(masks)
    params = np.asarray(params)

    h, w = masks.shape[-2:]

    pixel_normal_map = np.zeros((h, w, 3))
    pixel_offset_map = np.zeros((h, w))

    assert len(masks) == len(params)

    for mask, param in zip(masks, params):
        pixel_normal_map[mask, :] = param / (np.linalg.norm(param) + 1e-8)
        pixel_offset_map[mask] = 1. / (np.linalg.norm(param) + 1e-8)

    if vis:
        vis_pixel_normal_map = (pixel_normal_map + 1) / 2.0
        vis_pixel_normal_map = (vis_pixel_normal_map * 255).astype(np.uint8)

        cv2.imwrite('debug_pixel_normal.png', vis_pixel_normal_map)
        exit(1)

    pixel_normal_map = pixel_normal_map.transpose(2, 0, 1).astype(np.float32)

    plane_normal_maps = []
    for mask in masks:
        plane_normal_maps.append(mask * pixel_normal_map)

    return pixel_normal_map, pixel_offset_map, plane_normal_maps
