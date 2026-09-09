import math

import numpy as np

from project_detection.data.geometry import (
    camera_yaw,
    rot_x,
    rot_y,
    rot_z,
    vehicle_box_corners,
)
from tools.visualize_test import camera_box_corners


def test_calibrated_camera_corners_recover_vehicle_aligned_box():
    r_c2v = rot_z(-1.5756106) @ rot_y(-0.003602543) @ rot_x(-1.7742399)
    t_c2v = np.asarray([1.39806, 0.05145, 0.82208], dtype=np.float64)
    center_v = np.asarray([13.1912, -1.97244, 0.80688], dtype=np.float64)
    size_lwh = np.asarray([1.63742, 0.71757, 1.64854], dtype=np.float64)
    yaw_v = 0.01737
    center_c = r_c2v.T @ (center_v - t_c2v)
    stored_yaw = -camera_yaw(yaw_v, r_c2v)
    box = np.asarray(
        [*center_c, size_lwh[0], size_lwh[2], size_lwh[1], stored_yaw, 0, 0]
    )

    actual_corners_c = camera_box_corners(box, r_c2v, t_c2v)
    expected_corners_v = vehicle_box_corners(center_v, size_lwh, yaw_v)
    expected_corners_c = (
        r_c2v.T @ (expected_corners_v - t_c2v.reshape(1, 3)).T
    ).T

    np.testing.assert_allclose(actual_corners_c, expected_corners_c, atol=1e-10)


def test_uncalibrated_camera_corners_remain_available():
    box = np.asarray([1.0, 2.0, 10.0, 4.0, 2.0, 6.0, math.pi / 4, 0, 0])
    corners = camera_box_corners(box)

    np.testing.assert_allclose(corners.mean(axis=0), box[:3], atol=1e-12)
