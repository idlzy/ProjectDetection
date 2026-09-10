import math

import pytest

from tools.convert_kitti_to_mw3d import camera_rotation_to_vehicle_yaw


@pytest.mark.parametrize(
    ("rotation_y", "vehicle_yaw"),
    [
        (0.0, -math.pi / 2.0),
        (math.pi / 2.0, -math.pi),
        (-math.pi / 2.0, 0.0),
    ],
)
def test_camera_rotation_to_vehicle_yaw(rotation_y, vehicle_yaw):
    assert camera_rotation_to_vehicle_yaw(rotation_y) == pytest.approx(
        vehicle_yaw
    )
