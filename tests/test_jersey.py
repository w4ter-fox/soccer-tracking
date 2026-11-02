import numpy as np
import pytest

from soccer_tracking.config import DEFAULT_JERSEY_COLORS, GOALKEEPER, REFEREE, TEAM_BLUE, TEAM_WHITE, UNKNOWN
from soccer_tracking.jersey import classify_by_jersey_color


def player(shirt_bgr, number_bgr=None):
    """芝の背景に、胴体のシャツと腕（肌）を描いた選手の切り出し"""
    img = np.zeros((100, 40, 3), np.uint8)
    img[:] = (40, 140, 40)
    img[15:50, 8:32] = shirt_bgr
    if number_bgr:
        img[22:40, 16:24] = number_bgr
    img[18:45, 4:8] = (120, 160, 210)
    return img


@pytest.mark.parametrize(
    "shirt,number,expected",
    [
        ((235, 235, 235), None, TEAM_WHITE),
        ((165, 165, 160), None, TEAM_WHITE),  # 日陰の白
        ((160, 60, 20), None, TEAM_BLUE),
        ((90, 30, 10), (240, 240, 240), TEAM_BLUE),  # 紺に白い背番号
        ((0, 215, 255), None, REFEREE),
        ((30, 30, 210), None, GOALKEEPER),
        ((40, 140, 40), None, UNKNOWN),  # 芝しか写っていない
    ],
)
def test_classify(shirt, number, expected):
    assert classify_by_jersey_color(player(shirt, number), DEFAULT_JERSEY_COLORS) == expected


def test_empty_crop_is_unknown():
    assert classify_by_jersey_color(np.zeros((0, 0, 3), np.uint8), DEFAULT_JERSEY_COLORS) == UNKNOWN
