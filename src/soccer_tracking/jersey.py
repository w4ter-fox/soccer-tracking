"""ユニフォームの色によるチームの判定"""

import cv2
import numpy as np

from .config import UNKNOWN


def classify_by_jersey_color(crop, colors, min_ratio=0.08):
    """胴体部分のユニフォーム色を colors（{クラスID: [(HSVの下限, 上限), ...]}）のどれかに判定する。
    判定できなければ UNKNOWN。
    芝のピクセルは分母から除くので、細い選手や遠くの選手でも比率が薄まらない"""
    h, w = crop.shape[:2]
    # 胴体（上15〜50%、横中央50%）だけを見る。脚・芝・頭を避ける
    torso = crop[int(h * 0.15) : int(h * 0.5), int(w * 0.25) : int(w * 0.75)]
    if torso.size == 0:
        return UNKNOWN
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    total = torso.shape[0] * torso.shape[1] - cv2.countNonZero(grass)
    if total < 20:
        return UNKNOWN

    ratios = {}
    for cls, ranges in colors.items():
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, lo, hi)
        ratios[cls] = cv2.countNonZero(mask & ~grass) / total
    best = max(ratios, key=ratios.get)
    return best if ratios[best] >= min_ratio else UNKNOWN
