"""画像処理の部品（ピッチの領域、白線、カメラの動き、カット検出）"""

import cv2
import numpy as np


def get_pitch_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pitch_mask = np.zeros_like(mask)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(pitch_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    return pitch_mask


def white_line_mask(frame, pitch_mask, person_boxes):
    """芝の上の細く明るい線（白線）を抽出する。白ユニフォームと誤認しないよう選手の箱は消す"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    k = max(9, (frame.shape[1] // 80) | 1)
    tophat = cv2.morphologyEx(hsv[..., 2], cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    mask = ((tophat > 25) & (hsv[..., 1] < 80) & (pitch_mask > 0)).astype(np.uint8) * 255
    for x1, y1, x2, y2 in np.asarray(person_boxes).astype(int):
        mask[max(y1 - 3, 0) : y2 + 3, max(x1 - 3, 0) : x2 + 3] = 0
    return mask


def is_scene_cut(prev_hist, frame, threshold=0.5):
    """色ヒストグラムの急変でカット（リプレイ・別カメラへの切り替え）を検出する"""
    small = cv2.resize(frame, (160, 90))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    cut = prev_hist is not None and cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL) < threshold
    return cut, hist


class CameraMotionEstimator:
    """ORB特徴点のフレーム間マッチングで、現在フレーム→前フレームのホモグラフィを推定する"""

    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_kp, self.prev_des = None, None

    def update(self, frame, moving_boxes):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 選手やボールなど動く物体の上の特徴点はカメラの動きと無関係なので除外する
        mask = np.full(gray.shape, 255, np.uint8)
        for x1, y1, x2, y2 in moving_boxes.astype(int):
            mask[max(y1, 0) : y2, max(x1, 0) : x2] = 0
        kp, des = self.orb.detectAndCompute(gray, mask)

        H_cur2prev = np.eye(3)
        if self.prev_des is not None and des is not None and len(kp) >= 10:
            matches = self.matcher.match(des, self.prev_des)
            if len(matches) >= 15:
                src = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst = np.float32([self.prev_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
                if H is not None:
                    H_cur2prev = H
        self.prev_kp, self.prev_des = kp, des
        return H_cur2prev
