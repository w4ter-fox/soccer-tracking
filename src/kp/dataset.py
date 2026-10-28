"""学習データの読み込みと拡張（convert_soccernet.py の index_<split>.json を読む）"""
import json
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pitch  # noqa: E402

KEYS = [k for k, _, _ in pitch.keypoints()]


def flip_key(k):
    """左右反転したときに入れ替わるキーポイント（ピッチを x → L-x で反転。手前・奥はそのまま）"""
    if k.startswith("l_"):
        return "r_" + k[2:]
    if k.startswith("r_"):
        return "l_" + k[2:]
    return {"circle_left": "circle_right", "circle_right": "circle_left"}.get(k, k)


FLIP_INDEX = [KEYS.index(flip_key(k)) for k in KEYS]


def gaussian_heatmaps(points, valid, out_w, out_h, sigma):
    """各キーポイントの位置（出力解像度の座標）にガウスの山を置く。中心は 1 にする"""
    hm = np.zeros((len(points), out_h, out_w), np.float32)
    r = int(3 * sigma) + 1
    for i, ((x, y), ok) in enumerate(zip(points, valid)):
        if not ok:
            continue
        cx, cy = int(round(x)), int(round(y))
        x0, x1 = max(0, cx - r), min(out_w, cx + r + 1)
        y0, y1 = max(0, cy - r), min(out_h, cy + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        xs = np.arange(x0, x1)[None, :]
        ys = np.arange(y0, y1)[:, None]
        g = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma ** 2))
        hm[i, y0:y1, x0:x1] = np.maximum(hm[i, y0:y1, x0:x1], g)
        if 0 <= cx < out_w and 0 <= cy < out_h:
            hm[i, cy, cx] = 1.0
    return hm


class PitchKeypointDataset(Dataset):
    def __init__(self, index_files, in_w=640, in_h=352, stride=4, sigma=1.5, augment=True):
        self.items = []
        for f in index_files:
            with open(f) as fh:
                self.items += json.load(fh)
        self.in_w, self.in_h, self.stride, self.sigma, self.augment = in_w, in_h, stride, sigma, augment

    def __len__(self):
        return len(self.items)

    def _photometric(self, img, rng):
        """画質の粗いスマホ映像に近づける（明るさ・彩度・ぼけ・ノイズ・JPEG 圧縮）"""
        img = img.astype(np.float32)
        img = img * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25)
        hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-8, 8)) % 180
        hsv[..., 1] *= rng.uniform(0.6, 1.4)
        img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        if rng.random() < 0.4:
            k = int(rng.choice([3, 5]))
            img = cv2.GaussianBlur(img, (k, k), 0)
        if rng.random() < 0.3:   # 動きブレ
            k = int(rng.integers(3, 9))
            ker = np.zeros((k, k), np.float32)
            ker[k // 2, :] = 1.0 / k
            M = cv2.getRotationMatrix2D((k / 2, k / 2), rng.uniform(0, 180), 1)
            img = cv2.filter2D(img, -1, cv2.warpAffine(ker, M, (k, k)))
        if rng.random() < 0.4:
            img = np.clip(img + rng.normal(0, rng.uniform(2, 8), img.shape), 0, 255).astype(np.uint8)
        if rng.random() < 0.6:
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(25, 90))])
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img

    def __getitem__(self, i):
        it = self.items[i]
        img = cv2.imread(it["image"])
        h, w = img.shape[:2]
        pts = np.full((len(KEYS), 2), -1e4, np.float32)
        for k, (u, v) in it["keypoints"].items():
            if k in KEYS:
                pts[KEYS.index(k)] = (u, v)
        valid = pts[:, 0] > -1e3

        # 入力サイズへの変換（拡張時はランダムな拡大縮小・平行移動・わずかな回転を加える）
        sx, sy = self.in_w / w, self.in_h / h
        M = np.array([[sx, 0, 0], [0, sy, 0]], np.float64)
        rng = np.random.default_rng()
        if self.augment:
            s = rng.uniform(0.8, 1.3)
            ang = rng.uniform(-3, 3)
            tx, ty = rng.uniform(-0.1, 0.1) * self.in_w, rng.uniform(-0.1, 0.1) * self.in_h
            R = cv2.getRotationMatrix2D((self.in_w / 2, self.in_h / 2), ang, s)
            R[:, 2] += (tx, ty)
            M = R @ np.vstack([M, [0, 0, 1]])
        img = cv2.warpAffine(img, M, (self.in_w, self.in_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        pts = (np.c_[pts, np.ones(len(pts))] @ M.T).astype(np.float32)

        if self.augment and rng.random() < 0.5:   # 左右反転（キーポイントの左右も入れ替える）
            img = img[:, ::-1].copy()
            pts[:, 0] = self.in_w - 1 - pts[:, 0]
            pts, valid = pts[FLIP_INDEX], valid[FLIP_INDEX]
        if self.augment:
            img = self._photometric(img, rng)

        valid &= (pts[:, 0] >= 0) & (pts[:, 0] < self.in_w) & (pts[:, 1] >= 0) & (pts[:, 1] < self.in_h)
        ow, oh = self.in_w // self.stride, self.in_h // self.stride
        hm = gaussian_heatmaps(pts / self.stride, valid, ow, oh, self.sigma)

        x = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy()).float() / 255.0
        x = (x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        return x, torch.from_numpy(hm), torch.from_numpy(pts), torch.from_numpy(valid)
