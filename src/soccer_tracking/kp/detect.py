"""学習済みモデルでフレームからキーポイントを推定し、ホモグラフィを求める"""

import cv2
import numpy as np
import torch

from .. import pitch
from .model import PitchKeypointNet


class KeypointDetector:
    def __init__(self, weights, device=None):
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        self.cfg = ckpt["config"]
        self.keys = self.cfg["keys"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PitchKeypointNet(len(self.keys), self.cfg["encoder"], self.cfg["width"], self.cfg["use_ca"])
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]

    @torch.no_grad()
    def __call__(self, frame, min_score=0.3):
        """戻り値: {key: (u, v, score)}（元のフレームの座標）"""
        h, w = frame.shape[:2]
        iw, ih, st = self.cfg["in_w"], self.cfg["in_h"], self.cfg["stride"]
        img = cv2.cvtColor(cv2.resize(frame, (iw, ih), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(img).to(self.device).permute(2, 0, 1).float() / 255.0
        x = ((x - self.mean) / self.std)[None]
        hm = torch.sigmoid(self.model(x))[0]  # (K, oh, ow)
        K, oh, ow = hm.shape
        flat = hm.view(K, -1)
        score, idx = flat.max(dim=1)
        out = {}
        hm_np = hm.cpu().numpy()
        for k in range(K):
            s = float(score[k])
            if s < min_score:
                continue
            cy, cx = divmod(int(idx[k]), ow)
            # 3x3 の重み付き平均でサブピクセル位置にする
            y0, y1, x0, x1 = max(cy - 1, 0), min(cy + 2, oh), max(cx - 1, 0), min(cx + 2, ow)
            patch = hm_np[k, y0:y1, x0:x1]
            ys, xs = np.mgrid[y0:y1, x0:x1]
            px = (patch * xs).sum() / patch.sum()
            py = (patch * ys).sum() / patch.sum()
            u = (px * st + (st - 1) / 2) * w / iw
            v = (py * st + (st - 1) / 2) * h / ih
            out[self.keys[k]] = (float(u), float(v), s)
        return out


def homography_from_detections(dets, L, W, ransac_m=1.0, min_inliers=5):
    """検出したキーポイントから 画像→ピッチ（m）のホモグラフィを RANSAC で求める。
    戻り値: (H, 採用したキーの一覧) / 失敗時 (None, [])"""
    xy = {k: xy for k, _, xy in pitch.keypoints(L, W)}
    keys = [k for k in dets if k in xy]
    if len(keys) < 4:
        return None, []
    img = np.float32([dets[k][:2] for k in keys])
    pit = np.float32([xy[k] for k in keys])
    H, inl = cv2.findHomography(img, pit, cv2.RANSAC, ransac_m)
    if H is None:
        return None, []
    inl = inl.ravel().astype(bool)
    if inl.sum() < min_inliers:
        return None, []
    return H, [k for k, ok in zip(keys, inl) if ok]
