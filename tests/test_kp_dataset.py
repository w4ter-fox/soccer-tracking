import json

import cv2
import numpy as np
import pytest

pytest.importorskip("torch")

from soccer_tracking.kp.dataset import FLIP_INDEX, KEYS, PitchKeypointDataset  # noqa: E402


def test_flip_index_is_an_involution():
    assert all(FLIP_INDEX[FLIP_INDEX[i]] == i for i in range(len(KEYS)))


def test_labels_follow_augmentation(tmp_path):
    """キーポイントの位置に白い点を描いた画像で、拡張後もラベルが白い点の上にあること"""
    rng = np.random.default_rng(0)
    items = []
    for n in range(4):
        img = np.zeros((540, 960, 3), np.uint8)
        kps = {}
        for k in rng.choice(KEYS, 8, replace=False):
            u, v = float(rng.uniform(50, 910)), float(rng.uniform(50, 490))
            cv2.circle(img, (round(u), round(v)), 3, (255, 255, 255), -1)
            kps[str(k)] = [u, v]
        p = tmp_path / f"{n}.png"
        cv2.imwrite(str(p), img)
        items.append({"image": str(p), "width": 960, "height": 540, "keypoints": kps})
    idx = tmp_path / "index.json"
    idx.write_text(json.dumps(items))
    ds = PitchKeypointDataset([str(idx)], augment=True)
    ds._photometric = lambda img, rng: img
    for i in range(20):
        x, hm, pts, valid = ds[i % len(ds)]
        img = (x.numpy().transpose(1, 2, 0) * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
        for k in np.nonzero(valid.numpy())[0]:
            u, v = pts[k].numpy()
            patch = img[max(round(v) - 1, 0) : round(v) + 2, max(round(u) - 1, 0) : round(u) + 2].mean(2)
            assert patch.max() > 200
            oy, ox = np.unravel_index(hm[k].float().numpy().argmax(), hm[k].shape)
            assert abs(ox * 4 + 1.5 - u) < 6 and abs(oy * 4 + 1.5 - v) < 6
