"""動画のフレーム単位の読み込み

OpenCV のシーク（CAP_PROP_POS_FRAMES）はコーデックや可変フレームレートの動画で位置がずれることがあるので、
先頭から順に読み、track.py（supervision の get_video_frames_generator）と同じフレーム番号になるようにする。
"""
import os
from collections import OrderedDict

import cv2


class FrameReader:
    """フレーム番号を指定して読む。トラッキングセルと番号がずれないよう、シークせず先頭から順に読む。
    一度読んだフレームはキャッシュするので、行き来は速い。"""

    def __init__(self, path, cache_size=30):
        if not os.path.exists(path):
            raise FileNotFoundError(f"動画が見つかりません: {path}（Driveのマウントとファイル名を確認）")
        self.path = path
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self._open()
        if not self.cap.isOpened():
            raise RuntimeError(f"動画を開けません（コーデック非対応の可能性）: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    def _open(self):
        self.cap = cv2.VideoCapture(self.path)
        self.pos = 0  # 次に読むフレーム番号

    def get(self, idx):
        if idx in self.cache:
            self.cache.move_to_end(idx)
            return self.cache[idx]
        if idx < self.pos:  # 戻るときは開き直して先頭から
            self.cap.release()
            self._open()
        while self.pos < idx:
            if not self.cap.grab():
                return None
            self.pos += 1
        ok, frame = self.cap.read()
        if not ok:
            return None
        self.pos += 1
        self.cache[idx] = frame
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return frame
