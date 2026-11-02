import shutil

import cv2
import numpy as np
import pytest

from soccer_tracking.video_io import FFmpegWriter, FrameReader


def write_numbered_video(path, n=60):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (160, 120))
    for i in range(n):
        img = np.zeros((120, 160, 3), np.uint8)
        img[:, :] = (i * 4) % 256
        w.write(img)
    w.release()


def test_frame_reader_matches_sequential_read(tmp_path):
    path = tmp_path / "v.mp4"
    write_numbered_video(path)
    seq = []
    cap = cv2.VideoCapture(str(path))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        seq.append(f)
    reader = FrameReader(str(path))
    for i in [10, 3, 59, 58, 0, 30]:  # 前後に行き来しても同じフレームが返る
        assert np.array_equal(reader.get(i), seq[i])
    assert reader.get(500) is None


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg がない")
def test_ffmpeg_writer(tmp_path):
    out = tmp_path / "out.mp4"
    with FFmpegWriter(str(out), 161, 121, 25) as w:  # 奇数サイズでも書ける（偶数に余白を足す）
        for _ in range(10):
            w.write_frame(np.zeros((121, 161, 3), np.uint8))
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
