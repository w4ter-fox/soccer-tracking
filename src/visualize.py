import os
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from mplsoccer import Pitch

import argparse

ap = argparse.ArgumentParser(description="2D座標の静止画と動画を作る")
ap.add_argument("video", help="動画ファイル名（例: video1.mp4）")
ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
ap.add_argument("--snapshot-frame", type=int, default=None)
ap.add_argument("--no-video", action="store_true")
args = ap.parse_args()

BASE_DIR = os.path.abspath(args.data_dir)
TARGET_VIDEO = args.video
SNAPSHOT_FRAME = args.snapshot_frame
SMOOTH_WINDOW = 5       # 座標のガタつきを抑える移動平均のフレーム数（1で無効）
MAKE_VIDEO = not args.no_video

name = os.path.splitext(TARGET_VIDEO)[0]
df = pd.read_csv(os.path.join(BASE_DIR, f"{name}_positions.csv"))

# トラッキングセルが保存したfpsとスロー倍率（2D動画も1倍速で書き出す）
meta_path = os.path.join(BASE_DIR, f"{name}_meta.json")
if os.path.exists(meta_path):
    meta = json.load(open(meta_path))
    fps, slowdown = meta["file_fps"], meta["slowdown"]
else:
    cap = cv2.VideoCapture(os.path.join(BASE_DIR, TARGET_VIDEO))
    fps, slowdown = cap.get(cv2.CAP_PROP_FPS) or 30, 1
    cap.release()

# トラッキング動画のボックス色と合わせる（塗り, 縁）
STYLE = {
    0: ("#ffffff", "black", "Team White"),
    1: ("#1e5aff", "black", "Team Blue"),
    2: ("#ffd700", "black", "Referee"),
    3: ("#00c850", "black", "Goalkeeper"),
    4: ("#ff8c00", "black", "Ball"),
}

# 選手は同じトラッカーIDの中で移動平均をかける（ボールはIDが不安定なのでそのまま）
df = df.sort_values(["tracker_id", "frame"])
players = df["class_id"] != 4
if SMOOTH_WINDOW > 1:
    for col in ["pitch_x", "pitch_y"]:
        df.loc[players, col] = (df[players].groupby("tracker_id")[col]
                                .transform(lambda s: s.rolling(SMOOTH_WINDOW * slowdown, min_periods=1, center=True).mean()))

# ホモグラフィの誤差で大きくはみ出した点は除外する
df = df[df["pitch_x"].between(55, 125) & df["pitch_y"].between(-5, 85)]
frames = np.sort(df["frame"].unique())
print(f"2D座標のあるフレーム: {len(frames)}  ({frames.min()}〜{frames.max()})")

# 映っているゴールは常に右端に置いているので、右半面だけ描く
pitch = Pitch(pitch_type="statsbomb", half=True, pitch_color="#22312b", line_color="#c7d5cc")


def draw_snapshot(frame_no):
    fig, ax = pitch.draw(figsize=(10, 7))
    fig.patch.set_facecolor("#22312b")
    cur = df[df["frame"] == frame_no]
    for cid, (face, edge, label) in STYLE.items():
        sub = cur[cur["class_id"] == cid]
        if len(sub) == 0:
            continue
        size = 100 if cid == 4 else 200
        pitch.scatter(sub["pitch_x"], sub["pitch_y"], ax=ax, c=face, edgecolors=edge,
                      s=size, zorder=3, label=label)
        if cid != 4:
            for _, r in sub.iterrows():
                ax.text(r["pitch_x"], r["pitch_y"] - 2.5, str(int(r["tracker_id"])),
                        color="white", fontsize=8, ha="center", zorder=4)
    ax.legend(loc="upper right", facecolor="#22312b", edgecolor="white", labelcolor="white")
    ax.set_title(f"2D Tracking - frame {frame_no}", color="white", fontsize=16)
    out_png = os.path.join(BASE_DIR, f"{name}_2d_frame{frame_no}.png")
    plt.savefig(out_png, dpi=120, facecolor="#22312b")
    plt.close(fig)
    print(f"保存: {out_png}")


snapshot = SNAPSHOT_FRAME if SNAPSHOT_FRAME is not None else int(frames[len(frames) // 2])
draw_snapshot(snapshot)

if MAKE_VIDEO:
    fig, ax = pitch.draw(figsize=(10, 7))
    fig.patch.set_facecolor("#22312b")
    scatters = {}
    for cid, (face, edge, label) in STYLE.items():
        scatters[cid] = ax.scatter([], [], c=face, edgecolors=edge, s=100 if cid == 4 else 200,
                                   zorder=3, label=label)
    ax.legend(loc="upper right", facecolor="#22312b", edgecolor="white", labelcolor="white")
    title = ax.set_title("", color="white", fontsize=16)
    by_frame = dict(tuple(df.groupby("frame")))
    empty = np.empty((0, 2))

    def update(frame_no):
        cur = by_frame.get(frame_no)
        for cid, sc in scatters.items():
            if cur is None:
                sc.set_offsets(empty)
                continue
            sub = cur[cur["class_id"] == cid]
            sc.set_offsets(sub[["pitch_x", "pitch_y"]].to_numpy() if len(sub) else empty)
        title.set_text(f"2D Tracking - frame {frame_no}")
        return list(scatters.values()) + [title]

    all_frames = range(int(frames.min()), int(frames.max()) + 1, slowdown)  # 1倍速
    anim = animation.FuncAnimation(fig, update, frames=all_frames, blit=False)
    out_path = os.path.join(BASE_DIR, f"{name}_2d.mp4")
    anim.save(out_path, writer=animation.FFMpegWriter(fps=fps), savefig_kwargs={"facecolor": "#22312b"})
    plt.close(fig)
    print(f"保存: {out_path}")
