"""2D ピッチの静止画と動画を作る

使い方（リポジトリの直下で実行）:
  uv run soccer-visualize video3.mov
"""

import argparse
import json
import os

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402
from mplsoccer import Pitch  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="2D座標の静止画と動画を作る")
    ap.add_argument("video", help="動画ファイル名（例: video1.mp4）")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--snapshot-frame", type=int, default=None)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument(
        "--max-gap",
        type=float,
        default=1.5,
        help="選手が見えなかった区間をこの秒数（実時間）まで、前後の位置から補って描く（0で補わない）",
    )
    ap.add_argument(
        "--ball",
        choices=["none", "near", "all"],
        default="none",
        help="2Dにボールをどう出すか。none: 出さない / near: 選手の近く（地面にある可能性が高い）だけ / all: 全部",
    )
    ap.add_argument(
        "--ball-near-m", type=float, default=2.0, help="--ball near で、選手からこの距離（m）以内のときだけ出す"
    )
    args = ap.parse_args()

    BASE_DIR = os.path.abspath(args.data_dir)
    TARGET_VIDEO = args.video
    SNAPSHOT_FRAME = args.snapshot_frame
    SMOOTH_WINDOW = 5  # 座標のガタつきを抑える移動平均のフレーム数（1で無効）
    MAKE_VIDEO = not args.no_video

    name = os.path.splitext(TARGET_VIDEO)[0]
    df = pd.read_csv(os.path.join(BASE_DIR, f"{name}_positions.csv"))

    # トラッキングセルが保存したfpsとスロー倍率（2D動画も1倍速で書き出す）
    meta_path = os.path.join(BASE_DIR, f"{name}_meta.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        fps, slowdown = meta["file_fps"], meta["slowdown"]
        output_step = meta.get("output_step", slowdown)
        pitch_meta = meta.get("pitch", {"units": "statsbomb"})
    else:
        pitch_meta = {"units": "statsbomb"}
        cap = cv2.VideoCapture(os.path.join(BASE_DIR, TARGET_VIDEO))
        fps, slowdown = cap.get(cv2.CAP_PROP_FPS) or 30, 1
        output_step = 1
        cap.release()

    # トラッキング動画のボックス色と合わせる（塗り, 縁）
    STYLE = {
        0: ("#ffffff", "black", "Team White"),
        1: ("#1e5aff", "black", "Team Blue"),
        2: ("#ffd700", "black", "Referee"),
        3: ("#e61e1e", "black", "Goalkeeper"),
        4: ("#ff8c00", "black", "Ball"),
        5: ("#969696", "black", "Unknown"),
    }

    # 選手が見えなかった短い区間（隠れた、検出が一瞬抜けた等）を、前後の位置の直線補間で埋める。
    # 途切れが長いところは、どこにいたか分からないので埋めない
    real_fps = meta.get("real_fps", fps * slowdown) if os.path.exists(meta_path) else fps * slowdown
    max_gap_frames = int(args.max_gap * real_fps)
    if max_gap_frames > 0:
        filled = []
        n_added = 0
        for tid, g in df[df["class_id"] != 4].groupby("tracker_id"):
            g = g.drop_duplicates("frame").set_index("frame").sort_index()
            full = g.reindex(range(int(g.index.min()), int(g.index.max()) + 1))
            missing = full["pitch_x"].isna()
            # 欠けている区間ごとの長さを数え、長い区間は補間しない
            run_id = (missing != missing.shift()).cumsum()
            run_len = missing.groupby(run_id).transform("sum")
            fillable = missing & (run_len <= max_gap_frames)
            interp = full[["pitch_x", "pitch_y"]].interpolate(limit_area="inside")
            full.loc[fillable, ["pitch_x", "pitch_y"]] = interp.loc[fillable]
            full.loc[fillable, ["tracker_id", "class_id"]] = [tid, g["class_id"].iloc[0]]
            full["interpolated"] = fillable
            n_added += int(fillable.sum())
            filled.append(full[full["pitch_x"].notna()].reset_index().rename(columns={"index": "frame"}))
        df = pd.concat(filled + [df[df["class_id"] == 4].assign(interpolated=False)], ignore_index=True)
        df["tracker_id"] = df["tracker_id"].astype(int)
        df["class_id"] = df["class_id"].astype(int)
        print(f"選手の途切れを補間: {n_added} 点（{args.max_gap} 秒以内の途切れ）")

    # 選手は同じトラッカーIDの中で移動平均をかける（ボールはIDが不安定なのでそのまま）
    df = df.sort_values(["tracker_id", "frame"])
    players = df["class_id"] != 4
    if SMOOTH_WINDOW > 1:
        for col in ["pitch_x", "pitch_y"]:
            df.loc[players, col] = (
                df[players]
                .groupby("tracker_id")[col]
                .transform(lambda s: s.rolling(SMOOTH_WINDOW * slowdown, min_periods=1, center=True).mean())
            )

    # ホモグラフィの誤差で大きくはみ出した点は除外する
    if pitch_meta["units"] == "m":
        PL, PW = pitch_meta["length"], pitch_meta["width"]
        df = df[df["pitch_x"].between(-5, PL + 5) & df["pitch_y"].between(-5, PW + 5)]
    else:
        df = df[df["pitch_x"].between(55, 125) & df["pitch_y"].between(-5, 85)]
    # ボールの2D座標は、地面の平面に写しているので、浮いているボールは実際より奥に大きくずれる。
    # 選手の足元の近くにあるときは地面にある可能性が高いので、既定ではそのときだけ出す
    is_ball = df["class_id"] == 4
    if args.ball == "none":
        df = df[~is_ball]
    elif args.ball == "near":
        players_by_f = {f: g[["pitch_x", "pitch_y"]].to_numpy() for f, g in df[~is_ball].groupby("frame")}
        keep = []
        for idx, r in df[is_ball].iterrows():
            p = players_by_f.get(r["frame"])
            if (
                p is not None
                and len(p)
                and np.min(np.hypot(p[:, 0] - r["pitch_x"], p[:, 1] - r["pitch_y"])) <= args.ball_near_m
            ):
                keep.append(idx)
        df = pd.concat([df[~is_ball], df.loc[keep]]).sort_values(["frame", "tracker_id"])
        print(f"ボール: 選手の {args.ball_near_m} m 以内にある {len(keep)}/{int(is_ball.sum())} フレームだけ表示")
    frames = np.sort(df["frame"].unique())
    print(f"2D座標のあるフレーム: {len(frames)}  ({frames.min()}〜{frames.max()})")

    if pitch_meta["units"] == "m":
        # register.py の位置合わせ: 実寸（m）のピッチ全面。左下が原点、手前（カメラ側）が下
        pitch = Pitch(pitch_type="custom", pitch_length=PL, pitch_width=PW, pitch_color="#22312b", line_color="#c7d5cc")
    else:
        # 従来の方式: 映っているゴールは常に右端に置いているので、右半面だけ描く
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
            pitch.scatter(sub["pitch_x"], sub["pitch_y"], ax=ax, c=face, edgecolors=edge, s=size, zorder=3, label=label)
            if cid != 4:
                for _, r in sub.iterrows():
                    ax.text(
                        r["pitch_x"],
                        r["pitch_y"] - 2.5,
                        str(int(r["tracker_id"])),
                        color="white",
                        fontsize=8,
                        ha="center",
                        zorder=4,
                    )
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
            scatters[cid] = ax.scatter(
                [], [], c=face, edgecolors=edge, s=100 if cid == 4 else 200, zorder=3, label=label
            )
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

        all_frames = range(int(frames.min()), int(frames.max()) + 1, output_step)  # トラッキング動画と同じ速度
        anim = animation.FuncAnimation(fig, update, frames=all_frames, blit=False)
        out_path = os.path.join(BASE_DIR, f"{name}_2d.mp4")
        anim.save(out_path, writer=animation.FFMpegWriter(fps=fps), savefig_kwargs={"facecolor": "#22312b"})
        plt.close(fig)
        print(f"保存: {out_path}")


if __name__ == "__main__":
    main()
