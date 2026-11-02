"""選手・審判・ボールのトラッキングと2D座標の出力

1パス目: 検出 → ByteTrack → 色判定の集計、ボールの候補集め
全フレーム後: トラックのつなぎ直し → チームの決定（トラック単位）→ ボールの経路選択と弾道フィット
2パス目: 描画と2D座標の記録

2D座標は registration.py の結果（<動画名>_homography.npz）があるときだけ出す。

使い方（リポジトリの直下で実行）:
  uv run soccer-track video3.mov --slowdown 4 --kp-model models/pitch_kp/best.pt
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import cv2
import numpy as np
import pandas as pd

from . import pitch
from .ball import (
    BallTracker,
    ball_candidates,
    detect_ball_tiled,
    draw_ball,
    filter_ball_candidates,
    fit_ball_trajectory,
    merge_candidates,
    search_ball_roi,
    select_ball_path,
)
from .config import BALL, BOX_COLORS_RGB, TEAM_NAMES, UNKNOWN, Config
from .jersey import classify_by_jersey_color
from .linking import link_tracklets
from .teams import load_team_overrides, occlusion_flags, resolve_teams
from .video_io import FFmpegWriter
from .vision import get_pitch_mask


def load_registration(base):
    """registration.py の結果を読む。なければ None"""
    path = base + "_homography.npz"
    if not os.path.exists(path):
        return None
    reg = np.load(path)
    print(f"  位置合わせ: {os.path.basename(path)}（{int(reg['valid'].sum())}/{len(reg['valid'])} フレーム）")
    return reg


def merge_links(link, team_votes, spans, per_frame):
    """つなぎ直しの結果（古いID → 新しいID）を、票・区間・フレームごとのIDに反映する"""
    for old, new in link.items():
        team_votes[new].update(team_votes.pop(old, Counter()))
        so, sn = spans.pop(old), spans[new]
        if so["f0"] < sn["f0"]:
            sn["f0"], sn["start"] = so["f0"], so["start"]
        if so["f1"] > sn["f1"]:
            sn["f1"], sn["end"] = so["f1"], so["end"]
    return [
        (xyxy, np.array([link.get(int(t), int(t)) for t in tids], dtype=int), H, s) for xyxy, tids, H, s in per_frame
    ]


def process_video(input_path, output_path, cfg: Config):
    import supervision as sv
    from trackers import ByteTrackTracker
    from ultralytics import YOLO

    print(f"\n処理開始: {os.path.basename(input_path)}")
    model = YOLO(cfg.model)
    video_info = sv.VideoInfo.from_video_path(video_path=input_path)
    fps = video_info.fps or 30  # ファイル上のfps
    real_fps = fps * cfg.slowdown  # 実時間1秒あたりのフレーム数
    # 隠れて見えなくなった選手のIDを実時間で2秒間保持する
    tracker = ByteTrackTracker(frame_rate=real_fps, lost_track_buffer=int(real_fps * 2))
    ball_tracker = BallTracker(cfg)

    base = os.path.splitext(input_path)[0]
    reg = load_registration(base)
    if reg is not None:
        L, W = float(reg["pitch_length"]), float(reg["pitch_width"])
        spots = pitch.spots(L, W)
        pitch_meta = {"units": "m", "length": L, "width": W}
    else:
        print("  位置合わせの結果がないので、2D座標は出しません（--kp-model を付けるか registration を先に実行）。")
        spots, pitch_meta = None, None

    team_votes = defaultdict(Counter)  # トラックIDごとの色判定の集計
    spans = {}  # トラックの始まり・終わり
    track_pts = defaultdict(list)  # トラックIDごとの足元の位置の履歴（つなぎ直しに使う）
    per_frame = []  # (xyxy, tracker_ids, H, h_score)
    ball_cands = {}  # フレームごとのボール候補

    print(
        f"  1/2 推論とトラッキング（{video_info.total_frames} フレーム, 実時間 {video_info.total_frames / real_fps:.1f} 秒）..."
    )
    for frame_idx, frame in enumerate(sv.get_video_frames_generator(input_path)):
        pitch_mask = get_pitch_mask(frame)
        h_img, w_img = pitch_mask.shape[:2]
        results = model(
            frame,
            imgsz=cfg.imgsz,
            conf=min(cfg.person_conf, cfg.ball_conf),
            iou=cfg.nms_iou,
            classes=[0, 32],
            verbose=False,
        )[0]
        raw = sv.Detections.from_ultralytics(results)

        H, h_score = None, 0.0
        if reg is not None and frame_idx < len(reg["valid"]) and reg["valid"][frame_idx]:
            H, h_score = reg["H"][frame_idx], float(reg["score"][frame_idx])

        # ---- 選手 ----
        persons = raw[(raw.class_id == 0) & (raw.confidence >= cfg.person_conf)]
        keep, team_ids = [], []
        for i, xyxy in enumerate(persons.xyxy):
            feet_x = min(max(int((xyxy[0] + xyxy[2]) / 2), 0), w_img - 1)
            feet_y = min(max(int(xyxy[3]), 0), h_img - 1)
            if pitch_mask[feet_y, feet_x] == 0:  # ピッチ外（観客・ベンチ）
                continue
            x1, y1, x2, y2 = map(int, xyxy)
            # 色が判定できないフレームでも人としては追跡を続ける（票には数えない）
            team_ids.append(
                classify_by_jersey_color(
                    frame[max(y1, 0) : y2, max(x1, 0) : x2], cfg.jersey_colors, cfg.jersey_min_ratio
                )
            )
            keep.append(i)
        persons = persons[np.array(keep, dtype=int)]
        persons.class_id = np.array(team_ids, dtype=int)
        persons.data["occluded"] = occlusion_flags(persons.xyxy, cfg.occlusion_ratio)
        persons = tracker.update(persons)

        tracker_ids = persons.tracker_id if persons.tracker_id is not None else np.full(len(persons), -1)
        occluded = persons.data.get("occluded", np.zeros(len(persons), dtype=bool))
        confirmed = tracker_ids >= 0
        xyxy_c, tids_c = persons.xyxy[confirmed].copy(), tracker_ids[confirmed].copy()
        for box, cid, tid, occ in zip(xyxy_c, persons.class_id[confirmed], tids_c, occluded[confirmed]):
            tid = int(tid)
            # 他の選手と重なっているフレームの色判定は、隣の選手の色が混ざるので数えない
            if not occ and int(cid) != UNKNOWN:
                team_votes[tid][int(cid)] += 1
            foot = ((box[0] + box[2]) / 2, box[3])
            foot_pitch = tuple(cv2.perspectiveTransform(np.float32([[foot]]), H)[0, 0]) if H is not None else None
            pos = {"img": foot, "pitch": foot_pitch, "h": float(box[3] - box[1])}
            if tid not in spans:
                spans[tid] = {"f0": frame_idx, "start": pos}
            spans[tid]["f1"], spans[tid]["end"] = frame_idx, pos
            track_pts[tid].append(
                (
                    frame_idx,
                    foot[0],
                    foot[1],
                    foot_pitch[0] if foot_pitch else None,
                    foot_pitch[1] if foot_pitch else None,
                    float(box[3] - box[1]),
                )
            )
        # 表示と記録はIDが確定した選手だけ（低信頼の検出は追跡の継続にだけ使う）
        per_frame.append((xyxy_c, tids_c, H, h_score))

        # ---- ボール（候補を集めるだけ。採否は全フレーム後に経路選択で決める）----
        pred = ball_tracker.predict(frame_idx)
        balls = raw[raw.class_id == 32]
        cands = ball_candidates(balls.xyxy, balls.confidence) + detect_ball_tiled(model, frame, cfg)
        near = pred is not None and any(np.hypot(c[0] - pred[0], c[1] - pred[1]) <= ball_tracker.gate() for c in cands)
        if pred is not None and not near:
            cands += search_ball_roi(model, frame, pred, cfg)
        cands = filter_ball_candidates(merge_candidates(cands), H, spots, cfg)
        ball_tracker.update(pred, [c[:4] for c in cands])
        if cands:
            ball_cands[frame_idx] = cands

    # ---- トラックのつなぎ直しとチームの決定 ----
    link = link_tracklets(track_pts, team_votes, real_fps, cfg)
    if link:
        per_frame = merge_links(link, team_votes, spans, per_frame)
    print(f"  つなぎ直し: {len(link)} 個のトラックを前のトラックにつなげました（{len(spans)} トラックに）")
    team_of = resolve_teams(team_votes, spans, real_fps, cfg)
    for tid, team in load_team_overrides(base).items():
        if tid in spans:
            team_of[tid] = (team, "manual")
    src_count = Counter(s for _, s in team_of.values())
    print(
        f"  チーム: {len(team_of)} トラック（色判定で確定 {src_count['votes']} / 引き継ぎ {src_count['inherited']} / "
        f"判定少 {src_count['weak']} / 不明 {src_count['unknown']} / 手動 {src_count['manual']}）"
    )

    # ---- ボールの経路選択と弾道フィット ----
    ball_obs = select_ball_path(ball_cands, cfg)
    ball_fit = fit_ball_trajectory(ball_obs, cfg)
    n_total = len(per_frame)
    n_fit = sum(1 for v in ball_fit.values() if v[3] == "fit")
    print(
        f"  ボール: 実測 {len(ball_obs)}/{n_total} フレーム（{100 * len(ball_obs) / max(n_total, 1):.0f}%）"
        f" → 弾道で補完後 {len(ball_fit)} フレーム（{100 * len(ball_fit) / max(n_total, 1):.0f}%、うち2次式フィット {n_fit}）"
    )

    # ---- 2パス目: 描画と2D座標の記録 ----
    print("  2/2 描画と2D座標の記録...")
    palette = sv.ColorPalette(colors=[sv.Color(r=r, g=g, b=b) for r, g, b in BOX_COLORS_RGB])
    box_annotator = sv.BoxAnnotator(color=palette, thickness=2, color_lookup=sv.ColorLookup.CLASS)
    label_annotator = sv.LabelAnnotator(
        color=palette, text_color=sv.Color.BLACK, text_scale=0.5, color_lookup=sv.ColorLookup.CLASS
    )
    records = []
    with FFmpegWriter(output_path, video_info.width, video_info.height, fps, crf=cfg.output_crf) as sink:
        for frame_idx, frame in enumerate(sv.get_video_frames_generator(input_path)):
            if frame_idx >= len(per_frame):
                break
            xyxy, tids, H, score = per_frame[frame_idx]
            cls = np.array([team_of.get(int(t), (UNKNOWN, "unknown"))[0] for t in tids], dtype=int)
            src = [team_of.get(int(t), (UNKNOWN, "unknown"))[1] for t in tids]
            ball = ball_fit.get(frame_idx)
            t_sec = frame_idx / real_fps

            if H is not None:
                records += frame_records(frame_idx, t_sec, xyxy, tids, cls, src, ball, H, score)

            if cfg.output_speed == "realtime" and frame_idx % cfg.slowdown != 0:
                continue
            annotated = frame.copy()
            if len(xyxy) > 0:
                dets = sv.Detections(xyxy=xyxy, class_id=cls, tracker_id=tids)
                annotated = box_annotator.annotate(scene=annotated, detections=dets)
                annotated = label_annotator.annotate(
                    scene=annotated, detections=dets, labels=[str(int(t)) for t in tids]
                )
            trail = [
                ball_fit[f][:2]
                for f in range(max(0, frame_idx - cfg.ball_trail * cfg.slowdown), frame_idx + 1)
                if f in ball_fit
            ]
            draw_ball(annotated, ball, trail)
            sink.write_frame(annotated)

    speed_note = "入力と同じ速度" if cfg.output_speed == "input" else "実時間"
    print(f"  完了！ {os.path.basename(output_path)} を保存しました（{speed_note}）。")
    write_outputs(base, fps, real_fps, cfg, pitch_meta, team_of, spans, team_votes, records)


def frame_records(frame_idx, t_sec, xyxy, tids, cls, src, ball, H, score):
    """1フレーム分の2D座標の行（選手は足元、ボールは中心）"""
    rows = []
    if len(xyxy) > 0:
        feet = np.c_[(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]].astype(np.float32)
        pxy = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), H).reshape(-1, 2)
        for (px, py), (ix, iy), cid, tid, s in zip(pxy, feet, cls, tids, src):
            rows.append(
                {
                    "frame": frame_idx,
                    "time_s": t_sec,
                    "tracker_id": int(tid),
                    "class_id": int(cid),
                    "team_source": s,
                    "image_x": float(ix),
                    "image_y": float(iy),
                    "pitch_x": float(px),
                    "pitch_y": float(py),
                    "ball_kind": "",
                    "h_score": score,
                }
            )
    if ball is not None:
        bx, by, _, kind = ball
        px, py = cv2.perspectiveTransform(np.float32([[[bx, by]]]), H)[0, 0]
        rows.append(
            {
                "frame": frame_idx,
                "time_s": t_sec,
                "tracker_id": -1,
                "class_id": BALL,
                "team_source": "",
                "image_x": float(bx),
                "image_y": float(by),
                "pitch_x": float(px),
                "pitch_y": float(py),
                "ball_kind": kind,
                "h_score": score,
            }
        )
    return rows


def write_outputs(base, fps, real_fps, cfg, pitch_meta, team_of, spans, team_votes, records):
    with open(base + "_meta.json", "w") as f:
        json.dump(
            {
                "file_fps": fps,
                "slowdown": cfg.slowdown,
                "real_fps": real_fps,
                "output_step": cfg.slowdown if cfg.output_speed == "realtime" else 1,
                "pitch": pitch_meta,
            },
            f,
        )
    names = {v: k for k, v in TEAM_NAMES.items()}
    tracks = [
        {
            "tracker_id": tid,
            "team": names.get(team, "unknown"),
            "source": s,
            "start_frame": spans[tid]["f0"],
            "end_frame": spans[tid]["f1"],
            "votes": dict(team_votes.get(tid, {})),
        }
        for tid, (team, s) in sorted(team_of.items())
    ]
    pd.DataFrame(tracks).to_csv(base + "_tracks.csv", index=False)
    if records:
        pd.DataFrame(records).to_csv(base + "_positions.csv", index=False)
        print(f"  2D座標: {os.path.basename(base)}_positions.csv を保存しました（{len(records)} 行）。")


def main():
    cfg = Config()
    ap = argparse.ArgumentParser(description="選手・ボールのトラッキングと2D座標の出力")
    ap.add_argument("videos", nargs="+", help="data/ にある動画のファイル名")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--model", default=cfg.model)
    ap.add_argument("--imgsz", type=int, default=cfg.imgsz)
    ap.add_argument("--slowdown", type=int, default=cfg.slowdown, help="入力が実時間の何倍にスロー化されているか")
    ap.add_argument(
        "--output-speed",
        choices=["input", "realtime"],
        default=cfg.output_speed,
        help="出力動画の速度（input: 入力と同じ / realtime: slowdown で割って実時間に戻す）",
    )
    ap.add_argument("--ball-tiles", type=int, default=cfg.ball_tiles, help="ボール用のタイル分割数（0で無効、2で2×2）")
    ap.add_argument(
        "--kp-model",
        default=None,
        help="ピッチのキーポイント検出モデル。指定すると、位置合わせを自動で実行してから追跡する",
    )
    ap.add_argument("--kp-every", type=int, default=5, help="キーポイントを何フレームごとに推論するか")
    ap.add_argument("--re-register", action="store_true", help="位置合わせの結果があってもやり直す")
    args = ap.parse_args()
    cfg.model, cfg.imgsz, cfg.slowdown = args.model, args.imgsz, args.slowdown
    cfg.output_speed, cfg.ball_tiles = args.output_speed, args.ball_tiles
    data_dir = os.path.abspath(args.data_dir)

    for filename in args.videos:
        input_path = os.path.join(data_dir, filename)
        name = os.path.splitext(filename)[0]
        if not os.path.exists(input_path):
            print(f"\nエラー: {input_path} が見つかりません。")
            continue
        npz = os.path.join(data_dir, f"{name}_homography.npz")
        if args.kp_model and (args.re_register or not os.path.exists(npz)):
            from .registration import run as register

            print(f"\n位置合わせ: {filename}")
            register(filename, data_dir, kp_model=args.kp_model, kp_every=args.kp_every)
        process_video(input_path, os.path.join(data_dir, f"{name}_tracking.mp4"), cfg)


if __name__ == "__main__":
    main()
