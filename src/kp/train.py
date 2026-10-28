"""ピッチのキーポイント検出モデルの学習（SoccerNet からフルスクラッチ）

使い方（src/ で実行）:
  uv run python -m kp.train --data ~/data/sn_keypoints --out ../models/pitch_kp
  uv run python -m kp.train --data ~/data/sn_keypoints --out ../models/pitch_kp --encoder mnv3_small   （さらに軽く）
  uv run python -m kp.train --data ~/data/sn_keypoints --out ../models/pitch_kp --resume   （途中から再開）
  追加学習（自分の映像でアノテーションしたデータを混ぜる）:
  uv run python -m kp.train --data ~/data/sn_keypoints --extra ../data/video3_kp/index.json \
      --init ../models/pitch_kp/best.pt --out ../models/pitch_kp_ft --epochs 15 --lr 5e-4

評価指標:
  PCK@5px   正解のキーポイントのうち、入力画像上で 5px 以内に検出できた割合
  誤検出率  正解が画面外（存在しない）なのに、閾値以上のスコアで検出してしまった割合
  H可能率   1枚の画像で、ホモグラフィを作れる数（5点以上）を検出できた割合
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kp.dataset import KEYS, PitchKeypointDataset  # noqa: E402
from kp.model import PitchKeypointNet, focal_loss  # noqa: E402


@torch.no_grad()
def evaluate(model, loader, device, stride, thr=0.3, pck_px=5.0):
    model.eval()
    hit = total = fp = n_neg = h_ok = n_img = 0
    loss_sum = 0.0
    for x, hm, pts, valid in loader:
        x, hm = x.to(device), hm.to(device)
        logits = model(x)
        loss_sum += focal_loss(logits, hm).item() * len(x)
        p = torch.sigmoid(logits)
        B, K, oh, ow = p.shape
        score, idx = p.view(B, K, -1).max(dim=2)
        py = (idx // ow).float() * stride + (stride - 1) / 2
        px = (idx % ow).float() * stride + (stride - 1) / 2
        pred = torch.stack([px, py], dim=2).cpu()
        score = score.cpu()
        det = score >= thr
        d = (pred - pts).norm(dim=2)
        hit += ((d <= pck_px) & det & valid).sum().item()
        total += valid.sum().item()
        fp += (det & ~valid).sum().item()
        n_neg += (~valid).sum().item()
        h_ok += ((det & valid & (d <= pck_px)).sum(dim=1) >= 5).sum().item()
        n_img += B
    return {"loss": loss_sum / max(n_img, 1), "pck": hit / max(total, 1),
            "fp_rate": fp / max(n_neg, 1), "h_ok": h_ok / max(n_img, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="convert_soccernet.py の出力フォルダ（index_train.json など）")
    ap.add_argument("--extra", nargs="*", default=[], help="追加の学習データ（自分でアノテーションした index の json）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default="mnv3_large", choices=["mnv3_large", "mnv3_small"])
    ap.add_argument("--width", type=int, default=64, help="デコーダのチャンネル数")
    ap.add_argument("--no-ca", action="store_true", help="Coordinate Attention を使わない（比較実験用）")
    ap.add_argument("--in-w", type=int, default=640)
    ap.add_argument("--in-h", type=int, default=352)
    ap.add_argument("--sigma", type=float, default=1.5, help="ヒートマップのガウスの幅（出力解像度の px）")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", default=None, help="追加学習の元にする重み（SoccerNet で学習した best.pt など）")
    ap.add_argument("--extra-repeat", type=int, default=20,
                    help="--extra のデータを何回繰り返して混ぜるか（枚数が少ないので重みを上げる）")
    ap.add_argument("--max-train", type=int, default=0, help="動作確認用に学習データを減らす（0で全部）")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stride = 4
    train_files = [os.path.join(args.data, "index_train.json")] + args.extra
    val_file = os.path.join(args.data, "index_valid.json")
    ds_tr = PitchKeypointDataset(train_files, args.in_w, args.in_h, stride, args.sigma, augment=True)
    ds_va = PitchKeypointDataset([val_file] if os.path.exists(val_file) else train_files,
                                 args.in_w, args.in_h, stride, args.sigma, augment=False)
    if args.extra:
        n_extra = 0
        for f in args.extra:
            with open(f) as fh:
                extra_items = json.load(fh)
            ds_tr.items += extra_items * (args.extra_repeat - 1)
            n_extra += len(extra_items)
        print(f"追加データ {n_extra} 枚 × {args.extra_repeat} 回")
    if args.max_train:
        ds_tr.items = ds_tr.items[: args.max_train]
        ds_va.items = ds_va.items[: max(1, args.max_train // 4)]
    print(f"学習 {len(ds_tr)} 枚 / 検証 {len(ds_va)} 枚 / デバイス {device}")
    dl_tr = DataLoader(ds_tr, args.batch, shuffle=True, num_workers=args.workers, drop_last=len(ds_tr) > args.batch,
                       pin_memory=device == "cuda", persistent_workers=args.workers > 0)
    dl_va = DataLoader(ds_va, args.batch, shuffle=False, num_workers=args.workers)

    config = {"keys": KEYS, "encoder": args.encoder, "width": args.width, "use_ca": not args.no_ca,
              "in_w": args.in_w, "in_h": args.in_h, "stride": stride, "sigma": args.sigma}
    model = PitchKeypointNet(len(KEYS), args.encoder, args.width, not args.no_ca).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"パラメータ数: {n_params / 1e6:.2f} M")

    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"{args.init} から追加学習します")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * max(len(dl_tr), 1)
    warm = min(1000, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1))))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    start, best = 0, -1.0
    last_path, best_path = os.path.join(args.out, "last.pt"), os.path.join(args.out, "best.pt")
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, best = ck["epoch"] + 1, ck["best"]
        print(f"エポック {start} から再開（これまでの最高 PCK {best:.3f}）")

    log = open(os.path.join(args.out, "log.jsonl"), "a")
    for epoch in range(start, args.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for x, hm, _, _ in dl_tr:
            x, hm = x.to(device, non_blocking=True), hm.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(x)
            loss = focal_loss(logits, hm)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        m = evaluate(model, dl_va, device, stride)
        m.update(epoch=epoch, train_loss=tot / max(len(dl_tr), 1), sec=time.time() - t0, lr=opt.param_groups[0]["lr"])
        print(f"[{epoch + 1:3d}/{args.epochs}] train {m['train_loss']:.4f} / val {m['loss']:.4f}  "
              f"PCK@5px {m['pck']:.3f}  誤検出率 {m['fp_rate']:.3f}  H可能率 {m['h_ok']:.3f}  ({m['sec']:.0f}s)")
        log.write(json.dumps(m) + "\n")
        log.flush()
        state = {"model": model.state_dict(), "config": config, "metrics": m}
        if m["pck"] > best:
            best = m["pck"]
            torch.save(state, best_path)
        torch.save({**state, "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch, "best": best}, last_path)
    print(f"完了。最高 PCK@5px {best:.3f} → {best_path}")


if __name__ == "__main__":
    main()
