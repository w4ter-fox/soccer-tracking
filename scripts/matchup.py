"""トラッキング結果から、相手選手との近接（マッチアップ候補）を集計する。

使い方:
    uv run python scripts/matchup.py --prefix data/video3 --radius 3.0

入力:  <prefix>_positions.csv, <prefix>_tracks.csv
       <prefix>_id_map.csv（あれば。team と jersey_number で選手単位に集計する）
出力:  <prefix>_nearest.csv   各フレーム・各選手の最近傍の相手と距離
       <prefix>_matchups.csv  半径以内だったペアごとの合計時間
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EXCLUDE_TEAMS = ("referee",)


def load_id_map(prefix: Path) -> pd.DataFrame:
    path = Path(f"{prefix}_id_map.csv")
    if path.exists():
        m = pd.read_csv(path, dtype={"jersey_number": "string"})
    else:
        m = pd.read_csv(f"{prefix}_tracks.csv")[["tracker_id", "team"]]
        m["jersey_number"] = pd.NA
    fallback = "t" + m["tracker_id"].astype(str)
    m["player"] = m["team"] + "_" + m["jersey_number"].fillna(fallback)
    return m[["tracker_id", "team", "player"]]


def load_players(prefix: Path) -> pd.DataFrame:
    pos = pd.read_csv(f"{prefix}_positions.csv")
    p = pos[pos["tracker_id"] >= 0].merge(load_id_map(prefix), on="tracker_id")
    p = p[~p["team"].isin(EXCLUDE_TEAMS)].dropna(subset=["pitch_x", "pitch_y"])
    # 同じ選手に複数の ID を割り当てて同じフレームに重なった場合は平均を取る
    return p.groupby(["frame", "player", "team"], as_index=False).agg(
        time_s=("time_s", "first"), pitch_x=("pitch_x", "mean"), pitch_y=("pitch_y", "mean")
    )


def nearest_opponents(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for frame, g in p.groupby("frame"):
        if g["team"].nunique() < 2:
            continue
        xy = g[["pitch_x", "pitch_y"]].to_numpy()
        teams = g["team"].to_numpy()
        names = g["player"].to_numpy()
        t = g["time_s"].iloc[0]

        d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
        d[teams[:, None] == teams[None, :]] = np.inf  # 同じチーム（自分を含む）は除外
        j = d.argmin(axis=1)
        for i in range(len(g)):
            if np.isfinite(d[i, j[i]]):
                rows.append((frame, t, names[i], teams[i], names[j[i]], d[i, j[i]]))
    return pd.DataFrame(rows, columns=["frame", "time_s", "player", "team", "opponent", "dist_m"])


def summarize(nn: pd.DataFrame, radius: float) -> pd.DataFrame:
    dt = nn.drop_duplicates("frame")["time_s"].diff().median()
    close = nn[nn["dist_m"] <= radius]
    s = (
        close.groupby(["player", "team", "opponent"])
        .agg(
            frames=("frame", "size"),
            min_dist_m=("dist_m", "min"),
            mean_dist_m=("dist_m", "mean"),
        )
        .reset_index()
    )
    s["seconds"] = s["frames"] * dt
    return s.sort_values("seconds", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", type=Path, required=True, help="例: data/video3")
    ap.add_argument("--radius", type=float, default=3.0, help="近接とみなす距離 [m]")
    args = ap.parse_args()

    p = load_players(args.prefix)
    print("チームごとの検出数:")
    print(p["team"].value_counts(), "\n")

    nn = nearest_opponents(p)
    if nn.empty:
        print("2チームが同時に映っているフレームがありません。")
        return

    s = summarize(nn, args.radius)
    nn.to_csv(f"{args.prefix}_nearest.csv", index=False)
    s.to_csv(f"{args.prefix}_matchups.csv", index=False)
    print(f"半径 {args.radius} m 以内の近接ペア（上位15件）:")
    print(s.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
