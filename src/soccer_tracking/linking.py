"""途切れたトラックのつなぎ直し

見失った選手が再検出されると別のIDになるので、途切れたトラックの最後の位置と速度から
「そのまま進んでいたらどこにいるか」を予測し、その近くで新しく始まったトラックを同じ選手としてつなぐ。
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def link_tracklets(track_pts, team_votes, real_fps, cfg):
    """途切れたトラックと、その後に始まったトラックを同じ選手としてつなぐ。
    track_pts: {tid: [(frame, img_x, img_y, pitch_x, pitch_y, box_h), ...]}（pitch は無ければ None）
    戻り値: {古いID: つないだ先のID（チェーンの最初のID）}"""
    max_gap = cfg.link_max_gap_s * real_fps
    vel_n = max(2, int(cfg.link_vel_window_s * real_fps))

    def team_of(tid):
        c = team_votes.get(tid)
        if not c or sum(c.values()) < cfg.team_min_votes * cfg.slowdown:
            return None
        best, cnt = c.most_common(1)[0]
        return best if cnt / sum(c.values()) >= cfg.team_min_share else None

    def endpoint(pts, last):
        """トラックの端の位置と速度（1フレームあたり）。2D座標が揃っていればピッチ座標、なければ画像座標"""
        seg = pts[-vel_n:] if last else pts[:vel_n]
        use_pitch = all(p[3] is not None for p in seg)
        a = np.array([[p[0], p[3], p[4]] if use_pitch else [p[0], p[1], p[2]] for p in seg], dtype=np.float64)
        pos = a[-1, 1:] if last else a[0, 1:]
        vel = np.zeros(2)
        if len(a) >= 3 and np.ptp(a[:, 0]) > 0:
            vel = np.array([np.polyfit(a[:, 0], a[:, 1], 1)[0], np.polyfit(a[:, 0], a[:, 2], 1)[0]])
        return pos, vel, use_pitch, float(np.median([p[5] for p in seg]))

    tids = sorted(track_pts, key=lambda t: track_pts[t][0][0])
    ends = {t: endpoint(track_pts[t], True) for t in tids}
    starts = {t: endpoint(track_pts[t], False) for t in tids}
    INF = 1e9
    cost = np.full((len(tids), len(tids)), INF)
    for i, a in enumerate(tids):
        fa = track_pts[a][-1][0]
        pa, va, ua, ha = ends[a]
        for j, b in enumerate(tids):
            fb = track_pts[b][0][0]
            gap = fb - fa
            if a == b or gap <= 0 or gap > max_gap:
                continue
            pb, vb, ub, hb = starts[b]
            if ua != ub:  # 片方だけ2D座標がある組み合わせは比べられない
                continue
            ta, tb = team_of(a), team_of(b)
            if ta is not None and tb is not None and ta != tb:
                continue
            gap_s = gap / real_fps
            # 途切れた方は前へ、始まった方は後ろへ、それぞれの速度で外挿して、真ん中の時刻で比べる
            mid = gap / 2
            d = np.linalg.norm((pa + va * mid) - (pb - vb * mid))
            if ua:
                r = cfg.link_r0_m + cfg.link_r_per_s_m * gap_s
            else:
                r = (cfg.link_r0_box + cfg.link_r_per_s_box * gap_s) * max(ha, hb)
            if d <= r:
                cost[i, j] = d / r + 0.1 * gap_s  # 近いほど・途切れが短いほど優先
    rows, cols = linear_sum_assignment(np.where(cost >= INF, 1e6, cost))
    nxt = {tids[i]: tids[j] for i, j in zip(rows, cols) if cost[i, j] < INF}

    # チェーンをたどって、つながった全トラックを最初のIDにそろえる
    prev = {b: a for a, b in nxt.items()}
    mapping = {}
    for t in tids:
        root = t
        while root in prev:
            root = prev[root]
        if root != t:
            mapping[t] = root
    return mapping
