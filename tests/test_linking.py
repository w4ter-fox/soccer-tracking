from collections import Counter, defaultdict

from soccer_tracking.linking import link_tracklets


def runners(rng, n_players=8, n_frames=600, fps=60, p_lose=0.004):
    """直線で走る選手。ときどき見失い、0.3〜2秒後に別のIDで再検出される"""
    track_pts, votes, owner = defaultdict(list), defaultdict(Counter), {}
    next_id = 0
    for p in range(n_players):
        x0, y0, v, team = rng.uniform(20, 80), rng.uniform(10, 60), rng.normal(0, 4, 2), p % 2
        f, cur = 0, next_id
        owner[cur], next_id = p, next_id + 1
        while f < n_frames:
            if rng.random() < p_lose:
                f += int(rng.uniform(0.3, 2.0) * fps)
                cur = next_id
                owner[cur], next_id = p, next_id + 1
                continue
            t = f / fps
            x, y = x0 + v[0] * t + rng.normal(0, 0.15), y0 + v[1] * t + rng.normal(0, 0.15)
            track_pts[cur].append((f, 0.0, 0.0, x, y, 60.0))
            votes[cur][team] += 1
            f += 1
    return {k: v for k, v in track_pts.items() if v}, votes, owner


def test_links_broken_tracks_correctly(cfg, rng):
    cfg.slowdown = 1
    track_pts, votes, owner = runners(rng)
    link = link_tracklets(track_pts, votes, real_fps=60, cfg=cfg)
    n_breaks = len(track_pts) - len({owner[t] for t in track_pts})
    wrong = [a for a, b in link.items() if owner[a] != owner[b]]
    assert not wrong
    assert len(link) >= 0.8 * n_breaks


def test_does_not_link_different_teams(cfg):
    cfg.slowdown = 1
    # 同じ場所・同じ速さでも、チームが違えばつながない
    a = [(f, 0, 0, 10 + 0.05 * f, 30.0, 60.0) for f in range(0, 100)]
    b = [(f, 0, 0, 10 + 0.05 * f, 30.0, 60.0) for f in range(130, 230)]
    votes = {1: Counter({0: 100}), 2: Counter({1: 100})}
    assert link_tracklets({1: a, 2: b}, votes, real_fps=60, cfg=cfg) == {}
    votes = {1: Counter({0: 100}), 2: Counter({0: 100})}
    assert link_tracklets({1: a, 2: b}, votes, real_fps=60, cfg=cfg) == {2: 1}
