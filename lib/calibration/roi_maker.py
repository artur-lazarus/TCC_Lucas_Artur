import os
import sys
from typing import List, Tuple
import json

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

# Ensure the detection package modules are importable when running this file directly
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTION_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "detection"))
if DETECTION_DIR not in sys.path:
    sys.path.insert(0, DETECTION_DIR)

from dubska import detect_vanishing_points_debug
from detection_api import Detection
from typing import Optional


def _poly_mask(shape, poly_pts):
    """Return a binary mask (uint8) with polygon filled.
    poly_pts: Nx2 array or list of (x,y) in integer pixel coords
    """
    mask = np.zeros(shape[:2], dtype=np.uint8)
    pts = np.array([poly_pts], dtype=np.int32)
    cv2.fillPoly(mask, pts, 255)
    return mask


def _coverage_and_disabled(mask_bg: np.ndarray, poly_pts) -> Tuple[float, float, int, int]:
    """Compute coverage and disabled area metrics for polygon.

    Returns: (coverage_fraction_of_enabled, disabled_fraction_of_poly_area, enabled_inside, poly_area)
    """
    poly_mask = _poly_mask(mask_bg.shape, poly_pts) > 0
    enabled = mask_bg > 0
    enabled_total = int(enabled.sum())
    enabled_inside = int((enabled & poly_mask).sum())
    poly_area = int(poly_mask.sum())
    coverage = 0.0 if enabled_total == 0 else enabled_inside / enabled_total
    disabled_inside = int((poly_mask & (~enabled)).sum())
    disabled_frac = 0.0 if poly_area == 0 else disabled_inside / poly_area
    return coverage, disabled_frac, enabled_inside, poly_area


def _order_polygon_points(points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Order potentially unordered polygon vertices around their centroid (CCW).
    Keeps the same vertices but sorts by polar angle so downstream ops (neighbors, fillPoly)
    behave consistently even if input is scrambled.
    """
    if len(points) <= 2:
        return points
    arr = np.array(points, dtype=float)
    cx, cy = float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1]))
    angles = np.arctan2(arr[:, 1] - cy, arr[:, 0] - cx)
    order = np.argsort(angles)
    ordered = [(int(arr[i, 0]), int(arr[i, 1])) for i in order]
    return ordered


def _sample_uniform_valid_angle(x: int, y: int, r: float, w: int, h: int, rng: np.random.Generator,
                                angle_samples: int = 360, min_r: float = 1.0) -> Tuple[float, float]:
    """Sample an angle uniformly among directions that keep (x + r cos a, y + r sin a) inside the image.
    If no angle is valid for current r, progressively reduce r until at least one is valid or r < min_r.
    Returns (angle_in_radians, used_radius). If no valid angle found, returns (None, r).
    """
    two_pi = 2 * np.pi
    cur_r = float(r)
    for _ in range(20):  # reduce radius up to 20 times
        angles = np.linspace(0.0, two_pi, angle_samples, endpoint=False)
        xs = x + cur_r * np.cos(angles)
        ys = y + cur_r * np.sin(angles)
        valid = (xs >= 0) & (xs <= (w - 1)) & (ys >= 0) & (ys <= (h - 1))
        valid_idx = np.nonzero(valid)[0]
        if valid_idx.size > 0:
            idx = int(rng.integers(0, valid_idx.size))
            return float(angles[valid_idx[idx]]), cur_r
        cur_r *= 0.8
        if cur_r < min_r:
            break
    return None, r


def _sample_uniform_valid_nd_direction(vec: np.ndarray, mag: float, w: int, h: int, rng: np.random.Generator,
                                       max_tries: int = 400, shrink: float = 0.8, min_mag: float = 1.0) -> Tuple[np.ndarray, float]:
    """Sample a random unit direction d in R^(2N) uniformly and accept only if vec + mag*d stays inside bounds.
    Try up to max_tries; if none valid, shrink magnitude and retry. Returns (dir_unit, used_mag).
    If no valid direction found down to min_mag, returns (None, mag).
    Bounds alternate per component: even idx in [0,w-1], odd idx in [0,h-1].
    """
    used_mag = float(mag)
    nvars = len(vec)
    for _ in range(20):  # shrink cycles
        for _ in range(max_tries):
            d = rng.normal(size=nvars)
            nrm = float(np.linalg.norm(d))
            if nrm == 0:
                continue
            d /= nrm
            v_new = vec + used_mag * d
            ok = True
            for i in range(nvars):
                if (i % 2) == 0:  # x
                    if not (0 <= v_new[i] <= (w - 1)):
                        ok = False; break
                else:  # y
                    if not (0 <= v_new[i] <= (h - 1)):
                        ok = False; break
            if ok:
                return d, used_mag
        used_mag *= shrink
        if used_mag < min_mag:
            break
    return None, mag


def fit_polygon_to_mask(mask_bg: np.ndarray, sides: int, target_coverage: float = 0.95,
                        max_iters: int = 5000, step_frac: float = 0.02,
                        min_step_frac: float = 0.005, debug: bool = False) -> Tuple[List[Tuple[int,int]], dict, List[List[Tuple[int,int]]], List[dict]]:
    """Fit an N-sided polygon to include at least target_coverage of enabled mask while
    minimizing disabled area inside polygon.

    Strategy (greedy gradient-like): start with a polygon covering whole image. Iteratively
    move a single vertex slightly towards polygon centroid and accept moves that reduce
    disabled fraction while keeping coverage >= target_coverage. Choose moves that give
    largest disabled reduction per unit of coverage loss (prefer large FP decrease with small coverage loss).

    Returns polygon points (list of (x,y)) and dict with stats.
    """
    h, w = mask_bg.shape[:2]

    def _optimize_from(pts_start: List[Tuple[int, int]]):
        # Ensure ordering
        pts_loc = _order_polygon_points(pts_start.copy())
        coverage_loc, disabled_loc, enabled_loc, area_loc = _coverage_and_disabled(mask_bg, pts_loc)
        timeline: List[List[Tuple[int,int]]] = [pts_loc.copy()]
        step = max(w, h) * step_frac
        min_step = max(w, h) * min_step_frac
        iters = 0
        no_change_streak = 0
        eps = 1e-9
        # Phase 0: if below coverage threshold, first grow coverage with minimal disabled increase
        if coverage_loc < target_coverage:
            while iters < max_iters and coverage_loc < target_coverage and step >= min_step:
                iters += 1
                corner_best_expected = -1.0
                corner_best = None
                for vi, (vx, vy) in enumerate(pts_loc):
                    v = np.array([vx, vy], dtype=float)

                    def try_dir_growth(dx, dy):
                        cand = pts_loc.copy()
                        npnt = (int(round(v[0] + dx)), int(round(v[1] + dy)))
                        npnt = (max(0, min(w-1, npnt[0])), max(0, min(h-1, npnt[1])))
                        cand[vi] = npnt
                        cov2, dis2, en2, area2 = _coverage_and_disabled(mask_bg, cand)
                        d_cov = cov2 - coverage_loc
                        if d_cov <= 0:
                            return -np.inf, None, None
                        d_dis_inc = max(0.0, dis2 - disabled_loc)
                        score = d_cov / (d_dis_inc + eps)
                        return score, cand, (cov2, dis2, en2, area2)

                    s_up, cand_up, st_up = try_dir_growth(0, -step)
                    s_down, cand_down, st_down = try_dir_growth(0, step)
                    s_left, cand_left, st_left = try_dir_growth(-step, 0)
                    s_right, cand_right, st_right = try_dir_growth(step, 0)

                    if s_down >= s_up:
                        vy_comp = s_down; v_sign = 1.0
                    else:
                        vy_comp = s_up; v_sign = -1.0
                    if s_right >= s_left:
                        vx_comp = s_right; h_sign = 1.0
                    else:
                        vx_comp = s_left; h_sign = -1.0

                    if np.isfinite(vx_comp) or np.isfinite(vy_comp):
                        comp_x = 0.0 if (not np.isfinite(vx_comp) or vx_comp <= 0) else vx_comp * h_sign
                        comp_y = 0.0 if (not np.isfinite(vy_comp) or vy_comp <= 0) else vy_comp * v_sign
                        dir_vec = np.array([comp_x, comp_y], dtype=float)
                        norm = np.linalg.norm(dir_vec)
                    else:
                        norm = 0.0

                    if norm > 0:
                        dir_unit = dir_vec / norm
                        chosen = None
                        for factor in (1.0, 0.5, 0.25):
                            dx, dy = dir_unit * (step * factor)
                            sc, cand, st = try_dir_growth(dx, dy)
                            if np.isfinite(sc) and sc > 0:
                                chosen = (sc, cand, st, (dx, dy))
                                break
                        if chosen is not None:
                            sc, cand, st, move_vec = chosen
                            if sc > corner_best_expected:
                                corner_best_expected = sc
                                corner_best = (vi, move_vec, cand, st)

                if corner_best is not None:
                    vi, move_vec, new_pts, new_stats = corner_best
                    pts_loc = new_pts
                    coverage_loc, disabled_loc, enabled_loc, area_loc = new_stats
                    no_change_streak = 0
                    step = min(step * 1.05, max(w, h) * 0.5)
                    timeline.append(pts_loc.copy())
                else:
                    no_change_streak += 1
                    step *= 0.5
                    if no_change_streak > 6:
                        break
        while iters < max_iters and coverage_loc >= target_coverage and step >= min_step:
            iters += 1
            corner_best_expected = -1.0
            corner_best = None
            for vi, (vx, vy) in enumerate(pts_loc):
                v = np.array([vx, vy], dtype=float)

                def try_dir(dx, dy):
                    cand = pts_loc.copy()
                    npnt = (int(round(v[0] + dx)), int(round(v[1] + dy)))
                    npnt = (max(0, min(w-1, npnt[0])), max(0, min(h-1, npnt[1])))
                    cand[vi] = npnt
                    cov2, dis2, en_in2, poly_a2 = _coverage_and_disabled(mask_bg, cand)
                    if cov2 < target_coverage:
                        return -np.inf, None, None
                    d_disabled = disabled_loc - dis2
                    d_cov = coverage_loc - cov2
                    if d_disabled <= 0:
                        return -np.inf, None, None
                    score = d_disabled / (d_cov + eps)
                    return score, cand, (cov2, dis2, en_in2, poly_a2)

                s_up, cand_up, stats_up = try_dir(0, -step)
                s_down, cand_down, stats_down = try_dir(0, step)
                s_left, cand_left, stats_left = try_dir(-step, 0)
                s_right, cand_right, stats_right = try_dir(step, 0)

                if s_down >= s_up:
                    vy_comp = s_down; v_sign = 1.0
                else:
                    vy_comp = s_up; v_sign = -1.0
                if s_right >= s_left:
                    vx_comp = s_right; h_sign = 1.0
                else:
                    vx_comp = s_left; h_sign = -1.0

                if np.isfinite(vx_comp) or np.isfinite(vy_comp):
                    comp_x = 0.0 if not np.isfinite(vx_comp) or vx_comp < 0 else vx_comp * h_sign
                    comp_y = 0.0 if not np.isfinite(vy_comp) or vy_comp < 0 else vy_comp * v_sign
                    dir_vec = np.array([comp_x, comp_y], dtype=float)
                    norm = np.linalg.norm(dir_vec)
                else:
                    norm = 0.0

                if norm > 0:
                    dir_unit = dir_vec / norm
                    chosen = None
                    for factor in (1.0, 0.5, 0.25):
                        dx, dy = dir_unit * (step * factor)
                        sc, cand, st = try_dir(dx, dy)
                        if np.isfinite(sc) and sc > 0:
                            chosen = (sc, cand, st, (dx, dy))
                            break
                    if chosen is not None:
                        sc, cand, st, move_vec = chosen
                        if sc > corner_best_expected:
                            corner_best_expected = sc
                            corner_best = (vi, move_vec, cand, st)

            if corner_best is not None:
                vi, move_vec, new_pts, new_stats = corner_best
                pts_loc = new_pts
                coverage_loc, disabled_loc, enabled_loc, area_loc = new_stats
                no_change_streak = 0
                step = min(step * 1.05, max(w, h) * 0.5)
                timeline.append(pts_loc.copy())
            else:
                no_change_streak += 1
                step *= 0.5
                if no_change_streak > 6:
                    break
        stats_loc = dict(iters=iters, coverage=coverage_loc, disabled_frac=disabled_loc,
                         enabled_inside=enabled_loc, poly_area=area_loc)
        return pts_loc, stats_loc, timeline

    # Build initial polygon (user may reorder externally; ensure order here)
    starting_points = [(0,0),(0,h-1),(w-1,h-1),(w-1,0),(w//2,0),(0,h//2),(w-1,h//2),(w//2,h-1)]
    pts0 = _order_polygon_points(starting_points[:sides])
    print("Initial polygon points:", pts0)
    pts, stats, timeline = _optimize_from(pts0)
    print("Post-initial fit stats:", stats)

    # Track iteration counters: sum across initial + accepted kicks; also track all runs
    iters_opt_init = int(stats.get('iters', 0))
    iters_sum_accepted = iters_opt_init
    iters_sum_all_runs = iters_opt_init
    kicks_accepted = 0
    kicks_total = 0

    # Random kick scheme to escape local minima
    rng = np.random.default_rng()
    M = float(max(w, h))
    throw_frac = 0.25  # start with a big throw (25% of max dimension)
    min_throw_frac = 0.02
    consecutive_rejects = 0
    max_rejects = 20
    best_pts, best_stats = pts, stats
    best_timeline = timeline
    kicks: List[dict] = []

    while throw_frac >= min_throw_frac and consecutive_rejects < max_rejects:
        vi = int(rng.integers(0, len(best_pts)))
        mag = float(throw_frac * M)
        cand_pts = best_pts.copy()
        x, y = cand_pts[vi]
        angle, used_mag = _sample_uniform_valid_angle(x, y, mag, w, h, rng)
        if angle is None:
            # no valid direction for this magnitude: reduce throw and try again
            throw_frac *= 0.8
            continue
        dx = int(round(used_mag * np.cos(angle)))
        dy = int(round(used_mag * np.sin(angle)))
        newx = int(round(x + dx))
        newy = int(round(y + dy))
        newx = max(0, min(w-1, newx))
        newy = max(0, min(h-1, newy))
        cand_pts[vi] = (newx, newy)
        cand_pts = _order_polygon_points(cand_pts)
        # record initial kicked polygon (before fitting) with the timeline index at which it occurred
        kick_entry = {'poly': cand_pts.copy(), 'accepted': False, 'idx': int(max(0, len(best_timeline) - 1))}

        cand_fit_pts, cand_stats, cand_timeline = _optimize_from(cand_pts)
        kicks_total += 1
        iters_sum_all_runs += int(cand_stats.get('iters', 0))
        # Accept if coverage constraint met and disabled_frac improves
        if cand_stats['coverage'] >= target_coverage and cand_stats['disabled_frac'] + 1e-9 < best_stats['disabled_frac']:
            best_pts, best_stats = cand_fit_pts, cand_stats
            # concatenate timelines (do not separate by kick)
            best_timeline.extend(cand_timeline)
            kick_entry['accepted'] = True
            kicks_accepted += 1
            iters_sum_accepted += int(cand_stats.get('iters', 0))
            consecutive_rejects = 0
            print(f"Accepted random kick: throw_frac={throw_frac:.3f}, new disabled={best_stats['disabled_frac']:.6f}, coverage={best_stats['coverage']:.4f}")
        else:
            consecutive_rejects += 1
            print(f"Rejected random kick: throw_frac={throw_frac:.3f}")
        throw_frac *= 0.95  # progressively smaller throws
        kicks.append(kick_entry)

    # Report total iterations as sum across initial optimization + accepted kicks
    # Keep backwards compatibility by setting 'iters' to the total; also expose detailed counters
    best_stats = best_stats.copy()
    best_stats['iters'] = int(iters_sum_accepted)
    best_stats['iters_opt_init'] = int(iters_opt_init)
    best_stats['iters_sum_accepted_kicks'] = int(iters_sum_accepted - iters_opt_init)
    best_stats['iters_sum_all_runs'] = int(iters_sum_all_runs)
    best_stats['kicks_total'] = int(kicks_total)
    best_stats['kicks_accepted'] = int(kicks_accepted)

    return best_pts, best_stats, best_timeline, kicks


def fit_polygon_to_mask_joint(mask_bg: np.ndarray, sides: int, target_coverage: float = 0.95,
                              max_iters: int = 5000, step_frac: float = 0.01,
                              min_step_frac: float = 0.005) -> Tuple[List[Tuple[int,int]], dict, List[List[Tuple[int,int]]], List[dict]]:
    """Joint optimization treating polygon as a 2N vector (x0,y0,...,xN-1,yN-1).
    At each iteration, estimate a gradient-like direction by probing +/− step on each component,
    then move along the combined direction while keeping coverage >= target and minimizing disabled area.
    Includes random kicks in the full 2N space to escape local minima.
    """
    h, w = mask_bg.shape[:2]

    def vec_from_pts(pts):
        v = []
        for x, y in pts:
            v.extend([float(x), float(y)])
        return np.array(v, dtype=float)

    def pts_from_vec(vec):
        pts = []
        for i in range(0, len(vec), 2):
            x = int(round(max(0, min(w-1, vec[i]))))
            y = int(round(max(0, min(h-1, vec[i+1]))))
            pts.append((x, y))
        return _order_polygon_points(pts)

    def _optimize_from_vec(vec_start: np.ndarray):
        pts_loc = pts_from_vec(vec_start)
        vec = vec_from_pts(pts_loc)
        coverage_loc, disabled_loc, enabled_loc, area_loc = _coverage_and_disabled(mask_bg, pts_loc)
        timeline: List[List[Tuple[int,int]]] = [pts_loc.copy()]

        step = max(w, h) * step_frac
        min_step = max(w, h) * min_step_frac
        iters = 0
        no_change_streak = 0
        eps = 1e-9
        nvars = len(vec)
        # Phase 0: coverage-increasing walk if below target
        if coverage_loc < target_coverage:
            while iters < max_iters and coverage_loc < target_coverage and step >= min_step:
                iters += 1
                comps = np.zeros(nvars, dtype=float)
                # Probe each variable independently for coverage gain
                for i in range(nvars):
                    v_plus = vec.copy(); v_plus[i] += step
                    if i % 2 == 0:
                        v_plus[i] = max(0, min(w-1, v_plus[i]))
                    else:
                        v_plus[i] = max(0, min(h-1, v_plus[i]))
                    pts_p = pts_from_vec(v_plus)
                    cov_p, dis_p, en_p, area_p = _coverage_and_disabled(mask_bg, pts_p)
                    d_cov_p = cov_p - coverage_loc
                    d_dis_inc_p = max(0.0, dis_p - disabled_loc)
                    score_p = d_cov_p / (d_dis_inc_p + eps) if d_cov_p > 0 else -np.inf

                    v_minus = vec.copy(); v_minus[i] -= step
                    if i % 2 == 0:
                        v_minus[i] = max(0, min(w-1, v_minus[i]))
                    else:
                        v_minus[i] = max(0, min(h-1, v_minus[i]))
                    pts_m = pts_from_vec(v_minus)
                    cov_m, dis_m, en_m, area_m = _coverage_and_disabled(mask_bg, pts_m)
                    d_cov_m = cov_m - coverage_loc
                    d_dis_inc_m = max(0.0, dis_m - disabled_loc)
                    score_m = d_cov_m / (d_dis_inc_m + eps) if d_cov_m > 0 else -np.inf

                    if score_p > score_m and np.isfinite(score_p) and score_p > 0:
                        comps[i] = score_p
                    elif score_m > score_p and np.isfinite(score_m) and score_m > 0:
                        comps[i] = -score_m
                    else:
                        comps[i] = 0.0

                norm = np.linalg.norm(comps)
                if norm <= 0:
                    no_change_streak += 1
                    step *= 0.5
                    if no_change_streak > 6:
                        break
                    continue

                dir_unit = comps / norm
                chosen = None
                for factor in (1.0, 0.5, 0.25):
                    v_new = vec + dir_unit * (step * factor)
                    for i in range(nvars):
                        if i % 2 == 0:
                            v_new[i] = max(0, min(w-1, v_new[i]))
                        else:
                            v_new[i] = max(0, min(h-1, v_new[i]))
                    pts_new = pts_from_vec(v_new)
                    cov2, dis2, en2, area2 = _coverage_and_disabled(mask_bg, pts_new)
                    d_cov = cov2 - coverage_loc
                    d_dis_inc = max(0.0, dis2 - disabled_loc)
                    if d_cov <= 0:
                        continue
                    score = d_cov / (d_dis_inc + eps)
                    if np.isfinite(score) and score > 0:
                        chosen = (score, v_new, pts_new, (cov2, dis2, en2, area2))
                        break

                if chosen is not None:
                    _, vec, pts_loc, stats2 = chosen
                    coverage_loc, disabled_loc, enabled_loc, area_loc = stats2
                    no_change_streak = 0
                    step = min(step * 1.05, max(w, h) * 0.5)
                    timeline.append(pts_loc.copy())
                else:
                    no_change_streak += 1
                    step *= 0.5
                    if no_change_streak > 6:
                        break

        while iters < max_iters and coverage_loc >= target_coverage and step >= min_step:
            iters += 1
            comps = np.zeros(nvars, dtype=float)

            # Probe each variable independently
            for i in range(nvars):
                # plus
                v_plus = vec.copy()
                v_plus[i] += step
                # clamp single component
                if i % 2 == 0:
                    v_plus[i] = max(0, min(w-1, v_plus[i]))
                else:
                    v_plus[i] = max(0, min(h-1, v_plus[i]))
                pts_p = pts_from_vec(v_plus)
                cov_p, dis_p, en_p, area_p = _coverage_and_disabled(mask_bg, pts_p)
                if cov_p >= target_coverage:
                    d_dis_p = disabled_loc - dis_p
                    d_cov_p = coverage_loc - cov_p
                    score_p = d_dis_p / (d_cov_p + eps) if d_dis_p > 0 else -np.inf
                else:
                    score_p = -np.inf

                # minus
                v_minus = vec.copy()
                v_minus[i] -= step
                if i % 2 == 0:
                    v_minus[i] = max(0, min(w-1, v_minus[i]))
                else:
                    v_minus[i] = max(0, min(h-1, v_minus[i]))
                pts_m = pts_from_vec(v_minus)
                cov_m, dis_m, en_m, area_m = _coverage_and_disabled(mask_bg, pts_m)
                if cov_m >= target_coverage:
                    d_dis_m = disabled_loc - dis_m
                    d_cov_m = coverage_loc - cov_m
                    score_m = d_dis_m / (d_cov_m + eps) if d_dis_m > 0 else -np.inf
                else:
                    score_m = -np.inf

                # choose best direction for this variable
                if score_p > score_m and np.isfinite(score_p) and score_p > 0:
                    comps[i] = score_p
                elif score_m > score_p and np.isfinite(score_m) and score_m > 0:
                    comps[i] = -score_m
                else:
                    comps[i] = 0.0

            norm = np.linalg.norm(comps)
            if norm <= 0:
                no_change_streak += 1
                step *= 0.5
                if no_change_streak > 6:
                    break
                continue

            dir_unit = comps / norm
            chosen = None
            for factor in (1.0, 0.5, 0.25):
                v_new = vec + dir_unit * (step * factor)
                # clamp all components
                for i in range(nvars):
                    if i % 2 == 0:
                        v_new[i] = max(0, min(w-1, v_new[i]))
                    else:
                        v_new[i] = max(0, min(h-1, v_new[i]))
                pts_new = pts_from_vec(v_new)
                cov2, dis2, en2, area2 = _coverage_and_disabled(mask_bg, pts_new)
                if cov2 < target_coverage:
                    continue
                d_dis = disabled_loc - dis2
                d_cov = coverage_loc - cov2
                if d_dis <= 0:
                    continue
                score = d_dis / (d_cov + eps)
                if np.isfinite(score) and score > 0:
                    chosen = (score, v_new, pts_new, (cov2, dis2, en2, area2))
                    break

            if chosen is not None:
                _, vec, pts_loc, stats2 = chosen
                coverage_loc, disabled_loc, enabled_loc, area_loc = stats2
                no_change_streak = 0
                step = min(step * 1.05, max(w, h) * 0.5)
                timeline.append(pts_loc.copy())
            else:
                no_change_streak += 1
                step *= 0.5
                if no_change_streak > 6:
                    break

        stats_loc = dict(iters=iters, coverage=coverage_loc, disabled_frac=disabled_loc,
                         enabled_inside=enabled_loc, poly_area=area_loc)
        return vec, pts_loc, stats_loc, timeline

    # Initial polygon and vector
    starting_points = [(0,0),(0,h-1),(w-1,h-1),(w-1,0),(w//2,0),(0,h//2),(w-1,h//2),(w//2,h-1)]
    pts0 = _order_polygon_points(starting_points[:sides])
    vec0 = vec_from_pts(pts0)

    vec, pts, stats, timeline = _optimize_from_vec(vec0)

    # Track iteration counters: sum across initial + accepted kicks; also track all runs
    iters_opt_init = int(stats.get('iters', 0))
    iters_sum_accepted = iters_opt_init
    iters_sum_all_runs = iters_opt_init
    kicks_accepted = 0
    kicks_total = 0

    # Random kicks in full 2N space
    rng = np.random.default_rng()
    M = float(max(w, h))
    throw_frac = 0.25
    min_throw_frac = 0.02
    consecutive_rejects = 0
    max_rejects = 20
    best_vec, best_pts, best_stats = vec, pts, stats
    best_timeline = timeline
    kicks: List[dict] = []

    while throw_frac >= min_throw_frac and consecutive_rejects < max_rejects:
        mag = float(throw_frac * M)
        dir_unit, used_mag = _sample_uniform_valid_nd_direction(best_vec, mag, w, h, rng)
        if dir_unit is None:
            throw_frac *= 0.8
            continue
        v_kick = best_vec + dir_unit * used_mag
        pts_kick = pts_from_vec(v_kick)
        # record initial kicked polygon prior to fitting with the timeline index at which it occurred
        kick_entry = {'poly': pts_kick.copy(), 'accepted': False, 'idx': int(max(0, len(best_timeline) - 1))}
        v_fit, pts_fit, stats_fit, timeline_fit = _optimize_from_vec(v_kick)
        kicks_total += 1
        iters_sum_all_runs += int(stats_fit.get('iters', 0))

        if stats_fit['coverage'] >= target_coverage and stats_fit['disabled_frac'] + 1e-9 < best_stats['disabled_frac']:
            best_vec, best_pts, best_stats = v_fit, pts_fit, stats_fit
            best_timeline.extend(timeline_fit)
            kick_entry['accepted'] = True
            kicks_accepted += 1
            iters_sum_accepted += int(stats_fit.get('iters', 0))
            consecutive_rejects = 0
            print(f"Accepted joint random kick: throw_frac={throw_frac:.3f}, disabled={best_stats['disabled_frac']:.6f}, coverage={best_stats['coverage']:.4f}")
        else:
            consecutive_rejects += 1
            print(f"Rejected joint random kick: throw_frac={throw_frac:.3f}")
        throw_frac *= 0.95

        kicks.append(kick_entry)

    # Report total iterations as sum across initial optimization + accepted kicks
    best_stats = best_stats.copy()
    best_stats['iters'] = int(iters_sum_accepted)
    best_stats['iters_opt_init'] = int(iters_opt_init)
    best_stats['iters_sum_accepted_kicks'] = int(iters_sum_accepted - iters_opt_init)
    best_stats['iters_sum_all_runs'] = int(iters_sum_all_runs)
    best_stats['kicks_total'] = int(kicks_total)
    best_stats['kicks_accepted'] = int(kicks_accepted)

    return best_pts, best_stats, best_timeline, kicks

def load_or_compute_background(image_path: Optional[str] = None) -> np.ndarray:
    """Load ROI background mask from image if provided/exists, else compute from video pipeline."""
    # If image_path is not given, try default saved path first
    default_path = "complete_test_ROI_background_percentile_95.jpg"
    path_to_use = image_path
    if path_to_use is None and os.path.exists(default_path):
        path_to_use = default_path

    if path_to_use is not None:
        img = cv2.imread(path_to_use, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
        else:
            print(f"Warning: failed to read background image at '{path_to_use}', falling back to computation.")

    # Fallback: compute using the original video pipeline
    video_path = "dataset/session0_left/video.avi"
    d = Detection(video_path, max_frames=500)
    d.init_flows(dis_preset="FAST")
    d.init_background(method='percentile', percentile=50)
    _ = d.flow_subtract(hue_range=(10,30), value_min=6).and_(
        d.median_subtract(threshold_value=14).morphology.fill_holes()
    ).save("test_general_detector.mp4")
    b = d.flow_subtract(hue_range=(10,30), value_min=6).and_(
        d.median_subtract(threshold_value=14).morphology.fill_holes()
    )

    roi_d = Detection(b.masks, max_frames=500)
    roi_d.init_background(method='percentile', percentile=95)
    bg = roi_d._background.copy()
    cv2.imwrite(default_path, bg)
    return bg


def main(image_path: Optional[str] = None):
    # Load ROI background from image if provided/available, else compute it
    bg = load_or_compute_background(image_path=image_path)
    if bg is None:
        raise RuntimeError("Could not obtain ROI background mask.")
    # Save (or overwrite) a copy for traceability
    cv2.imwrite("complete_test_ROI_background_percentile_95.jpg", bg)

    # Fit polygons for 4,5,6 sides with both methods and compare
    results = {}
    target = 0.99
    # helper to draw timeline
    def draw_timeline(mask_bg: np.ndarray, timeline: List[List[Tuple[int,int]]], out_path: str):
        base = cv2.cvtColor((mask_bg>0).astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)
        n = max(1, len(timeline))
        for i, poly in enumerate(timeline):
            t = i / max(1, n-1)
            # color gradient from blue -> green -> red
            color = (
                int(255*(1-t)),            # Blue channel decreases
                int(255*min(1.0, t*2)),     # Green peaks in middle
                int(255*max(0.0, (t-0.5)*2)) # Red increases later
            )
            cv2.polylines(base, [np.array(poly, dtype=np.int32)], True, color, 1)
        cv2.imwrite(out_path, base)

    timeline_store = {"corner": {}, "joint": {}}
    for sides in [6]:
        start=time.time()
        pts_corner, stats_corner, tl_corner, kicks_corner = fit_polygon_to_mask(bg, sides, target_coverage=target)
        corner_end=time.time()
        pts_joint, stats_joint, tl_joint, kicks_joint = fit_polygon_to_mask_joint(bg, sides, target_coverage=target)
        joint_end=time.time()
        print(f"Sides={sides} CORNER fit time: {corner_end - start:.2f} seconds")
        print(f"Sides={sides}  JOINT fit time: {joint_end - corner_end:.2f} seconds")
        timeline_store["corner"][sides] = {"timeline": tl_corner, "kicks": kicks_corner}
        timeline_store["joint"][sides] = {"timeline": tl_joint, "kicks": kicks_joint}

        # Save overlays for both
        overlay_base = cv2.cvtColor((bg>0).astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)
        ov_corner = overlay_base.copy()
        ov_joint = overlay_base.copy()
        cv2.polylines(ov_corner, [np.array(pts_corner, dtype=np.int32)], True, (0,0,255), 2)
        cv2.polylines(ov_joint, [np.array(pts_joint, dtype=np.int32)], True, (255,0,0), 2)
        cv2.imwrite(f"roi_poly_{sides}_sides_corner.jpg", ov_corner)
        cv2.imwrite(f"roi_poly_{sides}_sides_joint.jpg", ov_joint)
        # Timelines (two per side)
        draw_timeline(bg, tl_corner, f"timeline_corner_{sides}.jpg")
        draw_timeline(bg, tl_joint, f"timeline_joint_{sides}.jpg")

        print(f"Sides={sides} CORNER: iters={stats_corner['iters']}, cov={stats_corner['coverage']:.4f}, dis={stats_corner['disabled_frac']:.4f}")
        print(f"Sides={sides}  JOINT: iters={stats_joint['iters']}, cov={stats_joint['coverage']:.4f}, dis={stats_joint['disabled_frac']:.4f}")

        # Choose better per-sides (min disabled_frac under coverage constraint; tie -> fewer iters)
        chosen_pts, chosen_stats, chosen_method = (pts_corner, stats_corner, 'corner')
        if (stats_joint['coverage'] >= target and
            (stats_joint['disabled_frac'] + 1e-9 < stats_corner['disabled_frac'] or
             (abs(stats_joint['disabled_frac'] - stats_corner['disabled_frac']) < 1e-9 and stats_joint['iters'] < stats_corner['iters']))):
            chosen_pts, chosen_stats, chosen_method = (pts_joint, stats_joint, 'joint')

        results[sides] = (chosen_pts, chosen_stats, chosen_method)
        # also save a per-sides chosen overlay
        ov_chosen = overlay_base.copy()
        cv2.polylines(ov_chosen, [np.array(chosen_pts, dtype=np.int32)], True, (0,255,255), 2)
        cv2.imwrite(f"roi_poly_{sides}_sides_chosen.jpg", ov_chosen)

    # choose smallest sided polygon that doesn't have a big jump in enabled coverage
    #coverages = {s: results[s][1]['coverage'] for s in results}
    #jump_thr = 0.03  # 3 percentage points
    #chosen = None
    #for s in (4,5):
    #    nexts = s+1
    #    if abs(coverages[s] - coverages[nexts]) <= jump_thr:
    #        chosen = s
    #        break
    #if chosen is None:
    #    chosen = 6
#
    #chosen_pts, chosen_stats, chosen_method = results[chosen]
    #print(f"Chosen polygon sides={chosen} via {chosen_method}, coverage={chosen_stats['coverage']:.4f}, disabled_frac={chosen_stats['disabled_frac']:.4f}")
    ## save chosen overlay
    #overlay = cv2.cvtColor((bg>0).astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)
    #cv2.polylines(overlay, [np.array(chosen_pts, dtype=np.int32)], isClosed=True, color=(0,255,0), thickness=3)
    #cv2.imwrite("roi_chosen_polygon.jpg", overlay)

    # Build an interactive HTML slider viewer for timelines
    def write_timeline_html(bg_path: str, width: int, height: int, timelines: dict, out_path: str):
        # Convert tuples to simple lists for JSON
        def to_jsonable_timeline(seq):
            return [[[int(x), int(y)] for (x,y) in poly] for poly in seq]
        def to_jsonable_kicks(seq):
            out = []
            for item in seq:
                poly = [[int(x), int(y)] for (x,y) in item.get('poly', [])]
                out.append({
                    "poly": poly,
                    "accepted": bool(item.get('accepted', False)),
                    "idx": int(item.get('idx', 0)),
                })
            return out
        json_data = {
            'corner': {str(k): {"timeline": to_jsonable_timeline(v.get('timeline', [])),
                                "kicks": to_jsonable_kicks(v.get('kicks', []))} for k, v in timelines['corner'].items()},
            'joint': {str(k): {"timeline": to_jsonable_timeline(v.get('timeline', [])),
                               "kicks": to_jsonable_kicks(v.get('kicks', []))} for k, v in timelines['joint'].items()},
        }
        html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <title>ROI Polygon Timelines</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 16px; }}
        .controls {{ margin-bottom: 12px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
        #stage {{ position: relative; width: {width}px; height: {height}px; }}
        canvas {{ position: absolute; left: 0; top: 0; }}
        #bg {{ position: absolute; left: 0; top: 0; width: {width}px; height: {height}px; }}
        .legend span {{ display: inline-block; width: 12px; height: 12px; margin-right: 6px; vertical-align: middle; }}
    </style>
</head>
<body>
    <h2>ROI Polygon Timelines</h2>
    <div class=\"controls\">
        <label>Fit type:
            <select id=\"fitType\">
                <option value=\"corner\">corner</option>
                <option value=\"joint\">joint</option>
            </select>
        </label>
        <label>Sides:
            <select id=\"sides\">
                <option value=\"4\">4</option>
                <option value=\"5\">5</option>
                <option value=\"6\">6</option>
            </select>
        </label>
        <label style=\"flex:1; min-width: 220px;\">Step:
            <input id=\"step\" type=\"range\" min=\"0\" max=\"0\" value=\"0\" style=\"width:100%;\" />
            <span id=\"stepLabel\">0</span>
        </label>
    </div>
    <div class=\"legend\">
        <div><span style=\"background:#00F\"></span> early</div>
        <div><span style=\"background:#0F0\"></span> middle</div>
        <div><span style=\"background:#F00\"></span> late</div>
    </div>
    <div id=\"stage\">
        <img id=\"bg\" src=\"{bg_path}\" alt=\"background\" />
        <canvas id=\"overlay\" width=\"{width}\" height=\"{height}\"></canvas>
    </div>
    <script>
        const TIMELINES = {json.dumps(json_data)};
        const canvas = document.getElementById('overlay');
        const ctx = canvas.getContext('2d');
        const fitTypeSel = document.getElementById('fitType');
        const sidesSel = document.getElementById('sides');
        const stepInput = document.getElementById('step');
        const stepLabel = document.getElementById('stepLabel');

            function getData() {{
            const ft = fitTypeSel.value;
            const sd = sidesSel.value;
                const data = (TIMELINES[ft] && TIMELINES[ft][sd]) ? TIMELINES[ft][sd] : {{ timeline: [], kicks: [] }};
                return data;
        }}

        function setSliderMax() {{
                const data = getData();
                const max = Math.max(0, data.timeline.length - 1);
            stepInput.max = String(max);
            if (parseInt(stepInput.value) > max) stepInput.value = String(max);
            stepLabel.textContent = stepInput.value;
        }}

            function colorForIndex(i, n) {{
            const t = (n <= 1) ? 0 : i / (n - 1);
            const b = Math.floor(255 * (1 - t));
            const g = Math.floor(255 * Math.min(1.0, t * 2));
            const r = Math.floor(255 * Math.max(0.0, (t - 0.5) * 2));
                return `rgb(${{r}},${{g}},${{b}})`;
        }}

        function draw() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
                    const data = getData();
                    const tl = data.timeline;
                    const kicks = data.kicks || [];
                    const n = tl.length;
            const idx = parseInt(stepInput.value);
            stepLabel.textContent = idx;
            if (n === 0) return;
            // draw all previous in faint lines
            ctx.lineWidth = 1;
            for (let i = 0; i <= idx && i < n; i++) {{
                const poly = tl[i];
                ctx.strokeStyle = colorForIndex(i, n);
                ctx.beginPath();
                for (let j = 0; j < poly.length; j++) {{
                    const [x, y] = poly[j];
                    if (j === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                }}
                if (poly.length > 0) ctx.closePath();
                ctx.stroke();
            }}
                    // draw kick seeds only up to the current step index (accepted cyan, rejected red)
                    for (const k of kicks) {{
                        const when = typeof k.idx === 'number' ? k.idx : 0;
                        if (when > idx) continue;
                        const poly = k.poly || [];
                        ctx.strokeStyle = k.accepted ? 'rgba(0,255,255,0.9)' : 'rgba(255,0,0,0.9)';
                        ctx.beginPath();
                        for (let j = 0; j < poly.length; j++) {{
                            const [x, y] = poly[j];
                            if (j === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                        }}
                        if (poly.length > 0) ctx.closePath();
                        ctx.stroke();
                    }}
        }}

        fitTypeSel.addEventListener('change', () => {{ setSliderMax(); draw(); }});
        sidesSel.addEventListener('change', () => {{ setSliderMax(); draw(); }});
        stepInput.addEventListener('input', () => {{ draw(); }});

        // init
        setSliderMax();
        draw();
    </script>
</body>
 </html>
 """
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(html)

    # Write interactive viewer to repo root for easy file:// viewing
    h, w = bg.shape[:2]
    write_timeline_html(
        bg_path="complete_test_ROI_background_percentile_95.jpg",
        width=w,
        height=h,
        timelines=timeline_store,
        out_path="timeline_viewer.html",
    )
if __name__ == "__main__":
    # Optional CLI: --bg-image path
    import argparse
    parser = argparse.ArgumentParser(description="Fit ROI polygons to a background mask")
    parser.add_argument("--bg-image", dest="bg_image", type=str, default=None,
                        help="Path to precomputed ROI background image (grayscale). If omitted, will try default saved file, else compute from video.")
    args = parser.parse_args()
    main(image_path=args.bg_image)