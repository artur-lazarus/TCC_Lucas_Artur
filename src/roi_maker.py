import numpy as np
import cv2
from typing import List, Tuple


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


def fit_polygon_to_mask_optimized(
    mask_bg: np.ndarray,
    sides: int,
    target_coverage: float = 0.95,
    max_iters: int = 5000,
    step_frac: float = 0.02,
    min_step_frac: float = 0.005,
    debug: bool = False,
    pyramid_levels: int = 2,         # <-- (5) coarse->fine levels (>=1). 2 is a good default.
):
    """
    Fit an N-sided polygon to include at least target_coverage of enabled mask while
    minimizing disabled area inside polygon.

    Optimizations implemented:
      (1) Precompute enabled mask and totals; reuse a single full-size scratch mask.
      (2) ROI-limited evaluation of candidate moves (tiny bounding box per moved vertex).
      (3) Incremental delta stats using ROI old/new polygon XOR instead of full-image re-fills.
      (5) Coarse->fine pyramid: solve on downsampled mask, then upscale and refine.

    Returns
    -------
    pts : List[(x,y)]
    stats : dict
    timeline : List[List[(x,y)]]
    kicks : List[dict]
    """

    def _build_pyramid(binmask: np.ndarray, levels: int):
        """(5) Build binary mask pyramid with nearest-neighbor to preserve labels."""
        levels = max(1, int(levels))
        pyr = [(binmask, 1.0, 1.0)]
        for _ in range(1, levels):
            h, w = pyr[-1][0].shape[:2]
            if min(h, w) <= 160:  # stop getting too tiny
                break
            ds = cv2.resize(pyr[-1][0], (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
            sx = (w // 2) / w
            sy = (h // 2) / h
            pyr.append((ds, sx * pyr[-1][1], sy * pyr[-1][2]))
        return list(reversed(pyr))  # coarse -> fine

    def _coverage_from_poly_full(pts, enabled_mask, enabled_total, scratch_mask):
        """One-time full-image fill to compute baseline stats (reuses scratch buffer)."""
        scratch_mask.fill(0)
        poly = np.array([pts], dtype=np.int32)
        cv2.fillPoly(scratch_mask, poly, 255)
        poly_area = int((scratch_mask > 0).sum())
        enabled_inside = int(((scratch_mask > 0) & enabled_mask).sum())
        coverage = 0.0 if enabled_total == 0 else enabled_inside / enabled_total
        disabled_frac = 0.0 if poly_area == 0 else (poly_area - enabled_inside) / poly_area
        return coverage, disabled_frac, enabled_inside, poly_area

    def _eval_move_delta_roi(
        pts, vi, new_pt,
        enabled_mask, enabled_total,
        cur_enabled_inside, cur_poly_area
    ):
        """(2)+(3) Fast local update: only fill old/new polygons within a small ROI and XOR them."""
        h, w = enabled_mask.shape
        n = len(pts)
        prev_pt = pts[(vi - 1) % n]
        cur_pt = pts[vi]
        next_pt = pts[(vi + 1) % n]

        # Tight ROI around prev/cur/next/new (with small pad)
        xs = [prev_pt[0], cur_pt[0], next_pt[0], new_pt[0]]
        ys = [prev_pt[1], cur_pt[1], next_pt[1], new_pt[1]]
        pad = 2
        x0 = max(0, min(xs) - pad)
        x1 = min(w, max(xs) + pad + 1)
        y0 = max(0, min(ys) - pad)
        y1 = min(h, max(ys) + pad + 1)
        roi_w, roi_h = x1 - x0, y1 - y0
        if roi_w <= 0 or roi_h <= 0:
            # no ROI change; stats unchanged
            cov = 0.0 if enabled_total == 0 else cur_enabled_inside / enabled_total
            dis = 0.0 if cur_poly_area == 0 else (cur_poly_area - cur_enabled_inside) / cur_poly_area
            return cov, dis, cur_enabled_inside, cur_poly_area

        # Build old/new polygons in ROI coords
        old_pts_roi = np.array([[(x - x0, y - y0) for (x, y) in pts]], dtype=np.int32)
        new_pts_full = pts.copy()
        new_pts_full[vi] = (int(new_pt[0]), int(new_pt[1]))
        new_pts_roi = np.array([[(x - x0, y - y0) for (x, y) in new_pts_full]], dtype=np.int32)

        # Rasterize only inside ROI
        old_roi = np.zeros((roi_h, roi_w), np.uint8)
        new_roi = np.zeros((roi_h, roi_w), np.uint8)
        cv2.fillPoly(old_roi, old_pts_roi, 255)
        cv2.fillPoly(new_roi, new_pts_roi, 255)

        # Compute entering/leaving pixels
        old_b = old_roi > 0
        new_b = new_roi > 0
        entering = new_b & (~old_b)   # pixels added to polygon
        leaving  = old_b & (~new_b)   # pixels removed from polygon

        # ROI-enabled view
        enabled_roi = enabled_mask[y0:y1, x0:x1]

        delta_area_add = int(entering.sum())
        delta_area_sub = int(leaving.sum())
        delta_en_add   = int((entering & enabled_roi).sum())
        delta_en_sub   = int((leaving  & enabled_roi).sum())

        poly_area_p      = cur_poly_area + delta_area_add - delta_area_sub
        enabled_inside_p = cur_enabled_inside + delta_en_add - delta_en_sub

        coverage = 0.0 if enabled_total == 0 else enabled_inside_p / enabled_total
        disabled_frac = 0.0 if poly_area_p == 0 else (poly_area_p - enabled_inside_p) / poly_area_p
        return coverage, disabled_frac, enabled_inside_p, poly_area_p

    # ------------------------
    # (5) Coarse -> fine loop
    # ------------------------
    pyr = _build_pyramid((mask_bg > 0).astype(np.uint8), pyramid_levels)

    # For reporting across levels
    final_best_pts = None
    final_best_stats = None
    final_best_timeline = []
    final_kicks = []

    for level_idx, (mask_lvl, sx, sy) in enumerate(pyr):
        h, w = mask_lvl.shape[:2]

        # (1) Precompute constants
        enabled = mask_lvl > 0
        enabled_total = int(enabled.sum())
        full_scratch = np.zeros((h, w), np.uint8)  # reused for occasional full eval

        # Step sizes scale with level
        step = max(w, h) * step_frac
        min_step = max(w, h) * min_step_frac

        # Init polygon: if coming from coarser level, upscale previous result
        if final_best_pts is not None:
            # upscale previous points to current level
            scale_x = w / prev_w
            scale_y = h / prev_h
            pts0 = [(int(round(x * scale_x)), int(round(y * scale_y))) for (x, y) in final_best_pts]
            pts0 = _order_polygon_points(pts0)
        else:
            # original initializer on this level
            starting_points = [(0,0),(0,h-1),(w-1,h-1),(w-1,0),(w//2,0),(0,h//2),(w-1,h//2),(w//2,h-1)]
            pts0 = _order_polygon_points(starting_points[:sides])

        if debug:
            print(f"[Level {level_idx+1}/{len(pyr)}] init points:", pts0)

        # Baseline stats (one full-image fill)
        cov0, dis0, en0, area0 = _coverage_from_poly_full(pts0, enabled, enabled_total, full_scratch)

        def _optimize_from(pts_start: List[Tuple[int, int]]):
            nonlocal step
            pts_loc = _order_polygon_points(pts_start.copy())

            # current stats
            coverage_loc, disabled_loc, enabled_loc, area_loc = _coverage_from_poly_full(
                pts_loc, enabled, enabled_total, full_scratch
            )

            timeline: List[List[Tuple[int, int]]] = [pts_loc.copy()]
            iters = 0
            no_change_streak = 0
            eps = 1e-9

            # Phase 0: grow coverage if needed (greedy, ROI/delta-based)
            if coverage_loc < target_coverage:
                while iters < max_iters and coverage_loc < target_coverage and step >= min_step:
                    iters += 1
                    corner_best_expected = -1.0
                    corner_best = None

                    for vi, (vx, vy) in enumerate(pts_loc):
                        v = np.array([vx, vy], dtype=float)

                        def try_dir_growth(dx, dy):
                            npnt = (int(round(v[0] + dx)), int(round(v[1] + dy)))
                            npnt = (max(0, min(w-1, npnt[0])), max(0, min(h-1, npnt[1])))
                            cov2, dis2, en2, area2 = _eval_move_delta_roi(
                                pts_loc, vi, npnt, enabled, enabled_total,
                                enabled_loc, area_loc
                            )
                            d_cov = cov2 - coverage_loc
                            if d_cov <= 0:
                                return -np.inf, None, None
                            d_dis_inc = max(0.0, dis2 - disabled_loc)
                            score = d_cov / (d_dis_inc + eps)
                            return score, npnt, (cov2, dis2, en2, area2)

                        # Coarse axis comps
                        s_up, np_up, st_up = try_dir_growth(0, -step)
                        s_down, np_down, st_down = try_dir_growth(0, step)
                        s_left, np_left, st_left = try_dir_growth(-step, 0)
                        s_right, np_right, st_right = try_dir_growth(step, 0)

                        if s_down >= s_up:
                            vy_comp = s_down; v_sign = 1.0; np_yc = (np_down, st_down)
                        else:
                            vy_comp = s_up;   v_sign = -1.0; np_yc = (np_up,   st_up)
                        if s_right >= s_left:
                            vx_comp = s_right; h_sign = 1.0; np_xc = (np_right, st_right)
                        else:
                            vx_comp = s_left;  h_sign = -1.0; np_xc = (np_left,  st_left)

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
                                score, npnt, st = try_dir_growth(dx, dy)
                                if np.isfinite(score) and score > 0:
                                    chosen = (score, npnt, st, (dx, dy))
                                    break
                            if chosen is not None:
                                sc, npnt, st, move_vec = chosen
                                if sc > corner_best_expected:
                                    corner_best_expected = sc
                                    corner_best = (vi, npnt, st)

                    if corner_best is not None:
                        vi, new_pt, new_stats = corner_best
                        # apply
                        pts_loc[vi] = new_pt
                        pts_loc = _order_polygon_points(pts_loc)  # keep simple/ordered
                        coverage_loc, disabled_loc, enabled_loc, area_loc = new_stats
                        no_change_streak = 0
                        # small step up when improving
                        timeline.append(pts_loc.copy())
                    else:
                        no_change_streak += 1
                        if no_change_streak > 6:
                            break
                        # shrink step on stall
                    step = min(step * 1.05, max(w, h) * 0.5) if corner_best is not None else step * 0.5

            # Phase 1: reduce disabled while keeping coverage >= target
            while iters < max_iters and coverage_loc >= target_coverage and step >= min_step:
                iters += 1
                corner_best_expected = -1.0
                corner_best = None

                for vi, (vx, vy) in enumerate(pts_loc):
                    v = np.array([vx, vy], dtype=float)

                    def try_dir(dx, dy):
                        npnt = (int(round(v[0] + dx)), int(round(v[1] + dy)))
                        npnt = (max(0, min(w-1, npnt[0])), max(0, min(h-1, npnt[1])))
                        cov2, dis2, en2, area2 = _eval_move_delta_roi(
                            pts_loc, vi, npnt, enabled, enabled_total,
                            enabled_loc, area_loc
                        )
                        if cov2 + 1e-12 < target_coverage:
                            return -np.inf, None, None
                        d_disabled = disabled_loc - dis2
                        d_cov = coverage_loc - cov2
                        if d_disabled <= 0:
                            return -np.inf, None, None
                        score = d_disabled / (d_cov + 1e-9)
                        return score, npnt, (cov2, dis2, en2, area2)

                    s_up, np_up, st_up = try_dir(0, -step)
                    s_down, np_down, st_down = try_dir(0, step)
                    s_left, np_left, st_left = try_dir(-step, 0)
                    s_right, np_right, st_right = try_dir(step, 0)

                    if s_down >= s_up:
                        vy_comp = s_down; v_sign = 1.0
                    else:
                        vy_comp = s_up;   v_sign = -1.0
                    if s_right >= s_left:
                        vx_comp = s_right; h_sign = 1.0
                    else:
                        vx_comp = s_left;  h_sign = -1.0

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
                            sc, npnt, st = try_dir(dx, dy)
                            if np.isfinite(sc) and sc > 0:
                                chosen = (sc, npnt, st)
                                break
                        if chosen is not None:
                            sc, npnt, st = chosen
                            if sc > corner_best_expected:
                                corner_best_expected = sc
                                corner_best = (vi, npnt, st)

                if corner_best is not None:
                    vi, new_pt, new_stats = corner_best
                    pts_loc[vi] = new_pt
                    pts_loc = _order_polygon_points(pts_loc)
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

        # Optimize from pts0
        pts, stats, timeline = _optimize_from(pts0)

        # Random kick scheme (unchanged externally, but uses fast eval internally through _optimize_from)
        iters_opt_init = int(stats.get('iters', 0))
        iters_sum_accepted = iters_opt_init
        iters_sum_all_runs = iters_opt_init
        kicks_accepted = 0
        kicks_total = 0
        rng = np.random.default_rng()
        M = float(max(w, h))
        throw_frac = 0.25
        min_throw_frac = 0.02
        consecutive_rejects = 0
        best_pts, best_stats = pts, stats
        best_timeline = timeline
        kicks: List[dict] = []

        while throw_frac >= min_throw_frac and consecutive_rejects < 20:
            vi = int(rng.integers(0, len(best_pts)))
            mag = float(throw_frac * M)
            x, y = best_pts[vi]
            angle, used_mag = _sample_uniform_valid_angle(x, y, mag, w, h, rng)
            if angle is None:
                throw_frac *= 0.8
                continue
            dx = int(round(used_mag * np.cos(angle)))
            dy = int(round(used_mag * np.sin(angle)))
            newx = max(0, min(w-1, int(round(x + dx))))
            newy = max(0, min(h-1, int(round(y + dy))))
            cand_pts = best_pts.copy()
            cand_pts[vi] = (newx, newy)
            cand_pts = _order_polygon_points(cand_pts)
            kick_entry = {'poly': cand_pts.copy(), 'accepted': False, 'idx': int(max(0, len(best_timeline) - 1))}

            cand_fit_pts, cand_stats, cand_timeline = _optimize_from(cand_pts)
            kicks_total += 1
            iters_sum_all_runs += int(cand_stats.get('iters', 0))
            if cand_stats['coverage'] >= target_coverage and cand_stats['disabled_frac'] + 1e-9 < best_stats['disabled_frac']:
                best_pts, best_stats = cand_fit_pts, cand_stats
                best_timeline.extend(cand_timeline)
                kick_entry['accepted'] = True
                kicks_accepted += 1
                iters_sum_accepted += int(cand_stats.get('iters', 0))
                consecutive_rejects = 0
                if debug:
                    print(f"[L{level_idx+1}] Accepted kick: throw={throw_frac:.3f}, disabled={best_stats['disabled_frac']:.6f}, cov={best_stats['coverage']:.4f}")
            else:
                consecutive_rejects += 1
                if debug:
                    print(f"[L{level_idx+1}] Rejected kick: throw={throw_frac:.3f}")
            throw_frac *= 0.95
            kicks.append(kick_entry)

        # Finalize stats accounting
        best_stats = best_stats.copy()
        best_stats['iters'] = int(iters_sum_accepted)
        best_stats['iters_opt_init'] = int(iters_opt_init)
        best_stats['iters_sum_accepted_kicks'] = int(iters_sum_accepted - iters_opt_init)
        best_stats['iters_sum_all_runs'] = int(iters_sum_all_runs)
        best_stats['kicks_total'] = int(kicks_total)
        best_stats['kicks_accepted'] = int(kicks_accepted)

        # Stash for next (finer) level
        final_best_pts = best_pts
        final_best_stats = best_stats
        final_best_timeline = best_timeline
        final_kicks += kicks
        prev_w, prev_h = w, h  # for next upscale

    # Return finest-level result
    return final_best_pts, final_best_stats, final_best_timeline, final_kicks