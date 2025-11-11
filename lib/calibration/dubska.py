# vp_debug.py
# pip install numpy scipy scikit-image matplotlib diamond_space

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.ndimage import gaussian_filter, maximum_filter, uniform_filter
from diamond_space import DiamondSpace

def detect_vanishing_points_debug(
    image,
    tensor_smoothing=False,
    show=True,
    save_dir=None,
    # Tuning parameters (you can override from code/CLI)
    space_size=128,
    use_dynamic_d=True,
    d_factor=1.0,
    d_static=256,
    edge_sigma_yx=(1.5, 1.5),
    edge_threshold=0.20,
    tensor_alpha=1.2,
    # Spatial thinning: prefer the strongest edgelet per neighborhood to avoid dense texture clutter.
    # Set to a positive radius in pixels (e.g., 5–15) to enable.
    edgelet_min_spacing_px=0,
    # Optional filter: keep only edges whose inclination (line direction) angle is within any provided ranges.
    # Angles are in radians and interpreted modulo pi (theta and theta+pi denote the same inclination).
    # Example: inclination_ranges=[(0.0, 0.3), (2.5, 3.1)]
    inclination_ranges=None,
    # Optional length/coherence filtering AFTER inclination and BEFORE NMS.
    # Keep a pixel only if there are at least K edge pixels within a (2R+1)x(2R+1) window around it.
    # This favors longer/denser structures and removes isolated speckles without requiring connectivity.
    length_min_neighbors_window_radius_px=2,
    length_min_neighbors_count=3,
    peak_threshold=0.55,
    peak_prominence=1.2,
    peak_min_dist=6,
    num_vp_to_return=3,
    # Visualization of supporting lines for each VP
    draw_support_lines=True,
    support_max_dist_px=2.5,
    support_top_k=None,  # e.g., 1000 to limit; None = all under threshold
    support_line_len=20,
    support_line_thickness=1,
    random_sample_if_dense=0,  # if >0, randomly sample this many supporting segments per VP for clarity
    # Manhattan-aware selection (try more peaks and pick the triple closest to orthogonal)
    select_manhattan_best=False,
    candidate_k=8,
    # Optional focal-length constraints
    # Provide fpx_bounds=(min_px,max_px) to constrain focal length in pixels directly.
    # Or provide sensor_width_mm and focal_mm_bounds=(min_mm,max_mm) to convert to pixels via fx = f_mm/img_width_mm * img_w.
    fpx_bounds=None,
    sensor_width_mm=None,
    focal_mm_bounds=None,
    # Inclination coloring
    color_edgelets_by_inclination=True,
    max_edgelets_color_show=20000,
    max_edgelets_ds_show=4000,
    ds_line_samples=64,
    # Figure toggles
    viz_nms_local_maxima=True,
    viz_nms_max_show=20000,
    viz_nms_marker_size=3.5,
    viz_nms_draw_windows=True,
    viz_nms_window_color='red',
    viz_nms_window_linewidth=0.9,
    viz_nms_show_all=False,
    viz_nms_suppressed_map=True,
    viz_inclination_prefilter=True,
    viz_inclination_lengthfilter=True,
    viz_input_image=True,
    viz_edges_and_gradients=True,
    viz_accumulator_and_peaks=False,
    viz_accumulator_all_candidates_numbered=True,
    viz_diamond_space_edgelets_inclination=False,  # Bad diamond colorful
    viz_folded_two_spaces=False,
    viz_overlay_vps_on_image=False,
    viz_all_candidates_on_image=False,
    viz_edgelets_inclination_on_image=True,
    viz_supporting_lines_combined=True,
    viz_supporting_lines_panels=True,
    viz_quadrants=False,
    viz_manhattan_check=False,
):
    """
    Loads an image, detects vanishing points using the Diamond Space accumulator,
    prints intermediary info, and shows/saves visualizations.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    show : bool
        If True, opens matplotlib windows. If False, only saves figures if save_dir is set.
    save_dir : str or None
        Directory to save figures. If None, nothing is saved.

    Returns
    -------
    vp : np.ndarray, shape (m, 2)
        Top m (≤ 3) vanishing points in image coordinates (x, y).
    """

    # ---------------------------------------------------
    # CONSTANTS (defaults can be overridden by function args)
    # ---------------------------------------------------
    # Accumulator / mapping
    SPACE_SIZE = int(space_size)       # per quadrant resolution (four quadrants internally)
    USE_DYNAMIC_D = bool(use_dynamic_d)  # choose diamond-space scale 'd' from image size
    D_FACTOR = float(d_factor)         # d = D_FACTOR * max(H, W) when USE_DYNAMIC_D
    D_STATIC = int(d_static)           # used if USE_DYNAMIC_D is False

    # Edge & gradient estimation
    EDGE_SIGMA_YX = tuple(edge_sigma_yx)  # Gaussian sigma for derivatives (y,x)
    EDGE_THRESH = float(edge_threshold)   # keep edges above this (after normalization to max=1)

    # Peak detection in accumulator
    PEAK_THRESHOLD = float(peak_threshold)
    PEAK_PROMINENCE = float(peak_prominence)
    PEAK_MIN_DIST = int(peak_min_dist)

    NUM_VP_TO_RETURN = int(num_vp_to_return)
    # ---------------------------------------------------

    t0 = time.perf_counter()

    # 1) Load image (grayscale)
    img_h, img_w = image.shape[:2]
    print(f"[info] image: {img_w}×{img_h}  dtype={image.dtype}")

    # 2) Edge magnitude and gradient orientation (Gaussian derivatives)
    # v_edges ~ d/dx, h_edges ~ d/dy
    Gx = gaussian_filter(image, sigma=EDGE_SIGMA_YX, order=(0, 1))  # d/dx
    Gy = gaussian_filter(image, sigma=EDGE_SIGMA_YX, order=(1, 0))  # d/dy
    gradient = np.arctan2(Gy, Gx)
    edges_grad = np.hypot(Gy, Gx)  # |∇I|

    if tensor_smoothing:
        # Structure tensor smoothing parameters
        TENSOR_SIGMA = 3.0
        ALPHA = 1.2  # coherence exponent ∈ [0.5, 2]

        # Compute smoothed structure tensor components
        Jxx = gaussian_filter(Gx * Gx, sigma=TENSOR_SIGMA)
        Jxy = gaussian_filter(Gx * Gy, sigma=TENSOR_SIGMA)
        Jyy = gaussian_filter(Gy * Gy, sigma=TENSOR_SIGMA)

        # Eigen decomposition
        tmp = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2)
        l1 = 0.5 * (Jxx + Jyy + tmp)
        l2 = 0.5 * (Jxx + Jyy - tmp)

        # Dominant orientation and coherence
        gradient = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)
        coherence = (l1 - l2) / (l1 + l2 + 1e-9)

        # Combine gradient magnitude and coherence
        edges = edges_grad * (coherence ** ALPHA)
    else:
        edges = edges_grad

    # normalize edge magnitudes to [0,1] for a stable threshold
    edges_norm = edges / (edges.max() + 1e-12)
    mask = edges_norm > EDGE_THRESH
    num_obs = int(mask.sum())
    print(
        f"[info] edges max={edges_norm.max():.3f}, threshold={EDGE_THRESH} "
        f"→ {num_obs} observations ({num_obs/(img_h*img_w):.1%} of pixels)"
    )

    # Optional inclination filtering BEFORE spatial NMS (angles in radians, modulo pi)
    # We compute inclination from the gradient as tangent angle = gradient + pi/2, modulo pi.
    mask_after_incl = None
    if inclination_ranges is not None:
        ang_mod_full = np.mod(gradient + (np.pi * 0.5), np.pi)
        valid_incl = np.zeros_like(ang_mod_full, dtype=bool)
        for rng in inclination_ranges:
            if not isinstance(rng, (list, tuple)) or len(rng) != 2:
                continue
            lo = float(rng[0]) % np.pi
            hi = float(rng[1]) % np.pi
            print("Low/High inclination range:" + str(lo) + " / " + str(hi))
            if hi >= lo:
                cond = (ang_mod_full >= lo) & (ang_mod_full <= hi)
            else:
                # wrapped interval (e.g., near pi->0 boundary)
                cond = (ang_mod_full >= lo) | (ang_mod_full <= hi)
            valid_incl |= cond
        before = int(mask.sum())
        mask = mask & valid_incl
        mask_after_incl = mask.copy()
        print(f"[info] inclination prefilter: kept {int(mask.sum())} / {before} after threshold")
    else:
        mask_after_incl = mask.copy()

    # Optional neighborhood-density filtering (after inclination, before NMS)
    mask_after_len = None
    if (length_min_neighbors_window_radius_px and length_min_neighbors_window_radius_px > 0) and (length_min_neighbors_count and length_min_neighbors_count > 0):
        base = mask.copy()
        before_len = int(base.sum())
        r = int(max(1, round(float(length_min_neighbors_window_radius_px))))
        win = 2 * r + 1
        # Count edge pixels in the local window via uniform_filter (mean * window_area = count)
        local_mean = uniform_filter(base.astype(np.float32), size=win, mode="constant", cval=0.0)
        local_count = local_mean * float(win * win)
        base = base & (local_count >= float(length_min_neighbors_count))
        after_len = int(base.sum())
        print(f"[info] neighborhood-density filter (win={win}, min={int(length_min_neighbors_count)}): kept {after_len} / {before_len}")
        mask = base
        mask_after_len = mask.copy()

    # Optional: spatial non-maximum suppression to keep edgelets far apart,
    # favoring the strongest response within a neighborhood of ~edgelet_min_spacing_px.
    if edgelet_min_spacing_px and edgelet_min_spacing_px > 0:
        rad = int(max(1, round(float(edgelet_min_spacing_px))))
        win = 2 * rad + 1
        # Add tiny deterministic jitter to break plateaus in maximum_filter comparisons
        rng = np.random.default_rng(12345)
        jitter = (rng.random(edges_norm.shape, dtype=np.float32) * 1e-6).astype(np.float32)
        edges_for_nms = (edges_norm + jitter).astype(np.float32)
        # Important: restrict the NMS competition to only allowed pixels
        # so disallowed orientations/thresholded-out pixels cannot suppress kept ones.
        edges_for_nms_masked = edges_for_nms.copy()
        edges_for_nms_masked[~mask] = -np.inf
        local_max = maximum_filter(edges_for_nms_masked, size=win, mode="nearest")
        keep = mask & (edges_for_nms_masked >= (local_max - 1e-12))
        print(f"[info] spatial NMS (win={win}): kept {int(keep.sum())} / {int(mask.sum())} edgelets after threshold")
        mask = keep

    # 3) Build homogeneous line parameters (A,B,C) for each oriented edge pixel
    #    A = cos θ, B = sin θ, C = -A*x - B*y
    yy, xx = np.meshgrid(
        np.arange(img_h, dtype=np.float32),
        np.arange(img_w, dtype=np.float32),
        indexing="ij",
    )
    cos_g = np.cos(gradient)
    sin_g = np.sin(gradient)

    A = cos_g[mask]
    B = sin_g[mask]
    C = -sin_g[mask] * yy[mask] - cos_g[mask] * xx[mask]
    # Canonicalize line direction: ensure A>0; if A==0, force B>=0. Flip (A,B,C) when needed.
    eps = 1e-12
    flip = (A < 0) | ((np.abs(A) <= eps) & (B < 0))
    if np.any(flip):
        A[flip] = -A[flip]
        B[flip] = -B[flip]
        C[flip] = -C[flip]
    # Prepare weights and geometry for optional filtering and visualization
    weights = edges_norm[mask].astype(np.float32)  # use edge strength as weight
    px = xx[mask]
    py = yy[mask]
    tx = -B  # tangent x component (perpendicular to normal)
    ty = A   # tangent y component

    # (Inclination filtering already applied pre-NMS at pixel level.)

    # Build final line set after optional filtering
    lines = np.stack([A, B, C], axis=1).astype(np.float32)
    print(f"[info] built {lines.shape[0]} line hypotheses")

    # 4) Diamond Space accumulation
    d_val = int(D_FACTOR * max(img_w, img_h)) if USE_DYNAMIC_D else int(D_STATIC)
    print(f"[info] diamond space: size={SPACE_SIZE}x{SPACE_SIZE} per quadrant (x4), d={d_val}")
    DS = DiamondSpace(d_val, SPACE_SIZE)

    # --- Debug: show which DS subspace(s) each line contributes to ---
    try:
        D_int, D_I, A_int, A_I = DS.get_intersection(lines.copy())
        # Subspace axis-pair mapping as in accumulator._generate_subspaces
        idx_a = [(0, 3), (1, 3), (1, 2), (0, 2)]  # ST, SS, TS, TT -> (-X,+Y), (+X,+Y), (+X,-Y), (-X,-Y)
        N = lines.shape[0]
        contrib = np.zeros((N, 4), dtype=bool)
        for k, (a0, a1) in enumerate(idx_a):
            has_d = D_I[:, k]
            has_a0 = A_I[:, a0]
            has_a1 = A_I[:, a1]
            # Equivalent to accumulator._generate_lines logic: a0&a1 or (d & (a0|a1))
            contrib[:, k] = (has_a0 & has_a1) | (has_d & (has_a0 | has_a1))
        counts = contrib.sum(axis=0)
        print(f"[debug] subspace routing counts ST,SS,TS,TT: {counts.tolist()}")

        # Sample a few lines to illustrate ty>=0 can still land in any subspace
        sample_k = min(10, N)
        if sample_k > 0:
            rng = np.random.default_rng(123)
            idx = rng.choice(np.arange(N), size=sample_k, replace=False)
            names = np.array(["ST", "SS", "TS", "TT"], dtype=object)
            for j in idx:
                a_, b_, c_ = float(lines[j, 0]), float(lines[j, 1]), float(lines[j, 2])
                tx_j, ty_j = -b_, a_
                hits = names[contrib[j]]
                print(f"  [debug] line#{j:05d}: A={a_:.4f}, B={b_:.4f}, C={c_:.1f}, t=({tx_j:.4f},{ty_j:.4f}) -> {','.join(hits) if len(hits) else 'none'}")
    except Exception as e:
        print(f"[debug] subspace routing log failed: {e}")

    DS.insert(lines, weights)  # or DS.insert(lines) for unweighted

    # 5) Peak finding (vanishing-point candidates)
    p, w, p_ds = DS.find_peaks(
        min_dist=PEAK_MIN_DIST, prominence=PEAK_PROMINENCE, t=PEAK_THRESHOLD
    )
    if p is None or len(p) == 0:
        print("[warn] no peaks found.")
        vp_xy = np.empty((0, 2), dtype=np.float32)
        vp_ds = None
        p_sorted = None
    else:
        order = np.argsort(-w)
        p_sorted = p[order].astype(np.float32)
        p_ds_sorted = p_ds[order].astype(np.float32) if p_ds is not None else None
        w_sorted = w[order]
        vp_full = p_sorted[:NUM_VP_TO_RETURN]
        vp_xy = vp_full[:, :2]
        selected_idx = np.arange(min(NUM_VP_TO_RETURN, p_sorted.shape[0]))
        vp_ds = p_ds_sorted[selected_idx, :2] if p_ds_sorted is not None else None
        ws = w_sorted[:NUM_VP_TO_RETURN]
        print("[info] peaks (image coords) sorted by response:")
        for i, (pt, wt) in enumerate(zip(vp_xy, ws), 1):
            print(f"  #{i}: (x={pt[0]:.1f}, y={pt[1]:.1f})  weight={wt:.2f}")

        # Manhattan-aware selection over more candidates (optional)
    if p_sorted is not None:
        if select_manhattan_best and p_sorted.shape[0] >= 3:
                import itertools
                cx, cy = (img_w - 1) * 0.5, (img_h - 1) * 0.5

                def implied_f2(v1, v2, c):
                    dx1, dy1 = v1[0] - c[0], v1[1] - c[1]
                    dx2, dy2 = v2[0] - c[0], v2[1] - c[1]
                    return -(dx1 * dx2 + dy1 * dy2)

                # Compute focal-length pixel bounds if given in mm
                fpx_min, fpx_max = None, None
                if focal_mm_bounds is not None and sensor_width_mm is not None and sensor_width_mm > 0:
                    fpx_min = float(focal_mm_bounds[0]) / float(sensor_width_mm) * float(img_w)
                    fpx_max = float(focal_mm_bounds[1]) / float(sensor_width_mm) * float(img_w)
                elif fpx_bounds is not None:
                    fpx_min, fpx_max = float(fpx_bounds[0]), float(fpx_bounds[1])

                def residual_sum(triple):
                    v1, v2, v3 = triple
                    f2s = [implied_f2(v1, v2, (cx, cy)), implied_f2(v1, v3, (cx, cy)), implied_f2(v2, v3, (cx, cy))]
                    pos = [v for v in f2s if v > 0]
                    if not pos:
                        return np.inf, None
                    f2 = float(np.median(pos))
                    # Optional focal-length bounds (in pixels)
                    if fpx_min is not None and fpx_max is not None:
                        f = float(np.sqrt(max(f2, 0.0)))
                        if not (fpx_min <= f <= fpx_max):
                            return np.inf, f2
                    # residuals around zero are better
                    r12 = (v1[0]-cx)*(v2[0]-cx) + (v1[1]-cy)*(v2[1]-cy) + f2
                    r13 = (v1[0]-cx)*(v3[0]-cx) + (v1[1]-cy)*(v3[1]-cy) + f2
                    r23 = (v2[0]-cx)*(v3[0]-cx) + (v2[1]-cy)*(v3[1]-cy) + f2
                    return abs(r12) + abs(r13) + abs(r23), f2

                K = int(min(candidate_k, p_sorted.shape[0]))
                cand = p_sorted[:K, :2]
                best = (np.inf, None, None)
                for i, j, k in itertools.combinations(range(K), 3):
                    triple = (cand[i], cand[j], cand[k])
                    score, f2 = residual_sum(triple)
                    if score < best[0]:
                        best = (score, f2, (i, j, k))
                if best[2] is not None:
                    i, j, k = best[2]
                    new_vps = np.stack([cand[i], cand[j], cand[k]], axis=0)
                    vp_xy = new_vps.astype(np.float32)
                    selected_idx = np.array([i, j, k], dtype=int)
                    vp_ds = p_ds_sorted[selected_idx, :2] if p_ds_sorted is not None else None
                    # Report f in pixels and optionally in mm if sensor width provided
                    f_report = ''
                    if best[1] is not None and best[1] > 0:
                        f_px = float(np.sqrt(best[1]))
                        f_report = f", f≈{f_px:.1f}px"
                        if sensor_width_mm is not None and sensor_width_mm > 0:
                            f_mm = f_px * float(sensor_width_mm) / float(img_w)
                            f_report += f" (~{f_mm:.2f}mm)"
                    print(f"[manhattan-select] chose indices {i},{j},{k} among top {K} with residual sum {best[0]:.1f} and f^2≈{best[1]:.1f}{f_report}")

    print(f"[done] total time: {time.perf_counter()-t0:.2f}s")

    # ---------------- Visualizations ----------------
    def _save(fig, name):
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, name), dpi=150)

    # (1) Input image
    if viz_input_image:
        fig1, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(image, cmap="gray")
        ax.set(title="Input image", xticks=[], yticks=[])
        fig1.tight_layout(); _save(fig1, "01_input.png")

    # (2) Edge magnitude + gradient orientation
    if viz_edges_and_gradients:
        print("AAAAAAA")
        fig2, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(edges_norm, cmap="Greys")
        axs[0].set(title="Edge map (normalized)", xticks=[], yticks=[])
        axs[1].imshow(gradient, cmap="twilight_shifted")
        axs[1].set(title="Gradient orientation (rad)", xticks=[], yticks=[])
        fig2.tight_layout(); _save(fig2, "02_edges_and_gradients.png")

    # (2a) Edge map with NMS local maxima spots
    # Shows the positions selected by spatial NMS ("keep") as yellow dots over the normalized edge map.
    if viz_nms_local_maxima and (edgelet_min_spacing_px and edgelet_min_spacing_px > 0) and 'keep' in locals() and keep is not None:
        fig2a, ax2a = plt.subplots(1, 1, figsize=(6, 6))
        # Show direction/length-filtered edge magnitude as background if available
        bg_edges = edges_norm
        if ('mask_after_len' in locals()) and (mask_after_len is not None):
            bg_edges = np.where(mask_after_len, edges_norm, 0.0)
        elif ('mask_after_incl' in locals()) and (mask_after_incl is not None):
            bg_edges = np.where(mask_after_incl, edges_norm, 0.0)
        ax2a.imshow(bg_edges, cmap="Greys")
        ys, xs = np.where(keep)
        total_kept = int(ys.size)
        if ys.size > 0:
            # Optional subsampling for readability
            if not viz_nms_show_all and (viz_nms_max_show is not None) and ys.size > int(viz_nms_max_show):
                rng = np.random.default_rng(31415)
                sel = rng.choice(np.arange(ys.size), size=int(viz_nms_max_show), replace=False)
                ys, xs = ys[sel], xs[sel]
            ax2a.plot(xs, ys, linestyle='none', marker='o', markersize=float(viz_nms_marker_size), color='yellow', alpha=0.9)
            # Optionally draw the NMS square window around each kept point
            if viz_nms_draw_windows and 'win' in locals():
                from matplotlib.patches import Rectangle
                r = max(0, int((int(win) - 1) // 2))
                for (x_i, y_i) in zip(xs, ys):
                    rect = Rectangle((x_i - r, y_i - r), 2*r + 1, 2*r + 1,
                                     fill=False,
                                     edgecolor=viz_nms_window_color,
                                     linewidth=float(viz_nms_window_linewidth),
                                     alpha=0.9)
                    ax2a.add_patch(rect)
        shown = int(xs.size)
        ax2a.set(title=f"NMS local maxima (win={win}) — showing {shown}/{total_kept}", xticks=[], yticks=[])
        fig2a.tight_layout(); _save(fig2a, "02a_nms_local_maxima.png")

    # (2a-extra) Map of suppressed vs kept after NMS (for debugging coverage)
    if viz_nms_suppressed_map and (edgelet_min_spacing_px and edgelet_min_spacing_px > 0) and 'keep' in locals() and keep is not None:
        fig2a_s, ax2a_s = plt.subplots(1, 1, figsize=(6, 6))
        # Show only direction/length-filtered edges; color kept vs suppressed
        # kept: yellow dots; suppressed: small red dots
        nms_base_mask = None
        if ('mask_after_len' in locals()) and (mask_after_len is not None):
            nms_base_mask = mask_after_len
        else:
            nms_base_mask = mask_after_incl
        bg_edges = np.where(nms_base_mask, edges_norm, 0.0)
        ax2a_s.imshow(bg_edges, cmap="Greys")
        ys_k, xs_k = np.where(keep)
        ys_s, xs_s = np.where(nms_base_mask & (~keep))
        if ys_s.size > 0:
            ax2a_s.plot(xs_s, ys_s, linestyle='none', marker='o', markersize=1.2, color='red', alpha=0.7)
        if ys_k.size > 0:
            ax2a_s.plot(xs_k, ys_k, linestyle='none', marker='o', markersize=2.2, color='yellow', alpha=0.9)
        ax2a_s.set(title="NMS kept (yellow) vs suppressed (red)", xticks=[], yticks=[])
        fig2a_s.tight_layout(); _save(fig2a_s, "02a_nms_suppressed_map.png")

    # (2b) Edge map after inclination prefilter (only edges within allowed inclination ranges)
    if viz_inclination_prefilter and ('mask_after_incl' in locals()) and (mask_after_incl is not None):
        fig2b, ax2b = plt.subplots(1, 1, figsize=(6, 6))
        masked_edges = np.where(mask_after_incl, edges_norm, 0.0)
        ax2b.imshow(masked_edges, cmap="Greys")
        ax2b.set(title="Edges after inclination prefilter", xticks=[], yticks=[])
        fig2b.tight_layout(); _save(fig2b, "02b_inclination_prefilter.png")

    # (2c) Edge map after neighborhood-density filtering
    if viz_inclination_lengthfilter and ('mask_after_len' in locals()) and (mask_after_len is not None):
        fig2c, ax2c = plt.subplots(1, 1, figsize=(6, 6))
        masked_edges_len = np.where(mask_after_len, edges_norm, 0.0)
        ax2c.imshow(masked_edges_len, cmap="Greys")
        ax2c.set(title="Edges after neighborhood-density filtering", xticks=[], yticks=[])
        fig2c.tight_layout(); _save(fig2c, "02c_inclination_lengthfilter.png")

    # (3) Accumulator (attached diamond space) + detected peaks
    A_img = DS.attach_spaces()
    extent = (
        (-DS.size + 0.5) / DS.scale,
        (DS.size - 0.5) / DS.scale,
        (DS.size - 0.5) / DS.scale,
        (-DS.size + 0.5) / DS.scale,
    )
    if viz_accumulator_and_peaks:
        fig3, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(A_img, cmap="Greys", extent=extent)
        if p_ds is not None and len(p_ds):
            ax.plot(p_ds[:, 0] / DS.scale, p_ds[:, 1] / DS.scale, "r+", alpha=0.5)
        # Annotate selected VPs with consistent numbering and colors
        colors_vp = ["tab:red", "tab:green", "tab:blue", "tab:orange"]
        if 'vp_ds' in locals() and vp_ds is not None and len(vp_xy):
            x_min, x_max = extent[0], extent[1]
            y_min, y_max = extent[3], extent[2]
            dx = 0.02 * (x_max - x_min)
            dy = 0.02 * (y_max - y_min)
            for i in range(min(vp_ds.shape[0], 4)):
                col = colors_vp[i % len(colors_vp)]
                x = vp_ds[i, 0] / DS.scale
                y = vp_ds[i, 1] / DS.scale
                ax.plot([x], [y], marker="+", color=col, markersize=10, mew=2)
                ax.text(x + dx, y - dy, f"VP{i+1}", color=col, fontsize=9)
        ax.set(title="Accumulator (diamond space)")
        ax.invert_yaxis()
        fig3.tight_layout(); _save(fig3, "03_accumulator_and_peaks.png")

    # Debug: print per-quadrant energy to clarify symmetry/mirroring behavior
    try:
        q0 = float(np.sum(DS.A[0]))  # ST
        q1 = float(np.sum(DS.A[1]))  # SS
        q2 = float(np.sum(DS.A[2]))  # TS
        q3 = float(np.sum(DS.A[3]))  # TT
        print(f"[debug] quadrant sums -> ST:{q0:.1f}  SS:{q1:.1f}  TS:{q2:.1f}  TT:{q3:.1f}")
    except Exception as e:
        print(f"[debug] could not compute quadrant sums: {e}")

    # Optional: show a folded, two-space view (sum opposite quadrants)
    if viz_folded_two_spaces:
        try:
            # Opposite pairs (by convention of attach_spaces in this package):
            fold_a = DS.A[1].astype(np.float32) + DS.A[3].astype(np.float32)  # SS + TT
            fold_b = DS.A[0].astype(np.float32) + DS.A[2].astype(np.float32)  # ST + TS
            vmax = max(float(fold_a.max()), float(fold_b.max()), 1.0)
            fig3d, axes3d = plt.subplots(1, 2, figsize=(8, 4))
            axes3d[0].imshow(fold_a, cmap="inferno", vmin=0, vmax=vmax)
            axes3d[0].set(title="Folded: SS + TT", xticks=[], yticks=[])
            axes3d[1].imshow(fold_b, cmap="inferno", vmin=0, vmax=vmax)
            axes3d[1].set(title="Folded: ST + TS", xticks=[], yticks=[])
            fig3d.tight_layout(); _save(fig3d, "03d_folded_two_spaces.png")
        except Exception as e:
            print(f"[debug] could not plot folded spaces: {e}")

    # (3c) Accumulator with ALL candidate peaks numbered (sorted by response)
    if viz_accumulator_all_candidates_numbered and 'p_ds_sorted' in locals() and p_ds_sorted is not None and len(p_ds_sorted):
        fig3c, ax3c = plt.subplots(1, 1, figsize=(6, 6))
        ax3c.imshow(A_img, cmap="Greys", extent=extent)
        ax3c.invert_yaxis()
        # plot all candidates as small markers
        ax3c.plot(p_ds_sorted[:, 0] / DS.scale, p_ds_sorted[:, 1] / DS.scale, marker='o', linestyle='none', markersize=2, color='cyan', alpha=0.8)
        # annotate with rank numbers (1-based)
        x_min, x_max = extent[0], extent[1]
        y_min, y_max = extent[3], extent[2]
        dx = 0.01 * (x_max - x_min)
        dy = 0.01 * (y_max - y_min)
        for i in range(p_ds_sorted.shape[0]):
            x = p_ds_sorted[i, 0] / DS.scale
            y = p_ds_sorted[i, 1] / DS.scale
            ax3c.text(x + dx, y - dy, str(i+1), color='yellow', fontsize=7)
        ax3c.set(title="Accumulator: all candidate peaks numbered (rank)")
        fig3c.tight_layout(); _save(fig3c, "03c_accumulator_all_candidates_numbered.png")

    # (3b) Accumulator with colored transformed edge constraints by inclination
    # Draw each edge's constraint line A x + B y + C = 0 in VP space, colored by its inclination hue.
    if viz_diamond_space_edgelets_inclination and color_edgelets_by_inclination and lines.shape[0] > 0:
        from matplotlib.collections import LineCollection
        # Compute hue per edge based on tangent direction angle, and convert to RGB with full S,V
        ang_edges = np.arctan2(ty, tx)
        # Use inclination modulo 180° so opposite directions share the same color
        h_edges = (ang_edges % np.pi) / np.pi
        hsv_edges = np.stack([h_edges, np.ones_like(h_edges), np.ones_like(h_edges)], axis=1)
        from matplotlib.colors import hsv_to_rgb
        rgb_edges = hsv_to_rgb(hsv_edges)

        # Subsample edges to plot in DS for readability
        N = lines.shape[0]
        ds_idx = np.arange(N)
        if max_edgelets_ds_show is not None and N > max_edgelets_ds_show:
            rng = np.random.default_rng(43)
            ds_idx = rng.choice(ds_idx, size=max_edgelets_ds_show, replace=False)

        # Prepare x or y samples in DS axis coordinates
        x_min, x_max = extent[0], extent[1]
        y_min, y_max = extent[3], extent[2]  # note extent is (xmin, xmax, ymax, ymin) before invert
        xs = np.linspace(x_min, x_max, ds_line_samples, dtype=np.float32)
        ys = np.linspace(y_min, y_max, ds_line_samples, dtype=np.float32)

        segs = []
        seg_cols = []
        for idx in ds_idx:
            a, b, c = lines[idx]
            col = rgb_edges[idx]
            # If |B| > |A|, solve y(x); else solve x(y)
            if abs(b) > abs(a) and abs(b) > 1e-6:
                # y_img = -(A*x_img + C)/B ; convert to DS axis by dividing by scale
                x_img = xs * DS.scale
                y_img = -(a * x_img + c) / b
                y_axis = y_img / DS.scale
                pts = np.stack([xs, y_axis], axis=1)
            elif abs(a) > 1e-6:
                # x_img = -(B*y_img + C)/A ; convert to DS axis by dividing by scale
                y_img = ys * DS.scale
                x_img = -(b * y_img + c) / a
                x_axis = x_img / DS.scale
                pts = np.stack([x_axis, ys], axis=1)
            else:
                continue
            # Clip to plotting bounds by removing NaNs/Infs and keeping inside extent box
            valid = (
                np.isfinite(pts[:,0]) & np.isfinite(pts[:,1]) &
                (pts[:,0] >= x_min) & (pts[:,0] <= x_max) &
                (pts[:,1] >= y_min) & (pts[:,1] <= y_max)
            )
            pts = pts[valid]
            if pts.shape[0] >= 2:
                # break into small segments for LineCollection coloring
                segs.extend(np.stack([pts[:-1], pts[1:]], axis=1))
                seg_cols.extend([col] * (pts.shape[0]-1))

        fig3b, ax3b = plt.subplots(1, 1, figsize=(6, 6))
        ax3b.imshow(A_img, cmap="Greys", extent=extent)
        if segs:
            lc = LineCollection(np.array(segs), colors=np.array(seg_cols), linewidths=0.6, alpha=0.9)
            ax3b.add_collection(lc)
        ax3b.set(title="Diamond Space: edge constraints colored by inclination")
        ax3b.invert_yaxis()
        fig3b.tight_layout(); _save(fig3b, "03b_diamond_space_edgelets_inclination.png")

    # (4) Overlay vanishing points on the original image
    if viz_overlay_vps_on_image:
        fig4, ax = plt.subplots(1, 1, figsize=(7, 7))
        ax.imshow(image, cmap="gray", extent=(-0.5, img_w - 0.5, img_h - 0.5, -0.5))
        colors_vp = ["tab:red", "tab:green", "tab:blue", "tab:orange"]
        if len(vp_xy):
            for i in range(min(vp_xy.shape[0], 4)):
                col = colors_vp[i % len(colors_vp)]
                x, y = vp_xy[i, 0], vp_xy[i, 1]
                ax.plot([x], [y], marker="+", color=col, markersize=12, mew=2)
                ax.text(x + 8, y - 8, f"VP{i+1}", color=col, fontsize=10)
        ax.set(title="Detected vanishing points (image coords)")
        ax.set_xlim(0, img_w); ax.set_ylim(img_h, 0)
        fig4.tight_layout(); _save(fig4, "04_overlay.png")

    # (4d) Original image with ALL candidate VPs numbered (sorted by response)
    if viz_all_candidates_on_image and 'p_sorted' in locals() and p_sorted is not None and len(p_sorted):
        fig4d, ax4d = plt.subplots(1, 1, figsize=(7, 7))
        ax4d.imshow(image, cmap="gray", extent=(-0.5, img_w - 0.5, img_h - 0.5, -0.5))

        # Helper: intersection of ray from center to (x,y) with image border
        cx, cy = (img_w - 1) * 0.5, (img_h - 1) * 0.5
        def _intersect_border(cx, cy, x, y, w, h):
            dx = x - cx; dy = y - cy
            ts = []
            eps = 1e-9
            # left x=0
            if abs(dx) > eps:
                t = (0 - cx) / dx
                yy = cy + t * dy
                if t > 0 and 0 - 1e-6 <= yy <= (h - 1) + 1e-6:
                    ts.append((t, 0.0, yy))
                # right x=w-1
                t = ((w - 1) - cx) / dx
                yy = cy + t * dy
                if t > 0 and 0 - 1e-6 <= yy <= (h - 1) + 1e-6:
                    ts.append((t, float(w - 1), yy))
            # top y=0
            if abs(dy) > eps:
                t = (0 - cy) / dy
                xx = cx + t * dx
                if t > 0 and 0 - 1e-6 <= xx <= (w - 1) + 1e-6:
                    ts.append((t, xx, 0.0))
                # bottom y=h-1
                t = ((h - 1) - cy) / dy
                xx = cx + t * dx
                if t > 0 and 0 - 1e-6 <= xx <= (w - 1) + 1e-6:
                    ts.append((t, xx, float(h - 1)))
            if not ts:
                return None
            tmin, xi, yi = min(ts, key=lambda z: z[0])
            return float(xi), float(yi)

        # Plot all candidates with numbering; show off-screen as border ticks + labels
        ax4d.set_xlim(0, img_w); ax4d.set_ylim(img_h, 0)
        for i in range(p_sorted.shape[0]):
            x = float(p_sorted[i, 0]); y = float(p_sorted[i, 1])
            inside = (0 <= x <= (img_w - 1)) and (0 <= y <= (img_h - 1))
            if inside:
                ax4d.plot([x], [y], marker='o', linestyle='none', markersize=2, color='cyan', alpha=0.9)
                ax4d.text(x + 6, y - 6, str(i + 1), color='yellow', fontsize=8)
            else:
                hit = _intersect_border(cx, cy, x, y, img_w, img_h)
                if hit is None:
                    continue
                xi, yi = hit
                # draw a short tick pointing outward along the ray direction
                dx = x - cx; dy = y - cy
                norm = (dx * dx + dy * dy) ** 0.5
                if norm > 1e-6:
                    ux, uy = dx / norm, dy / norm
                else:
                    ux, uy = 1.0, 0.0
                tick_len = 10.0
                x0, y0 = xi - ux * tick_len, yi - uy * tick_len
                ax4d.plot([x0, xi], [y0, yi], color='cyan', linewidth=1.2)
                # label slightly inside the border
                ax4d.text(x0 - ux * 4, y0 - uy * 4, str(i + 1), color='yellow', fontsize=8)
        ax4d.set(title="All candidate VPs numbered (rank); off-screen shown at border")
        fig4d.tight_layout(); _save(fig4d, "04d_all_candidates_on_image.png")

    # (4a) Edgelets colored by inclination (line direction angle) on the original image
    if viz_edgelets_inclination_on_image and color_edgelets_by_inclination and mask.any():
        # angle of the tangent direction of the line: atan2(ty, tx)
        angles = np.arctan2(ty, tx)
        # Use inclination modulo 180°: directions θ and θ+π map to the same hue
        hues = (angles % np.pi) / np.pi
        from matplotlib.colors import hsv_to_rgb
        colors = hsv_to_rgb(np.stack([hues, np.ones_like(hues), np.ones_like(hues)], axis=1))

        # Build short segments centered at (px,py)
        half_len = max(1.0, support_line_len * 0.5)
        dx = tx * half_len
        dy = ty * half_len

        # Optionally subsample to avoid drawing too many segments
        N = px.shape[0]
        idx = np.arange(N)
        if max_edgelets_color_show is not None and N > max_edgelets_color_show:
            rng = np.random.default_rng(42)
            idx = rng.choice(idx, size=max_edgelets_color_show, replace=False)

        from matplotlib.collections import LineCollection
        segs = np.stack([
            np.stack([px[idx] - dx[idx], py[idx] - dy[idx]], axis=1),
            np.stack([px[idx] + dx[idx], py[idx] + dy[idx]], axis=1)
        ], axis=1)  # (M,2,2)
        lc = LineCollection(segs, colors=colors[idx], linewidths=max(0.5, support_line_thickness), alpha=1.0)
        fig4a, ax4a = plt.subplots(1, 1, figsize=(7, 7))
        ax4a.imshow(image, cmap="gray", extent=(-0.5, img_w - 0.5, img_h - 0.5, -0.5))
        ax4a.add_collection(lc)
        ax4a.set_xlim(0, img_w); ax4a.set_ylim(img_h, 0)
        ax4a.set(title="Edgelets colored by inclination (HSV)")
        fig4a.tight_layout(); _save(fig4a, "04a_edgelets_inclination.png")

    # (4b) Supporting lines for each VP
    if viz_supporting_lines_combined and draw_support_lines and len(vp_xy):
        colors = ["tab:red", "tab:green", "tab:blue", "tab:orange"]
        # distances from each line to each vp: |A*xv + B*yv + C|
        VPn = vp_xy.shape[0]
        Ax_v = lines[:, 0][:, None] * vp_xy[None, :, 0]
        By_v = lines[:, 1][:, None] * vp_xy[None, :, 1]
        C_v = lines[:, 2][:, None]
        dists = np.abs(Ax_v + By_v + C_v)  # (L, VPn)

        def draw_segments(ax_, sel_idx, color):
            if sel_idx.size == 0:
                return
            # optional random sampling to avoid overplot
            if random_sample_if_dense and sel_idx.size > random_sample_if_dense:
                rng = np.random.default_rng(123)
                sel_idx = rng.choice(sel_idx, size=random_sample_if_dense, replace=False)
            half_len = support_line_len * 0.5
            dx = tx[sel_idx] * half_len
            dy = ty[sel_idx] * half_len
            x0 = px[sel_idx] - dx
            y0 = py[sel_idx] - dy
            x1 = px[sel_idx] + dx
            y1 = py[sel_idx] + dy
            for x_a, y_a, x_b, y_b in zip(x0, y0, x1, y1):
                ax_.plot(
                    [x_a, x_b],
                    [y_a, y_b],
                    color=color,
                    linewidth=support_line_thickness,
                    alpha=0.8,
                )

        # Combined overlay for all VPs
        fig4b, ax4b = plt.subplots(1, 1, figsize=(7, 7))
        ax4b.imshow(image, cmap="gray", extent=(-0.5, img_w - 0.5, img_h - 0.5, -0.5))
        for i in range(VPn):
            color = colors[i % len(colors)]
            sel = np.where(dists[:, i] <= support_max_dist_px)[0]
            if support_top_k is not None and sel.size > support_top_k:
                order_sel = np.argsort(-weights[sel])[:support_top_k]
                sel = sel[order_sel]
            draw_segments(ax4b, sel, color)
            ax4b.plot(vp_xy[i, 0], vp_xy[i, 1], marker="+", color=color, markersize=12, mew=2)
            print(f"[info] VP#{i+1}: {sel.size} supporting lines within {support_max_dist_px}px")
        ax4b.set(title="Supporting lines per VP (colored)")
        ax4b.set_xlim(0, img_w); ax4b.set_ylim(img_h, 0)
        fig4b.tight_layout(); _save(fig4b, "04b_supporting_lines.png")

        # Per-VP panels
        if viz_supporting_lines_panels:
            cols = min(3, VPn)
            rows = int(np.ceil(VPn / cols))
            fig4c, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])
            axes = axes.ravel()
            for i in range(VPn):
                ax_i = axes[i]
                ax_i.imshow(image, cmap="gray", extent=(-0.5, img_w - 0.5, img_h - 0.5, -0.5))
                color = colors[i % len(colors)]
                sel = np.where(dists[:, i] <= support_max_dist_px)[0]
                if support_top_k is not None and sel.size > support_top_k:
                    order_sel = np.argsort(-weights[sel])[:support_top_k]
                    sel = sel[order_sel]
                draw_segments(ax_i, sel, color)
                ax_i.plot(vp_xy[i, 0], vp_xy[i, 1], marker="+", color=color, markersize=12, mew=2)
                ax_i.set(title=f"VP#{i+1}: {sel.size} supporting lines")
                ax_i.set_xlim(0, img_w); ax_i.set_ylim(img_h, 0)
            for j in range(VPn, axes.size):
                axes[j].axis("off")
            fig4c.tight_layout(); _save(fig4c, "04c_supporting_lines_panels.png")

    # (5) Separate quadrants (SS/ST/TS/TT) for debugging
    if viz_quadrants:
        Amax = A_img.max()
        fig5, ax = plt.subplots(2, 2, figsize=(5, 5))
        for a in ax.ravel():
            a.axis("off")
        ax[0, 0].imshow(DS.A[0], cmap="inferno", vmin=0, vmax=Amax)
        ax[0, 0].invert_yaxis(); ax[0, 0].invert_xaxis(); ax[0, 0].text(10, 12, "ST", color="w")
        ax[0, 1].imshow(DS.A[1], cmap="inferno", vmin=0, vmax=Amax)
        ax[0, 1].invert_yaxis(); ax[0, 1].text(10, 12, "SS", color="w")
        ax[1, 1].imshow(DS.A[2], cmap="inferno", vmin=0, vmax=Amax)
        ax[1, 1].text(10, 12, "TS", color="w")
        ax[1, 0].imshow(DS.A[3], cmap="inferno", vmin=0, vmax=Amax)
        ax[1, 0].invert_xaxis(); ax[1, 0].text(10, 12, "TT", color="w")
        fig5.tight_layout(); _save(fig5, "05_quadrants.png")

    # (6) Optional: Manhattan orthogonality sanity check (principal point at image center)
    if viz_manhattan_check and len(vp_xy) >= 2:
        cx, cy = (img_w - 1) * 0.5, (img_h - 1) * 0.5

        def implied_f2(v1, v2, c):
            dx1, dy1 = v1[0] - c[0], v1[1] - c[1]
            dx2, dy2 = v2[0] - c[0], v2[1] - c[1]
            # For orthogonal directions i ⟂ j: (dx1*dx2 + dy1*dy2) + f^2 = 0
            return -(dx1 * dx2 + dy1 * dy2)

        def dot_ortho_residual(v1, v2, f2, c):
            dx1, dy1 = v1[0] - c[0], v1[1] - c[1]
            dx2, dy2 = v2[0] - c[0], v2[1] - c[1]
            return (dx1 * dx2 + dy1 * dy2 + f2)

        # Compute f from the first two VPs and check residuals
        f2_12 = implied_f2(vp_xy[0], vp_xy[1], (cx, cy))
        f2_13 = implied_f2(vp_xy[0], vp_xy[2], (cx, cy)) if len(vp_xy) >= 3 else None
        f2_23 = implied_f2(vp_xy[1], vp_xy[2], (cx, cy)) if len(vp_xy) >= 3 else None

        def fmt_f2(name, val):
            if val is None:
                return f"{name}: n/a"
            return f"{name}: f^2={val:.1f} ({'valid' if val>0 else 'invalid'})"

        print("[manhattan] principal point ~ center:", (cx, cy))
        print("[manhattan] implied focal-squared from pairs:")
        print("  "+fmt_f2("f2(1,2)", f2_12))
        print("  "+fmt_f2("f2(1,3)", f2_13))
        print("  "+fmt_f2("f2(2,3)", f2_23))

        # Choose a positive f^2 (if any) for residual reporting
        candidates = [v for v in [f2_12, f2_13, f2_23] if (v is not None and v > 0)]
        if candidates:
            f2 = float(np.median(candidates))
            print(f"[manhattan] using f^2≈{f2:.1f} to report orthogonality residuals")
            # Residuals should be near 0 for orthogonal pairs
            if len(vp_xy) >= 2:
                r12 = dot_ortho_residual(vp_xy[0], vp_xy[1], f2, (cx, cy))
                print(f"  residual(1,2)={r12:.1f}")
            if len(vp_xy) >= 3:
                r13 = dot_ortho_residual(vp_xy[0], vp_xy[2], f2, (cx, cy))
                r23 = dot_ortho_residual(vp_xy[1], vp_xy[2], f2, (cx, cy))
                print(f"  residual(1,3)={r13:.1f}")
                print(f"  residual(2,3)={r23:.1f}")

            # Annotated figure
            fig6, ax6 = plt.subplots(1, 1, figsize=(7, 7))
            ax6.imshow(image, cmap="gray", extent=(-0.5, img_w - 0.5, img_h - 0.5, -0.5))
            ax6.plot([cx], [cy], marker="x", color="yellow", mew=2)
            ax6.text(cx+5, cy+5, "c", color="yellow")
            colors = ["tab:red", "tab:green", "tab:blue"]
            for i in range(len(vp_xy)):
                col = colors[i % len(colors)]
                ax6.plot(vp_xy[i, 0], vp_xy[i, 1], marker="+", color=col, markersize=12, mew=2)
                ax6.plot([cx, vp_xy[i, 0]], [cy, vp_xy[i, 1]], color=col, alpha=0.6)
                ax6.text(vp_xy[i, 0]+5, vp_xy[i, 1]+5, f"VP{i+1}", color=col)
            ax6.set(title="VPs with center and rays (Manhattan check)")
            ax6.set_xlim(0, img_w); ax6.set_ylim(img_h, 0)
            fig6.tight_layout(); _save(fig6, "06_manhattan_check.png")

    if show:
        plt.show()
    else:
        plt.close("all")

    return vp_xy


if __name__ == "__main__":
    # quick CLI usage example
    image_path = "dataset/session0_left/screen.png"
    image=imread(image_path, as_gray=True).astype(np.float32)
    detect_vanishing_points_debug(
        image,
        show=True,
        save_dir="test_video/vp_debug",
        # Example tuning overrides
        edgelet_min_spacing_px=20,
        edge_sigma_yx=(1.0, 1.0),
        edge_threshold=0.30,
        peak_threshold=0.35,
        peak_prominence=0.9,
        peak_min_dist=8,
        draw_support_lines=True,
        support_max_dist_px=10,
        support_top_k=2500,
        random_sample_if_dense=0,
        # Manhattan-aware selection with focal constraint at 40mm on 1/2.3" sensor (~6.17mm width)
        select_manhattan_best=True,
        candidate_k=10,
        sensor_width_mm=6.17,
        focal_mm_bounds=(40.0, 40.0),
        # Ensure inclination coloring visuals are enabled
        color_edgelets_by_inclination=True,
        max_edgelets_color_show=20000,
        max_edgelets_ds_show=4000,
        ds_line_samples=64,
        inclination_ranges = [(8*np.pi/180, 38*np.pi/180)]
    )
