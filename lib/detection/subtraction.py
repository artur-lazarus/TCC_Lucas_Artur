from email.mime import image
from tracking import Tracker
import time
import os
import cv2
from detection_api import Detection, BlobOps, MaskResult
import utils
import numpy as np
import matplotlib.pyplot as plt

# NOTE (RGB tracking overlay):
#   The tracking output videos now use color (isColor=True) and frames are
#   converted from grayscale to BGR before drawing overlays. Each track gets a
#   deterministic color based on its ID. Below every track ID we render the
#   horizontal speed (vx) in pixels/second, derived directly from the Kalman
#   state (vx component, since the motion model uses dt in seconds). The
#   background visual appearance stays grayscale because we only expand the
#   channels; no recoloring of underlying pixel intensities occurs.


def select_greedy_hue_window(counts, window_size=10):
    """
    Greedy window selection on a circular histogram (orientation folded).

    - counts: 1D array of length N (e.g., N=90 for hue orientation bins)
    - window_size: number of consecutive bins to allow in the window (circular)

    Algorithm (matches the requested behavior):
      1) Sort bins by descending count and add them one-by-one.
      2) Maintain the minimal circular arc covering the selected bins.
      3) Continue while that minimal arc length <= window_size.
      4) Stop when adding the next (smaller) bin would require expanding the arc
         beyond window_size — i.e., to fit that smaller bin you'd need to push a
         bigger one outside the window.

    Returns (start_idx, end_idx, chosen_indices):
      - start_idx/end_idx are inclusive indices in [0, N-1], circular; when
        start_idx <= end_idx, the window is [start_idx, end_idx]; if start_idx > end_idx
        it wraps: [start_idx..N-1] U [0..end_idx].
      - chosen_indices are the indices actually inside the final window.
    """
    counts = np.asarray(counts, dtype=float)
    N = counts.size
    if N == 0:
        return (0, -1, [])

    order = np.argsort(counts)[::-1]  # desc by weight
    selected = []

    def minimal_cover_arc_length(idxs):
        if not idxs:
            return 0, 0, 0  # length, start, end
        arr = np.sort(np.array(idxs))
        # Consider gaps between consecutive points on the circle; the maximum gap
        # determines the complementary minimal covering arc.
        diffs = np.diff(np.r_[arr, arr[0] + N])  # circular differences
        max_gap_idx = np.argmax(diffs)
        max_gap = diffs[max_gap_idx]
        # Minimal arc covers the rest of the circle
        length = N - max_gap
        # Window starts right after the biggest gap
        start = (arr[(max_gap_idx + 1) % arr.size]) % N
        end = (start + length - 1) % N
        return int(length), int(start), int(end)

    # Greedily add while the minimal cover arc fits the window
    best_len, best_start, best_end = 0, 0, -1
    for idx in order:
        trial = selected + [int(idx)]
        length, start, end = minimal_cover_arc_length(trial)
        if length <= window_size:
            selected = trial
            best_len, best_start, best_end = length, start, end
        else:
            # Adding this smaller bin would force arc > window_size => stop.
            break

    # Ensure selected bins actually sit inside the final window
    def within_window(i, s, e):
        return (s <= e and s <= i <= e) or (s > e and (i >= s or i <= e))
    chosen = [i for i in selected if within_window(i, best_start, best_end)]

    return best_start, best_end, chosen

def _draw_track_trail(frame, track, max_len=25):
    """Draw a short trail of a track's recent positions."""
    pts = track.history[-max_len:]
    for a, b in zip(pts[:-1], pts[1:]):
        a = tuple(map(int, a))
        b = tuple(map(int, b))
        cv2.line(frame, a, b, (255, 255, 255), 2, cv2.LINE_AA)

def _draw_tracks(frame, tracker):
    """Draw tracked objects with ID and horizontal speed (pixels/s).

    Background frame can be grayscale or BGR; ensure color drawing works.
    """
    for t in tracker.tracks:
        x, y = t.position()
        vx, _vy = t.velocity()  # vx already in pixels/second due to model using dt.
        pid = t.id
        # Deterministic vivid-ish color mapping by ID
        color = (
            (37 * (pid % 7) + 50) % 256,
            (83 * (pid % 5) + 50) % 256,
            (127 * (pid % 3) + 50) % 256,
        )
        p = (int(x), int(y))
        cv2.circle(frame, p, 6, color, -1, cv2.LINE_AA)

        # Prepare text lines: ID and speed below it
        id_text = f"ID {pid}"
        speed_text = f"vx: {vx:.1f} px/s"
        # Compute baseline offsets for neat stacking
        (id_size_w, id_size_h), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        # Draw ID slightly above point
        id_org = (p[0] + 8, p[1] - 8)
        speed_org = (id_org[0], id_org[1] + id_size_h + 4)  # 4px gap below ID
        cv2.putText(frame, id_text, id_org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        cv2.putText(frame, speed_text, speed_org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        _draw_track_trail(frame, t, max_len=25)

if __name__ == "__main__":
    #Flows initialized (preset=VERY_FAST) overrides={'dis_preset': 'VERY_FAST'} count=399
    #Initialized flows with preset VERY_FAST in 5.11 seconds.
    #Flows initialized (preset=FAST) overrides={'dis_preset': 'FAST'} count=399
    #Initialized flows with preset FAST in 5.56 seconds.
    #Flows initialized (preset=MEDIUM) overrides={'dis_preset': 'MEDIUM'} count=399
    #Initialized flows with preset MEDIUM in 9.78 seconds.

    
    # Notes:
    #  - Larger patch_size reduces noise but merges nearby motions.
    #  - More iterations can improve accuracy at cost of speed.
    #  - Hue range (e.g., 140-170) and value_min threshold tune motion masking.
    video_path = "test_video/session0_left_birdview.mp4"

    # Possibility to add consistency to flow?
    d = Detection(video_path, max_frames = 8000, color=False, frame_interval=5)
    #d.downscale(0.8)
    #d.set_framerate(2)
    d.init_background(method='median')
    mask=d.median_subtract_normalized(threshold_value=16).morphology.fill_holes()
    blob = BlobOps(mask)
    min_area = 1600
    save_filename = f"test_detection_birdview_framerate_10_median_16_fill_normalized_minarea_{min_area}_blob_detection.mp4"
    result = blob.save_detection(min_area=min_area, filename=save_filename)

    # Build 8 histograms of bbox corners: (TL, TR, BL, BR) for each coordinate (x, y)
    # bboxes is a list per frame -> flatten
    all_bboxes = [bbox for frame_bboxes in result.get('bboxes', []) for bbox in frame_bboxes]

    # Prepare containers
    corners = {
        'TL': {'x': [], 'y': []},
        'TR': {'x': [], 'y': []},
        'BL': {'x': [], 'y': []},
        'BR': {'x': [], 'y': []},
    }

    for (x, y, w, h) in all_bboxes:
        # Top-left
        corners['TL']['x'].append(x)
        corners['TL']['y'].append(y)
        # Top-right
        corners['TR']['x'].append(x + w)
        corners['TR']['y'].append(y)
        # Bottom-left
        corners['BL']['x'].append(x)
        corners['BL']['y'].append(y + h)
        # Bottom-right
        corners['BR']['x'].append(x + w)
        corners['BR']['y'].append(y + h)

    # Determine image dimensions for bin ranges if available
    bins = 50
    x_range = None
    y_range = None
    if mask.frames and len(mask.frames) > 0 and mask.frames[0] is not None:
        H, W = mask.frames[0].shape[:2]
        x_range = (0, W)
        y_range = (0, H)

    # Plot histograms
    fig, axs = plt.subplots(2, 4, figsize=(18, 8))
    titles = [
        ('TL.x', 'TL.y'),
        ('TR.x', 'TR.y'),
        ('BL.x', 'BL.y'),
        ('BR.x', 'BR.y'),
    ]

    data_pairs = [
        (corners['TL']['x'], corners['TL']['y']),
        (corners['TR']['x'], corners['TR']['y']),
        (corners['BL']['x'], corners['BL']['y']),
        (corners['BR']['x'], corners['BR']['y']),
    ]

    tl_y_counts = None
    tl_y_bin_edges = None
    tl_y_peaks = []

    for col, ((dx, dy), (tx, ty)) in enumerate(zip(data_pairs, titles)):
        # X histogram on first row
        axs[0, col].hist(dx, bins=bins, range=x_range, color='steelblue', alpha=0.8)
        axs[0, col].set_title(tx)
        axs[0, col].set_xlabel('x')
        axs[0, col].set_ylabel('count')
        # Y histogram on second row; for TL.y capture raw counts for peak finding
        hist_vals, bin_edges, _ = axs[1, col].hist(dy, bins=bins, range=y_range, color='indianred', alpha=0.8)
        axs[1, col].set_title(ty)
        axs[1, col].set_xlabel('y')
        axs[1, col].set_ylabel('count')
        if ty == 'TL.y':
            tl_y_counts = hist_vals
            tl_y_bin_edges = bin_edges

    # Peak detection for TL.y histogram (distinct distributions)
    if tl_y_counts is not None and tl_y_bin_edges is not None and tl_y_counts.sum() > 0:
        # Smooth counts with a small binomial kernel (approx Gaussian)
        kernel = np.array([1, 4, 6, 4, 1], dtype=float)
        kernel = kernel / kernel.sum()
        padded = np.pad(tl_y_counts, (len(kernel)//2, len(kernel)//2), mode='edge')
        smooth = np.convolve(padded, kernel, mode='valid')

        # Debug plot: raw (original), padded, smooth
        try:
            centers = 0.5 * (tl_y_bin_edges[:-1] + tl_y_bin_edges[1:])
            fig_dbg, axes_dbg = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
            axes_dbg[0].bar(centers, tl_y_counts, width=(centers[1]-centers[0]) if len(centers) > 1 else 1.0, color='gray', alpha=0.8)
            axes_dbg[0].set_title('TL.y raw counts')
            axes_dbg[0].set_ylabel('count')
            axes_dbg[1].bar(np.arange(len(padded)), padded, color='slateblue', alpha=0.8)
            axes_dbg[1].set_title('TL.y padded counts (index space)')
            axes_dbg[1].set_ylabel('padded')
            axes_dbg[2].plot(centers, smooth, color='orange', linewidth=2)
            axes_dbg[2].set_title('TL.y smoothed counts')
            axes_dbg[2].set_xlabel('y')
            axes_dbg[2].set_ylabel('smooth')
        except Exception as e:
            print('Debug plot generation failed:', e)

        # Local maxima candidates where slope changes from + to -
        candidates = [i for i in range(1, len(smooth)-1) if smooth[i] > smooth[i-1] and smooth[i] >= smooth[i+1]]

        # Windowed prominence against nearby minima to avoid splitting a single distribution
        n = len(smooth)
        window = max(3, int(0.01 * n))
        rng = smooth.max() - smooth.min()
        min_prominence = max(5.0, 0.01 * rng)
        min_separation_bins = max(3, int(0.04 * n))

        print("Candidate peaks at indices:", candidates)
        scored = []
        for i in candidates:
            left_start = max(0, i - window)
            right_end = min(n, i + 1 + window)
            left_min = smooth[left_start:i].min() if i - left_start > 0 else smooth[i]
            right_min = smooth[i+1:right_end].min() if right_end - (i+1) > 0 else smooth[i]
            base = max(left_min, right_min)
            prominence = smooth[i] - base
            if prominence >= min_prominence:
                scored.append((i, prominence, smooth[i]))

        # Keep well-separated strongest peaks
        scored.sort(key=lambda t: (t[1], t[2]), reverse=True)
        accepted = []
        for idx, prom, height in scored:
            if all(abs(idx - a) >= min_separation_bins for a in accepted):
                accepted.append(idx)
        accepted.sort()

        # Convert bin indices to y coordinate centers
        for idx in accepted:
            y_center = 0.5 * (tl_y_bin_edges[idx] + tl_y_bin_edges[idx+1])
            tl_y_peaks.append(y_center)

        # Annotate TL.y subplot (second row, first column)
        tl_ax = axs[1, 0]
        for peak_y in tl_y_peaks:
            tl_ax.axvline(peak_y, color='gold', linestyle='--', linewidth=1.5)
        if tl_y_peaks:
            tl_ax.text(0.02, 0.95, f'Peaks: {", ".join(f"{p:.1f}" for p in tl_y_peaks)}', transform=tl_ax.transAxes,
                       fontsize=10, color='gold', verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.3))
        print('TL.y peaks:', tl_y_peaks)

        # Black out the rows at peak indices in the mask and rerun blob detection
        peak_rows = [int(round(p)) for p in tl_y_peaks]
        # Clamp rows within image bounds
        if mask.frames and len(mask.frames) > 0 and mask.frames[0] is not None:
            H, W = mask.frames[0].shape[:2]
            peak_rows = [r for r in peak_rows if 0 <= r < H]
        if peak_rows:
            new_masks = []
            for m in mask.masks:
                if m is None:
                    new_masks.append(m)
                    continue
                mm = m.copy()
                for r in peak_rows:
                    try:
                        mm[r, :] = 0
                    except Exception:
                        pass
                new_masks.append(mm)
            lanes_mask = MaskResult(new_masks, frames=mask.frames)
            blob_lanes = BlobOps(lanes_mask)
            name, ext = os.path.splitext(save_filename)
            save_filename_lanes = f"{name}_lanes{ext}"
            print(f"Saving lanes-suppressed blobs to: {save_filename_lanes}, min_area={min_area}")
            result_lanes = blob_lanes.save_detection(min_area=min_area, filename=save_filename_lanes)
            
    else:
        print('No TL.y data for peak detection.')

    plt.tight_layout()
    plt.show()

    tracker = Tracker(
        dt=1/30,              # set to your frame period
        sigma_a=8.0,          # tweak: higher for more jittery motion
        sigma_z=4.0,          # tweak: measurement noise (pixels)
        distance_threshold=60,# tweak: max association distance (pixels)
        max_age=8,            # frames to keep an unobserved track
        min_hits=2            # suppress 1-frame ghosts
    )

    fps_out = 10.0
    dt = 1.0 / fps_out

    # Guard: need original frames to overlay
    if not (hasattr(mask, "frames") and mask.frames and mask.frames[0] is not None):
        raise RuntimeError("mask.frames is empty; need original frames to draw overlays.")

    H, W = mask.frames[0].shape[:2]

    # Filenames as requested
    tracked_name = f"test_detection_birdview_framerate_10_median_16_fill_normalized_minarea_{min_area}_blob_detection_tracked.mp4"
    tracked_lanes_name = f"test_detection_birdview_framerate_10_median_16_fill_normalized_minarea_{min_area}_blob_detection_lanes_tracked.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # Switch to color output so per-track overlays are colorful while keeping background gray.
    # We'll convert frames to BGR before drawing overlays.
    writer_tracked = cv2.VideoWriter(tracked_name, fourcc, fps_out, (W, H), isColor=True)

    # Build one tracker for the normal detection run
    tracker = Tracker(
        dt=dt,
        sigma_a=8.0,
        sigma_z=4.0,
        distance_threshold=60,
        max_age=8,
        min_hits=2
    )

    # We'll iterate over the same frame count used in result['bboxes']
    bboxes_per_frame = result.get('bboxes', [])
    num_frames = min(len(mask.frames), len(bboxes_per_frame))

    for i in range(num_frames):
        bbox_frame = bboxes_per_frame[i] if i < len(bboxes_per_frame) else []
        cars = Tracker.bboxes_to_points(bbox_frame)  # [(x, y+h)]
        tracker.update(cars)

        # Draw on a copy of the ORIGINAL frame (not the mask)
        frame_vis = mask.frames[i].copy()
        # Ensure 3-channel BGR for colored drawing if source is grayscale
        if len(frame_vis.shape) == 2 or frame_vis.shape[2] == 1:
            frame_vis = cv2.cvtColor(frame_vis, cv2.COLOR_GRAY2BGR)
        # Optional: draw raw detection points (small hollow circles) for reference
        for (x, y, w, h) in bbox_frame:
            p = (int(x), int(y + h))
            cv2.circle(frame_vis, p, 4, (0, 255, 255), 2, cv2.LINE_AA)

        _draw_tracks(frame_vis, tracker)
        writer_tracked.write(frame_vis)

    writer_tracked.release()
    print(f"Saved: {tracked_name}")

    # ====== Optional: lanes-suppressed tracking video (if available) ======
    if 'result_lanes' in locals() and result_lanes is not None and 'bboxes' in result_lanes:
        writer_tracked_lanes = cv2.VideoWriter(tracked_lanes_name, fourcc, fps_out, (W, H), isColor=True)

        tracker_lanes = Tracker(
            dt=dt,
            sigma_a=8.0,
            sigma_z=4.0,
            distance_threshold=60,
            max_age=8,
            min_hits=2
        )

        bboxes_per_frame_lanes = result_lanes.get('bboxes', [])
        num_frames_lanes = min(len(mask.frames), len(bboxes_per_frame_lanes))

        for i in range(num_frames_lanes):
            bbox_frame = bboxes_per_frame_lanes[i] if i < len(bboxes_per_frame_lanes) else []
            cars = Tracker.bboxes_to_points(bbox_frame)
            tracker_lanes.update(cars)

            frame_vis = mask.frames[i].copy()
            if len(frame_vis.shape) == 2 or frame_vis.shape[2] == 1:
                frame_vis = cv2.cvtColor(frame_vis, cv2.COLOR_GRAY2BGR)
            # Optional: show the lanes-suppressed detections too
            for (x, y, w, h) in bbox_frame:
                p = (int(x), int(y + h))
                cv2.circle(frame_vis, p, 4, (0, 200, 255), 2, cv2.LINE_AA)

            _draw_tracks(frame_vis, tracker_lanes)
            writer_tracked_lanes.write(frame_vis)

        writer_tracked_lanes.release()
        print(f"Saved: {tracked_lanes_name}")
    else:
        print("No lanes-suppressed result available; skipping the _lanes_tracked.mp4.")



    

    #for i in [2,4,8,16,32]:
    #    for j in [50,100,200,400,800]:
    #        mask=d.median_subtract_normalized(threshold_value=i).morphology.fill_holes().save(f"test_detection_birdview_median_{i}_fill_normalized.mp4")
    #        blob = BlobOps(mask)
    #        blob.save_detection(min_area=j, filename = f"test_detection_birdview_median_{i}_fill_normalized_minarea_{j}_blob_detection.mp4")

    #for k in ["VERY_FAST","FAST","MEDIUM"]:
    #for k in ["VERY FAST"]:
    #    init_flows_start_time = time.perf_counter()
    #    d.init_flows(dis_preset=k)
    #    print(f"Initialized flows with preset {k} in {time.perf_counter() - init_flows_start_time:.2f} seconds.")
    #    for j in [2,4,8,16,32]:
    #        for i,f in [[70,100],[80,100],[70,90],[80,90]]:
    #            flow_subtract_start_time = time.perf_counter()
    #            d.flow_subtract(hue_range=(i,f), value_min=j)#.save(f"testFlowBird_hue_range_{i}_{f}_value_min_{j}_preset_{k}.mp4")
    #            print(f"  Flow subtraction with hue_range=({i},{f}), value_min={j} took {time.perf_counter() - flow_subtract_start_time:.2f} seconds.")
    #d.flow_subtract(hue_range=(140,170),value_min=12).save("testFlow_final1.mp4")

    # Median subtraction threshold between 10 and 20 seems to work best
    #print("time before anything= " + str(time.perf_counter()))



    #roi = cv2.imread("ROI_background_percentile_80_big.jpg")
    #d.init_background(method='percentile', percentile=50)
    #d.init_flows(dis_preset="MEDIUM")
    #print("time after init flows= " + str(time.perf_counter()))



    # Lower quality
    #d.downscale(0.8)
    #d.set_framerate(2)


    

    # Save the full optical-flow HSV visualization as color video
    #if d._hsv_flows is not None and len(d._hsv_flows) > 0:
    #    hsv_bgr_frames = [cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) for hsv in d._hsv_flows]
    #    utils.saveColor(hsv_bgr_frames, video_path.split("/")[-2] + "_optical_flow_medium_dis.mp4")
    #print("time after saving hsv flow video= " + str(time.perf_counter()))
    # Subtraction
    #d.flow_subtract(hue_range=(160,170),value_min=50).save("test_detection_topdown_flow_160_170.mp4")
    #d.median_subtract(threshold_value=20).save("test_detection_topdown_median_20.mp4")
    #d.flow_subtract(hue_range=(140,170),value_min=30).and_(d.median_subtract(threshold_value=14).morphology.fill_holes()).save("test_final_with_warp.mp4")

    # Save background
    #cv2.imwrite("ROI_background_percentile_80_big.jpg", d._background)

    