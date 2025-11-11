import cv2
import numpy as np

def build_median_background(frames):
    """Build a background by taking the median across time.

    Works for both color frames (H, W, C) and grayscale frames (H, W).
    """
    stack = np.stack(frames, axis=-1)
    return np.median(stack, axis=-1).astype(np.uint8)

def build_mean_background(frames):
    """Build a background by taking the mean across time.

    Works for both color frames (H, W, C) and grayscale frames (H, W).
    """
    stack = np.stack(frames, axis=-1)
    return np.mean(stack, axis=-1).astype(np.uint8)

def build_mode_background(frames, bins=256):
    """Build a background by taking the per-pixel mode across time.

    Supports both grayscale (H, W) and color (H, W, C) frames.
    Note: This implementation is simple and may be slow for large images.
    """
    stack = np.stack(frames, axis=-1)

    if stack.ndim == 3:
        # (H, W, N)
        h, w, n = stack.shape
        mode_bg = np.zeros((h, w), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                hist, bin_edges = np.histogram(stack[i, j, :], bins=bins, range=(0, 256))
                mode_bg[i, j] = bin_edges[np.argmax(hist)]
        return mode_bg
    else:
        # (H, W, C, N)
        h, w, c, n = stack.shape
        mode_bg = np.zeros((h, w, c), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    hist, bin_edges = np.histogram(stack[i, j, k, :], bins=bins, range=(0, 256))
                    mode_bg[i, j, k] = bin_edges[np.argmax(hist)]
        return mode_bg

def build_weighted_background(frames, weights=None):
    if weights is None:
        weights = [1.0 / len(frames)] * len(frames)
    weights = np.array(weights) / np.sum(weights)
    weighted_sum = np.zeros_like(frames[0], dtype=np.float64)
    for frame, weight in zip(frames, weights):
        weighted_sum += frame.astype(np.float64) * weight
    return weighted_sum.astype(np.uint8)

def build_percentile_background(frames, percentile=90):
    """Build a background by taking the given percentile across time.

    Works for both color frames (H, W, C) and grayscale frames (H, W).
    Time axis is the last axis.
    """
    if not frames:
        return None
    stack = np.stack(frames, axis=-1)
    return np.percentile(stack, percentile, axis=-1).astype(np.uint8)