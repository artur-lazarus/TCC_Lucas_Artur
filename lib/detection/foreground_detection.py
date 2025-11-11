import cv2
import numpy as np

def detect_foreground_value_based(frames, background_image, threshold_value=15):
    """
    Foreground by absolute difference on the value/brightness channel.

    Supports both inputs:
      - Color (BGR) frames and background: uses HSV V channel.
      - Grayscale frames and background: uses the grayscale directly.
    """
    # Determine background value image
    if background_image.ndim == 2:
        background_value = background_image
    else:
        bg_hsv = cv2.cvtColor(background_image, cv2.COLOR_BGR2HSV)
        background_value = bg_hsv[:, :, 2]

    foreground_masks = []
    for frame in frames:
        if frame.ndim == 2:
            value = frame
        else:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            value = hsv[:, :, 2]
        diff = cv2.absdiff(background_value, value)
        _, mask = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
        mask = cv2.medianBlur(mask, 5)
        foreground_masks.append(mask)
    return foreground_masks

def detect_foreground_value_separated(frames, background_image, threshold_value=15):
    # Background V
    if background_image.ndim == 2:
        background_value = background_image
    else:
        bg_hsv = cv2.cvtColor(background_image, cv2.COLOR_BGR2HSV)
        background_value = bg_hsv[:, :, 2]
    over_masks, under_masks, combined_masks = [], [], []
    for frame in frames:
        if frame.ndim == 2:
            value = frame
        else:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            value = hsv[:, :, 2]
        diff_signed = value.astype(np.int16) - background_value.astype(np.int16)
        over_mask = np.where(diff_signed > threshold_value, 255, 0).astype(np.uint8)
        under_mask = np.where(diff_signed < -threshold_value, 255, 0).astype(np.uint8)
        combined_mask = cv2.bitwise_or(over_mask, under_mask)
        over_masks.append(cv2.medianBlur(over_mask, 5))
        under_masks.append(cv2.medianBlur(under_mask, 5))
        combined_masks.append(cv2.medianBlur(combined_mask, 5))
    return over_masks, under_masks, combined_masks

def detect_foreground_rgb_based(frames, background_image, threshold_value=15):
    foreground_masks = []
    for frame in frames:
        diff_b = cv2.absdiff(frame[:, :, 0], background_image[:, :, 0])
        diff_g = cv2.absdiff(frame[:, :, 1], background_image[:, :, 1])
        diff_r = cv2.absdiff(frame[:, :, 2], background_image[:, :, 2])
        _, mask_b = cv2.threshold(diff_b, threshold_value, 255, cv2.THRESH_BINARY)
        _, mask_g = cv2.threshold(diff_g, threshold_value, 255, cv2.THRESH_BINARY)
        _, mask_r = cv2.threshold(diff_r, threshold_value, 255, cv2.THRESH_BINARY)
        combined_mask = cv2.bitwise_or(cv2.bitwise_or(mask_b, mask_g), mask_r)
        foreground_masks.append(cv2.medianBlur(combined_mask, 5))
    return foreground_masks

def detect_foreground_gaussian_mixture(frames, learning_rate=0.01, threshold=16):
    backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=False, varThreshold=threshold)
    foreground_masks = []
    for frame in frames:
        # OpenCV handles both grayscale and color
        fgMask = backSub.apply(frame, learningRate=learning_rate)
        foreground_masks.append(fgMask)
    return foreground_masks

def detect_foreground_knn(frames, learning_rate=0.01, threshold=400):
    backSub = cv2.createBackgroundSubtractorKNN(detectShadows=False, dist2Threshold=threshold)
    foreground_masks = []
    for frame in frames:
        # OpenCV handles both grayscale and color
        fgMask = backSub.apply(frame, learningRate=learning_rate)
        foreground_masks.append(fgMask)
    return foreground_masks

def detect_foreground_value_based_normalized(
    frames,
    background_image,
    threshold_value=15,
    robust=True,
    percentiles=(10, 90),
):
    """
    Value-channel foreground detection with per-frame illumination normalization.

    Normalizes global lighting in each frame before subtraction so gradual or sudden
    brightness changes (e.g. clouds) reduce false positives.

    Strategy:
      If robust=True: perform percentile-based linear scaling of the frame V channel
      to match the background V channel distribution between p_low and p_high.
      Else: simple median-based gain correction.

    Args:
      frames: list of BGR frames.
      background_image: single BGR background frame (already built by init_background).
      threshold_value: absolute difference threshold applied after normalization.
      robust: whether to use percentile scaling (recommended).
      percentiles: (low, high) percentiles for robust scaling.

    Returns:
      List of uint8 foreground masks (255=foreground, 0=background).
    """
    # Background V (supports grayscale background)
    if background_image.ndim == 2:
        bg_v = background_image.astype(np.float32)
    else:
        bg_hsv = cv2.cvtColor(background_image, cv2.COLOR_BGR2HSV)
        bg_v = bg_hsv[:, :, 2].astype(np.float32)
    p_low, p_high = percentiles
    # Precompute robust stats for background
    if robust:
        bg_v_low = np.percentile(bg_v, p_low)
        bg_v_high = np.percentile(bg_v, p_high)
        bg_v_range = max(1.0, bg_v_high - bg_v_low)
        bg_v_median = np.median(bg_v)
    else:
        bg_v_median = np.median(bg_v)

    masks = []
    for frame in frames:
        if frame.ndim == 2:
            v = frame.astype(np.float32)
        else:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            v = hsv[:, :, 2].astype(np.float32)
        if robust:
            v_low = np.percentile(v, p_low)
            v_high = np.percentile(v, p_high)
            v_range = max(1.0, v_high - v_low)
            # Linear scale into background percentile span
            v_norm = (v - v_low) * (bg_v_high - bg_v_low) / v_range + bg_v_low
        else:
            v_median = np.median(v)
            gain = bg_v_median / max(1.0, v_median)
            v_norm = v * gain

        # Clip and convert back to uint8
        v_norm = np.clip(v_norm, 0, 255).astype(np.uint8)
        diff = cv2.absdiff(bg_v.astype(np.uint8), v_norm)
        _, mask = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
        mask = cv2.medianBlur(mask, 5)
        masks.append(mask)

    return masks