import cv2
import numpy as np

def calculate_optical_flow(
    frames,
    levels=None,
    patch_size=None,
    iterations=None,
    dis_preset="FAST",
    variational_refinement=None,  # None means keep preset's decision
    finest_scale_max=5,
    refinement_iters=None,
    refinement_alpha=None,
    refinement_delta=None,
    refinement_gamma=None,
):
    """Compute dense optical flow sequence using OpenCV DISOpticalFlow.

    Override behavior: All tuning parameters default to None. When None, the corresponding
    preset-provided value is left untouched. Only explicitly supplied (non-None) values
    trigger a setter call, preserving the semantics of the chosen preset.

    Parameters:
      frames: list of BGR images.
      levels: logical scale depth (finestScale = clamp(levels-1)).
      patch_size: DIS patch size.
      iterations: gradient descent iterations per scale.
      dis_preset: one of {"ULTRAFAST","FAST","MEDIUM"}.
      variational_refinement: True/False to force enable/disable; None keeps preset.
      refinement_*: only applied if variational_refinement ends up enabled and the value is not None.
    Returns: list of flow arrays (H,W,2) between consecutive frames.
    """
    if len(frames) < 2:
        return []

    preset_map = {
        "ULTRAFAST": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
        "FAST": cv2.DISOPTICAL_FLOW_PRESET_FAST,
        "MEDIUM": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
    }
    preset_cv = preset_map.get(dis_preset.upper(), cv2.DISOPTICAL_FLOW_PRESET_FAST)
    dis = cv2.DISOpticalFlow_create(preset_cv)

    try:
        applied = {}
        if patch_size is not None:
            dis.setPatchSize(int(max(1, patch_size)))
            applied['patch_size'] = dis.getPatchSize()
        if iterations is not None:
            dis.setGradientDescentIterations(int(max(1, iterations)))
            applied['iterations'] = dis.getGradientDescentIterations()
        if levels is not None:
            finest = int(max(0, min(finest_scale_max, levels - 1)))
            dis.setFinestScale(finest)
            applied['finestScale'] = dis.getFinestScale()
        # Always keep spatial propagation & mean normalization (common useful defaults)
        dis.setUseSpatialPropagation(True)
        dis.setUseMeanNormalization(True)

        # Variational refinement handling
        if variational_refinement is not None:
            if variational_refinement:
                # Enable (OpenCV enables when iterations > 0)
                if refinement_iters is None:
                    refinement_iters = 5
                dis.setVariationalRefinementIterations(int(refinement_iters))
                if refinement_alpha is not None:
                    dis.setVariationalRefinementAlpha(float(refinement_alpha))
                if refinement_delta is not None:
                    dis.setVariationalRefinementDelta(float(refinement_delta))
                if refinement_gamma is not None:
                    dis.setVariationalRefinementGamma(float(refinement_gamma))
                applied['variational_refinement'] = 'enabled'
            else:
                # Disable by setting iterations to 0
                dis.setVariationalRefinementIterations(0)
                applied['variational_refinement'] = 'disabled'
        print(f"DIS overrides applied: {applied}" if applied else "DIS overrides: none (preset only)")
    except Exception as e:  # pragma: no cover - defensive
        print(f"WARNING configuring DISOpticalFlow: {e}")

    flows = []
    # Accept both BGR and already-grayscale frames
    first = frames[0]
    if first.ndim == 2:
        prev_gray = first
    else:
        prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    print(f"Calculating DIS optical flow (preset={dis_preset})")
    for i in range(1, len(frames)):
        f = frames[i]
        if f.ndim == 2:
            gray = f
        else:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        flow = dis.calc(prev_gray, gray, None)
        flows.append(flow)
        prev_gray = gray
    return flows

def flow_to_hsv(flow, brightness_increase=50):
    h, w = flow.shape[:2]
    fx, fy = flow[..., 0], flow[..., 1]
    magnitude, angle = cv2.cartToPolar(fx, fy)
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2][hsv[..., 2] != 0] = np.clip(hsv[..., 2][hsv[..., 2] != 0] + brightness_increase, 0, 255)
    return hsv

def optical_flow_to_motion_masks(flows, magnitude_threshold=2.0):
    masks = []
    for flow in flows:
        fx, fy = flow[..., 0], flow[..., 1]
        magnitude = np.sqrt(fx**2 + fy**2)
        mask = (magnitude > magnitude_threshold).astype(np.uint8) * 255
        masks.append(mask)
    return masks

def flow_direction_mask(flows, direction_degrees, tolerance_degrees=30, magnitude_threshold=2.0):
    masks = []
    target_rad = np.radians(direction_degrees)
    tolerance_rad = np.radians(tolerance_degrees)
    for flow in flows:
        fx, fy = flow[..., 0], flow[..., 1]
        magnitude = np.sqrt(fx**2 + fy**2)
        angle = np.arctan2(fy, fx)
        angle_diff = np.abs(angle - target_rad)
        angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)
        direction_mask = (angle_diff <= tolerance_rad) & (magnitude > magnitude_threshold)
        masks.append((direction_mask.astype(np.uint8) * 255))
    return masks

def hue_range_mask(hsv_flows, hue_min, hue_max, value_min=20):
    masks = []
    for hsv in hsv_flows:
        if hue_min <= hue_max:
            hue_mask = cv2.inRange(hsv[..., 0], hue_min, hue_max)
        else:
            mask1 = cv2.inRange(hsv[..., 0], hue_min, 179)
            mask2 = cv2.inRange(hsv[..., 0], 0, hue_max)
            hue_mask = cv2.bitwise_or(mask1, mask2)
        value_mask = cv2.inRange(hsv[..., 2], value_min, 255)
        mask = cv2.bitwise_and(hue_mask, value_mask)
        masks.append(mask)
    return masks