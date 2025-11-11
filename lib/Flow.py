import cv2
import numpy as np
import lib.detection.utils as utils

def opticalFlow(
    frames,
    levels=None,
    patch_size=None,
    iterations=None,
    dis_preset="FAST",
    variational_refinement=None,
    finest_scale_max=5,
    refinement_iters=None,
    refinement_alpha=None,
    refinement_delta=None,
    refinement_gamma=None,
):
    """Compute dense optical flow sequence using OpenCV DISOpticalFlow (nullable overrides).

    Mirrors detection.optical_flow.calculate_optical_flow but kept here as a convenience wrapper.
    Only non-None parameters override preset values.
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
        dis.setUseSpatialPropagation(True)
        dis.setUseMeanNormalization(True)
        if variational_refinement is not None:
            if variational_refinement:
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
                dis.setVariationalRefinementIterations(0)
                applied['variational_refinement'] = 'disabled'
        print(f"Flow overrides applied: {applied}" if applied else "Flow overrides: none (preset only)")
    except Exception as e:  # pragma: no cover
        print(f"WARNING configuring DISOpticalFlow: {e}")

    flows = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for i in range(1, len(frames)):
        print(i)
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        flow = dis.calc(prev_gray, gray, None)
        flows.append(flow)
        prev_gray = gray
    return flows

def flow_to_bgr(flow, brightness_increase=50):
    h, w = flow.shape[:2]
    fx, fy = flow[...,0], flow[...,1]
    magnitude, angle = cv2.cartToPolar(fx, fy)
    
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[...,0] = angle * 180 / np.pi / 2  # Hue = direction
    hsv[...,1] = 255                      # Saturation
    hsv[...,2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Increase brightness for non-zero pixels
    hsv[...,2][hsv[...,2] != 0] = np.clip(hsv[...,2][hsv[...,2] != 0] + brightness_increase, 0, 255)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

if __name__ == "__main__":
    # Simple demo: compute flow once and save visualization
    video_path = "small_video.mp4"
    frames = utils.loadVideo(video_path, 120)
    if frames:
        # Example override: only patch_size provided, others inherit preset
        flows = opticalFlow(frames, patch_size=21, dis_preset="FAST")
        flow_imgs = [flow_to_bgr(f, 30) for f in flows]
        utils.saveColor(flow_imgs, "output_flow_demo.mp4")