import cv2
import numpy as np
import numba

def calculate_optical_flow_multiple(
    frames,
    dis_preset="FAST"):
    """Compute dense optical flow sequence using OpenCV DISOpticalFlow."""
    if len(frames) < 2:
        return []
    
    preset_map = {
        "ULTRAFAST": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
        "FAST": cv2.DISOPTICAL_FLOW_PRESET_FAST,
        "MEDIUM": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
    }

    preset_cv = preset_map.get(dis_preset.upper(), cv2.DISOPTICAL_FLOW_PRESET_FAST)
    dis = cv2.DISOpticalFlow_create(preset_cv)

    flows = []
    prev = frames[0]
    for i in range(1, len(frames)):
        f = frames[i]
        flow = dis.calc(prev, f, None)
        flows.append(flow)
        prev = f

    return flows

def calculate_optical_flow(first_frame, second_frame, dis_preset="FAST"):
    """Compute dense optical flow between two frames using OpenCV DISOpticalFlow."""
    preset_map = {
        "ULTRAFAST": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
        "FAST": cv2.DISOPTICAL_FLOW_PRESET_FAST,
        "MEDIUM": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
    }

    preset_cv = preset_map.get(dis_preset.upper(), cv2.DISOPTICAL_FLOW_PRESET_FAST)
    dis = cv2.DISOpticalFlow_create(preset_cv)

    flow = dis.calc(first_frame, second_frame, None)
    return flow

def flow_to_polar(flow): 
    fx, fy = flow[..., 0], flow[..., 1] 
    magnitude, angle = cv2.cartToPolar(fx, fy) 
    return magnitude, angle

def flow_to_polar_multiple(flows):
    flows_polar = []
    for flow in flows:
        mag, ang = flow_to_polar(flow)
        flows_polar.append((mag, ang))
    return flows_polar

def flow_subtract(flow_polar, direction_range, threshold, save=False):
        dir_min, dir_max = direction_range
        if dir_min>dir_max:
            dir_min, dir_max = dir_max, dir_min
        dir_mask = cv2.inRange(flow_polar[1], dir_min, dir_max)
        mag_mask = cv2.inRange(flow_polar[0], threshold, 1e6)
        cv2.imwrite("test_output/direction_mask.png", dir_mask)
        # cv2.waitKey(1)
        cv2.imwrite("test_output/magnitude_mask.png", mag_mask)
        # cv2.waitKey(1)
        combined_mask = cv2.bitwise_and(dir_mask, mag_mask)
        return combined_mask


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

if __name__ == "__main__":
    import video_stream

    test_video = video_stream.VideoStream()
    test_video.set_config("dataset/session0_left/video.avi", None, colour=False)
    frames = [test_video.get_frame()[1]]
    for i in range(10):
        _, frame = test_video.get_frame()
        frames.append(frame)
        calculate_optical_flow(frames[i], frames[i+1])
