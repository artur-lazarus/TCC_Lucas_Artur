import cv2
import numpy as np

def extract_hsv_channels(frames):
    hue_frames, saturation_frames, value_frames = [], [], []
    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue_frames.append(hsv[:, :, 0])
        saturation_frames.append(hsv[:, :, 1])
        value_frames.append(hsv[:, :, 2])
    return hue_frames, saturation_frames, value_frames

def create_hsv_range_mask(frames, hue_range, sat_range=(50, 255), val_range=(50, 255)):
    masks = []
    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([hue_range[0], sat_range[0], val_range[0]])
        upper = np.array([hue_range[1], sat_range[1], val_range[1]])
        mask = cv2.inRange(hsv, lower, upper)
        masks.append(mask)
    return masks

def create_hue_mask(frames, target_hue, tolerance=10):
    hue_min = max(0, target_hue - tolerance)
    hue_max = min(179, target_hue + tolerance)
    return create_hsv_range_mask(frames, (hue_min, hue_max), (0, 255), (0, 255))

def create_hsv_comparison_image(hue_frames, saturation_frames, value_frames):
    comparison_frames = []
    for hue, sat, val in zip(hue_frames, saturation_frames, value_frames):
        h, w = hue.shape
        comparison = np.zeros((h, w * 3), dtype=np.uint8)
        comparison[:, 0:w] = hue
        comparison[:, w:2*w] = sat
        comparison[:, 2*w:3*w] = val
        comparison_frames.append(comparison)
    return comparison_frames