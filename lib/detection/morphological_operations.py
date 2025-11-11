import cv2
import numpy as np

def apply_erosion(masks, kernel_size=(3, 3), iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    return [cv2.erode(mask, kernel, iterations=iterations) for mask in masks]

def apply_dilation(masks, kernel_size=(3, 3), iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    return [cv2.dilate(mask, kernel, iterations=iterations) for mask in masks]

def apply_opening(masks, kernel_size=(4, 4)):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    return [cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) for mask in masks]

def apply_closing(masks, kernel_size=(4, 4)):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    return [cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) for mask in masks]

def apply_opening_then_closing(masks, kernel_size=(4, 4)):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    result = []
    for mask in masks:
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        result.append(closed)
    return result

def remove_small_components(masks, min_area=50):
    filtered = []
    for mask in masks:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        output = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                output[labels == i] = 255
        filtered.append(output)
    return filtered

def apply_median_filter(masks, kernel_size=5):
    return [cv2.medianBlur(mask, kernel_size) for mask in masks]

def fill_enclosed_regions(masks):
    filled = []
    for mask in masks:
        h, w = mask.shape
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        im_flood = mask.copy()
        cv2.floodFill(im_flood, flood_mask, (0, 0), 255)
        im_flood_inv = cv2.bitwise_not(im_flood)
        filled.append(cv2.bitwise_or(mask, im_flood_inv))
    return filled