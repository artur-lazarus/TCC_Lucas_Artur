import cv2
import numpy as np

def mask_and(mask1, mask2):
    return cv2.bitwise_and(mask1, mask2)

def mask_or(mask1, mask2):
    return cv2.bitwise_or(mask1, mask2)

def mask_xor(mask1, mask2):
    return cv2.bitwise_xor(mask1, mask2)

def mask_subtract(mask1, mask2):
    return cv2.bitwise_and(mask1, cv2.bitwise_not(mask2))

def mask_not(mask):
    return cv2.bitwise_not(mask)

def masks_and_pairwise(masks1, masks2):
    if isinstance(masks2, np.ndarray):
        m2 = masks2
        if m2.ndim == 3 and m2.shape[2] == 3:
            m2 = cv2.cvtColor(m2, cv2.COLOR_BGR2GRAY)
        h, w = masks1[0].shape[:2]
        if m2.shape[:2] != (h, w):
            m2 = cv2.resize(m2, (w, h), interpolation=cv2.INTER_NEAREST)
        return [mask_and(m, m2) for m in masks1]
    min_len = min(len(masks1), len(masks2))
    return [mask_and(masks1[i], masks2[i]) for i in range(min_len)]

def masks_or_pairwise(masks1, masks2):
    min_len = min(len(masks1), len(masks2))
    return [mask_or(masks1[i], masks2[i]) for i in range(min_len)]

def masks_xor_pairwise(masks1, masks2):
    min_len = min(len(masks1), len(masks2))
    return [mask_xor(masks1[i], masks2[i]) for i in range(min_len)]

def masks_subtract_pairwise(masks1, masks2):
    min_len = min(len(masks1), len(masks2))
    return [mask_subtract(masks1[i], masks2[i]) for i in range(min_len)]

def masks_weighted_combination(masks1, masks2, weight1=0.5, weight2=0.5):
    min_len = min(len(masks1), len(masks2))
    combined = []
    for i in range(min_len):
        weighted = cv2.addWeighted(masks1[i], weight1, masks2[i], weight2, 0)
        _, thresholded = cv2.threshold(weighted, 127, 255, cv2.THRESH_BINARY)
        combined.append(thresholded)
    return combined

def masks_area_list(masks):
    return [np.sum(mask == 255) for mask in masks]

def masks_centroid_list(masks):
    centroids = []
    for mask in masks:
        moments = cv2.moments(mask)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            centroids.append((cx, cy))
        else:
            centroids.append((0, 0))
    return centroids