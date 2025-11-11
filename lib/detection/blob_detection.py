import cv2
import numpy as np

def detect_blobs_simple(mask, min_area=50, max_area=None):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    valid_centroids, valid_areas, valid_bboxes = [], [], []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area and (max_area is None or area <= max_area):
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            valid_centroids.append(centroids[i])
            valid_areas.append(area)
            valid_bboxes.append((x, y, w, h))
    return valid_centroids, valid_areas, valid_bboxes

def process_blob_detection_sequence(binary_masks, min_area=50, draw_boxes=True, draw_centroids=True, show_info=False):
    output_images, all_centroids, all_areas, all_bboxes = [], [], [], []
    for mask in binary_masks:
        output_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        centroids, areas, bboxes = detect_blobs_simple(mask, min_area)
        all_centroids.append(centroids)
        all_areas.append(areas)
        all_bboxes.append(bboxes)
        for i, (bbox, centroid) in enumerate(zip(bboxes, centroids)):
            x, y, w, h = bbox
            if draw_boxes:
                cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if draw_centroids:
                cx, cy = int(centroid[0]), int(centroid[1])
                cv2.circle(output_img, (cx, cy), 3, (255, 0, 0), -1)
        output_images.append(output_img)
    return output_images, all_centroids, all_areas, all_bboxes

def get_blob_statistics(areas_list, centroids_list):
    total_blobs = sum(len(areas) for areas in areas_list)
    frames_with_blobs = sum(1 for areas in areas_list if len(areas) > 0)
    all_areas = [area for areas in areas_list for area in areas]
    avg_area = np.mean(all_areas) if all_areas else 0
    max_area = np.max(all_areas) if all_areas else 0
    return {
        'total_blobs': total_blobs,
        'frames_with_blobs': frames_with_blobs,
        'num_frames': len(areas_list),
        'avg_blobs_per_frame': total_blobs / len(areas_list) if areas_list else 0,
        'avg_area': avg_area,
        'max_area': max_area
    }