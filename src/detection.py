import optical_flow
import numpy as np
import cv2
import tracking
from video_stream import video
import background
import time
import os

class Detection:
    def __init__(self):
        self._background: background.Background = None
        self.video_writer = None
        self.initial_frame_count = 0

    def insert_calibration(self, H_matrix, roi_polygon, H_out, W_out, lanes_y_pxs, scale_lambda, fps):
        self.H_matrix = H_matrix
        self.roi_polygon = roi_polygon
        self.H_out = H_out
        self.W_out = W_out
        self.lanes_y_pxs = lanes_y_pxs
        self.scale_lambda = scale_lambda
        self.fps = fps

    def start_tracker(self,scale_lambda, kalman_sigma_a, kalman_sigma_z,
                      max_association_distance, max_age, min_hits):
        self.tracker = tracking.Tracker(
            dt = 1.0 / self.fps,
            scale_lambda=scale_lambda,
            sigma_a=kalman_sigma_a,
            sigma_z=kalman_sigma_z,
            distance_threshold=max_association_distance,
            max_age=max_age,
            min_hits=min_hits,
        )

    def append_car_images_to_tracks(self, original_frame, frame_count):
        for t in self.tracker.tracks:
            if len(t.bboxes_with_framecount) == 0:
                continue
            if t.bboxes_with_framecount[-1][1] == frame_count:
                bbox_warped = t.bboxes_with_framecount[-1][0]
                x, y, w, h = bbox_warped
                
                # Transform bbox corners: [top-left, top-right, bottom-right, bottom-left]
                bbox_corners = np.array([[[x, y],
                                         [x + w, y],
                                         [x + w, y + h],
                                         [x, y + h]]], dtype=np.float32)
                bbox_original = cv2.perspectiveTransform(bbox_corners, np.linalg.inv(self.H_matrix))
                
                # Extract all x and y coordinates to find bounding rectangle
                xs = bbox_original[0, :, 0]
                ys = bbox_original[0, :, 1]
                x_min, x_max = int(np.floor(xs.min())), int(np.ceil(xs.max()))
                y_min, y_max = int(np.floor(ys.min())), int(np.ceil(ys.max()))
                
                # Clip to frame boundaries
                h_frame, w_frame = original_frame.shape[:2]
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(w_frame, x_max)
                y_max = min(h_frame, y_max)
                
                # Crop and append if valid
                if y_max > y_min and x_max > x_min:
                    original_frame_cropped = original_frame[y_min:y_max, x_min:x_max]
                    t.frames.append(original_frame_cropped)

    def process_frame(self, visualize=True):
        min_car_area_m2 = 11
        speed_limit_km_h = 65
        min_car_area_px = int((min_car_area_m2 / (self.scale_lambda ** 2)))

        if not hasattr(self, 'H_matrix'):
            print("ERROR: Calibration data not set. Call insert_calibration() first.")
            return None
        if not hasattr(self, 'tracker'):
            print("ERROR: Tracker not initialized. Call start_tracker() first.")
            return None
        if not self._background or self._background.loaded < self._background.size:
            print("ERROR: Background not initialized or not enough frames loaded. Call init_background() first.")
            return None
        if self.video_writer is None and visualize:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter('test_output/detection_debug/detection_output.mp4', fourcc, self.fps, (self.W_out, self.H_out))
        timec0 = time.perf_counter()
        frame_count, original_frame = video.get_frame()
        timec1 = time.perf_counter()
        print(f"Get frame time: {timec1 - timec0:.4f} seconds")
        calibrated_frame = cv2.warpPerspective(original_frame, self.H_matrix, (self.W_out, self.H_out),
                                               flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,
                                                borderValue=0)
        timec2 = time.perf_counter()
        print(f"Warp perspective time: {timec2 - timec1:.4f} seconds")
        self._background.update(calibrated_frame)
        timec3 = time.perf_counter()
        print(f"Background update time: {timec3 - timec2:.4f} seconds")

        for lane_y in self.lanes_y_pxs:
            cv2.line(calibrated_frame, (0, lane_y), (self.W_out, lane_y), (0), 1)
        timec4 = time.perf_counter()
        print(f"Draw lanes time: {timec4 - timec3:.4f} seconds")
        calibrated_frame_mask = fill_holes(self._background.background_subtract(calibrated_frame, threshold=16, normalize=False))
        timec5 = time.perf_counter()
        print(f"Background subtraction and fill holes time: {timec5 - timec4:.4f} seconds")
        output_image, bboxes, areas = detect_blobs(calibrated_frame_mask, min_area=min_car_area_px, draw_boxes=True)
        BL_corners = [(bbox[0], bbox[1]+bbox[3]) for bbox in bboxes] # (x, y+h)
        timec6 = time.perf_counter()
        print(f"Blob detection time: {timec6 - timec5:.4f} seconds")
        self.tracker.update(bboxes, frame_count)
        timec7 = time.perf_counter()
        print(f"Tracker update time: {timec7 - timec6:.4f} seconds")
        self.append_car_images_to_tracks(original_frame, frame_count)
        if self.tracker.new_finished_tracks:
            finished_tracks = self.tracker.retrieve_finished_tracks()
            for finished_track in finished_tracks:
                if finished_track[1]*self.scale_lambda*3.6 > speed_limit_km_h:
                    print(f"Speeding detected! Track ID {finished_track[0].id} average speed: {finished_track[1]*self.scale_lambda*3.6:.2f} km/h")
                    if not os.path.exists("test_output/speeding_cars"):
                        os.makedirs("test_output/speeding_cars")
                    if len(finished_track[0].frames) > 0:
                        for i in range(len(finished_track[0].frames)):
                            if not os.path.exists(f"test_output/speeding_cars/track_{finished_track[0].id}"):
                                os.makedirs(f"test_output/speeding_cars/track_{finished_track[0].id}")
                            cv2.imwrite(f"test_output/speeding_cars/track_{finished_track[0].id}/frame_{i}.png", finished_track[0].frames[i])
                else:
                    print(f"Track ID {finished_track[0].id} average speed: {finished_track[1]*self.scale_lambda*3.6:.2f} km/h")
                    finished_track[0].frames = []  # Clear frames to save memory
        timec8 = time.perf_counter()
        print(f"Process finished tracks time: {timec8 - timec7:.4f} seconds")
        if visualize:
            if self.initial_frame_count == 0:
                self.initial_frame_count = frame_count
            print("Frame count:", (frame_count-self.initial_frame_count)/5)
        
            frame_vis = cv2.cvtColor(calibrated_frame, cv2.COLOR_GRAY2BGR)
            for (x, y, w, h) in bboxes:
                p = (int(x), int(y + h))
                cv2.circle(frame_vis, p, 4, (0, 255, 255), 2, cv2.LINE_AA)
                _draw_tracks(frame_vis, self.tracker)
            self.video_writer.write(frame_vis)
            timec9 = time.perf_counter()
            print(f"Visualization time: {timec9 - timec8:.4f} seconds")
            print(f"Total frame processing time: {timec9 - timec0:.4f} seconds")
            return frame_vis
        else:
            return None
    
def _draw_track_trail(frame, track, max_len=25):
    """Draw a short trail of a track's recent positions."""
    pts = track.history[-max_len:]
    for a, b in zip(pts[:-1], pts[1:]):
        # Extract position (index 1) from history entry [frame_count, (x, y), vel_x]
        a_pos = a[1]  # (x, y) tuple
        b_pos = b[1]  # (x, y) tuple
        a = tuple(map(int, a_pos))
        b = tuple(map(int, b_pos))
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

def fill_holes(mask):
        h, w = mask.shape
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        im_flood = mask.copy()
        cv2.floodFill(im_flood, flood_mask, (0, 0), 255)
        im_flood_inv = cv2.bitwise_not(im_flood)
        return cv2.bitwise_or(mask, im_flood_inv)

def detect_blobs(mask, min_area, max_area=None, draw_boxes=True):
        output_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        valid_bboxes, valid_areas = [], []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            #if area>1000: print("Blob area:", area)
            if area >= min_area and (max_area is None or area <= max_area):
                x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                valid_bboxes.append((x, y, w, h))
                valid_areas.append(area)
        for bbox in valid_bboxes:
            x, y, w, h = bbox   
            if draw_boxes:
                cv2.rectangle(mask, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return output_img, valid_bboxes, valid_areas

def detect_blobs_multiple(masks, min_area, max_area=None, draw_boxes=True):
        output_images, all_bboxes, all_areas = [], [], []
        for mask in masks:
            output_img, valid_bboxes, valid_areas = detect_blobs(mask, min_area, max_area, draw_boxes)
            output_images.append(output_img)
            all_bboxes.append(valid_bboxes)
            all_areas.append(valid_areas)
        return output_images, all_bboxes, all_areas
