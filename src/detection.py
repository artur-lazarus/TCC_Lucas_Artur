import optical_flow
import numpy as np
import cv2
import tracking

class Detection:
    def __init__(self):
        self._flows = None
        self._flows_polar = None
        self._background = None

    def insert_calibration(self, H_matrix, roi_polygon, H_out, W_out, lanes_y_pxs, fps):
        self.H_matrix = H_matrix
        self.roi_polygon = roi_polygon
        self.H_out = H_out
        self.W_out = W_out
        self.lanes_y_pxs = lanes_y_pxs
        self.fps = fps

    def start_tracker(self,kalman_sigma_a, kalman_sigma_z,
                      max_association_distance, max_age, min_hits):
        self.tracker = tracking.Tracker(
            dt = 1.0 / self.fps,
            sigma_a=kalman_sigma_a,
            sigma_z=kalman_sigma_z,
            distance_threshold=max_association_distance,
            max_age=max_age,
            min_hits=min_hits
        )

    def process_frame(self, frame, visualize=True):
        min_car_area = 1600

        if not hasattr(self, 'H_matrix'):
            print("ERROR: Calibration data not set. Call insert_calibration() first.")
            return None
        if not hasattr(self, 'tracker'):
            print("ERROR: Tracker not initialized. Call start_tracker() first.")
            return None
        calibrated_frame = cv2.warpPerspective(
            frame,
            np.array(self.H_matrix),
            (self.W_out, self.H_out),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )


        for lane_y in self.lanes_y_pxs:
            cv2.line(calibrated_frame, (0, lane_y), (self.W_out, lane_y), (0), 1)
        
        calibrated_frame_mask = self.fill_holes(self.background_subtract(calibrated_frame, threshold=16, normalize=True))
        output_image, bboxes, areas = self.detect_blobs(calibrated_frame_mask, min_area=min_car_area, draw_boxes=True)
        BL_corners = [(bbox[0], bbox[1]+bbox[3]) for bbox in bboxes] # (x, y+h)

        self.tracker.update(BL_corners)

        if visualize:
            frame_vis = cv2.cvtColor(calibrated_frame, cv2.COLOR_GRAY2BGR)
            for (x, y, w, h) in bboxes:
                p = (int(x), int(y + h))
                cv2.circle(frame_vis, p, 4, (0, 255, 255), 2, cv2.LINE_AA)
                _draw_tracks(frame_vis, self.tracker)
            return frame_vis
        else:
            return None

    def init_flows(self, frames, dis_preset="FAST"):
        # Initialize optical flows for the given frames
        self._flows = optical_flow.calculate_optical_flow_multiple(frames, dis_preset=dis_preset)
        self._flows_polar = optical_flow.flow_to_polar_multiple(self._flows)

    def init_background_populated(self, frames):
        self.init_background(frames[0].shape[1], frames[0].shape[0], window_size=len(frames))
        for frame in frames:
            self._background.update(frame)
    
    def init_background(self, W, H, window_size):
        self._background = Background(W, H, size=window_size)
        
    def get_background(self, percentile):
        if self._background is None:
            print("ERROR: Background not initialized. Call init_background() first.")
            return None
        return self._background.get_background_percentile(percentile)
    
    def flow_subtract(self, index, direction_range, threshold,save=False):
        if self._flows is None or self._flows_polar is None:
            print("ERROR: Flows not initialized. Call init_flows() first.")
            return None
        dir_min, dir_max = direction_range
        if dir_min>dir_max:
            dir_min, dir_max = dir_max, dir_min
        dir_mask = cv2.inRange(self._flows_polar[index][1], dir_min, dir_max)
        mag_mask = cv2.inRange(self._flows_polar[index][0], threshold, 1e6)
        combined_mask = cv2.bitwise_and(dir_mask, mag_mask)
        return combined_mask
    
    def flow_subtract_multiple(self, n_frames, direction_range, threshold, save=False):
        masks = []
        for index in range(n_frames):
            print("Processing flow subtract for frame index:", index)
            mask = self.flow_subtract(index, direction_range, threshold,save)
            masks.append(mask)
        return masks
    
    def background_subtract(self, frame, threshold, subtract_percentile = 50, normalize=False, norm_percentiles=(10,90)):
        if self._background is None:
            print("ERROR: Background not initialized. Call init_background() first.")
            return None
        bg_v = self._background.get_background_percentile(subtract_percentile).astype(np.float32)
        v = frame.astype(np.float32)
        
        
        if normalize:
            p_low, p_high = norm_percentiles
            bg_v_low = np.percentile(bg_v, p_low)
            bg_v_high = np.percentile(bg_v, p_high)
            v_low = np.percentile(v, p_low)
            v_high = np.percentile(v, p_high)
            v_range = max(1.0, v_high - v_low)
            v_norm = (v - v_low) * (bg_v_high - bg_v_low) / v_range + bg_v_low
        else:
            v_norm = v

        v_norm = np.clip(v_norm, 0, 255).astype(np.uint8)
        print("AAAAAAAA. bg_v size:", bg_v.shape, " v_norm size:", v_norm.shape)
        diff = cv2.absdiff(bg_v.astype(np.uint8), v_norm)
        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.medianBlur(mask, 5)  
        return mask
    
    def background_subtract_multiple(self, frames, threshold, normalize=False, percentiles=(10,90)):
        masks = []
        for frame in frames:
            mask = self.background_subtract(frame=frame, threshold=threshold, normalize=normalize, norm_percentiles=percentiles)
            masks.append(mask)
        return masks
    
    def fill_holes(self, mask):
        h, w = mask.shape
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        im_flood = mask.copy()
        cv2.floodFill(im_flood, flood_mask, (0, 0), 255)
        im_flood_inv = cv2.bitwise_not(im_flood)
        return cv2.bitwise_or(mask, im_flood_inv)
    
    def fill_holes_multiple(self, masks):
        filled = []
        for mask in masks:
            filled_mask = self.fill_holes(mask)
            filled.append(filled_mask)
        return filled
    
    def detect_blobs(self, mask, min_area, max_area=None, draw_boxes=True):
        output_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        valid_bboxes, valid_areas = [], []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area and (max_area is None or area <= max_area):
                x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                valid_bboxes.append((x, y, w, h))
                valid_areas.append(area)
        for bbox in valid_bboxes:
            x, y, w, h = bbox   
            if draw_boxes:
                cv2.rectangle(mask, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return output_img, valid_bboxes, valid_areas

    def detect_blobs_multiple(self, masks, min_area, max_area=None, draw_boxes=True):
        output_images, all_bboxes, all_areas = [], [], []
        for mask in masks:
            output_img, valid_bboxes, valid_areas = self.detect_blobs(mask, min_area, max_area, draw_boxes)
            output_images.append(output_img)
            all_bboxes.append(valid_bboxes)
            all_areas.append(valid_areas)
        return output_images, all_bboxes, all_areas
    
def _draw_track_trail(frame, track, max_len=25):
    """Draw a short trail of a track's recent positions."""
    pts = track.history[-max_len:]
    for a, b in zip(pts[:-1], pts[1:]):
        a = tuple(map(int, a))
        b = tuple(map(int, b))
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
    
    