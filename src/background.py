import numpy as np
import cv2
import time
from ultralytics import YOLO

# Optional acceleration with numba
try:
    import numba

    HAVE_NUMBA = True

    @numba.njit(parallel=True, fastmath=True)
    def _percentile_from_hist_numba(hist, target):
        """
        hist: (N, 256) uint16
        target: int (0..size)
        returns: (N,) uint8
        """
        npx, nbins = hist.shape
        out = np.empty(npx, np.uint8)
        for i in numba.prange(npx):
            h = hist[i]
            cum = 0
            val = 0
            for b in range(nbins):
                cum += h[b]
                if cum >= target:
                    val = b
                    break
            out[i] = val
        return out

    @numba.njit(parallel=True, fastmath=True)
    def _update_warm_numba(hist, ring, ring_head, frame_flat):
        """
        Warm-up phase update: only add new frame to hist + ring.
        """
        npx = frame_flat.shape[0]
        for i in numba.prange(npx):
            v = int(frame_flat[i])
            hist[i, v] += 1
            ring[i, ring_head] = v

    @numba.njit(parallel=True, fastmath=True)
    def _update_steady_numba(hist, ring, ring_head, frame_flat):
        """
        Steady-state update: remove old frame, add new frame.
        """
        npx = frame_flat.shape[0]
        for i in numba.prange(npx):
            old_v = int(ring[i, ring_head])
            new_v = int(frame_flat[i])
            hist[i, old_v] -= 1
            hist[i, new_v] += 1
            ring[i, ring_head] = new_v

except ImportError:
    HAVE_NUMBA = False
    _percentile_from_hist_numba = None
    _update_warm_numba = None
    _update_steady_numba = None


class Background:
    def __init__(self, W, H, size):
        self.W = W
        self.H = H
        self.NPX = W * H
        self.size = size  # sliding window length

        self.hist = np.zeros((self.NPX, 256), dtype=np.uint16)
        self.ring = np.zeros((self.NPX, self.size), dtype=np.uint8)
        self.ring_head = 0

        self.last_bg_computed = None
        self.last_bg_computed_percentile = None
        self.updated_since_last_median = False

        self.loaded = 0  # how many frames have been ingested

        # Precompute pixel indices for the NumPy fallback (avoids reallocating every update)
        self._idx = np.arange(self.NPX, dtype=np.int32)

    # ---------------------------------------------------------
    def update(self, frame):
        """Add one new frame to the sliding histogram."""
        time0 = time.perf_counter()

        # Ensure uint8 and contiguous 1D view
        f = np.asarray(frame, dtype=np.uint8).ravel()

        if self.loaded < self.size:
            # WARM-UP PHASE: no removals
            if HAVE_NUMBA and _update_warm_numba is not None:
                _update_warm_numba(self.hist, self.ring, self.ring_head, f)
            else:
                # NumPy fallback
                self.hist[self._idx, f] += 1
                self.ring[:, self.ring_head] = f

            self.ring_head = (self.ring_head + 1) % self.size
            self.loaded += 1

            # print("Background update (warm-up) took: " + str(time.perf_counter() - time0))
            return

        # STEADY STATE: sliding window remove + add
        if HAVE_NUMBA and _update_steady_numba is not None:
            _update_steady_numba(self.hist, self.ring, self.ring_head, f)
        else:
            # NumPy fallback
            old_vals = self.ring[:, self.ring_head]
            self.hist[self._idx, old_vals] -= 1
            self.hist[self._idx, f] += 1
            self.ring[:, self.ring_head] = f

        self.ring_head = (self.ring_head + 1) % self.size
        self.updated_since_last_median = True

        # print("Background update (steady) took: " + str(time.perf_counter() - time0))

    # ---------------------------------------------------------
    def _compute_background_percentile_image(self, percentile: float) -> np.ndarray:
        """
        Internal helper: compute percentile image from hist using
        fast numba loop if available, otherwise fall back to cumsum+argmax.
        Returns (H, W) uint8.
        """

        # Target count for the percentile
        target = int((percentile / 100.0) * self.size)

        if HAVE_NUMBA and _percentile_from_hist_numba is not None:
            # Fast compiled path: scan each row until cumulative >= target
            values_flat = _percentile_from_hist_numba(self.hist, target)
        else:
            # Fallback: original logic (slower, but same result)
            c = np.cumsum(self.hist, axis=1, dtype=np.uint32)
            values_flat = np.argmax(c >= target, axis=1).astype(np.uint8)

        img = values_flat.reshape(self.H, self.W)
        return img

    # ---------------------------------------------------------
    def get_background_percentile(self, percentile):
        """
        Returns the background percentile image (e.g., 50 = median).
        If not enough frames have been loaded, prints a warning and returns None.
        """
        time0 = time.perf_counter()
        if self.loaded < self.size:
            print(f"[Background] Not enough frames yet ({self.loaded}/{self.size}). Returning None.")
            return None

        if (not self.updated_since_last_median) and \
           (self.last_bg_computed is not None) and \
           (self.last_bg_computed_percentile == percentile):
            # Cached value still valid
            return self.last_bg_computed

        # Fast percentile computation
        bg_img = self._compute_background_percentile_image(percentile)

        self.last_bg_computed = bg_img.astype(np.uint8)
        self.last_bg_computed_percentile = percentile
        self.updated_since_last_median = False
        # print("Background percentile computation took: " + str(time.perf_counter() - time0))
        return self.last_bg_computed

    # ---------------------------------------------------------
    def background_subtract(self, frame, threshold,
                            subtract_percentile=50,
                            normalize=False,
                            norm_percentiles=(10, 90)):
        if self is None or self.loaded < self.size:
            print("ERROR: Background not initialized or not enough frames loaded. Call init_background() first.")
            return None

        bg_v_u8 = self.get_background_percentile(subtract_percentile)
        if bg_v_u8 is None:
            return None

        # Convert to float32 only if we need normalization
        bg_v = bg_v_u8.astype(np.float32)
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

        v_norm_u8 = np.clip(v_norm, 0, 255).astype(np.uint8)

        diff = cv2.absdiff(bg_v_u8, v_norm_u8)
        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.medianBlur(mask, 5)
        return mask
    
    def background_subtract_yolo(self, frame, conf_threshold=0.8):
        vehicle_classes = [2]
        model_path = "resources/yolov8l-seg.pt"
        
        model = YOLO(model_path)
        iou_threshold=0.7
        
        if len(frame.shape) == 2:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_bgr = frame
        
        results = model.predict(
            frame_bgr,
            conf=conf_threshold,
            iou=iou_threshold,
            classes=vehicle_classes,
            verbose=False
        )
        
        # Create blank mask
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Extract segmentation masks and bounding boxes from results
        if results and len(results) > 0:
            result = results[0]
            if result.masks is not None and result.boxes is not None:
                # Get the segmentation masks and bounding boxes
                masks_data = result.masks.data.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()
                
                # Process each detected object
                for mask_data, box in zip(masks_data, boxes):
                    # Resize mask to frame size if needed
                    if mask_data.shape != (h, w):
                        mask_resized = cv2.resize(
                            mask_data, (w, h), 
                            interpolation=cv2.INTER_LINEAR
                        )
                    else:
                        mask_resized = mask_data
                    
                    # Threshold to get binary mask
                    binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
                    
                    # Crop mask to bounding box
                    x1, y1, x2, y2 = box.astype(int)
                    # Clamp box coordinates to frame bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    # Create bounding box mask
                    bbox_mask = np.zeros((h, w), dtype=np.uint8)
                    bbox_mask[y1:y2, x1:x2] = 255
                    
                    # Apply bounding box crop to segmentation
                    cropped_mask = cv2.bitwise_and(binary_mask, bbox_mask)
                    
                    # Add to combined mask
                    mask = cv2.bitwise_or(mask, cropped_mask)
        
        return mask

def fill_holes(mask):
        h, w = mask.shape
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        im_flood = mask.copy()
        cv2.floodFill(im_flood, flood_mask, (0, 0), 255)
        im_flood_inv = cv2.bitwise_not(im_flood)
        return cv2.bitwise_or(mask, im_flood_inv)
