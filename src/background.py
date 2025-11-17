import numpy as np

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

    # ---------------------------------------------------------
    def update(self, frame):
        """Add one new frame to the sliding histogram."""
        f = frame.reshape(-1)

        if self.loaded < self.size:
            # WARM-UP PHASE: no removals
            self.hist[np.arange(self.NPX), f] += 1
            self.ring[:, self.ring_head] = f

            self.ring_head = (self.ring_head + 1) % self.size
            self.loaded += 1
            return

        # STEADY STATE: sliding window remove + add
        old_vals = self.ring[:, self.ring_head]

        self.hist[np.arange(self.NPX), old_vals] -= 1
        self.hist[np.arange(self.NPX), f]       += 1

        self.ring[:, self.ring_head] = f
        self.ring_head = (self.ring_head + 1) % self.size

        self.updated_since_last_median = True

    # ---------------------------------------------------------
    def get_background_percentile(self, percentile):
        """
        Returns the background percentile image (e.g., 50 = median).
        If not enough frames have been loaded, prints a warning and returns None.
        """
        if self.loaded < self.size:
            print(f"[Background] Not enough frames yet ({self.loaded}/{self.size}). Returning None.")
            return None
        
        if (not self.updated_since_last_median) and (self.last_bg_computed_percentile == percentile):
            return self.last_bg_computed

        # Compute cumulative histogram per pixel
        c = np.cumsum(self.hist, axis=1)

        # Target count for the percentile
        target = int((percentile / 100.0) * self.size)

        # argmax finds first bin where cumulative >= target
        values = np.argmax(c >= target, axis=1)

        self.last_bg_computed = values.reshape(self.H, self.W).astype(np.uint8)
        self.last_bg_computed_percentile = percentile
        self.updated_since_last_median = False

        return values.reshape(self.H, self.W).astype(np.uint8)
    
def background_subtract(frame, background_object, threshold, subtract_percentile = 50, normalize=False, norm_percentiles=(10,90)):
        if background_object is None or background_object.loaded < background_object.size:
            print("ERROR: Background not initialized or not enough frames loaded. Call init_background() first.")
            return None
        bg_v = background_object.get_background_percentile(subtract_percentile).astype(np.float32)
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