import cv2
import background

class VideoStream:
    def __init__(self):
        self.roi_mask = None

    def set_config(self, video_path, frame_interval=1, colour=True, make_background=False):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        self.frame_interval = frame_interval
        self.frame_count = 0
        self.colour = colour
        self.make_background = make_background

    def start_background(self, window_size, W, H):
        self._background = background.Background(W, H, size=window_size)

    def get_frame(self):
        if self.make_background and self._background is None:
            raise ValueError("Background not initialized. Call start_background() first.")
        
        for _ in range(self.frame_interval):
            ret, frame = self.cap.read()
            self.frame_count += 1
            if not ret:
                return None
        if not self.colour and frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.make_background and frame is not None:
            self._background.update(frame)
        if self.roi_mask is not None:
            frame = cv2.bitwise_and(frame, mask=self.roi_mask)
        return self.frame_count, frame
    
    def set_warping_configs(self, H_matrix, W_out, H_out):
        self.H_matrix=H_matrix
        self.warped_H_out = H_out
        self.warped_W_out = W_out
        
    def set_roi_mask(self, mask):
        self.roi_mask = mask

    def get_frame_warped(self):
        frame_count, frame = self.get_frame()
        warped_frame = cv2.warpPerspective(frame, 
                                         self.H_matrix, 
                                         (self.warped_W_out, self.warped_H_out),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=0)
        return warped_frame
    
    def get_frame_background_subtracted(self, threshold, subtract_percentile = 50, normalize=False, norm_percentiles=(10,90)):
        frame_count, frame = self.get_frame()
        mask = self._background.background_subtract(frame, threshold, subtract_percentile, normalize, norm_percentiles)
        bg_subtracted = cv2.bitwise_and(frame, frame, mask=mask)
        return frame_count, bg_subtracted
    
video = VideoStream()