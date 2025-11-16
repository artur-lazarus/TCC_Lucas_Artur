import background_generation as bg_gen
import foreground_detection as fg_det
import optical_flow as opt_flow
import morphological_operations as morph_ops
import mask_operations as mask_ops
import blob_detection as blob_det
import hsv_processing as hsv_proc
import utils
import cv2

class MaskResult:
    def __init__(self, masks, frames=None):
        self.masks = masks
        self.frames = frames
    
    def and_(self, other):
        masks = other.masks if isinstance(other, MaskResult) else other
        return MaskResult(mask_ops.masks_and_pairwise(self.masks, masks), self.frames)

    def and_with_image(self, image):
        return MaskResult(mask_ops.masks_and_pairwise(self.masks, image), self.frames)
    
    def or_(self, other):
        masks = other.masks if isinstance(other, MaskResult) else other
        return MaskResult(mask_ops.masks_or_pairwise(self.masks, masks), self.frames)
    
    def subtract(self, other):
        masks = other.masks if isinstance(other, MaskResult) else other
        return MaskResult(mask_ops.masks_subtract_pairwise(self.masks, masks), self.frames)
    
    @property
    def morphology(self):
        return MorphologyOps(self)
    
    @property
    def blobs(self):
        return BlobOps(self)
    
    def save(self, filename):
        utils.saveGrayscale(self.masks, filename)
        return self
    
    def count(self):
        return len(self.masks)

class MorphologyOps:
    def __init__(self, mask_result):
        self.mask_result = mask_result
    
    def opening(self, kernel_size=(4, 4)):
        return MaskResult(morph_ops.apply_opening(self.mask_result.masks, kernel_size), self.mask_result.frames)
    
    def closing(self, kernel_size=(4, 4)):
        return MaskResult(morph_ops.apply_closing(self.mask_result.masks, kernel_size), self.mask_result.frames)
    
    def both(self, kernel_size=(4, 4)):
        return MaskResult(morph_ops.apply_opening_then_closing(self.mask_result.masks, kernel_size), self.mask_result.frames)
    
    def clean(self, min_area=50):
        return MaskResult(morph_ops.remove_small_components(self.mask_result.masks, min_area), self.mask_result.frames)
    
    def fill_holes(self):
        return MaskResult(morph_ops.fill_enclosed_regions(self.mask_result.masks), self.mask_result.frames)

class BlobOps:
    def __init__(self, mask_result):
        self.mask_result = mask_result
    
    def detect(self, min_area=50, draw_boxes=True, draw_centroids=True):
        blob_images, centroids, areas, bboxes = blob_det.process_blob_detection_sequence(
            self.mask_result.masks, min_area, draw_boxes, draw_centroids
        )
        return {
            'images': blob_images,
            'centroids': centroids,
            'areas': areas,
            'bboxes': bboxes,
            'stats': blob_det.get_blob_statistics(areas, centroids)
        }
    
    def save_detection(self, filename, min_area=50):
        result = self.detect(min_area)
        utils.saveColor(result['images'], filename)
        return result

class Detection:
    def __init__(self, video_path_or_frames, max_frames=400, color=True, start_frame: int = 0, frame_interval: int = 1):
        """
        Initialize detection on a sequence of frames or a video file.

        Args:
          video_path_or_frames: list of frames (BGR or grayscale) or path string.
          max_frames: cap on frames to load/process from video input.
          color: if True, keep frames in color (BGR). If False, store as grayscale
                 to reduce memory footprint. Color-only methods will be disabled.
          start_frame: when video_path_or_frames is a path str, start reading at
                       this zero-based frame index and load at most max_frames.
          frame_interval: sample one every N frames during initial load (>=1).
                          For example, frame_interval=5 loads frames
                          start_frame, start_frame+5, start_frame+10, ...
        """
        self._color_mode = bool(color)
        self._frame_interval = int(frame_interval) if frame_interval and frame_interval > 0 else 1
        if isinstance(video_path_or_frames, list):
            # Copy slice and normalize to desired color mode
            # Apply interval sampling if a list was provided directly
            src_all = video_path_or_frames
            if self._frame_interval > 1:
                src_all = src_all[::self._frame_interval]
            src = src_all[:max_frames]
            if self._color_mode:
                self.frames = src
            else:
                # Ensure single-channel storage
                gray_frames = []
                for f in src:
                    if f is None:
                        continue
                    if f.ndim == 2:
                        gray_frames.append(f)
                    else:
                        gray_frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
                self.frames = gray_frames
        elif type(video_path_or_frames) == str:
            # Load directly in desired mode to avoid peak memory usage
            self.frames = utils.loadVideo(
                video_path_or_frames,
                max_frames=max_frames,
                color=self._color_mode,
                start_frame=start_frame,
                frame_interval=self._frame_interval,
            )
        else:
            raise ValueError("Input must be a list of frames or a video file path.")
        self._background = None
        self._flows = None
        self._hsv_flows = None
    
    def resize(self, width, height):
        if width <= 0 or height <= 0:
            print("ERROR: width and height must be positive")
            return self
        resized_frames = []
        for frame in self.frames:
            resized = cv2.resize(frame, (width, height))
            resized_frames.append(resized)
        self.frames = resized_frames
        self._background = None
        self._flows = None
        self._hsv_flows = None
        print(f"Resized to {width}x{height}: {len(self.frames)} frames")
        return self
    
    def downscale(self, scale_factor):
        if scale_factor <= 0 or scale_factor > 1:
            print("ERROR: scale_factor must be between 0 and 1")
            return self
        downscaled_frames = []
        for frame in self.frames:
            h, w = frame.shape[:2]
            new_w, new_h = int(w * scale_factor), int(h * scale_factor)
            downscaled = cv2.resize(frame, (new_w, new_h))
            downscaled_frames.append(downscaled)
        self.frames = downscaled_frames
        self._background = None
        self._flows = None
        self._hsv_flows = None
        print(f"Downscaled by {scale_factor}: {len(self.frames)} frames at {new_w}x{new_h}")
        return self
    
    def set_framerate(self, frame_interval):
        if frame_interval <= 0:
            print("ERROR: frame_interval must be positive")
            return self
        if frame_interval == 1:
            print("Frame interval is 1, no change needed")
            return self
        
        # Resample already loaded frames; update stored interval
        self.frames = self.frames[::frame_interval]
        self._frame_interval = frame_interval
        self._background = None
        self._flows = None
        self._hsv_flows = None
        print(f"Applied frame interval {frame_interval}: {len(self.frames)} frames remaining")
        return self
    
    def init_background(self, method='median', percentile=90):
        if method == 'median':
            self._background = bg_gen.build_median_background(self.frames)
        elif method == 'mean':
            self._background = bg_gen.build_mean_background(self.frames)
        elif method == 'mode':
            self._background = bg_gen.build_mode_background(self.frames)
        elif method == 'percentile':
            self._background = bg_gen.build_percentile_background(self.frames, percentile)
        print(f"Background initialized ({method}): {self._background.shape}")
        return self
    
    def init_flows(
        self,
        levels=None,
        patch_size=None,
        iterations=None,
        dis_preset="FAST",
        variational_refinement=None,
        finest_scale_max=5,
        refinement_iters=None,
        refinement_alpha=None,
        refinement_delta=None,
        refinement_gamma=None,
    ):
        args = dict(
            frames=self.frames,
            levels=levels,
            patch_size=patch_size,
            iterations=iterations,
            dis_preset=dis_preset,
            variational_refinement=variational_refinement,
            finest_scale_max=finest_scale_max,
            refinement_iters=refinement_iters,
            refinement_alpha=refinement_alpha,
            refinement_delta=refinement_delta,
            refinement_gamma=refinement_gamma,
        )
        # calculate_optical_flow expects frames as first positional arg
        frames = args.pop('frames')
        self._flows = opt_flow.calculate_optical_flow(frames, **args)
        self._hsv_flows = [opt_flow.flow_to_hsv(flow, 0) for flow in self._flows]
        provided = {k: v for k, v in args.items() if v is not None and k not in ('finest_scale_max',)}
        print(
            f"Flows initialized (preset={dis_preset}) overrides={provided if provided else 'none'} count={len(self._flows)}"
        )
        return self
    
    def flow_subtract(self, hue_range=(140, 170), value_min=6):
        if self._flows is None or self._hsv_flows is None:
            print("ERROR: Flows not initialized. Call init_flows() first.")
            return None
        masks = opt_flow.hue_range_mask(self._hsv_flows, hue_range[0], hue_range[1], value_min)
        return MaskResult(masks, self.frames)
    
    def flow_motion(self, magnitude_threshold=2.0):
        if self._flows is None:
            print("ERROR: Flows not initialized. Call init_flows() first.")
            return None
        masks = opt_flow.optical_flow_to_motion_masks(self._flows, magnitude_threshold)
        return MaskResult(masks, self.frames)
    
    def median_subtract(self, threshold_value=15, method='value'):
        if self._background is None and method in ['value', 'rgb']:
            print("ERROR: Background not initialized. Call init_background() first.")
            return None
        frames_subset = self.frames[1:] if method in ['value', 'rgb'] else self.frames
        if method == 'value':
            masks = fg_det.detect_foreground_value_based(frames_subset, self._background, threshold_value)
        elif method == 'rgb':
            if not self._color_mode:
                print("ERROR: RGB subtraction requires color mode. Reinitialize Detection with color=True.")
                return None
            masks = fg_det.detect_foreground_rgb_based(frames_subset, self._background, threshold_value)
        elif method == 'mog':
            masks = fg_det.detect_foreground_gaussian_mixture(frames_subset)
        elif method == 'knn':
            masks = fg_det.detect_foreground_knn(frames_subset)
        return MaskResult(masks, frames_subset)

    def median_subtract_normalized(self, threshold_value=15, robust=True, percentiles=(10, 90)):
        """
        Like median_subtract(method='value') but with per-frame illumination normalization
        against the initialized background to reduce false positives from lighting changes.

        Args:
          threshold_value: pixel difference threshold after normalization.
          robust: use percentile-based scaling (recommended).
          percentiles: (low, high) percentiles for robust scaling when robust=True.
        """
        if self._background is None:
            print("ERROR: Background not initialized. Call init_background() first.")
            return None
        frames_subset = self.frames[1:]
        masks = fg_det.detect_foreground_value_based_normalized(
            frames_subset,
            self._background,
            threshold_value=threshold_value,
            robust=robust,
            percentiles=percentiles,
        )
        return MaskResult(masks, frames_subset)
    
    def hsv_color_range(self, hue_range, sat_range=(50, 255), val_range=(50, 255)):
        if not self._color_mode:
            print("ERROR: HSV color range requires color mode. Reinitialize Detection with color=True.")
            return None
        masks = hsv_proc.create_hsv_range_mask(self.frames, hue_range, sat_range, val_range)
        return MaskResult(masks, self.frames)

    def yolo_subtract(self, conf_threshold=0.5, iou_threshold=0.7):
        """
        "Goes Nuts" background subtraction using YOLOv8 segmentation.
        
        Detects and segments vehicles (cars, motorcycles, buses, trucks) using YOLOv8,
        creating foreground masks based on the segmentation results.
        
        Args:
            model_path: path to YOLOv8 segmentation model (.pt file).
            conf_threshold: confidence threshold for detections (0-1).
            iou_threshold: IoU threshold for NMS (0-1).
        
        Returns:
            MaskResult containing the foreground masks.
        """
        if not self._color_mode:
            print("ERROR: YOLO segmentation requires color mode. Reinitialize Detection with color=True.")
            return None
        masks = fg_det.detect_foreground_yolo_segmentation(
            self.frames,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        return MaskResult(masks, self.frames)

    @property
    def color_mode(self):
        """Return True if frames are kept in color, False if grayscale."""
        return self._color_mode
