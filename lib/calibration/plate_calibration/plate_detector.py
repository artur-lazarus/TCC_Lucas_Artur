import yolov5
import cv2
import torch
import time
import numpy as np
import os

class PlateDetector:
    """
    A class to load the YOLOv5 license plate detector,
    find plates, and find their 4 corners efficiently.
    """
    def __init__(self, model_name='keremberke/yolov5m-license-plate'):
        print("Initializing Plate Detector...")
        
        # --- Monkey patch for torch.load ---
        print("Applying monkey patch for torch.load...")
        original_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        torch.load = patched_load
        
        print(f"Loading YOLOv5 model: {model_name}...")
        self.model = yolov5.load(model_name)
        torch.load = original_load # Restore
        
        self.model.conf = 0.25  # NMS confidence threshold
        self.model.iou = 0.45   # NMS IoU threshold
        self.model.max_det = 20 # Maximum detections per image
        print("Plate Detector initialized and ready.")

    def _find_corners_in_region(self, full_image, box, save_crop=False, filename=""):
        """
        (Private helper) Finds the 4 corners of the license plate 
        within a specific bounding box region of the full image.
        
        Returns 4 *global* (x,y) corner points or None.
        """
        x1, y1, x2, y2 = map(int, box)
        
        # Ensure coordinates are valid
        y1, y2 = max(0, y1), min(full_image.shape[0], y2)
        x1, x2 = max(0, x1), min(full_image.shape[1], x2)
            
        if x1 >= x2 or y1 >= y2:
            return None # Skip invalid boxes
            
        # Create a NumPy "view" of the crop (memory-efficient, not a copy)
        plate_crop_view = full_image[y1:y2, x1:x2]
        
        # --- Save crop if requested (for debugging) ---
        if save_crop:
            cv2.imwrite(filename, plate_crop_view)
            
        h, w = plate_crop_view.shape[:2]
        if h == 0 or w == 0:
            return None
            
        # --- Run CV operations on the crop view ---
        gray_crop = cv2.cvtColor(plate_crop_view, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_crop, (5, 5), 0)
        canny = cv2.Canny(blurred, 100, 200)

        contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in contours:
            if cv2.contourArea(cnt) < (w * h * 0.2):
                continue
                
            epsilon = 0.018 * cv2.arcLength(cnt, True)
            approx_poly = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx_poly) == 4 and cv2.isContourConvex(approx_poly):
                # We found the 4 corners *local* to the crop
                local_corners = approx_poly.squeeze()
                
                # Sort them
                local_corners = local_corners[np.argsort(local_corners[:, 1])]
                top_corners = local_corners[:2][np.argsort(local_corners[:2, 0])]
                bottom_corners = local_corners[2:][np.argsort(local_corners[2:, 0])]
                ordered_local_corners = np.array([top_corners[0], top_corners[1], bottom_corners[1], bottom_corners[0]], dtype=np.float32)
                
                # --- Translate local corners to global image coordinates ---
                global_corners = ordered_local_corners + np.array([x1, y1], dtype=np.float32)
                
                return global_corners

        return None # No 4-sided contour found

    def detect(self, img, size=640, save_crops=False, save_dir='output'):
        """
        Runs full detection pipeline on a single image.
        
        Returns:
            list: A list of detection dictionaries. Each dict contains:
                  {'box': [x1, y1, x2, y2], 
                   'conf': float, 
                   'corners': np.array(4, 2) or None}
        """
        # Create output dir if saving
        if save_crops:
            os.makedirs(save_dir, exist_ok=True)
            
        # Run YOLO inference
        results = self.model(img, size=size)
        predictions = results.pred[0]
        
        boxes = predictions[:, :4].cpu().numpy()
        confs = predictions[:, 4].cpu().numpy()
        
        detection_results = []

        for i, (box, conf) in enumerate(zip(boxes, confs)):
            
            # --- Prepare debug info (if saving) ---
            crop_filename = ""
            if save_crops:
                crop_filename = os.path.join(save_dir, f'plate_{i+1}_crop.jpg')
            
            # --- Find corners in the region ---
            global_corners = self._find_corners_in_region(
                img, 
                box, 
                save_crop=save_crops, 
                filename=crop_filename
            )
            
            # Store the result
            detection_results.append({
                'box': box, 
                'conf': conf, 
                'corners': global_corners
            })

        return detection_results

if __name__ == '__main__':
    # ... (The test block remains the same and will work) ...
    print("Testing PlateDetector module...")
    detector = PlateDetector()
    image_path = '../../../assets/transito-do-Rio.jpg'
    test_img = cv2.imread(image_path)
    
    if test_img is not None:
        print(f"Running test detection on {image_path} with save_crops=True...")
        detections = detector.detect(test_img, save_crops=True, save_dir='test_output')
        
        print(f"\nTest detection complete. Found {len(detections)} plates.")
        for i, det in enumerate(detections):
            has_corners = "Yes" if det['corners'] is not None else "No"
            print(f"  Plate {i+1}: Conf: {det['conf']:.2f}, Has Corners: {has_corners}")
    else:
        print(f"Could not load test image at {image_path}. Skipping test.")