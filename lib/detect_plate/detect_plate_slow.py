# plate_detector.py
import yolov5
import cv2
import torch
import time
import numpy as np
import os # <-- Import OS for creating directories

class PlateDetector:
    """
    A class to load the YOLOv5 license plate detector 
    and run inference.
    """
    def __init__(self, model_name='keremberke/yolov5m-license-plate'):
        """
        Initializes and loads the YOLOv5 model.
        """
        print("Initializing Plate Detector...")
        
        # --- Monkey patch for torch.load ---
        print("Applying monkey patch for torch.load...")
        original_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)

        torch.load = patched_load
        # -------------------------------------

        # Load the pre-trained model
        print(f"Loading YOLOv5 model: {model_name}...")
        start_model = time.time()
        self.model = yolov5.load(model_name)
        end_model = time.time()
        print(f"Model loaded in {end_model - start_model:.2f} seconds")

        # Restore original torch.load
        torch.load = original_load

        # Set model parameters
        self.model.conf = 0.25  # NMS confidence threshold
        self.model.iou = 0.45   # NMS IoU threshold
        self.model.max_det = 20 # Maximum detections per image
        print("Plate Detector initialized and ready.")

    def detect(self, img, size=640, save_crops=False, save_dir='output'): # <-- ADDED ARGS
        """
        Runs inference on a single image.
        
        Args:
            img (np.array): The input image in BGR format.
            size (int): The inference size.
            save_crops (bool): If True, save cropped plates to save_dir.
            save_dir (str): Directory to save cropped images.
            
        Returns:
            np.array: An array of bounding boxes [x1, y1, x2, y2].
        """
        # Run inference
        results = self.model(img, size=size)
        
        # Extract predictions
        predictions = results.pred[0]
        
        # Get just the bounding boxes (xyxy)
        boxes = predictions[:, :4].cpu().numpy()
        
        # --- ADDED: Save crop logic ---
        if save_crops:
            # Create save directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)
            print(f"Saving {len(boxes)} cropped plates to '{save_dir}/'...")
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                
                # Ensure coordinates are within image bounds
                y1, y2 = max(0, y1), min(img.shape[0], y2)
                x1, x2 = max(0, x1), min(img.shape[1], x2)
                
                if x1 >= x2 or y1 >= y2:
                    continue # Skip invalid crops
                    
                # Crop the plate region
                plate_crop = img[y1:y2, x1:x2]
                
                # Save the cropped plate
                plate_filename = os.path.join(save_dir, f'plate_{i+1}_crop.jpg')
                cv2.imwrite(plate_filename, plate_crop)
        # ------------------------------
        
        return boxes

if __name__ == '__main__':
    # This block runs ONLY if you execute this file directly
    print("Testing PlateDetector module...")
    
    # Initialize the detector
    detector = PlateDetector()
    
    # Load a test image
    image_path = '../../assets/transito-do-Rio.jpg'
    test_img = cv2.imread(image_path)
    
    if test_img is not None:
        print(f"Running test detection on {image_path} with save_crops=True...")
        # --- Test the new feature ---
        test_boxes = detector.detect(test_img, save_crops=True, save_dir='test_output')
        print(f"Test detection complete. Found {len(test_boxes)} boxes.")
        print(f"Cropped images saved to 'test_output/'")
        print(test_boxes)
    else:
        print(f"Could not load test image at {image_path}. Skipping test.")