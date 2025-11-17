import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import yolov5
import cv2
import torch
import time
import os

class PlateDetector:
    """
    A class to load the YOLOv5 license plate detector 
    and run inference.
    """
    def __init__(self, conf_threshold=0.25):
        """
        Initializes and loads the YOLOv5 model from local weights.
        
        Args:
            conf_threshold (float): NMS confidence threshold (default: 0.25)
        """
        original_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        torch.load = patched_load
        
        
        model_path = 'resources/plate_detector.pt'
        self.model = yolov5.load(model_path)
        torch.load = original_load
        
        self.model.conf = conf_threshold
        self.model.iou = 0.45
        self.model.max_det = 20

    def detect(self, img, size=640, save_crops=False, save_dir='test_output/plate_detection'):
        """
        Runs inference on a single image.
        
        Args:
            img (np.array): The input image
            size (int): The inference size
            save_crops (bool): If True, save cropped plates to save_dir
            save_dir (str): Directory to save cropped images
            
        Returns:
            np.array: An array of bounding boxes [x1, y1, x2, y2]
        """
        results = self.model(img, size=size)
        predictions = results.pred[0]
        boxes = predictions[:, :4].cpu().numpy()
        
        if save_crops and len(boxes) > 0:
            os.makedirs(save_dir, exist_ok=True)
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                y1, y2 = max(0, y1), min(img.shape[0], y2)
                x1, x2 = max(0, x1), min(img.shape[1], x2)
                
                if x1 >= x2 or y1 >= y2:
                    continue
                
                plate_crop = img[y1:y2, x1:x2]
                plate_filename = os.path.join(save_dir, f'plate_{i+1}_crop.jpg')
                cv2.imwrite(plate_filename, plate_crop)
        
        return boxes

if __name__ == '__main__':
    print("Testing PlateDetector module...")
    
    detector = PlateDetector()
    
    image_path = 'assets/transito-do-Rio.jpg'
    test_img = cv2.imread(image_path)
    
    if test_img is not None:
        print(f"Running test detection on {image_path} with save_crops=True...")
        time1 = time.perf_counter()
        test_boxes = detector.detect(test_img, save_crops=True, save_dir='test_output')
        time2 = time.perf_counter()
        print(f"Test detection took {time2 - time1:.3f} seconds")
        print(f"Test detection complete. Found {len(test_boxes)} boxes.")
        print(f"Cropped images saved to 'test_output/'")
        print(test_boxes)
    else:
        print(f"Could not load test image at {image_path}. Skipping test.")
