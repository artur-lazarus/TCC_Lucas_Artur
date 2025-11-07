from ultralytics import YOLO
import yolov5
import cv2
import torch
import time

print("Starting license plate detection...")
start_total = time.time()

# Monkey patch torch.load to use weights_only=False
print("Applying monkey patch for torch.load...")
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)

torch.load = patched_load

# Load the pre-trained model from Hugging Face
print("Loading YOLOv5 model...")
start_model = time.time()
model = yolov5.load('keremberke/yolov5n-license-plate')
end_model = time.time()
print(f"Model loaded in {end_model - start_model:.2f} seconds")

# Restore original torch.load
torch.load = original_load

# Set model parameters (optional, but recommended)
# model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45   # NMS IoU threshold
model.max_det = 20  # Maximum detections per image

# Load your test image
image_path = '../../assets/transito-do-Rio.jpg'
print(f"Loading image: {image_path}")
start_load = time.time()
img = cv2.imread(image_path)
end_load = time.time()
print(f"Image loaded in {end_load - start_load:.4f} seconds")

if img is None:
    print(f"Error: Could not load image at {image_path}")
else:
    # Get original image dimensions
    original_height, original_width = img.shape[:2]
    print(f"Original image dimensions: {original_width}x{original_height}")
    
    # Run inference on the image
    print("Running inference...")
    start_inference = time.time()
    results = model(img, size=640)
    end_inference = time.time()
    print(f"Inference completed in {end_inference - start_inference:.4f} seconds")

    # Extract predictions
    print("Processing results...")
    start_process = time.time()
    predictions = results.pred[0]  # Get the first (and only) result
    
    # Calculate scale factors to adjust coordinates
    # YOLOv5 resizes the image maintaining aspect ratio
    scale = min(640 / original_width, 640 / original_height)
    scaled_width = int(original_width * scale)
    scaled_height = int(original_height * scale)
    
    # Calculate padding
    pad_x = (640 - scaled_width) // 2
    pad_y = (640 - scaled_height) // 2
    
    print(f"Applied scale: {scale:.4f}")
    print(f"Scaled dimensions: {scaled_width}x{scaled_height}")
    print(f"Padding: x={pad_x}, y={pad_y}")
    
    # Initialize cropped_plates list
    cropped_plates = []
    
    if len(predictions) > 0:
        boxes = predictions[:, :4]  # xyxy coordinates
        scores = predictions[:, 4]
        categories = predictions[:, 5]
        
        print(f"Detected {len(predictions)} license plates in the image")

        # Crop and save each detected plate
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = map(int, box)
            
            # Crop the plate region from the original image
            plate_crop = img[y1:y2, x1:x2]
            
            # Save the cropped plate
            plate_filename = f'output/plate_{i+1}.jpg'
            cv2.imwrite(plate_filename, plate_crop)
            cropped_plates.append(plate_filename)
            
            print(f"  Plate {i+1}: confidence {score:.4f}, position ({x1},{y1}) -> ({x2},{y2})")
            print(f"    Cropped plate saved at: {plate_filename}")
    else:
        print("No license plates detected in the image.")
    
    end_process = time.time()
    print(f"Results processing in {end_process - start_process:.4f} seconds")

    # Save information about detected plates
    if len(cropped_plates) > 0:
        print(f"\nSummary:")
        print(f"Total detected plates: {len(cropped_plates)}")
        print(f"Cropped plates saved at:")
        for plate_file in cropped_plates:
            print(f"  - {plate_file}")
    
    end_total = time.time()
    print(f"\nTotal execution time: {end_total - start_total:.2f} seconds")
