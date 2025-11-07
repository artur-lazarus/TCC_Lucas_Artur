import yolov5
import torch
import time
import os
from ultralytics import YOLO
    
def detect_plate_setup(slow = False):
    if slow:
        print("Loading YOLOv5 medium model...")
        start_model = time.time()
        slow_model = YOLO("yolov5m-detect-plate.pt")
        end_model = time.time()
        print(f"Medium model loaded in {end_model - start_model:.2f} seconds")
        
        return slow_model
    
    print("Loading YOLOv5 nano model...")
    start_model = time.time()
    fast_model = YOLO("yolov5n-detect-plate.onnx", task="detect")
    end_model = time.time()
    print(f"Nano model loaded in {end_model - start_model:.2f} seconds.")
    
    return fast_model
    

    
if __name__ == "__main__":
    detect_plate_setup()