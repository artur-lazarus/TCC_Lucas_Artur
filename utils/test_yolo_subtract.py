"""
Test script for the "Goes Nuts" YOLO-based background subtraction.

Usage:
    python test_yolo_subtract.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from detection_api import Detection

def test_yolo_subtract():
    """Test YOLO segmentation-based background subtraction."""
    
    # Example usage - adjust video path as needed
    video_path = "assets/video.avi"
    
    if not os.path.exists(video_path):
        print(f"Test video not found at: {video_path}")
        print("Please adjust the video_path in this script to point to a valid video file.")
        return
    
    print("=" * 60)
    print("Testing YOLO-based Background Subtraction")
    print("=" * 60)
    
    # Initialize detection with color frames (required for YOLO)
    print("\n1. Loading video...")
    d = Detection(video_path, max_frames=1000, color=True, frame_interval=5)
    print(f"   Loaded {len(d.frames)} frames")
    
    # Test basic YOLO subtraction
    print("\n2. Running YOLO segmentation...")
    mask = d.yolo_subtract()
    print(f"   Generated {mask.count()} masks")
    
    # Save the result
    output_file = "test_yolo_subtract_output.mp4"
    print(f"\n3. Saving masks to {output_file}...")
    mask.save(output_file)
    print(f"   Saved successfully!")
    
    # Test with morphological operations
    print("\n4. Testing with morphological operations...")
    mask_cleaned = d.yolo_subtract().morphology.fill_holes()
    output_file_cleaned = "test_yolo_subtract_cleaned.mp4"
    mask_cleaned.save(output_file_cleaned)
    print(f"   Saved cleaned masks to {output_file_cleaned}")
    
    # Test blob detection
    print("\n5. Testing blob detection on YOLO masks...")
    blob_result = mask_cleaned.blobs.detect(min_area=100)
    print(f"   Detected blobs - Stats: {blob_result['stats']}")
    
    # Save blob detection visualization
    output_file_blobs = "test_yolo_subtract_blobs.mp4"
    blob_result_saved = mask_cleaned.blobs.save_detection(output_file_blobs, min_area=100)
    print(f"   Saved blob detection to {output_file_blobs}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
    print("\nGenerated files:")
    print(f"  - {output_file}")
    print(f"  - {output_file_cleaned}")
    print(f"  - {output_file_blobs}")

if __name__ == "__main__":
    test_yolo_subtract()
