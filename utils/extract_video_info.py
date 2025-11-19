#!/usr/bin/env python3
"""
Video Information Extraction Script
Extracts detailed information from video files including FPS, frame count, duration, etc.
"""

import cv2
import os
import sys
from pathlib import Path

def extract_video_info(video_path):
    """
    Extract comprehensive information from a video file.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        dict: Dictionary containing video information
    """
    # Check if file exists
    if not os.path.exists(video_path):
        return {"error": f"Video file not found: {video_path}"}
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return {"error": f"Could not open video file: {video_path}"}
    
    try:
        # Extract video properties
        info = {}
        
        # Basic properties
        info['file_path'] = video_path
        info['file_size_mb'] = round(os.path.getsize(video_path) / (1024 * 1024), 2)
        
        # Video properties
        info['fps'] = cap.get(cv2.CAP_PROP_FPS)
        info['total_frames'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate duration
        if info['fps'] > 0:
            info['duration_seconds'] = info['total_frames'] / info['fps']
            info['duration_formatted'] = format_duration(info['duration_seconds'])
        else:
            info['duration_seconds'] = 0
            info['duration_formatted'] = "Unknown"
        
        # Additional properties
        info['codec'] = get_fourcc_string(cap.get(cv2.CAP_PROP_FOURCC))
        info['aspect_ratio'] = round(info['width'] / info['height'], 2) if info['height'] > 0 else 0
        
        # Try to get additional metadata if available
        try:
            info['brightness'] = cap.get(cv2.CAP_PROP_BRIGHTNESS)
            info['contrast'] = cap.get(cv2.CAP_PROP_CONTRAST)
            info['saturation'] = cap.get(cv2.CAP_PROP_SATURATION)
            info['hue'] = cap.get(cv2.CAP_PROP_HUE)
        except:
            # These properties might not be available for all video sources
            pass
            
        return info
        
    finally:
        cap.release()

def get_fourcc_string(fourcc):
    """Convert fourcc code to string format."""
    try:
        return "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])
    except:
        return "Unknown"

def format_duration(seconds):
    """Format duration in seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def print_video_info(info):
    """Print video information in a formatted way."""
    if "error" in info:
        print(f"Error: {info['error']}")
        return
    
    print("=" * 50)
    print("VIDEO INFORMATION")
    print("=" * 50)
    print(f"File Path: {info['file_path']}")
    print(f"File Size: {info['file_size_mb']} MB")
    print()
    print("VIDEO PROPERTIES:")
    print(f"  Resolution: {info['width']} x {info['height']}")
    print(f"  Aspect Ratio: {info['aspect_ratio']}:1")
    print(f"  FPS (Frames Per Second): {info['fps']:.2f}")
    print(f"  Total Frames: {info['total_frames']:,}")
    print(f"  Duration: {info['duration_formatted']} ({info['duration_seconds']:.2f} seconds)")
    print(f"  Codec: {info['codec']}")
    print()
    
    # Additional properties if available
    additional_props = ['brightness', 'contrast', 'saturation', 'hue']
    has_additional = any(prop in info for prop in additional_props)
    
    if has_additional:
        print("ADDITIONAL PROPERTIES:")
        for prop in additional_props:
            if prop in info:
                print(f"  {prop.capitalize()}: {info[prop]}")
        print()

def save_info_to_file(info, output_file):
    """Save video information to a text file."""
    try:
        with open(output_file, 'w') as f:
            f.write("VIDEO INFORMATION REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated for: {info['file_path']}\n\n")
            
            if "error" not in info:
                f.write(f"File Size: {info['file_size_mb']} MB\n")
                f.write(f"Resolution: {info['width']} x {info['height']}\n")
                f.write(f"Aspect Ratio: {info['aspect_ratio']}:1\n")
                f.write(f"FPS: {info['fps']:.2f}\n")
                f.write(f"Total Frames: {info['total_frames']:,}\n")
                f.write(f"Duration: {info['duration_formatted']} ({info['duration_seconds']:.2f} seconds)\n")
                f.write(f"Codec: {info['codec']}\n")
                
                # Additional properties
                additional_props = ['brightness', 'contrast', 'saturation', 'hue']
                for prop in additional_props:
                    if prop in info:
                        f.write(f"{prop.capitalize()}: {info[prop]}\n")
            else:
                f.write(f"Error: {info['error']}\n")
        
        print(f"Information saved to: {output_file}")
        
    except Exception as e:
        print(f"Could not save to file: {e}")

def main():
    """Main function to handle command line arguments and execute extraction."""
    if len(sys.argv) != 2:
        print("Usage: python extract_video_info.py <video_path>")
        print("Example: python extract_video_info.py assets/video1.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    print(f"Extracting information from: {video_path}")
    print()
    
    # Extract video information
    info = extract_video_info(video_path)
    
    # Print results
    print_video_info(info)
    
    # Also save to a text file
    if "error" not in info:
        output_file = f"{Path(video_path).stem}_info.txt"
        save_info_to_file(info, output_file)

if __name__ == "__main__":
    main()
