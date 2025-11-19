#!/usr/bin/env python3
"""
Video Frame Extraction Script

This script extracts a single frame from a video file and saves it as an image.
You can specify either a frame number or a timestamp to extract the frame.
"""

import cv2
import argparse
import os
import sys


def extract_frame_by_number(video_path, frame_number, output_path=None):
    """
    Extract a frame by its frame number.
    
    Args:
        video_path (str): Path to the input video file
        frame_number (int): Frame number to extract (0-based)
        output_path (str): Path to save the extracted frame (optional)
    
    Returns:
        bool: True if successful, False otherwise
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return False
    
    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video has {total_frames} frames")
    
    if frame_number >= total_frames or frame_number < 0:
        print(f"Error: Frame number {frame_number} is out of range (0-{total_frames-1})")
        cap.release()
        return False
    
    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame {frame_number}")
        cap.release()
        return False
    
    # Generate output filename if not provided
    if output_path is None:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f"{video_name}_frame_{frame_number:06d}.jpg"
    
    # Save the frame
    success = cv2.imwrite(output_path, frame)
    if success:
        print(f"Frame {frame_number} extracted and saved as '{output_path}'")
        print(f"Frame dimensions: {frame.shape[1]}x{frame.shape[0]}")
    else:
        print(f"Error: Could not save frame to '{output_path}'")
    
    cap.release()
    return success


def extract_frame_by_time(video_path, timestamp, output_path=None):
    """
    Extract a frame by timestamp in seconds.
    
    Args:
        video_path (str): Path to the input video file
        timestamp (float): Time in seconds to extract the frame
        output_path (str): Path to save the extracted frame (optional)
    
    Returns:
        bool: True if successful, False otherwise
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video duration: {duration:.2f} seconds ({fps:.2f} FPS)")
    
    if timestamp > duration or timestamp < 0:
        print(f"Error: Timestamp {timestamp} is out of range (0-{duration:.2f} seconds)")
        cap.release()
        return False
    
    # Calculate frame number from timestamp
    frame_number = int(timestamp * fps)
    
    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame at timestamp {timestamp}")
        cap.release()
        return False
    
    # Generate output filename if not provided
    if output_path is None:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f"{video_name}_time_{timestamp:.2f}s.jpg"
    
    # Save the frame
    success = cv2.imwrite(output_path, frame)
    if success:
        print(f"Frame at {timestamp}s (frame #{frame_number}) extracted and saved as '{output_path}'")
        print(f"Frame dimensions: {frame.shape[1]}x{frame.shape[0]}")
    else:
        print(f"Error: Could not save frame to '{output_path}'")
    
    cap.release()
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Extract a single frame from a video file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract frame 1000 from video.mp4
  python extract_frame.py video.mp4 --frame 1000
  
  # Extract frame at 30.5 seconds
  python extract_frame.py video.mp4 --time 30.5
  
  # Extract frame with custom output filename
  python extract_frame.py video.mp4 --frame 500 --output my_frame.png
        """
    )
    
    parser.add_argument("video", help="Path to the input video file")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--frame", "-f", type=int, 
                      help="Frame number to extract (0-based)")
    group.add_argument("--time", "-t", type=float,
                      help="Timestamp in seconds to extract the frame")
    
    parser.add_argument("--output", "-o", type=str,
                       help="Output filename for the extracted frame")
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video):
        print(f"Error: Video file '{args.video}' not found")
        sys.exit(1)
    
    # Extract frame based on the specified method
    if args.frame is not None:
        success = extract_frame_by_number(args.video, args.frame, args.output)
    else:
        success = extract_frame_by_time(args.video, args.time, args.output)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
