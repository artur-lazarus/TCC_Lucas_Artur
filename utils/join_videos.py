#!/usr/bin/env python3
"""
Video Joining Utility
Joins multiple MP4 videos into a single output video file.
Supports both OpenCV and FFmpeg methods for video concatenation.
"""

import cv2
import os
import sys
import subprocess
from pathlib import Path
import tempfile

def join_videos_opencv(video_paths, output_path, method='copy_properties'):
    """
    Join videos using OpenCV.
    
    Args:
        video_paths (list): List of input video file paths
        output_path (str): Output video file path
        method (str): 'copy_properties' or 'auto_detect'
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not video_paths:
        print("Error: No video files provided")
        return False
    
    # Check if all files exist
    for video_path in video_paths:
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return False
    
    # Get properties from first video
    first_cap = cv2.VideoCapture(video_paths[0])
    if not first_cap.isOpened():
        print(f"Error: Could not open first video: {video_paths[0]}")
        return False
    
    # Get video properties
    fps = first_cap.get(cv2.CAP_PROP_FPS)
    width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Create output video writer
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not create output video: {output_path}")
        first_cap.release()
        return False
    
    print(f"Joining {len(video_paths)} videos...")
    print(f"Output properties: {width}x{height} @ {fps:.2f} fps")
    
    total_frames_written = 0
    
    try:
        for i, video_path in enumerate(video_paths):
            print(f"Processing video {i+1}/{len(video_paths)}: {video_path}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Could not open video: {video_path}")
                continue
            
            # Get current video properties
            curr_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            curr_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            curr_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"  Input: {curr_width}x{curr_height} @ {curr_fps:.2f} fps, {total_frames} frames")
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize frame if dimensions don't match
                if curr_width != width or curr_height != height:
                    frame = cv2.resize(frame, (width, height))
                
                out.write(frame)
                frame_count += 1
                total_frames_written += 1
                
                # Progress indicator
                if frame_count % 100 == 0:
                    print(f"  Processed {frame_count}/{total_frames} frames", end='\r')
            
            print(f"  Completed: {frame_count} frames written")
            cap.release()
        
        print(f"\nSuccessfully joined videos!")
        print(f"Total frames written: {total_frames_written}")
        print(f"Output saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error during video joining: {e}")
        return False
        
    finally:
        first_cap.release()
        out.release()

def join_videos_ffmpeg(video_paths, output_path):
    """
    Join videos using FFmpeg (more reliable for different formats).
    
    Args:
        video_paths (list): List of input video file paths
        output_path (str): Output video file path
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not video_paths:
        print("Error: No video files provided")
        return False
    
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: FFmpeg not found. Please install FFmpeg or use OpenCV method.")
        return False
    
    # Check if all files exist
    for video_path in video_paths:
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return False
    
    # Create temporary file list for ffmpeg
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        temp_list_file = f.name
        for video_path in video_paths:
            # Convert to absolute path and escape for ffmpeg
            abs_path = os.path.abspath(video_path)
            f.write(f"file '{abs_path}'\n")
    
    try:
        print(f"Joining {len(video_paths)} videos using FFmpeg...")
        
        # FFmpeg command to concatenate videos
        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', temp_list_file,
            '-c', 'copy',  # Copy streams without re-encoding (faster)
            '-y',  # Overwrite output file
            output_path
        ]
        
        print("Running FFmpeg command...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Successfully joined videos using FFmpeg!")
            print(f"Output saved to: {output_path}")
            return True
        else:
            print(f"FFmpeg error: {result.stderr}")
            
            # Try with re-encoding if copy failed
            print("Trying with re-encoding...")
            cmd_reencode = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', temp_list_file,
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-y',
                output_path
            ]
            
            result = subprocess.run(cmd_reencode, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Successfully joined videos with re-encoding!")
                print(f"Output saved to: {output_path}")
                return True
            else:
                print(f"FFmpeg re-encoding error: {result.stderr}")
                return False
    
    except Exception as e:
        print(f"Error running FFmpeg: {e}")
        return False
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_list_file)
        except:
            pass

def get_video_info_summary(video_paths):
    """Get summary information about input videos."""
    print("\nInput Videos Summary:")
    print("=" * 50)
    
    total_duration = 0
    total_frames = 0
    
    for i, video_path in enumerate(video_paths):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frames / fps if fps > 0 else 0
            
            print(f"{i+1}. {os.path.basename(video_path)}")
            print(f"   Resolution: {width}x{height}")
            print(f"   FPS: {fps:.2f}")
            print(f"   Frames: {frames:,}")
            print(f"   Duration: {duration:.2f}s")
            print()
            
            total_duration += duration
            total_frames += frames
            cap.release()
        else:
            print(f"{i+1}. {os.path.basename(video_path)} - Could not read")
    
    print(f"Total Duration: {total_duration:.2f}s ({total_duration/60:.2f} minutes)")
    print(f"Total Frames: {total_frames:,}")
    print("=" * 50)

def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 4:
        print("Usage: python join_videos.py <method> <output_file> <input_video1> <input_video2> [input_video3] ...")
        print()
        print("Methods:")
        print("  opencv  - Use OpenCV (works without external dependencies)")
        print("  ffmpeg  - Use FFmpeg (more reliable, requires FFmpeg installation)")
        print()
        print("Examples:")
        print("  python join_videos.py opencv output.mp4 video1.mp4 video2.mp4 video3.mp4")
        print("  python join_videos.py ffmpeg combined.mp4 assets/video1.mp4 assets/video2.mp4")
        sys.exit(1)
    
    method = sys.argv[1].lower()
    output_path = sys.argv[2]
    video_paths = sys.argv[3:]
    
    if method not in ['opencv', 'ffmpeg']:
        print("Error: Method must be 'opencv' or 'ffmpeg'")
        sys.exit(1)
    
    print(f"Video Joining Utility")
    print(f"Method: {method.upper()}")
    print(f"Output: {output_path}")
    print(f"Input videos: {len(video_paths)}")
    
    # Show input video information
    get_video_info_summary(video_paths)
    
    # Join videos based on selected method
    if method == 'opencv':
        success = join_videos_opencv(video_paths, output_path)
    else:  # ffmpeg
        success = join_videos_ffmpeg(video_paths, output_path)
    
    if success:
        # Show output file info
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"\nOutput file size: {file_size:.2f} MB")
        print("Video joining completed successfully!")
    else:
        print("Video joining failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
