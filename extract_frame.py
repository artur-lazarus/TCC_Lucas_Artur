import cv2
import sys

def extract_frame(video_path, frame_number, output_path, mask_path):
    """
    Extract a specific frame from a video file.
    
    Args:
        video_path: Path to the input video file
        frame_number: Frame number to extract (0-indexed)
        output_path: Path to save the extracted frame
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return False
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video properties:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {total_frames/fps:.2f} seconds")
    
    # Check if frame number is valid
    if frame_number >= total_frames:
        print(f"Error: Frame {frame_number} is out of range (video has {total_frames} frames)")
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
    
    mask = cv2.imread(mask_path)
    final_image = cv2.bitwise_and(frame, mask)
    
    # Save the frame
    success = cv2.imwrite(output_path, final_image)
    
    if success:
        print(f"Successfully extracted frame {frame_number} to {output_path}")
        print(f"Frame dimensions: {frame.shape[1]}x{frame.shape[0]}")
    else:
        print(f"Error: Could not save frame to {output_path}")
    
    # Release the video capture object
    cap.release()
    
    return success

if __name__ == "__main__":
    video_path = "assets/video.avi"
    frame_number = 550
    mask_path = "assets/video_mask.png"
    output_path = "assets/frame_280.png"
    
    extract_frame(video_path, frame_number, output_path, mask_path)
