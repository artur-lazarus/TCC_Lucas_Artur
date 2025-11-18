import cv2

def save_video_portion(n_frames, path):
    cap = cv2.VideoCapture(path)
    
    # Check if video opened successfully FIRST
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    
    # Read first frame to get dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return None
    
    # Determine if frame is color or grayscale
    is_color = len(frame.shape) == 3
    
    # Set up VideoWriter with correct color setting
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("session1_right.mp4", fourcc, 50.0, 
                          (frame.shape[1], frame.shape[0]), isColor=is_color)
    
    # Write the first frame
    out.write(frame)
    
    # Write remaining frames
    for i in range(1, n_frames):
        if i % 100 == 0:
            print(f"Frame {i}")
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    cap.release()
    out.release()
    print(f"Video saved successfully with {i} frames")
    
if __name__ == "__main__":
    save_video_portion(50000, "assets/video.avi")
