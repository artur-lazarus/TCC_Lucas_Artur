import cv2
import numpy as np

input_video = "../assets/video.avi"
output_video = "../assets/test_video.mp4"
mask_path = "../assets/video_mask.png"

cap = cv2.VideoCapture(input_video)
fps = 10
duration = 5 * 60
total_frames = fps * duration

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video, cv2.VideoWriter.fourcc(*'mp4v'), fps, (width, height))

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
mask = cv2.resize(mask, (width, height))
mask = mask / 255.0

original_fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(original_fps / fps)

print(f"Original FPS: {original_fps}, Target FPS: {fps}, Frame interval: {frame_interval}")
print(f"Processing {total_frames} frames...")

frame_count = 0
written_frames = 0

while written_frames < total_frames:
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_count % frame_interval == 0:
        masked_frame = (frame * mask[:, :, np.newaxis]).astype(np.uint8)
        out.write(masked_frame)
        written_frames += 1
        if written_frames % 100 == 0:
            print(f"Progress: {written_frames}/{total_frames} frames written")
    
    frame_count += 1

cap.release()
out.release()
print(f"Processed {written_frames} frames at {fps} fps")
