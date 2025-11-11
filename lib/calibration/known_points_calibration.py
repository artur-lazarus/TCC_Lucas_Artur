import os
import cv2
from perspective import unwarp

# Video calibration pipeline using predefined points (no interactive UI).
#
# How it works:
# - Reads frames from INPUT_VIDEO
# - Applies perspective unwarp using POINTS (in source video coordinates)
# - Writes a new video with the top-down view to OUTPUT_VIDEO
#
# Set your defaults here by editing the constants below.

# Example default input video (adjust to your dataset)
INPUT_VIDEO = "dataset/session0_left/video.avi"

# If None, will save next to input as "<name>_topdown.<ext>"
OUTPUT_VIDEO = None

# Declare calibration points here, in source image coordinates (x,y), 4 points.
# Order can be arbitrary; they will be ordered internally as [top-left, top-right, bottom-right, bottom-left].
# If left as None or empty, the full frame corners are used (identity transform).
POINTS = [(9, 334), (755, 1072), (1909, 643), (724, 292)]  # e.g., [(120, 200), (520, 190), (540, 420), (100, 430)]

MAX_FRAMES = 400

# Quality/encoding settings
# Use a less lossy codec by default: 'MJPG' (AVI). For MP4, use 'mp4v'.
CODEC = 'mp4v'
# Warp interpolation: INTER_LINEAR (fast), INTER_CUBIC (good), INTER_LANCZOS4 (best but slower)
INTERPOLATION = cv2.INTER_LANCZOS4
# Optionally resize the warped output to this width (keeps aspect ratio). Set to None to keep native size.
RESIZE_OUTPUT_WIDTH = None

def _auto_output_path(input_path: str) -> str:
    root, ext = os.path.splitext(input_path)
    # Choose extension based on codec for convenience
    if CODEC.upper() == 'MJPG':
        return f"{root}_topdown.avi"
    else:
        return f"{root}_topdown.mp4"

def _open_writer(sample_frame, output_path: str, fps: float):
    h, w = sample_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*CODEC)
    # If fps is invalid, fall back to 20.0 (same as detection utils)
    if not fps or fps <= 0:
        fps = 20.0
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h), isColor=(len(sample_frame.shape) == 3))
    return writer

def process_video(input_path: str, points, output_path: str | None = None, max_frames: int | None = None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {input_path}")

    # Try to get input FPS for smoother output; fallback handled in _open_writer
    fps = cap.get(cv2.CAP_PROP_FPS)

    ret, first = cap.read()
    if not ret or first is None:
        cap.release()
        raise SystemExit("Failed to read first frame from input video")

    # If points not provided, use full-frame corners (identity-like transform)
    if not points:
        h, w = first.shape[:2]
        points = [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]

    out_path = output_path or _auto_output_path(input_path)

    # Compute warp using high-quality interpolation
    warped_first = unwarp(first, points, interpolation=INTERPOLATION)
    # Optional output resize to a target width
    if RESIZE_OUTPUT_WIDTH is not None and RESIZE_OUTPUT_WIDTH > 0 and warped_first.shape[1] != RESIZE_OUTPUT_WIDTH:
        h, w = warped_first.shape[:2]
        new_h = int(h * (RESIZE_OUTPUT_WIDTH / w))
        warped_first = cv2.resize(warped_first, (RESIZE_OUTPUT_WIDTH, new_h), interpolation=INTERPOLATION)
    writer = _open_writer(warped_first, out_path, fps)

    # Write first frame
    writer.write(warped_first)

    # Process remaining frames
    frame_idx = 1
    while (frame_idx < max_frames) if (max_frames is not None) else True:
        ret, frame = cap.read()
        if not ret:
            break
        warped = unwarp(frame, points, interpolation=INTERPOLATION)
        if RESIZE_OUTPUT_WIDTH is not None and RESIZE_OUTPUT_WIDTH > 0 and warped.shape[1] != RESIZE_OUTPUT_WIDTH:
            h, w = warped.shape[:2]
            new_h = int(h * (RESIZE_OUTPUT_WIDTH / w))
            warped = cv2.resize(warped, (RESIZE_OUTPUT_WIDTH, new_h), interpolation=INTERPOLATION)
        writer.write(warped)
        frame_idx += 1

    writer.release()
    cap.release()
    print(f"Saved top-down video: {out_path} ({frame_idx} frames)")

def main():
    # Use the configured constants at the top of this file.
    # Set INPUT_VIDEO, OUTPUT_VIDEO (or leave as None), and POINTS.
    process_video(INPUT_VIDEO, POINTS, OUTPUT_VIDEO, MAX_FRAMES)

if __name__ == "__main__":
    main()
