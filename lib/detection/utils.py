import cv2
import numpy as np

def loadVideo(path, max_frames=None, color=True, start_frame: int = 0, frame_interval: int = 1):
    """
    Load a video file into a list of frames.

    Args:
      path: path to the video file.
      max_frames: optional cap on number of frames to read.
      color: if True, return BGR frames; if False, return single-channel grayscale frames.
      start_frame: zero-based index of the first frame to read. Reads at most
                   max_frames frames starting from start_frame.
      frame_interval: keep one every N frames (>=1). For example, frame_interval=5
                      returns frames [start, start+5, start+10, ...].

    Returns:
      list of frames (np.ndarray). Returns None if the video cannot be opened.
    """
    frames = []
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    # Seek to the desired starting frame (best-effort; may be approximate depending on codec)
    if start_frame and start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))

    if frame_interval is None or frame_interval < 1:
        frame_interval = 1

    read_limit = int(max_frames) if max_frames is not None else None
    loaded = 0
    while True:
        if read_limit is not None and loaded >= read_limit:
            break
        ret, frame = cap.read()
        if not ret:
            break
        if color:
            frames.append(frame)
        else:
            # Convert immediately to grayscale to avoid keeping color frames in memory
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
        loaded += 1

        # Skip the next frame_interval-1 frames quickly
        if frame_interval > 1:
            to_skip = frame_interval - 1
            for _ in range(to_skip):
                # Advance without decoding to save time when possible
                if not cap.grab():
                    break
    cap.release()
    return frames

def load_video(path, max_frames=None, color=True, start_frame: int = 0, frame_interval: int = 1):
    """Snake_case alias for loadVideo for convenience/backward compatibility."""
    return loadVideo(
        path,
        max_frames=max_frames,
        color=color,
        start_frame=start_frame,
        frame_interval=frame_interval,
    )

def show(frames):
    for frame in frames:
        cv2.imshow("Frame", frame)
        if cv2.waitKey(30) & 0xFF == 27:  # ESC to exit
            break
    cv2.destroyAllWindows()

def saveGrayscale(frames, path):
    """
    Save a sequence of frames as a single-channel (grayscale) MP4.

    - Accepts frames that are already grayscale (H, W) or color (H, W, 3).
    - Color frames are converted to grayscale before writing.
    - Boolean masks are converted to uint8 (0/255).
    """
    if not frames:
        raise ValueError("No frames to save")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Normalize the first frame to grayscale to determine size
    first = frames[0]
    if first is None:
        raise ValueError("First frame is None")

    if first.ndim == 3 and first.shape[2] in (3, 4):
        first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    elif first.ndim == 2:
        first_gray = first
    else:
        raise ValueError(f"Unsupported frame shape for grayscale save: {first.shape}")

    # Ensure dtype uint8
    if first_gray.dtype == bool:
        first_gray = (first_gray.astype('uint8')) * 255
    elif first_gray.dtype != np.uint8:
        first_gray = cv2.normalize(first_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    height, width = first_gray.shape
    out = cv2.VideoWriter(path, fourcc, 20.0, (width, height), isColor=False)

    # Write first frame then the rest
    out.write(first_gray)

    for frame in frames[1:]:
        if frame is None:
            continue
        if frame.ndim == 3 and frame.shape[2] in (3, 4):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 2:
            gray = frame
        else:
            raise ValueError(f"Unsupported frame shape for grayscale save: {frame.shape}")

        if gray.dtype == bool:
            gray = (gray.astype('uint8')) * 255
        elif gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Resize if any frame size differs to avoid writer failure
        if gray.shape != (height, width):
            gray = cv2.resize(gray, (width, height))

        out.write(gray)
    out.release()

def saveColor(frames, path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(path, fourcc, 20.0, (width, height), isColor=True)

    for frame in frames:
        out.write(frame)
    out.release()