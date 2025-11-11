import cv2
import lib.detection.utils as utils

def subtractBackground(frames, history=500, varThreshold=32):
    detectShadows = True
    excludeShadows = True

    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=history,          # Number of frames to build background model
        varThreshold=varThreshold,      # Variance threshold for pixel classification
        detectShadows=detectShadows    # Detect shadows (gray vs white foreground)
    )

    for i in range(len(frames)):
        frames[i] = fgbg.apply(frames[i])

    if excludeShadows:
        for i in range(len(frames)):
            frames[i][frames[i] == 127] = 0

def morphologicalCleanup(frames, shape=cv2.MORPH_ELLIPSE, size=(2,2), operations=[cv2.MORPH_OPEN, cv2.MORPH_CLOSE]):
    kernel = cv2.getStructuringElement(shape, size)
    for i in range(len(frames)):
        for op in operations:
            frames[i] = cv2.morphologyEx(frames[i], op, kernel)

if __name__ == "__main__":
    histories = [100, 250, 500, 750, 1000]
    varThresholds = [8, 16, 32, 64, 128]
    shapes = [cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, cv2.MORPH_CROSS]
    sizes = [(2,2), (3,3), (4,4), (5,5)]
    ops = [[cv2.MORPH_ERODE], 
           [cv2.MORPH_DILATE], 
           [cv2.MORPH_OPEN], 
           [cv2.MORPH_CLOSE], 
           [cv2.MORPH_OPEN, cv2.MORPH_CLOSE], 
           [cv2.MORPH_CLOSE, cv2.MORPH_OPEN],
           [cv2.MORPH_BLACKHAT],
           [cv2.MORPH_TOPHAT],
           [cv2.MORPH_GRADIENT]
           ]
    
    video_path = "small_video.mp4"
    frames = utils.loadVideo(video_path, 30)
    if frames is not None:
        for i in range(len(histories)):
            for j in range(len(varThresholds)):
                for k in range(len(shapes)):
                    for l in range(len(sizes)):
                        for m in range(len(ops)):
                            temp_frames = frames.copy()
                            subtractBackground(temp_frames, histories[i], varThresholds[j])
                            morphologicalCleanup(temp_frames, shapes[k], sizes[l], ops[m])
                            utils.saveGrayscale(temp_frames, f"output_mog_h{i}_v{j}_s{k}_sz{l}_op{m}.mp4")
        subtractBackground(frames)
        morphologicalCleanup(frames)
        utils.saveGrayscale(frames, "SandPandG_22_mog_500_32.mp4")
        #show(frames)