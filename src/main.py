import time
import calibration
import detection
import cv2
import numpy as np
import os

class VideoStream:
    def __init__(self, video_path, frame_interval=1, colour=True):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        self.frame_interval = frame_interval
        self.frame_count = 0
        self.colour = colour

    def get_frame(self):
        for _ in range(self.frame_interval):
            ret, frame = self.cap.read()
            if not ret:
                return None
        if not self.colour and frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame
        

def main():
    time0 = time.perf_counter()
    input_video_path = "dataset/session0_left/video.avi"
    colour = False
    start_frame = 100
    original_fps = 50
    target_fps = 10
    frame_interval = original_fps // target_fps
    calibration_frame_count, detection_frame_count = 1000,1000
    visualize_detection = True

    kalman_sigma_a = 8.0
    kalman_sigma_z = 4.0
    kalman_max_association_distance = 60
    kalman_max_age = 8
    kalman_min_hits = 2

    time1 = time.perf_counter()
    print(f"Setup time: {time1 - time0:.3f} seconds")


    video = VideoStream(input_video_path, frame_interval=frame_interval, colour=colour)
    time2 = time.perf_counter()
    print(f"Video load time: {time2 - time1:.3f} seconds")
    # Calibration
    video_frames_calibration = []
    for i in range(calibration_frame_count):
        video_frames_calibration.append(video.get_frame())
    time3 = time.perf_counter()
    print(f"Calibration frame extraction time: {time3 - time2:.3f} seconds")
    print(f"Average time per frame: {(time3 - time2)/calibration_frame_count:.4f} seconds")
    H_matrix, roi_polygon, H_out, W_out, lanes_y_pxs, calibration_background_object, vp_road, vp_vertical, K_matrix, r1, r2, r3, focal_length = calibration.calibrate(video_frames_calibration)
    time4 = time.perf_counter()
    print(f"Calibration computation time: {time4 - time3:.3f} seconds")
    # Detection
    detected_frame_count = 0
    d = detection.Detection()
    d._background = calibration_background_object
    d.insert_calibration(H_matrix, roi_polygon, H_out, W_out, lanes_y_pxs, target_fps)
    d.start_tracker(kalman_sigma_a, kalman_sigma_z,
                    kalman_max_association_distance, kalman_max_age, kalman_min_hits)
    
    os.makedirs("final_debug", exist_ok=True)
    np.savez("final_debug/calibration_params.npz",
             vp_road=vp_road, vp_vertical=vp_vertical,
             K_matrix=K_matrix, focal_length=focal_length,
             r1=r1, r2=r2, r3=r3,
             H_matrix=H_matrix, roi_polygon=roi_polygon,
             H_out=H_out, W_out=W_out, lanes_y_pxs=lanes_y_pxs)
    print("Saved: final_debug/calibration_params.npz")
    
    with open("final_debug/calibration_info.txt", "w") as fout:
        fout.write("CALIBRATION PARAMETERS\n")
        fout.write("=" * 50 + "\n\n")
        fout.write(f"Vanishing Points:\n")
        fout.write(f"  Road VP: ({vp_road[0]:.2f}, {vp_road[1]:.2f})\n")
        fout.write(f"  Vertical VP: ({vp_vertical[0]:.2f}, {vp_vertical[1]:.2f})\n\n")
        fout.write(f"Focal Length: {focal_length:.2f} px\n\n")
        fout.write(f"K Matrix:\n{K_matrix}\n\n")
        fout.write(f"Rotation Vectors:\n")
        fout.write(f"  r1 (road): {r1}\n")
        fout.write(f"  r2 (across): {r2}\n")
        fout.write(f"  r3 (up): {r3}\n\n")
        fout.write(f"Homography Matrix:\n{H_matrix}\n\n")
        fout.write(f"Output size: {W_out} x {H_out}\n")
        fout.write(f"ROI polygon: {roi_polygon.tolist()}\n")
        fout.write(f"Lane Y-pixels: {lanes_y_pxs}\n")
    print("Saved: final_debug/calibration_info.txt")
    
    first_frame = video_frames_calibration[0]
    
    
    roi_area = cv2.contourArea(roi_polygon)
    M = cv2.moments(roi_polygon)
    cx_roi = M['m10'] / M['m00'] if M['m00'] != 0 else 0
    cy_roi = M['m01'] / M['m00'] if M['m00'] != 0 else 0
    with open("final_debug/roi_stats.txt", "w") as fout:
        fout.write("ROI POLYGON STATISTICS\n")
        fout.write("=" * 50 + "\n\n")
        fout.write(f"Number of vertices: {len(roi_polygon)}\n")
        fout.write(f"Area: {roi_area:.2f} px²\n")
        fout.write(f"Centroid: ({cx_roi:.2f}, {cy_roi:.2f})\n")
        fout.write(f"Vertices:\n")
        for i, pt in enumerate(roi_polygon):
            fout.write(f"  {i}: ({pt[0]}, {pt[1]})\n")
    print("Saved: final_debug/roi_on_mask.png, final_debug/roi_stats.txt")
    
    first_warped = cv2.warpPerspective(first_frame, H_matrix, (W_out, H_out))
    cv2.imwrite("final_debug/first_warped.png", first_warped)
    print("Saved: final_debug/first_warped.png")
    
    vp_visual = cv2.cvtColor(first_frame, cv2.COLOR_GRAY2BGR)
    if 0 <= vp_road[0] < first_frame.shape[1] and 0 <= vp_road[1] < first_frame.shape[0]:
        cv2.circle(vp_visual, (int(vp_road[0]), int(vp_road[1])), 10, (0, 0, 255), -1)
        cv2.putText(vp_visual, "Road VP", (int(vp_road[0])+15, int(vp_road[1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    if 0 <= vp_vertical[0] < first_frame.shape[1] and 0 <= vp_vertical[1] < first_frame.shape[0]:
        cv2.circle(vp_visual, (int(vp_vertical[0]), int(vp_vertical[1])), 10, (255, 0, 0), -1)
        cv2.putText(vp_visual, "Vertical VP", (int(vp_vertical[0])+15, int(vp_vertical[1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.imwrite("final_debug/vanishing_points.png", vp_visual)
    print("Saved: final_debug/vanishing_points.png")
    while(detected_frame_count < detection_frame_count):
        frame = video.get_frame()
        if frame is not None:
            time_before_detection = time.perf_counter()
            debug_frame = d.process_frame(frame, visualize=visualize_detection)
            time_after_detection = time.perf_counter()
            print(f"Detection time for frame {detected_frame_count}: {time_after_detection - time_before_detection:.4f} seconds")
            if visualize_detection and debug_frame is not None:
                cv2.imshow("Detection", debug_frame)
                cv2.waitKey(1)
        if d.tracker.new_finished_tracks > 0:
            finished_tracks = d.tracker.retrieve_finished_tracks()
            for track, avg_speed in finished_tracks:
                print(f"Track {track.id} finished with average speed {avg_speed:.3f} px/s")
        detected_frame_count += 1
    
    all_tracks = d.tracker.get_average_velocity_per_track()
    with open("final_debug/tracks_summary.txt", "w") as fout:
        fout.write("Track ID | Detections | Avg Speed (px/s)\n")
        fout.write("---------|------------|------------------\n")
        for track_id, num_detections, avg_speed in all_tracks:
            fout.write(f"{track_id:8d} | {num_detections:10d} | {avg_speed:16.3f}\n")
    print(f"Saved: final_debug/tracks_summary.txt ({len(all_tracks)} tracks)")


        

if __name__ == "__main__":
    global video
    main()