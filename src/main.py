import time
import calibration
import detection
import cv2
import numpy as np
import os
import json
from video_stream import video
        

def main():
    time0 = time.perf_counter()
    input_video_path = "assets/video.avi"
    colour = False
    start_frame = 100
    original_fps = 50
    target_fps = 10
    frame_interval = original_fps // target_fps
    detection_frame_count = 1000
    video_background_window_size = 800
    visualize_detection = True
    video_resolution = (1920, 1080)  # (W, H)

    kalman_sigma_a = 8.0
    kalman_sigma_z = 4.0
    kalman_max_association_distance = 60
    kalman_max_age = 8
    kalman_min_hits = 2

    time1 = time.perf_counter()
    print(f"Setup time: {time1 - time0:.3f} seconds")

    video.set_config(input_video_path, frame_interval=frame_interval, colour=colour, make_background=True)
    video.start_background(window_size=video_background_window_size, W=video_resolution[0], H=video_resolution[1])
    time2 = time.perf_counter()
    print(f"Video instantiation time: {time2 - time1:.3f} seconds")

    # Background population
    for _ in range(video_background_window_size):
        video.get_frame()
    time3 = time.perf_counter()

    # Calibration
    print(f"Initial background population time: {time3 - time2:.3f} seconds")
    H_matrix, roi_polygon, H_out, W_out, lanes_y_pxs, calibration_background_object = calibration.calibrate()
    time4 = time.perf_counter()
    print(f"Calibration computation time: {time4 - time3:.3f} seconds")

    # Detection
    detected_frame_count = 0
    d = detection.Detection()
    d._background = calibration_background_object
    d.insert_calibration(H_matrix, roi_polygon, H_out, W_out, lanes_y_pxs, target_fps)
    d.start_tracker(kalman_sigma_a, kalman_sigma_z,
                    kalman_max_association_distance, kalman_max_age, kalman_min_hits)
    
    while(detected_frame_count < detection_frame_count):
        time_before_detection = time.perf_counter()
        debug_frame = d.process_frame(visualize=visualize_detection)
        time_after_detection = time.perf_counter()
        print(f"Detection time for frame {detected_frame_count}: {time_after_detection - time_before_detection:.4f} seconds")

        if visualize_detection and debug_frame is not None:
            cv2.imshow("Detection", debug_frame)
            cv2.waitKey(1)

        detected_frame_count += 1
    
    all_tracks = d.tracker.get_average_velocity_per_track()
    with open("final_debug/tracks_summary.txt", "w") as fout:
        fout.write("Track ID | Detections | Avg Speed (px/s)\n")
        fout.write("---------|------------|------------------\n")
        for track_id, num_detections, avg_speed in all_tracks:
            fout.write(f"{track_id:8d} | {num_detections:10d} | {avg_speed:16.3f}\n")
    print(f"Saved: final_debug/tracks_summary.txt ({len(all_tracks)} tracks)")

    # Create detailed tracks JSON
    tracks_data = {"cars": []}
    
    # Get all tracks including finished ones
    all_track_objects = d.tracker._tracks + [t for t, _ in d.tracker._finished_tracks]
    
    for track in all_track_objects:
        if len(track.history) < 1:
            continue
            
        # Calculate average velocity
        avg_vel_x = d.tracker.get_track_average_velocity(track)
        
        # Build frame stats from history
        frame_stats = []
        for frame_count, pos, vel_x in track.history:
            frame_stats.append({
                "frame_count": frame_count,
                "pos_x": pos[0],
                "pos_y": pos[1],
                "vel_x": vel_x
            })
        
        car_data = {
            "id": track.id,
            "average_velocity": avg_vel_x,
            "frame_stats": frame_stats
        }
        tracks_data["cars"].append(car_data)
    
    with open("final_debug/tracks.json", "w") as fout:
        json.dump(tracks_data, fout, indent=2)
    print(f"Saved: final_debug/tracks.json ({len(tracks_data['cars'])} cars)")


        

if __name__ == "__main__":
    main()