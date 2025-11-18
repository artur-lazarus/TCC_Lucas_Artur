import time
import calibration
import detection
import cv2
import numpy as np
import subprocess
import json
from video_stream import video
        

def start_cpp_detection(json_calibration_path):
    detection_file_path = "cpp/src/main_app"
    subprocess.run(["./" + detection_file_path, json_calibration_path])

def write_calibration_file(
    path,
    H_matrix,
    roi_polygon,
    H_out,
    W_out,
    lanes_y_pxs,
    scale_lambda
):
    """
    Writes calibration parameters to `path` in the exact JSON format
    expected by the C++ Calibration::from_json_file() reader.

    Parameters
    ----------
    path : str
        Output JSON file path.
    H_matrix : list[list[float]] or numpy 3x3 array
        3×3 homography matrix.
    roi_polygon : list[(x, y)]
        Polygon points in pixel coordinates.
    H_out : int
        Output height for dewarped view.
    W_out : int
        Output width for dewarped view.
    lanes_y_pxs : list[int]
        List of Y positions per lane in the BEV image.
    scale_lambda : float
        Scale parameter.
    """

    # Convert numpy arrays to Python lists if needed
    H_matrix_list = [[float(v) for v in row] for row in H_matrix]

    roi_list = [[float(x), float(y)] for (x, y) in roi_polygon]

    data = {
        "H_matrix": H_matrix_list,
        "roi_polygon": roi_list,
        "H_out": int(H_out),
        "W_out": int(W_out),
        "lanes_y_pxs": [int(v) for v in lanes_y_pxs],
        "scale_lambda": float(scale_lambda)
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Calibration JSON written to {path}")


def main():
    cpp_detection = False  # Set to True to run C++ detection instead of Python detection
    cpp_file = "cpp/src/tcc"

    time0 = time.perf_counter()
    input_video_path = "dataset/session2_left/video.avi"
    json_calibration_path = "calibration.json"
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
    H_matrix, roi_polygon, H_out, W_out, lanes_y_pxs, scale_lambda, calibration_background_object = calibration.calibrate(show_video=True)
    time4 = time.perf_counter()
    print(f"Calibration computation time: {time4 - time3:.3f} seconds")

    if not cpp_detection:
        # Detection
        detected_frame_count = 0
        d = detection.Detection()
        d._background = calibration_background_object
        d.insert_calibration(H_matrix, roi_polygon, H_out, W_out, lanes_y_pxs, scale_lambda, target_fps)
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
    else:
        write_calibration_file(
            json_calibration_path,
            H_matrix,
            roi_polygon,
            H_out,
            W_out,
            lanes_y_pxs,
            scale_lambda
        )
        start_cpp_detection(json_calibration_path)

        

if __name__ == "__main__":
    main()
