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

def load_calibration_file(path):
    """
    Loads calibration parameters from a JSON file.

    Parameters
    ----------
    path : str
        Input JSON file path.

    Returns
    -------
    tuple
        (H_matrix, roi_polygon, H_out, W_out, lanes_y_pxs, scale_lambda)
        where H_matrix is a numpy 3x3 array and roi_polygon is a list of tuples.
    """
    with open(path, "r") as f:
        data = json.load(f)
    
    # Convert H_matrix to numpy array
    H_matrix = np.array(data["H_matrix"], dtype=np.float64)
    
    # Convert roi_polygon to list of tuples
    roi_polygon = [(float(x), float(y)) for x, y in data["roi_polygon"]]
    
    H_out = int(data["H_out"])
    W_out = int(data["W_out"])
    lanes_y_pxs = [int(v) for v in data["lanes_y_pxs"]]
    scale_lambda = float(data["scale_lambda"])
    
    print(f"Calibration loaded from {path}")
    
    return H_matrix, roi_polygon, H_out, W_out, lanes_y_pxs, scale_lambda


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
    # Configuration flags
    cpp_detection = False  # Set to True to run C++ detection instead of Python detection
    load_calibration_from_file = False  # Set to True to load calibration from JSON instead of running calibration
    cpp_file = "cpp/src/tcc"

    time0 = time.perf_counter()
    input_video_path = "assets/combined_all_videos.mp4"
    json_calibration_path = "test_output/eldorado.json"
    colour = False
    start_frame = 0
    target_fps = 10
    detection_frame_count = 300
    video_background_window_size = 800
    visualize_detection = True
    video_resolution = (1920, 1080)  # (W, H)
    max_road_length = 40

    kalman_sigma_a = 400.0
    kalman_sigma_z = 2.0
    kalman_max_association_distance_m = 2.6
    kalman_max_age = 8
    kalman_min_hits = 2

    time1 = time.perf_counter()
    print(f"Setup time: {time1 - time0:.3f} seconds")

    video.set_config(input_video_path, target_fps, colour=colour, make_background=True)
    video.start_background(window_size=video_background_window_size, W=video_resolution[0], H=video_resolution[1])
    if start_frame > 0:
        video.jump_to_frame(start_frame)
    time2 = time.perf_counter()
    print(f"Video instantiation time: {time2 - time1:.3f} seconds")

    # Background population
    last_time = time.perf_counter()
    for _ in range(video_background_window_size):
        if _ % 50 == 0:
            print(f"Background population: {_}/{video_background_window_size} - {time.perf_counter() - last_time:.3f}sec")
            last_time = time.perf_counter()
        video.get_frame()
    time3 = time.perf_counter()

    # Calibration - either run calibration or load from file
    print(f"Initial background population time: {time3 - time2:.3f} seconds")
    
    if load_calibration_from_file:
        # Load calibration from JSON file
        print(f"Loading calibration from {json_calibration_path}...")
        H_matrix, roi_polygon, H_out, W_out, lanes_y_pxs, scale_lambda = load_calibration_file(json_calibration_path)
        calibration_background_object = video._background  # Use video's background object
        time4 = time.perf_counter()
        print(f"Calibration load time: {time4 - time3:.3f} seconds")
    else:
        # Run calibration
        H_matrix, roi_polygon, H_out, W_out, lanes_y_pxs, scale_lambda, calibration_background_object = calibration.calibrate(show_video=True, max_width_meters=max_road_length)
        time4 = time.perf_counter()
        print(f"Calibration computation time: {time4 - time3:.3f} seconds")
        
        # Save calibration to JSON file
        write_calibration_file(
            json_calibration_path,
            H_matrix,
            roi_polygon,
            H_out,
            W_out,
            lanes_y_pxs,
            scale_lambda
        )

    if not cpp_detection:
        # Detection
        detected_frame_count = 0
        d = detection.Detection()
        d._background = calibration_background_object
        d.insert_calibration(H_matrix, roi_polygon, H_out, W_out, lanes_y_pxs, scale_lambda, target_fps)
        kalman_max_association_distance_px = kalman_max_association_distance_m / scale_lambda
        d.start_tracker(scale_lambda, kalman_sigma_a, kalman_sigma_z,
                        kalman_max_association_distance_px, kalman_max_age, kalman_min_hits)
        
        while(detected_frame_count < detection_frame_count):
            time_before_detection = time.perf_counter()
            debug_frame = d.process_frame(visualize=visualize_detection)
            time_after_detection = time.perf_counter()
            #print(f"Detection time for frame {detected_frame_count}: {time_after_detection - time_before_detection:.4f} seconds")

            if visualize_detection and debug_frame is not None:
                cv2.imshow("Detection", debug_frame)
                cv2.waitKey(1)

            detected_frame_count += 1
            #time.sleep(1)
        if d.video_writer is not None:
            d.video_writer.release()
        
        all_tracks = d.tracker.get_average_velocity_per_track()
        with open("final_debug/tracks_summary.txt", "w") as fout:
            fout.write("Track ID | Detections | Avg Speed (px/s) | Avg Speed (km/h)\n")
            fout.write("---------|------------|------------------|------------------\n")
            for track_id, num_detections, avg_speed in all_tracks:
                avg_speed_kmh = avg_speed * 3.6*scale_lambda  # Convert px/s to km/h assuming scale_lambda is in meters/pixel
                fout.write(f"{track_id:8d} | {num_detections:10d} | {avg_speed:16.3f} | {avg_speed_kmh:16.3f}\n")
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
        #start_cpp_detection(json_calibration_path)
        pass

        

if __name__ == "__main__":
    main()
