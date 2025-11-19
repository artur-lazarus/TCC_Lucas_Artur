#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <chrono>

#include "calibration.hpp"
#include "background.hpp"
#include "detection.hpp"
#include "tracker.hpp"
#include <omp.h>

using my_clock_t = std::chrono::high_resolution_clock;
using ms         = std::chrono::duration<double, std::milli>;

struct StageTimer {
    ms total{0};
    std::size_t count = 0;

    void add(ms dt) {
        total += dt;
        ++count;
    }

    double avg_ms() const {
        return count > 0 ? total.count() / static_cast<double>(count) : 0.0;
    }
};

static constexpr uint8_t  min_car_area_m2             = 11;
static constexpr uint8_t  speed_limit_km_h            = 65;

static constexpr double   kalman_sigma_a              = 400.0;
static constexpr double   kalman_sigma_z              = 2.0;
static constexpr double   kalman_max_association_distance_m = 2.6;
static constexpr int      kalman_max_age              = 8;
static constexpr int      kalman_min_hits             = 2;

static constexpr uint8_t  frameIntervalBackgroundUpdate = 3;
static constexpr uint8_t  frameInterval                 = 5;


int main(int argc, char** argv) {
    // Just to verify OMP is working
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nt  = omp_get_num_threads();
        #pragma omp critical
        {
            std::cout << "Hello from thread " << tid << " / " << nt << "\n";
        }
    }

    const bool kEnableVisualization = true;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <calibration.json> [video_path]\n";
        return 1;
    }

    try {
        Calibration calib = Calibration::from_json_file(argv[1]);

        // Strict validation
        if (calib.H_matrix.empty() || calib.H_matrix.rows != 3 || calib.H_matrix.cols != 3) {
            throw std::runtime_error("Invalid calibration: H_matrix must be a valid 3x3 matrix");
        }
        if (calib.W_out <= 0 || calib.H_out <= 0) {
            throw std::runtime_error("Invalid calibration: W_out and H_out must be positive integers");
        }
        if (calib.roi_polygon.empty()) {
            throw std::runtime_error("Invalid calibration: roi_polygon must not be empty");
        }
        if (calib.lanes_y_pxs.empty()) {
            throw std::runtime_error("Invalid calibration: lanes_y_pxs must not be empty");
        }
        if (calib.scale_lambda <= 0.0) {
            throw std::runtime_error("Invalid calibration: scale_lambda must be positive");
        }

        uint16_t min_car_area_px = static_cast<uint16_t>(
            float(min_car_area_m2) / (calib.scale_lambda * calib.scale_lambda)
        );

        const int W_out = calib.W_out;
        const int H_out = calib.H_out;
        const int fps   = 10;

        // Background model in warped space
        const int bg_window = 200;
        Background background(W_out, H_out, bg_window, 64);  // 64 bins

        // Tracker
        Tracker tracker(1.0 / fps, calib.scale_lambda, kalman_sigma_a,
                        kalman_sigma_z, kalman_max_association_distance_m,
                        kalman_max_age, kalman_min_hits);

        // Video source
        cv::VideoCapture cap;
        if (argc >= 3) {
            cap.open(argv[2]);
        } else {
            cap.open(0);
        }
        if (!cap.isOpened()) {
            std::cerr << "Failed to open video source.\n";
            return 1;
        }

        // Reusable containers
        std::vector<Point> detections;
        detections.reserve(64);

        int frame_count = 0;
        cv::Mat raw, gray, gray_roi, warped, mask, mask_filled, vis;

        // ROI mask in ORIGINAL camera coordinates
        cv::Mat roi_mask_src;
        bool roi_initialized = false;

        cv::setUseOptimized(true);
        cv::setNumThreads(1);

        // ---- Timers ----
        StageTimer timer_read;
        StageTimer timer_gray;
        StageTimer timer_roi;
        StageTimer timer_warp;
        StageTimer timer_bg_update;
        StageTimer timer_bg_subtract;
        StageTimer timer_blobs;      // fill_holes + detect_blobs
        StageTimer timer_tracking;
        StageTimer timer_visualize;
        StageTimer timer_total;

        bool profiling_enabled = false;  // set to true after background warmup is done

        while (true) {
            auto frame_start = my_clock_t::now();
            std::cout << "A";

            // ---- READ (with frame skipping by frameInterval) ----
            bool ok = true;
            auto t_read_start = my_clock_t::now();
            for (int i = 0; i < frameInterval; ++i) {
                if (!cap.read(raw) || raw.empty()) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                std::cout << "End of video or capture error.\n";
                break;
            }
            if (profiling_enabled) {
                timer_read.add(ms(my_clock_t::now() - t_read_start));
            }

            // ---- GRAYSCALE ----
            auto t_gray_start = my_clock_t::now();
            if (raw.channels() == 3) {
                cv::cvtColor(raw, gray, cv::COLOR_BGR2GRAY);
            } else if (raw.channels() == 4) {
                cv::cvtColor(raw, gray, cv::COLOR_BGRA2GRAY);
            } else {
                gray = raw;
            }
            if (profiling_enabled) {
                timer_gray.add(ms(my_clock_t::now() - t_gray_start));
            }
            std::cout << "B";

            // ---- ROI MASK INITIALIZATION (one-time) ----
            if (!roi_initialized) {
                roi_mask_src = cv::Mat(gray.rows, gray.cols, CV_8UC1, cv::Scalar(0));
                std::vector<cv::Point> roi_pts;
                roi_pts.reserve(calib.roi_polygon.size());
                for (const auto& p : calib.roi_polygon) {
                    roi_pts.emplace_back(static_cast<int>(p.x), static_cast<int>(p.y));
                }
                std::vector<std::vector<cv::Point>> pts_vec{roi_pts};
                cv::fillPoly(roi_mask_src, pts_vec, cv::Scalar(255));
                roi_initialized = true;
            }
            std::cout << "C";

            // ---- APPLY ROI ----
            auto t_roi_start = my_clock_t::now();
            cv::bitwise_and(gray, roi_mask_src, gray_roi);
            if (profiling_enabled) {
                timer_roi.add(ms(my_clock_t::now() - t_roi_start));
            }

            // ---- WARP ----
            auto t_warp_start = my_clock_t::now();
            cv::warpPerspective(gray_roi, warped, calib.H_matrix,
                                cv::Size(W_out, H_out),
                                cv::INTER_LINEAR,
                                cv::BORDER_CONSTANT);
            if (profiling_enabled) {
                timer_warp.add(ms(my_clock_t::now() - t_warp_start));
            }
            std::cout << "D";

            // ---- BACKGROUND UPDATE ----
            if (frame_count % frameIntervalBackgroundUpdate == 0) {
                auto t_bg_upd_start = my_clock_t::now();
                background.update(warped);
                if (profiling_enabled) {
                    timer_bg_update.add(ms(my_clock_t::now() - t_bg_upd_start));
                }
            }

            // ---- BACKGROUND WARMUP CHECK ----
            if (background.loaded() < background.window_size()) {
                ++frame_count;
                continue;  // don't profile warmup frames
            }

            // First time we get past warmup, enable profiling for next frames
            if (!profiling_enabled) {
                profiling_enabled = true;
            }

            std::cout << "E";

            // ---- BACKGROUND SUBTRACT ----
            auto t_bg_sub_start = my_clock_t::now();
            mask = background.background_subtract(warped, 16, 50, false, 10, 90);
            if (profiling_enabled) {
                timer_bg_subtract.add(ms(my_clock_t::now() - t_bg_sub_start));
            }
            if (mask.empty()) {
                std::cerr << "background_subtract returned empty mask\n";
                ++frame_count;
                continue;
            }
            std::cout << "F";

            // ---- KNIFE LINES BETWEEN LANES ----
            for (int lane_y : calib.lanes_y_pxs) {
                if (lane_y >= 0 && lane_y < H_out) {
                    cv::line(mask,
                             cv::Point(0, lane_y),
                             cv::Point(W_out, lane_y),
                             cv::Scalar(0),
                             1);
                }
            }

            // ---- FILL HOLES + BLOB DETECTION (timed together as "blobs") ----
            auto t_blobs_start = my_clock_t::now();

            mask_filled = fill_holes(mask);
            if (mask_filled.empty()) {
                std::cerr << "fill_holes returned empty mask\n";
                ++frame_count;
                continue;
            }
            std::cout << "G";

            BlobDetectionResult blobs = detect_blobs(mask_filled, min_car_area_px, -1, /*draw_boxes=*/false);

            detections.clear();
            for (const auto& r : blobs.bboxes) {
                detections.emplace_back(
                    static_cast<double>(r.x),
                    static_cast<double>(r.y + r.height)
                );
            }

            if (profiling_enabled) {
                timer_blobs.add(ms(my_clock_t::now() - t_blobs_start));
            }

            // ---- TRACKING ----
            auto t_track_start = my_clock_t::now();
            tracker.update(detections, frame_count);
            if (profiling_enabled) {
                timer_tracking.add(ms(my_clock_t::now() - t_track_start));
            }
            std::cout << "H";

            // ---- VISUALIZATION ----
            auto t_vis_start = my_clock_t::now();
            if (kEnableVisualization) {
                // Start visualization from warped mask (after fill_holes)
                cv::cvtColor(mask_filled, vis, cv::COLOR_GRAY2BGR);

                // Draw boxes + bottom points
                for (const auto& r : blobs.bboxes) {
                    cv::rectangle(vis, r, cv::Scalar(0, 255, 0), 2);
                    cv::Point bottom_pt(r.x, r.y + r.height);
                    cv::circle(vis, bottom_pt, 4,
                               cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
                }

                // Draw tracks
                const auto& tracks = tracker.tracks_ref();
                for (const auto& t : tracks) {
                    Point pos = t.position();
                    Point vel = t.velocity();
                    int pid = t.id;

                    cv::Point p(
                        static_cast<int>(pos.x()),
                        static_cast<int>(pos.y())
                    );
                    cv::Scalar color(
                        (37 * (pid % 7) + 50) & 0xFF,
                        (83 * (pid % 5) + 50) & 0xFF,
                        (127 * (pid % 3) + 50) & 0xFF
                    );
                    cv::circle(vis, p, 6, color, -1, cv::LINE_AA);

                    std::string id_text    = "ID " + std::to_string(pid);
                    std::string speed_text = "vx: " + std::to_string(vel.x());
                    int base = 0;
                    cv::Size id_size = cv::getTextSize(
                        id_text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &base
                    );

                    cv::Point id_org(p.x + 8, p.y - 8);
                    cv::Point speed_org(id_org.x, id_org.y + id_size.height + 4);
                    cv::putText(vis, id_text,   id_org,   cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv::LINE_AA);
                    cv::putText(vis, speed_text, speed_org, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);

                    // Short trail (last up to 25 positions)
                    int max_len    = 25;
                    int history_sz = static_cast<int>(t.history.size());
                    int start      = std::max(0, history_sz - max_len);
                    for (int i = start; i + 1 < history_sz; ++i) {
                        cv::Point a(
                            static_cast<int>(t.history[i].pos.x()),
                            static_cast<int>(t.history[i].pos.y())
                        );
                        cv::Point b(
                            static_cast<int>(t.history[i+1].pos.x()),
                            static_cast<int>(t.history[i+1].pos.y())
                        );
                        cv::line(vis, a, b, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
                    }
                }

                cv::imshow("Detection", vis);
                int key = cv::waitKey(1);
                if (key == 27) {
                    // ESC pressed; you can uncomment if you want to quit:
                    // break;
                }
            }
            if (profiling_enabled) {
                timer_visualize.add(ms(my_clock_t::now() - t_vis_start));
            }
            std::cout << "I";

            // ---- TOTAL FRAME TIME ----
            if (profiling_enabled) {
                timer_total.add(ms(my_clock_t::now() - frame_start));
            }

            ++frame_count;

            // ---- PRINT EVERY 30 FULLY PROFILED FRAMES ----
            if (profiling_enabled && timer_total.count > 0 && (timer_total.count % 30 == 0)) {
                std::cout
                    << "\n------------------------------\n"
                    << "After " << timer_total.count << " processed frames\n"
                    << "Average timing (ms):\n"
                    << "read:        " << timer_read.avg_ms()        << "\n"
                    << "grayscale:   " << timer_gray.avg_ms()        << "\n"
                    << "roi:         " << timer_roi.avg_ms()         << "\n"
                    << "warp:        " << timer_warp.avg_ms()        << "\n"
                    << "bg_update*:  " << timer_bg_update.avg_ms()   << "  (*per update call)\n"
                    << "bg_subtract: " << timer_bg_subtract.avg_ms() << "\n"
                    << "blobs+fill:  " << timer_blobs.avg_ms()       << "\n"
                    << "tracking:    " << timer_tracking.avg_ms()    << "\n"
                    << "visualize:   " << timer_visualize.avg_ms()   << "\n"
                    << "TOTAL:       " << timer_total.avg_ms()       << "\n\n";
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
