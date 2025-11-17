#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "calibration.hpp"
#include "background.hpp"
#include "detection.hpp"
#include "tracker.hpp"

int main(int argc, char** argv) {
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

        const int W_out = calib.W_out;
        const int H_out = calib.H_out;
        const int fps   = 30;

        // Background model in warped space
        const int bg_window = 120;
        Background background(W_out, H_out, bg_window);

        // Tracker
        Tracker tracker(1.0 / fps, 5.0, 3.0, 50.0, 10, 2);

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

        while (true) {
            if (!cap.read(raw) || raw.empty()) {
                break;
            }

            // Convert to grayscale
            if (raw.channels() == 3) {
                cv::cvtColor(raw, gray, cv::COLOR_BGR2GRAY);
            } else if (raw.channels() == 4) {
                cv::cvtColor(raw, gray, cv::COLOR_BGRA2GRAY);
            } else {
                gray = raw;
            }

            // Lazily build source-space ROI mask once, based on original frame size
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

            // Apply ROI in original space
            cv::bitwise_and(gray, roi_mask_src, gray_roi);

            // Warp to rectified road coordinates
            cv::warpPerspective(
                gray_roi, warped,
                calib.H_matrix,
                cv::Size(W_out, H_out),
                cv::INTER_LINEAR,
                cv::BORDER_CONSTANT
            );

            // Update background with warped ROI image (no lane lines yet)
            background.update(warped);
            if (background.loaded() < background.window_size()) {
                ++frame_count;
                continue;
            }

            // Background subtraction on warped+lanes image
            mask = background.background_subtract(warped, 16, 50, true, 10, 90);
            if (mask.empty()) {
                ++frame_count;
                continue;
            }

            // Draw lane "knife" lines directly to separate blobs from cars side by side
            for (int lane_y : calib.lanes_y_pxs) {
                if (lane_y >= 0 && lane_y < H_out) {
                    cv::line(
                        mask,
                        cv::Point(0, lane_y),
                        cv::Point(W_out, lane_y),
                        cv::Scalar(0),  // black on gray
                        1
                    );
                }
            }

            // Fill holes in the mask
            mask_filled = fill_holes(mask);

            // Blob detection (do NOT draw inside detect_blobs, we'll draw ourselves)
            constexpr int kMinCarArea = 1600;
            BlobDetectionResult blobs =
                detect_blobs(mask_filled, kMinCarArea, -1, /*draw_boxes=*/false);

            // Prepare detections: bottom points of bounding boxes
            detections.clear();
            for (const auto& r : blobs.bboxes) {
                detections.emplace_back(
                    static_cast<double>(r.x),
                    static_cast<double>(r.y + r.height)
                );
            }

            // Tracker update
            tracker.update(detections, frame_count);

            if (kEnableVisualization) {
                // Start visualization from warped (already ROI’d + lane lines)
                cv::cvtColor(warped, vis, cv::COLOR_GRAY2BGR);

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
                    int   pid = t.id;

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
                    cv::Point speed_org(id_org.x,
                                        id_org.y + id_size.height + 4);

                    cv::putText(vis, id_text, id_org,
                                cv::FONT_HERSHEY_SIMPLEX, 0.6,
                                color, 2, cv::LINE_AA);
                    cv::putText(vis, speed_text, speed_org,
                                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                                color, 1, cv::LINE_AA);

                    // Short trail (last up to 25 positions)
                    const int max_len = 25;
                    int history_sz = static_cast<int>(t.history.size());
                    int start = std::max(0, history_sz - max_len);
                    for (int i = start; i + 1 < history_sz; ++i) {
                        cv::Point a(
                            static_cast<int>(t.history[i].pos.x()),
                            static_cast<int>(t.history[i].pos.y())
                        );
                        cv::Point b(
                            static_cast<int>(t.history[i+1].pos.x()),
                            static_cast<int>(t.history[i+1].pos.y())
                        );
                        cv::line(vis, a, b,
                                 cv::Scalar(255, 255, 255),
                                 2, cv::LINE_AA);
                    }
                }

                cv::imshow("Detection", vis);
                int key = cv::waitKey(1);
                if (key == 27) { // ESC
                    break;
                }
            }

            ++frame_count;
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
