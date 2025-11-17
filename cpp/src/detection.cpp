#include "detection.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>  // optional, but harmless
#include <algorithm>
#include <iostream>

BlobDetectionResult detect_blobs(const cv::Mat& mask,
                                 int min_area,
                                 int max_area,
                                 bool draw_boxes)
{
    BlobDetectionResult result;

    // Expect single-channel 8-bit
    CV_Assert(mask.type() == CV_8UC1);

    // Convert to BGR once
    cv::cvtColor(mask, result.output_img, cv::COLOR_GRAY2BGR);

    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(
        mask, labels, stats, centroids, 8, CV_32S
    );

    result.bboxes.clear();
    result.areas.clear();
    result.bboxes.reserve(std::max(0, num_labels - 1));
    result.areas.reserve(std::max(0, num_labels - 1));

    // Label 0 is background
    for (int i = 1; i < num_labels; ++i) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area < min_area) {
            continue;
        }
        if (max_area > 0 && area > max_area) {
            continue;
        }

        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        result.bboxes.emplace_back(x, y, w, h);
        result.areas.push_back(area);
    }

    if (draw_boxes) {
        for (const auto& r : result.bboxes) {
            cv::rectangle(result.output_img, r, cv::Scalar(0, 255, 0), 2);
        }
    }

    return result;
}

cv::Mat fill_holes(const cv::Mat& mask)
{
    CV_Assert(mask.type() == CV_8UC1);

    const int h = mask.rows;
    const int w = mask.cols;

    // floodMask must be 2 pixels larger in each dimension
    cv::Mat flood_mask(h + 2, w + 2, CV_8UC1, cv::Scalar(0));

    cv::Mat im_flood = mask.clone();
    // Fill from (0,0) with value 255
    cv::floodFill(im_flood, flood_mask, cv::Point(0, 0), 255);

    cv::Mat im_flood_inv;
    cv::bitwise_not(im_flood, im_flood_inv);

    cv::Mat out;
    cv::bitwise_or(mask, im_flood_inv, out);
    return out;
}
