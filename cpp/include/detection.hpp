#pragma once

#include <opencv2/core.hpp>
#include <vector>

// Result of blob detection
struct BlobDetectionResult {
    // BGR image derived from the input mask, with rectangles if draw_boxes == true
    cv::Mat output_img;
    std::vector<cv::Rect> bboxes;
    std::vector<int>      areas;
};

/**
 * Fast blob detection based on connectedComponentsWithStats.
 *
 * mask       : CV_8UC1 binary image
 * min_area   : minimum area in pixels
 * max_area   : maximum area in pixels; if < 0, no upper limit
 * draw_boxes : if true, draw green rectangles on output_img
 *
 * Returns: BlobDetectionResult { output_img (BGR), bboxes, areas }
 */
BlobDetectionResult detect_blobs(const cv::Mat& mask,
                                 int min_area,
                                 int max_area = -1,
                                 bool draw_boxes = true);

/**
 * Hole filling using flood fill, equivalent to the Python fill_holes().
 *
 * mask : CV_8UC1 binary image
 *
 * Returns: CV_8UC1 image with holes filled.
 */
cv::Mat fill_holes(const cv::Mat& mask);