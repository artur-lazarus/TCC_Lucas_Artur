#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>

class Calibration {
public:
    cv::Mat H_matrix;                        // 3x3
    std::vector<cv::Point2f> roi_polygon;    // [ [x,y], ... ]
    int H_out = 0;
    int W_out = 0;
    std::vector<int> lanes_y_pxs;
    double scale_lambda = 1.0;

    static Calibration from_json_file(const std::string& path);
};