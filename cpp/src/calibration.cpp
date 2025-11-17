#include "calibration.hpp"
#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

Calibration Calibration::from_json_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open calibration file: " + path);
    }

    json j;
    in >> j;

    Calibration c;

    // H_matrix: 3x3
    c.H_matrix = cv::Mat(3, 3, CV_64F);
    auto Hj = j.at("H_matrix");
    for (int r = 0; r < 3; ++r) {
        for (int k = 0; k < 3; ++k) {
            c.H_matrix.at<double>(r, k) = Hj.at(r).at(k).get<double>();
        }
    }

    // roi_polygon: [[x, y], ...]
    c.roi_polygon.clear();
    for (const auto& pt : j.at("roi_polygon")) {
        float x = pt.at(0).get<float>();
        float y = pt.at(1).get<float>();
        c.roi_polygon.emplace_back(x, y);
    }

    c.H_out        = j.at("H_out").get<int>();
    c.W_out        = j.at("W_out").get<int>();
    c.lanes_y_pxs  = j.at("lanes_y_pxs").get<std::vector<int>>();
    c.scale_lambda = j.at("scale_lambda").get<double>();

    return c;
}