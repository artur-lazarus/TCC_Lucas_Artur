#pragma once

#include <cstdint>
#include <vector>
#include <opencv2/core.hpp>

class Background {
public:
    // W, H: image width/height; window_size: sliding window length (#frames)
    Background(int width, int height, int window_size);

    // Add one new frame (must be CV_8UC1, size HxW)
    void update(const cv::Mat& frame);

    // Get background percentile image (e.g. 50 = median). Returns empty Mat if not ready.
    cv::Mat get_background_percentile(int percentile);

    // Foreground mask via background subtraction.
    // threshold: abs-diff threshold (uint8).
    // subtract_percentile: percentile of background to use (e.g. 50).
    // normalize: if true, match global brightness percentiles between frame and background.
    // norm_p_low / norm_p_high: percentiles for normalization (e.g. 10, 90).
    cv::Mat background_subtract(const cv::Mat& frame,
                                int threshold,
                                int subtract_percentile = 50,
                                bool normalize = false,
                                int norm_p_low = 10,
                                int norm_p_high = 90);

    int loaded() const      { return loaded_; }
    int window_size() const { return size_; }

private:
    int W_;
    int H_;
    int NPX_;
    int size_;                // sliding window length

    // hist_[pixel * 256 + bin]
    std::vector<std::uint16_t> hist_;
    // ring_[pixel * size_ + t]
    std::vector<std::uint8_t>  ring_;

    int  ring_head_;
    int  loaded_;
    bool updated_since_last_bg_;

    int     last_bg_percentile_;
    cv::Mat last_bg_;         // cached background image (CV_8UC1)

    // Compute percentile image into last_bg_ and return it.
    cv::Mat compute_background_percentile_image(int percentile);

    // Compute two percentiles from a uint8 buffer via histogram (fast).
    static void compute_two_percentiles_uint8(const std::uint8_t* data,
                                              int N,
                                              int p_low,
                                              int p_high,
                                              int& out_low,
                                              int& out_high);
};