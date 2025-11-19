#pragma once

#include <cstdint>      // std::uint8_t, std::uint16_t
#include <vector>       // std::vector
#include <opencv2/core.hpp>  // cv::Mat

class Background {
public:
    // Old API for compatibility: uses 256 bins internally (full precision)
    Background(int width, int height, int window_size);

    // New API: choose arbitrary number of histogram bins in [1, 256]
    Background(int width, int height, int window_size, int num_bins);

    // Ingest one grayscale frame (CV_8UC1, fixed size W_ x H_)
    void update(const cv::Mat& frame);

    // Get background percentile image (e.g., 50 = median); returns CV_8UC1
    cv::Mat get_background_percentile(int percentile);

    // Utility: compute two percentiles of a flat uint8 buffer (used by normalization)
    static void compute_two_percentiles_uint8(const std::uint8_t* data,
                                              int N,
                                              int p_low,
                                              int p_high,
                                              int& out_low,
                                              int& out_high);

    // Background subtraction: returns binary mask (CV_8UC1)
    cv::Mat background_subtract(const cv::Mat& frame,
                                int threshold,
                                int subtract_percentile,
                                bool normalize,
                                int norm_p_low,
                                int norm_p_high);

    // --- NEW: small getters to keep old code working ---
    int loaded() const        { return loaded_; }
    int window_size() const   { return size_; }
    int num_bins() const      { return num_bins_; }  // optional, but handy
    // ---------------------------------------------------

private:
    int W_;
    int H_;
    int NPX_;
    int size_;
    int num_bins_;

    std::vector<std::uint16_t> hist_;
    std::vector<std::uint8_t>  ring_;

    int  ring_head_;
    int  loaded_;
    bool updated_since_last_bg_;
    int  last_bg_percentile_;
    cv::Mat last_bg_;

    cv::Mat compute_background_percentile_image(int percentile);
};
