#include "background.hpp"

#include <opencv2/imgproc.hpp>
#include <iostream>
#include <algorithm>
#include <cmath>

Background::Background(int width, int height, int window_size)
    : W_(width),
      H_(height),
      NPX_(width * height),
      size_(window_size),
      hist_(static_cast<std::size_t>(width) *
                static_cast<std::size_t>(height) * 256u,
            0),
      ring_(static_cast<std::size_t>(width) *
                static_cast<std::size_t>(height) *
                static_cast<std::size_t>(window_size),
            0),
      ring_head_(0),
      loaded_(0),
      updated_since_last_bg_(false),
      last_bg_percentile_(-1),
      last_bg_() {}

// ---------------------------------------------------------
void Background::update(const cv::Mat& frame) {
    // Expect grayscale 8-bit image with fixed size.
    CV_Assert(frame.type() == CV_8UC1);
    CV_Assert(frame.rows == H_ && frame.cols == W_);

    const std::uint8_t* f = frame.ptr<std::uint8_t>();

    // WARM-UP PHASE: fill window, no removals
    if (loaded_ < size_) {
        for (int i = 0; i < NPX_; ++i) {
            std::uint8_t v = f[i];
            hist_[static_cast<std::size_t>(i) * 256u + v] += 1;
            ring_[static_cast<std::size_t>(i) * static_cast<std::size_t>(size_) +
                  ring_head_] = v;
        }

        ring_head_ = (ring_head_ + 1) % size_;
        ++loaded_;
        // No need to touch updated_since_last_bg_ yet; we won't serve BG until ready.
        return;
    }

    // STEADY STATE: remove oldest, add newest
    for (int i = 0; i < NPX_; ++i) {
        std::size_t base_idx =
            static_cast<std::size_t>(i) * static_cast<std::size_t>(size_);
        std::uint8_t& old_v = ring_[base_idx + ring_head_];
        std::uint8_t  new_v = f[i];

        hist_[static_cast<std::size_t>(i) * 256u + old_v] -= 1;
        hist_[static_cast<std::size_t>(i) * 256u + new_v] += 1;
        old_v = new_v;
    }

    ring_head_ = (ring_head_ + 1) % size_;
    updated_since_last_bg_ = true;
}

// ---------------------------------------------------------
cv::Mat Background::compute_background_percentile_image(int percentile) {
    if (last_bg_.empty() ||
        last_bg_.rows != H_ || last_bg_.cols != W_ ||
        last_bg_.type() != CV_8UC1) {
        last_bg_.create(H_, W_, CV_8UC1);
    }

    std::uint8_t* out = last_bg_.ptr<std::uint8_t>();

    // Target count for this percentile
    const int target = static_cast<int>((percentile / 100.0) * size_);

    for (int i = 0; i < NPX_; ++i) {
        const std::uint16_t* row_hist =
            &hist_[static_cast<std::size_t>(i) * 256u];

        std::uint32_t cum = 0;
        std::uint8_t  val = 255;

        for (int b = 0; b < 256; ++b) {
            cum += row_hist[b];
            if (cum >= static_cast<std::uint32_t>(target)) {
                val = static_cast<std::uint8_t>(b);
                break;
            }
        }
        out[i] = val;
    }

    return last_bg_;
}

// ---------------------------------------------------------
cv::Mat Background::get_background_percentile(int percentile) {
    if (loaded_ < size_) {
        std::cerr << "[Background] Not enough frames yet (" << loaded_
                  << "/" << size_ << ").\n";
        return cv::Mat();
    }

    if (!updated_since_last_bg_ &&
        !last_bg_.empty() &&
        last_bg_percentile_ == percentile) {
        // Cached value still valid
        return last_bg_;
    }

    cv::Mat bg = compute_background_percentile_image(percentile);
    last_bg_percentile_   = percentile;
    updated_since_last_bg_ = false;
    return bg;
}

// ---------------------------------------------------------
void Background::compute_two_percentiles_uint8(const std::uint8_t* data,
                                               int N,
                                               int p_low,
                                               int p_high,
                                               int& out_low,
                                               int& out_high)
{
    // Clamp percentiles
    p_low  = std::max(0, std::min(100, p_low));
    p_high = std::max(0, std::min(100, p_high));

    std::uint32_t hist[256] = {0};

    for (int i = 0; i < N; ++i) {
        ++hist[data[i]];
    }

    std::uint32_t target_low  =
        static_cast<std::uint32_t>((p_low  / 100.0) * N);
    std::uint32_t target_high =
        static_cast<std::uint32_t>((p_high / 100.0) * N);

    std::uint32_t cum = 0;
    bool got_low  = false;
    bool got_high = false;
    out_low  = 0;
    out_high = 0;

    for (int v = 0; v < 256; ++v) {
        cum += hist[v];

        if (!got_low && cum >= target_low) {
            out_low = v;
            got_low = true;
        }
        if (!got_high && cum >= target_high) {
            out_high = v;
            got_high = true;
        }
        if (got_low && got_high) break;
    }

    if (!got_low)  out_low  = 255;
    if (!got_high) out_high = 255;
}

// ---------------------------------------------------------
cv::Mat Background::background_subtract(const cv::Mat& frame,
                                        int threshold,
                                        int subtract_percentile,
                                        bool normalize,
                                        int norm_p_low,
                                        int norm_p_high)
{
    if (loaded_ < size_) {
        std::cerr << "ERROR: Background not initialized or not enough frames"
                     " loaded.\n";
        return cv::Mat();
    }

    cv::Mat bg = get_background_percentile(subtract_percentile);
    if (bg.empty()) {
        return cv::Mat();
    }

    CV_Assert(frame.type() == CV_8UC1);
    CV_Assert(frame.rows == H_ && frame.cols == W_);

    cv::Mat v_norm;

    if (normalize) {
        v_norm.create(H_, W_, CV_8UC1);

        const std::uint8_t* bg_ptr = bg.ptr<std::uint8_t>();
        const std::uint8_t* v_ptr  = frame.ptr<std::uint8_t>();
        std::uint8_t*       out    = v_norm.ptr<std::uint8_t>();

        int bg_low, bg_high;
        int v_low,  v_high;

        compute_two_percentiles_uint8(bg_ptr, NPX_, norm_p_low, norm_p_high,
                                      bg_low, bg_high);
        compute_two_percentiles_uint8(v_ptr,  NPX_, norm_p_low, norm_p_high,
                                      v_low,  v_high);

        float v_range = std::max(1.0f, static_cast<float>(v_high - v_low));
        float scale   = static_cast<float>(bg_high - bg_low) / v_range;
        float offset  = static_cast<float>(bg_low) - scale * v_low;

        for (int i = 0; i < NPX_; ++i) {
            float val_f = v_ptr[i] * scale + offset;
            int   iv    = static_cast<int>(std::lround(val_f));
            if (iv < 0)      iv = 0;
            else if (iv > 255) iv = 255;
            out[i] = static_cast<std::uint8_t>(iv);
        }
    } else {
        // No normalization: use frame as-is (shallow copy)
        v_norm = frame;
    }

    cv::Mat diff;
    cv::absdiff(bg, v_norm, diff);

    cv::Mat mask;
    cv::threshold(diff, mask, threshold, 255, cv::THRESH_BINARY);
    cv::medianBlur(mask, mask, 5);

    return mask;
}
