#include "background.hpp"

#include <opencv2/imgproc.hpp>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <chrono>

// Small alias for timing
using my_clock_t = std::chrono::high_resolution_clock;
using ms         = std::chrono::duration<double, std::milli>;

// Portable restrict macro (helps auto-vectorization / alias analysis)
#if defined(__GNUC__) || defined(__clang__)
#  define BG_RESTRICT __restrict__
#else
#  define BG_RESTRICT
#endif

// ---- Small helper functions for quantization ----

// Map intensity v in [0,255] to a bin in [0, num_bins-1]
static inline constexpr int intensity_to_bin(std::uint8_t v, int num_bins)
{
    // Scale linearly: bin = floor(v * num_bins / 256)
    // Implemented as multiply + shift to avoid division.
    // Guarantees 0 <= bin < num_bins for 0 <= v <= 255, 1 <= num_bins <= 256
    return (static_cast<unsigned>(v) * static_cast<unsigned>(num_bins)) >> 8;
}

// Map a bin index in [0, num_bins-1] back to a representative intensity in [0,255]
static inline constexpr std::uint8_t bin_to_intensity(int bin, int num_bins)
{
    // Use the center of the bin:
    // center ≈ ( (2*bin+1) / (2*num_bins) ) * 256
    int iv = ( (2 * bin + 1) * 256 ) / (2 * num_bins);
    if (iv < 0)        iv = 0;
    else if (iv > 255) iv = 255;
    return static_cast<std::uint8_t>(iv);
}

// ---------------------------------------------------------
// Old 3-arg ctor kept for compatibility: defaults to 256 bins (full precision)
Background::Background(int width, int height, int window_size)
    : Background(width, height, window_size, 256)
{}

// New ctor with explicit bin count: 1 <= num_bins <= 256
Background::Background(int width, int height, int window_size, int num_bins)
    : W_(width),
      H_(height),
      NPX_(width * height),
      size_(window_size),
      num_bins_(std::max(1, std::min(256, num_bins))),  // clamp to [1,256]
      // Histogram: [NPX_ x num_bins_], counts in uint16_t
      hist_(static_cast<std::size_t>(width) *
                static_cast<std::size_t>(height) *
                static_cast<std::size_t>(num_bins_),
            0),
      // Ring buffer: [time][pixel], raw intensities per frame (0..255)
      ring_(static_cast<std::size_t>(width) *
                static_cast<std::size_t>(height) *
                static_cast<std::size_t>(window_size),
            0),
      ring_head_(0),
      loaded_(0),
      updated_since_last_bg_(false),
      last_bg_percentile_(-1),
      last_bg_()
{}

// ---------------------------------------------------------
void Background::update(const cv::Mat& frame) {
    // Expect grayscale 8-bit image with fixed size.
    auto tb0 = my_clock_t::now();

    CV_Assert(frame.type() == CV_8UC1);
    CV_Assert(frame.rows == H_ && frame.cols == W_);
    auto tb1 = my_clock_t::now();

    const std::uint8_t* BG_RESTRICT f = frame.ptr<std::uint8_t>();
    auto tb2 = my_clock_t::now();

    if (loaded_ < size_) {
        // WARM-UP: only add, no removals
        const int size   = size_;
        const int head   = ring_head_;
        const int nbins  = num_bins_;
        const int npx    = NPX_;

        std::uint16_t* BG_RESTRICT hist_ptr = hist_.data();
        std::uint8_t*  BG_RESTRICT ring_ptr = ring_.data();

        // Ring layout is [time][pixel]
        std::uint8_t* BG_RESTRICT ring_col =
            ring_ptr + static_cast<std::size_t>(head) *
                        static_cast<std::size_t>(npx);

        auto tb3 = my_clock_t::now();

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < npx; ++i) {
            std::uint8_t v  = f[i];                       // raw intensity 0..255
            int bin         = intensity_to_bin(v, nbins); // 0..nbins-1

            std::uint16_t* hist_row = hist_ptr +
                                      static_cast<std::size_t>(i) *
                                      static_cast<std::size_t>(nbins);

            hist_row[bin] += 1;
            // ring[head][i] = v
            ring_col[i] = v;  // keep raw intensity in ring
        }
        auto tb4 = my_clock_t::now();

        ring_head_ = (ring_head_ + 1) % size_;
        ++loaded_;
        auto tb5 = my_clock_t::now();
        return;
    }

    // STEADY STATE: remove oldest, add newest
    auto tb3 = my_clock_t::now();
    const int size   = size_;
    const int head   = ring_head_;
    const int nbins  = num_bins_;
    const int npx    = NPX_;

    std::uint16_t* BG_RESTRICT hist_ptr = hist_.data();
    std::uint8_t*  BG_RESTRICT ring_ptr = ring_.data();

    // ring[head][i] row (oldest time slice for all pixels)
    std::uint8_t* BG_RESTRICT ring_col =
        ring_ptr + static_cast<std::size_t>(head) *
                    static_cast<std::size_t>(npx);

    auto tb4 = my_clock_t::now();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < npx; ++i) {
        std::uint16_t* hist_row = hist_ptr +
                                  static_cast<std::size_t>(i) *
                                  static_cast<std::size_t>(nbins);

        // Old value from ring: raw intensity 0..255
        std::uint8_t& old_v = ring_col[i];  // ring[head][i]
        std::uint8_t  new_v = f[i];

        int old_bin = intensity_to_bin(old_v, nbins);
        int new_bin = intensity_to_bin(new_v, nbins);

        hist_row[old_bin] -= 1;
        hist_row[new_bin] += 1;

        // Keep ring storing raw intensity for compatibility
        old_v = new_v; // ring[head][i] = new_v
    }
    auto tb5 = my_clock_t::now();

    ring_head_ = (ring_head_ + 1) % size_;
    updated_since_last_bg_ = true;
    auto tb6 = my_clock_t::now();
}

// ---------------------------------------------------------
cv::Mat Background::compute_background_percentile_image(int percentile) {
    if (last_bg_.empty() ||
        last_bg_.rows != H_ || last_bg_.cols != W_ ||
        last_bg_.type() != CV_8UC1) {
        last_bg_.create(H_, W_, CV_8UC1);
    }

    std::uint8_t* BG_RESTRICT out = last_bg_.ptr<std::uint8_t>();

    const int   nbins  = num_bins_;
    const int   npx    = NPX_;
    const int   size   = size_;
    // Target count for this percentile (keep same semantics as before)
    const std::uint32_t target =
        static_cast<std::uint32_t>((percentile / 100.0) * size);

    const std::uint16_t* BG_RESTRICT hist_ptr = hist_.data();

    // Precompute bin -> intensity LUT once per call (cheap vs NPX_*nbins)
    std::uint8_t bin_lut[256];  // num_bins_ is clamped to <= 256
    for (int b = 0; b < nbins; ++b) {
        bin_lut[b] = bin_to_intensity(b, nbins);
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < npx; ++i) {
        const std::uint16_t* row_hist =
            hist_ptr + static_cast<std::size_t>(i) * static_cast<std::size_t>(nbins);

        std::uint32_t cum = 0;
        std::uint8_t  val = 255;

        // Use pointer increment instead of row_hist[b]
        const std::uint16_t* h = row_hist;
        for (int b = 0; b < nbins; ++b) {
            cum += *h++;
            if (cum >= target) {
                val = bin_lut[b];
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
    last_bg_percentile_    = percentile;
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

    // This is a separate 256-bin histogram for a flat uint8 buffer
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
    std::cout << "1";
    auto tbs0 = my_clock_t::now();
    if (loaded_ < size_) {
        std::cerr << "ERROR: Background not initialized or not enough frames"
                     " loaded.\n";
        return cv::Mat();
    }
    std::cout << "2";
    cv::Mat bg = get_background_percentile(subtract_percentile);
    if (bg.empty()) {
        return cv::Mat();
    }
    auto tbs1 = my_clock_t::now();
    std::cout << "3";

    CV_Assert(frame.type() == CV_8UC1);
    CV_Assert(frame.rows == H_ && frame.cols == W_);

    cv::Mat v_norm;
    auto tbs2 = my_clock_t::now();
    std::cout << "4";
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
            if (iv < 0)        iv = 0;
            else if (iv > 255) iv = 255;
            out[i] = static_cast<std::uint8_t>(iv);
        }
    } else {
        // No normalization: use frame as-is (shallow copy)
        v_norm = frame;
    }
    auto tbs3 = my_clock_t::now();
    std::cout << "5";
    cv::Mat diff;
    cv::absdiff(bg, v_norm, diff);
    std::cout << "6";
    auto tbs4 = my_clock_t::now();
    cv::Mat mask;
    cv::threshold(diff, mask, threshold, 255, cv::THRESH_BINARY);
    std::cout << "7";
    auto tbs5 = my_clock_t::now();
    cv::medianBlur(mask, mask, 3);
    std::cout << "8";
    auto tbs6 = my_clock_t::now();

    return mask;
}
