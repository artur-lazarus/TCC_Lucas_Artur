#pragma once

#include <opencv2/opencv.hpp>

class IVideoStream {
public:
    virtual ~IVideoStream() = default;

    // Open underlying source.
    virtual bool open() = 0;

    // Close underlying source.
    virtual void close() = 0;

    // Check if source is opened.
    virtual bool isOpened() const = 0;

    // Get next frame. Returns false on EOS / error.
    virtual bool read(cv::Mat &frame) = 0;

    // Optional helpers.
    virtual double getFps() const { return 0.0; }
    virtual cv::Size getFrameSize() const { return {}; }
};
