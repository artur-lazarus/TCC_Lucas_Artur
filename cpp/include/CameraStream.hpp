#pragma once

#include "IVideoStream.hpp"
#include "camera_interface.hpp"
#include <opencv2/opencv.hpp>

class CameraStream : public IVideoStream {
public:
    CameraStream() = default;

    bool open() override;
    void close() override;
    bool isOpened() const override;
    bool read(cv::Mat &frame) override;

    double getFps() const override { return fps_; }
    cv::Size getFrameSize() const override {
        return cv::Size(width_, height_);
    }

private:
    CameraInterface camera_;
    bool opened_{false};
    double fps_{10.0};   // matches CameraInterface::FRAMERATE
    int width_{1920};
    int height_{1080};
};
