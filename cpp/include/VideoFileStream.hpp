#pragma once

#include "IVideoStream.hpp"
#include <opencv2/opencv.hpp>
#include <string>

class VideoFileStream : public IVideoStream {
public:
    // For file-based capture
    explicit VideoFileStream(const std::string &path)
        : path_(path), device_index_(-1) {}

    // For camera index-based capture (OpenCV backend)
    explicit VideoFileStream(int device_index)
        : device_index_(device_index) {}

    bool open() override;
    void close() override;
    bool isOpened() const override;
    bool read(cv::Mat &frame) override;

    double getFps() const override;
    cv::Size getFrameSize() const override;

private:
    std::string path_;
    int device_index_{-1};
    cv::VideoCapture cap_;
};
