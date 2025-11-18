#include "VideoFileStream.hpp"

bool VideoFileStream::open() {
    if (!path_.empty()) {
        cap_.open(path_);
    } else if (device_index_ >= 0) {
        cap_.open(device_index_);
    }
    return cap_.isOpened();
}

void VideoFileStream::close() {
    if (cap_.isOpened()) {
        cap_.release();
    }
}

bool VideoFileStream::isOpened() const {
    return cap_.isOpened();
}

bool VideoFileStream::read(cv::Mat &frame) {
    if (!cap_.isOpened()) return false;
    return cap_.read(frame);
}

double VideoFileStream::getFps() const {
    if (!cap_.isOpened()) return 0.0;
    return cap_.get(cv::CAP_PROP_FPS);
}

cv::Size VideoFileStream::getFrameSize() const {
    if (!cap_.isOpened()) return {};
    int w = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    int h = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    return {w, h};
}