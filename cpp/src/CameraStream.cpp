#include "CameraStream.hpp"

CameraStream::CameraStream()
    : opened_(false) {
}

CameraStream::~CameraStream() {
    close();
}

bool CameraStream::open() {
    // Initialize and start capture if not already done.
    if (!camera_.initialize()) {
        return false;
    }
    if (!camera_.startCapture()) {
        return false;
    }
    opened_ = true;
    return true;
}

void CameraStream::close() {
    if (!opened_) return;
    camera_.stopCapture();
    opened_ = false;
}

bool CameraStream::isOpened() const {
    return opened_;
}

bool CameraStream::read(cv::Mat &frame) {
    // Blocking read from CameraInterface; returns false on failure.
    return camera_.read(frame);
}