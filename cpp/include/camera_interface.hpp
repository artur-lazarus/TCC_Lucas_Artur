#ifndef CAMERA_INTERFACE_HPP
#define CAMERA_INTERFACE_HPP

#include <libcamera/libcamera.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <memory>
#include <vector>
#include <string>

class CameraInterface {
private:
    std::unique_ptr<libcamera::CameraManager> camera_manager_;
    std::shared_ptr<libcamera::Camera> camera_;
    std::unique_ptr<libcamera::CameraConfiguration> config_;
    libcamera::FrameBufferAllocator *allocator_;
    std::vector<std::unique_ptr<libcamera::Request>> requests_;
    size_t frameSize_;
    
    // Timing statistics
    std::chrono::high_resolution_clock::time_point last_frame_time_;
    double avg_capture_time_ms_ = 0.0;
    int frame_count_ = 0;

    void printTiming(const std::string& operation, 
                    std::chrono::high_resolution_clock::time_point start);

public:
    CameraInterface();
    ~CameraInterface();
    
    bool initialize();
    bool startCapture();
    void run(int num_frames = 300);
    
private:
    void requestComplete(libcamera::Request *request);
};

#endif // CAMERA_INTERFACE_HPP
