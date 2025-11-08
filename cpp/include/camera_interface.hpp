#ifndef CAMERA_INTERFACE_HPP
#define CAMERA_INTERFACE_HPP

#include <libcamera/libcamera.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <chrono>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <map>
#include <thread>
#include <vector>
#include <atomic>

class CameraInterface {
public:
    CameraInterface();
    ~CameraInterface();

    bool initialize();
    bool startCapture();
    void stopCapture();
    void run(int num_frames);

private:
    void setupCamera();
    void configureStreams();
    void allocateBuffers();
    void processFrame(const uint8_t* data, size_t length);
    void mockProcessing(cv::Mat& frame);
    void requestComplete(libcamera::Request* request);
    void saveWorker();

    std::unique_ptr<libcamera::CameraManager> camera_manager_;
    std::shared_ptr<libcamera::Camera> camera_;
    std::unique_ptr<libcamera::CameraConfiguration> config_;
    std::vector<std::unique_ptr<libcamera::Request>> requests_;
    std::unique_ptr<libcamera::FrameBufferAllocator> allocator_;
    
    libcamera::Stream* stream_;
    int frame_count_;
    int width_;
    int height_;
    bool camera_started_;
    bool camera_acquired_;
    
    // Frame buffer management
    std::map<libcamera::FrameBuffer*, std::vector<libcamera::Span<uint8_t>>> mapped_buffers_;
    std::queue<libcamera::Request*> completed_requests_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    // Async save queue
    struct SaveTask {
        cv::Mat frame;
        int frame_number;
    };
    std::queue<SaveTask> save_queue_;
    std::mutex save_mutex_;
    std::condition_variable save_cv_;
    std::vector<std::thread> save_threads_;
    std::atomic<bool> save_running_;
    std::atomic<int> frames_saved_;
    
    static constexpr int FRAMERATE = 10;
    static constexpr int FRAME_INTERVAL_MS = 100; // 1000ms / 10fps
    static constexpr int NUM_SAVE_THREADS = 2;
};

#endif // CAMERA_INTERFACE_HPP
