#include "camera_interface.hpp"
#include <iostream>
#include <iomanip>
#include <sys/mman.h>
#include <thread>
#include <string>

using namespace libcamera;
using namespace std::chrono;

CameraInterface::CameraInterface() : allocator_(nullptr) {}

CameraInterface::~CameraInterface() {
    if (camera_) {
        camera_->stop();
        camera_->release();
    }
    if (allocator_) {
        delete allocator_;
    }
}

void CameraInterface::printTiming(const std::string& operation, high_resolution_clock::time_point start) {
    auto end = high_resolution_clock::now();
    auto duration_us = duration_cast<microseconds>(end - start).count();
    std::cout << "[TIMING] " << operation << ": " 
              << std::fixed << std::setprecision(3) 
              << duration_us / 1000.0 << " ms" << std::endl;
}

bool CameraInterface::initialize() {
    auto start_total = high_resolution_clock::now();
    
    // Step 1: Initialize camera manager
    auto start = high_resolution_clock::now();
    camera_manager_ = std::make_unique<CameraManager>();
    camera_manager_->start();
    printTiming("Camera manager initialization", start);

    // Step 2: Get camera
    start = high_resolution_clock::now();
    if (camera_manager_->cameras().empty()) {
        std::cerr << "No cameras available!" << std::endl;
        return false;
    }
    camera_ = camera_manager_->cameras()[0];
    camera_->acquire();
    printTiming("Camera acquisition", start);

    // Step 3: Configure camera for 10 FPS target
    start = high_resolution_clock::now();
    config_ = camera_->generateConfiguration({StreamRole::VideoRecording});
    
    StreamConfiguration &streamConfig = config_->at(0);
    streamConfig.size = Size(1920, 1080);
    streamConfig.pixelFormat = formats::YUV420;
    
    // Target 10 FPS = 100ms per frame = 100,000 microseconds
    int64_t frame_duration_us = 100000;  // 10 FPS for processing budget
    
    config_->validate();
    camera_->configure(config_.get());

    frameSize_ = streamConfig.frameSize;
    std::cout << "[INFO] Actual Frame Size (bytes): " << frameSize_ << std::endl;
    
    // Set frame rate controls AFTER configure
    auto controls = std::make_unique<libcamera::ControlList>();
    controls->set(libcamera::controls::FrameDurationLimits, 
                  libcamera::Span<const int64_t, 2>({frame_duration_us, frame_duration_us}));
    
    printTiming("Camera configuration", start);

    // Step 4: Allocate buffers
    start = high_resolution_clock::now();
    allocator_ = new FrameBufferAllocator(camera_);
    Stream *stream = streamConfig.stream();
    allocator_->allocate(stream);
    printTiming("Buffer allocation", start);

    // Step 5: Create requests with frame duration control
    start = high_resolution_clock::now();
    const std::vector<std::unique_ptr<FrameBuffer>> &buffers = allocator_->buffers(stream);
    for (unsigned int i = 0; i < buffers.size(); ++i) {
        std::unique_ptr<Request> request = camera_->createRequest();
        if (!request) {
            std::cerr << "Failed to create request" << std::endl;
            return false;
        }
        
        FrameBuffer *buffer = buffers[i].get();
        request->addBuffer(stream, buffer);
        
        // Apply frame rate control to each request
        request->controls().merge(*controls);
        
        requests_.push_back(std::move(request));
    }
    printTiming("Request creation", start);

    printTiming("TOTAL initialization", start_total);
    
    std::cout << "\n[INFO] Camera configured:" << std::endl;
    std::cout << "  Resolution: " << streamConfig.size.width << "x" 
              << streamConfig.size.height << std::endl;
    std::cout << "  Format: " << streamConfig.pixelFormat.toString() << std::endl;
    std::cout << "  Target FPS: " << (1000000.0 / frame_duration_us) << std::endl;
    std::cout << "  Frame budget: " << (frame_duration_us / 1000.0) << " ms" << std::endl;
    std::cout << "  Buffers: " << buffers.size() << std::endl;
    
    return true;
}

bool CameraInterface::startCapture() {
    auto start = high_resolution_clock::now();
    
    // Register request completed callback
    camera_->requestCompleted.connect(this, &CameraInterface::requestComplete);
    
    // IMPORTANT: Start camera FIRST, then queue requests
    // The camera must be in "Running" state before accepting requests
    camera_->start();
    printTiming("Camera start", start);
    
    // Now queue all requests (camera is running)
    for (auto &request : requests_) {
        camera_->queueRequest(request.get());
    }
    
    last_frame_time_ = high_resolution_clock::now();
    return true;
}

void CameraInterface::requestComplete(Request *request) {
    auto frame_start = high_resolution_clock::now();
    
    if (request->status() == Request::RequestCancelled)
        return;

    auto time_since_last = duration_cast<microseconds>(frame_start - last_frame_time_).count();
    double fps = 1000000.0 / time_since_last;
    
    auto process_start = high_resolution_clock::now();
    const Request::BufferMap &buffers = request->buffers();

    long long mmap_time_us = 0;
    long long mat_creation_time_us = 0;
    long long imwrite_time_us = 0;
    long long munmap_time_us = 0;
    
    for (auto bufferPair : buffers) {
        FrameBuffer *buffer = bufferPair.second;
        const FrameBuffer::Plane &plane = buffer->planes()[0];
        
        auto mmap_start = high_resolution_clock::now();
        void *data = mmap(nullptr, frameSize_, PROT_READ, MAP_SHARED, plane.fd.get(), 0);
        mmap_time_us = duration_cast<microseconds>(high_resolution_clock::now() - mmap_start).count();

        if (data != MAP_FAILED) {
            // --- START OF YUV TEST ---
            int width = 1920;
            int height = 1080;

            // 1. Create a Mat 'wrapper' ONLY for the Y plane (grayscale)
            // The Y plane is the first (width * height) bytes of the 'data' buffer
            auto mat_creation_start = high_resolution_clock::now();
            cv::Mat grayscale_frame(height, width, CV_8UC1, data);
            mat_creation_time_us = duration_cast<microseconds>(high_resolution_clock::now() - mat_creation_start).count();

            // 2. Save this grayscale frame, but ONLY ONCE
            // (Saving every frame (10x/sec) would cause I/O bottleneck)
            if (frame_count_ == 10 || frame_count_ == 20 || frame_count_ == 30) { // Save the 10th, 20th, and 30th frames
                auto imwrite_start = high_resolution_clock::now();
                std::string filename = "/home/tcc/tcc/TCC_Lucas_Artur/cpp/test_frame_grayscale_" + std::to_string(frame_count_) + ".jpg";
                bool success = cv::imwrite(filename, grayscale_frame);
                imwrite_time_us = duration_cast<microseconds>(high_resolution_clock::now() - imwrite_start).count();
                if (success) {
                    std::cout << "\n[YUV TEST] Test image saved as '" << filename << "'\n" << std::endl;
                } else {
                    std::cout << "\n[YUV TEST] FAILED to save test image '" << filename << "'.\n" << std::endl;
                }
            }
            
            // 3. Unmap the buffer
            auto munmap_start = high_resolution_clock::now();
            munmap(data, frameSize_);
            munmap_time_us = duration_cast<microseconds>(high_resolution_clock::now() - munmap_start).count();
        }
    }
    auto process_time = duration_cast<microseconds>(high_resolution_clock::now() - process_start).count();

    frame_count_++;
    double total_frame_time = duration_cast<microseconds>(high_resolution_clock::now() - frame_start).count() / 1000.0;
    avg_capture_time_ms_ = (avg_capture_time_ms_ * (frame_count_ - 1) + total_frame_time) / frame_count_;

    if (frame_count_ % 10 == 0) {  // Report every 10 frames (every 1 second at 10 FPS)
        double time_budget_ms = 100.0;  // 10 FPS = 100ms budget
        double available_for_processing = time_budget_ms - (process_time / 1000.0);
        double budget_used_percent = (process_time / 1000.0 / time_budget_ms) * 100.0;
        
        std::cout << "\n[FRAME " << frame_count_ << " STATS]" << std::endl;
        std::cout << "  Actual FPS: " << std::fixed << std::setprecision(2) << fps << std::endl;
        std::cout << "  Time since last: " << std::setprecision(2) 
                  << time_since_last / 1000.0 << " ms" << std::endl;
        std::cout << "  Total processing: " << std::setprecision(3) 
                  << process_time / 1000.0 << " ms" << std::endl;
        std::cout << "    - mmap: " << mmap_time_us / 1000.0 << " ms" << std::endl;
        std::cout << "    - Mat creation: " << mat_creation_time_us / 1000.0 << " ms" << std::endl;
        
        std::cout << "    - imwrite (frame " << frame_count_ << "): " << imwrite_time_us / 1000.0 << " ms" << std::endl;
        
        std::cout << "    - munmap: " << munmap_time_us / 1000.0 << " ms" << std::endl;
        std::cout << "  Processing budget available: " << std::setprecision(2)
                  << available_for_processing << " ms (" 
                  << (100.0 - budget_used_percent) << "% free)" << std::endl;
        
        // Warning if approaching budget limit
        if (budget_used_percent > 80.0) {
            std::cout << "  ⚠ WARNING: Using " << budget_used_percent 
                      << "% of frame budget!" << std::endl;
        }
    }

    request->reuse(Request::ReuseBuffers);
    camera_->queueRequest(request);
    
    last_frame_time_ = frame_start;
}

void CameraInterface::run(int num_frames) {
    std::cout << "\n[INFO] Starting capture for " << num_frames << " frames..." << std::endl;
    std::cout << "[INFO] At 10 FPS, this will take ~" << (num_frames / 10.0) << " seconds" << std::endl;
    
    while (frame_count_ < num_frames) {
        std::this_thread::sleep_for(milliseconds(1));
    }
    
    std::cout << "\n[SUMMARY]" << std::endl;
    std::cout << "  Total frames captured: " << frame_count_ << std::endl;
    std::cout << "  Average frame processing time: " << avg_capture_time_ms_ << " ms" << std::endl;
    std::cout << "  Average processing budget available: " 
              << (100.0 - avg_capture_time_ms_) << " ms per frame" << std::endl;
    std::cout << "  Achieved FPS: " << (frame_count_ / (frame_count_ / 10.0)) << std::endl;
}