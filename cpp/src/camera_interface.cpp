#include "camera_interface.hpp"
#include <iostream>
#include <thread>
#include <iomanip>
#include <sys/mman.h>

CameraInterface::CameraInterface()
    : frame_count_(0)
    , width_(1920)
    , height_(1080)
    , stream_(nullptr)
    , camera_started_(false)
    , camera_acquired_(false)
    , save_running_(false)
    , frames_saved_(0) {
}

CameraInterface::~CameraInterface() {
    stopCapture();
    
    // Stop save threads
    save_running_ = false;
    save_cv_.notify_all();
    for (auto& thread : save_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    if (camera_acquired_ && camera_) {
        camera_->release();
        camera_acquired_ = false;
    }
}

bool CameraInterface::initialize() {
    auto init_start = std::chrono::high_resolution_clock::now();
    
    try {
        // Initialize camera manager
        auto cm_start = std::chrono::high_resolution_clock::now();
        camera_manager_ = std::make_unique<libcamera::CameraManager>();
        int ret = camera_manager_->start();
        if (ret) {
            std::cerr << "Failed to start camera manager: " << ret << std::endl;
            return false;
        }
        auto cm_end = std::chrono::high_resolution_clock::now();
        auto cm_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cm_end - cm_start).count();
        std::cout << "  ⏱️  Camera manager start: " << cm_duration << " ms" << std::endl;

        // Find IMX477 camera
        auto cameras = camera_manager_->cameras();
        if (cameras.empty()) {
            std::cerr << "No cameras available" << std::endl;
            return false;
        }

        // Select first camera (typically IMX477 on RPi)
        camera_ = cameras[0];
        std::cout << "Using camera: " << camera_->id() << std::endl;

        // Acquire camera
        auto acquire_start = std::chrono::high_resolution_clock::now();
        if (camera_->acquire()) {
            std::cerr << "Failed to acquire camera" << std::endl;
            return false;
        }
        camera_acquired_ = true;
        auto acquire_end = std::chrono::high_resolution_clock::now();
        auto acquire_duration = std::chrono::duration_cast<std::chrono::milliseconds>(acquire_end - acquire_start).count();
        std::cout << "  ⏱️  Camera acquire: " << acquire_duration << " ms" << std::endl;

        auto setup_start = std::chrono::high_resolution_clock::now();
        setupCamera();
        auto setup_end = std::chrono::high_resolution_clock::now();
        auto setup_duration = std::chrono::duration_cast<std::chrono::milliseconds>(setup_end - setup_start).count();
        std::cout << "  ⏱️  Camera setup: " << setup_duration << " ms" << std::endl;
        
        auto init_end = std::chrono::high_resolution_clock::now();
        auto total_init = std::chrono::duration_cast<std::chrono::milliseconds>(init_end - init_start).count();
        std::cout << "  ⏱️  Total initialization: " << total_init << " ms" << std::endl;
        
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Exception during initialization: " << e.what() << std::endl;
        return false;
    }
}

void CameraInterface::setupCamera() {
    // Generate configuration for video recording
    config_ = camera_->generateConfiguration({ libcamera::StreamRole::VideoRecording });
    if (!config_) {
        throw std::runtime_error("Failed to generate camera configuration");
    }

    // Configure stream for BGR format directly (no conversion needed!)
    libcamera::StreamConfiguration& streamConfig = config_->at(0);
    streamConfig.size.width = width_;
    streamConfig.size.height = height_;
    streamConfig.pixelFormat = libcamera::formats::BGR888;
    streamConfig.bufferCount = 4;

    // Color space will be set automatically by libcamera during validation

    // Validate configuration
    config_->validate();
    
    std::cout << "Stream configuration: " 
              << streamConfig.size.width << "x" << streamConfig.size.height
              << " @ " << streamConfig.pixelFormat.toString() << std::endl;

    // Apply configuration
    if (camera_->configure(config_.get()) < 0) {
        throw std::runtime_error("Failed to configure camera");
    }

    stream_ = streamConfig.stream();
    
    // Allocate buffers
    allocateBuffers();
}

void CameraInterface::allocateBuffers() {
    // Allocate frame buffers for the stream
    allocator_ = std::make_unique<libcamera::FrameBufferAllocator>(camera_);
    
    if (allocator_->allocate(stream_) < 0) {
        throw std::runtime_error("Failed to allocate buffers");
    }

    std::cout << "Allocated " << allocator_->buffers(stream_).size() << " buffers" << std::endl;

    // Map buffers to user space
    const std::vector<std::unique_ptr<libcamera::FrameBuffer>>& buffers = allocator_->buffers(stream_);
    for (const auto& buffer : buffers) {
        size_t buffer_size = 0;
        for (unsigned i = 0; i < buffer->planes().size(); i++) {
            const libcamera::FrameBuffer::Plane& plane = buffer->planes()[i];
            buffer_size += plane.length;
        }
        
        // Map the buffer
        std::vector<libcamera::Span<uint8_t>> mapped_planes;
        for (unsigned i = 0; i < buffer->planes().size(); i++) {
            const libcamera::FrameBuffer::Plane& plane = buffer->planes()[i];
            void* memory = mmap(NULL, plane.length, PROT_READ | PROT_WRITE, 
                               MAP_SHARED, plane.fd.get(), 0);
            if (memory == MAP_FAILED) {
                throw std::runtime_error("Failed to mmap buffer");
            }
            mapped_planes.push_back(libcamera::Span<uint8_t>(
                static_cast<uint8_t*>(memory), plane.length));
        }
        mapped_buffers_[buffer.get()] = std::move(mapped_planes);
    }

    // Create requests
    for (unsigned int i = 0; i < buffers.size(); ++i) {
        std::unique_ptr<libcamera::Request> request = camera_->createRequest();
        if (!request) {
            throw std::runtime_error("Failed to create request");
        }

        if (request->addBuffer(stream_, buffers[i].get()) < 0) {
            throw std::runtime_error("Failed to add buffer to request");
        }

        requests_.push_back(std::move(request));
    }
}

bool CameraInterface::startCapture() {
    try {
        // Set camera controls for 10fps
        libcamera::ControlList controls;
        
        // Set frame duration for 10fps (100,000 microseconds = 100ms)
        int64_t frame_time = 100000; // microseconds
        controls.set(libcamera::controls::FrameDurationLimits, 
                    libcamera::Span<const int64_t, 2>({ frame_time, frame_time }));

        // Connect request completed signal
        camera_->requestCompleted.connect(this, &CameraInterface::requestComplete);

        // Start save worker threads
        save_running_ = true;
        for (int i = 0; i < NUM_SAVE_THREADS; ++i) {
            save_threads_.emplace_back(&CameraInterface::saveWorker, this);
        }

        // Start camera with controls
        if (camera_->start(&controls)) {
            std::cerr << "Failed to start camera" << std::endl;
            return false;
        }

        camera_started_ = true;

        // Queue all requests
        for (auto& request : requests_) {
            if (camera_->queueRequest(request.get()) < 0) {
                std::cerr << "Failed to queue request" << std::endl;
                return false;
            }
        }

        std::cout << "Camera started at " << FRAMERATE << " fps" << std::endl;
        std::cout << "Save threads: " << NUM_SAVE_THREADS << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Exception during start: " << e.what() << std::endl;
        return false;
    }
}

void CameraInterface::stopCapture() {
    if (camera_started_) {
        camera_->stop();
        camera_->requestCompleted.disconnect(this, &CameraInterface::requestComplete);
        camera_started_ = false;
        std::cout << "Camera stopped" << std::endl;
    }
    
    // Stop save threads
    save_running_ = false;
    save_cv_.notify_all();
    for (auto& thread : save_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    save_threads_.clear();
    
    std::cout << "Total frames saved: " << frames_saved_.load() << std::endl;
    
    // Unmap buffers
    for (auto& [buffer, spans] : mapped_buffers_) {
        for (auto& span : spans) {
            munmap(span.data(), span.size());
        }
    }
    mapped_buffers_.clear();
}

void CameraInterface::requestComplete(libcamera::Request* request) {
    if (request->status() == libcamera::Request::RequestCancelled) {
        return;
    }
    
    // Immediately queue the request for processing - don't hold locks longer than needed
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        completed_requests_.push(request);
    }
    // Notify immediately after unlocking
    queue_cv_.notify_one();
}

void CameraInterface::run(int num_frames) {
    auto start_time = std::chrono::steady_clock::now();
    int frames_processed = 0;
    int frames_dropped = 0;
    
    // Timing statistics
    long long total_processing_time = 0;
    uint64_t last_frame_timestamp = 0;

    std::cout << "Starting capture of " << num_frames << " frames..." << std::endl;
    std::cout << "Target: " << FRAMERATE << " fps (" << FRAME_INTERVAL_MS << "ms per frame)" << std::endl;

    while (frames_processed < num_frames) {
        // Wait for completed request - use longer timeout
        libcamera::Request* request = nullptr;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            // Wait indefinitely for frame (camera controls timing)
            queue_cv_.wait(lock, [this] { return !completed_requests_.empty(); });
            
            if (!completed_requests_.empty()) {
                request = completed_requests_.front();
                completed_requests_.pop();
            }
        }

        if (!request) {
            frames_dropped++;
            continue;
        }

        // Measure frame processing time
        auto process_start = std::chrono::high_resolution_clock::now();
        
        // Get the buffer from the request
        libcamera::FrameBuffer* buffer = request->buffers().begin()->second;
        
        // Get actual frame timestamp for accurate timing
        uint64_t frame_timestamp = buffer->metadata().timestamp;
        
        auto it = mapped_buffers_.find(buffer);
        if (it != mapped_buffers_.end() && !it->second.empty()) {
            const uint8_t* data = it->second[0].data();
            size_t length = it->second[0].size();
            
            // Process the frame
            processFrame(data, length);
        }
        
        frames_processed++;
        
        auto process_end = std::chrono::high_resolution_clock::now();
        auto processing_time = std::chrono::duration_cast<std::chrono::microseconds>(process_end - process_start).count();
        
        total_processing_time += processing_time;

        // Calculate actual frame interval
        float frame_interval_ms = 0.0f;
        if (last_frame_timestamp > 0) {
            frame_interval_ms = (frame_timestamp - last_frame_timestamp) / 1000000.0f; // ns to ms
        }
        last_frame_timestamp = frame_timestamp;

        // Re-queue the request immediately to maintain timing
        request->reuse(libcamera::Request::ReuseBuffers);
        camera_->queueRequest(request);

        // Display timing for every frame
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - start_time).count();
        float actual_fps = (frames_processed * 1000.0f) / elapsed;
        float avg_processing = (float)total_processing_time / frames_processed / 1000.0f; // convert to ms
        
        std::cout << "  🔍 Frame " << std::setw(4) << frames_processed << "/" << num_frames 
                  << " | FPS: " << std::fixed << std::setprecision(2) << std::setw(6) << actual_fps
                  << " | Frame interval: " << std::setw(6) << std::setprecision(1) << frame_interval_ms << "ms"
                  << " | Elapsed: " << std::setw(6) << std::setprecision(1) << elapsed / 1000.0f << "s"
                  << " | Proc: " << std::setw(5) << std::setprecision(1) << processing_time / 1000.0f << "ms"
                  << std::endl;
    }

    auto end_time = std::chrono::steady_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    std::cout << "\n[CAPTURE SUMMARY]" << std::endl;
    std::cout << "  Frames captured: " << frames_processed << std::endl;
    std::cout << "  Total time: " << std::fixed << std::setprecision(2) 
              << total_time / 1000.0f << " seconds" << std::endl;
    std::cout << "  Average FPS: " << std::setprecision(2) 
              << (frames_processed * 1000.0f) / total_time << std::endl;
    std::cout << "  Frames dropped: " << frames_dropped << std::endl;
    std::cout << "\n[PROCESSING TIMING]" << std::endl;
    std::cout << "  Average: " << std::setprecision(2) 
              << (float)total_processing_time / frames_processed / 1000.0f << " ms/frame" << std::endl;
    std::cout << "  CPU utilization: " << std::setprecision(1)
              << (100.0f * total_processing_time / 1000.0f) / total_time << "%" << std::endl;
}

void CameraInterface::processFrame(const uint8_t* data, size_t /*length*/) {
    // Data is already in BGR888 format - wrap it directly in cv::Mat
    // Clone the data so it can be saved asynchronously
    cv::Mat bgr_frame(height_, width_, CV_8UC3, (void*)data);
    
    // Apply mock processing
    mockProcessing(bgr_frame);
    
    frame_count_++;
}

void CameraInterface::mockProcessing(cv::Mat& frame) {
    if (frame.empty()) {
        std::cerr << "Warning: Empty frame received" << std::endl;
        return;
    }

    // Skip first 10 frames (camera initialization/warmup period)
    if (frame_count_ < 10) {
        if (frame_count_ == 0) {
            std::cout << "  🔥 Warmup: Skipping first 10 frames for camera initialization..." << std::endl;
        }
        return;
    }

    // Clone frame and queue for async saving
    SaveTask task;
    task.frame = frame.clone();
    task.frame_number = frame_count_;
    
    {
        std::lock_guard<std::mutex> lock(save_mutex_);
        save_queue_.push(std::move(task));
    }
    save_cv_.notify_one();
}

void CameraInterface::saveWorker() {
    // JPEG compression parameters - lower quality for speed
    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
    compression_params.push_back(50); // Reduced quality for faster saving
    compression_params.push_back(cv::IMWRITE_JPEG_OPTIMIZE);
    compression_params.push_back(0); // Disable optimization for speed
    
    while (save_running_) {
        SaveTask task;
        
        {
            std::unique_lock<std::mutex> lock(save_mutex_);
            save_cv_.wait(lock, [this] { 
                return !save_queue_.empty() || !save_running_; 
            });
            
            if (!save_running_ && save_queue_.empty()) {
                break;
            }
            
            if (!save_queue_.empty()) {
                task = std::move(save_queue_.front());
                save_queue_.pop();
            } else {
                continue;
            }
        }
        
        // Save outside of lock
        std::string filename = "/home/tcc/tcc/TCC_Lucas_Artur/cpp/frames/frame_" + 
                              std::to_string(task.frame_number) + ".jpg";
        
        if (cv::imwrite(filename, task.frame, compression_params)) {
            frames_saved_++;
            // Only log every 10 frames
            if (task.frame_number % 10 == 0) {
                std::cout << "  💾 Saved: frame_" << task.frame_number << ".jpg" << std::endl;
            }
        } else {
            std::cerr << "  ❌ Failed to save: frame_" << task.frame_number << ".jpg" << std::endl;
        }
    }
}
