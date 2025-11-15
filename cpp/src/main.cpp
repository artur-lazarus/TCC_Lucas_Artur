#include "camera_interface.hpp"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    std::cout << "=== Traffic Speed Detection System ===" << std::endl;
    std::cout << "=== IMX477 Camera Performance Test ===\n" << std::endl;
    
    // Parse command line arguments
    std::string test_mode = "quick";
    int num_frames = 30;
    
    if (argc > 1) {
        test_mode = argv[1];
    }
    if (argc > 2) {
        num_frames = std::stoi(argv[2]);
    }
    
    // Show help
    if (test_mode == "help" || test_mode == "-h" || test_mode == "--help") {
        std::cout << "Usage: " << argv[0] << " [mode] [frames]\n" << std::endl;
        std::cout << "Available Test Modes:" << std::endl;
        std::cout << "  basic      - Standard test (default: 300 frames, ~10s)" << std::endl;
        std::cout << "  quick      - Quick validation (30 frames, ~1s)" << std::endl;
        std::cout << "  stress     - Stress test (1000 frames, ~33s)" << std::endl;
        std::cout << "  endurance  - Long-running test (3000 frames, ~100s)" << std::endl;
        std::cout << "\nExamples:" << std::endl;
        std::cout << "  " << argv[0] << " quick" << std::endl;
        std::cout << "  " << argv[0] << " basic 500" << std::endl;
        std::cout << "  " << argv[0] << " stress" << std::endl;
        return 0;
    }
    
    // Adjust frame count based on mode
    if (test_mode == "quick" && argc <= 2) {
        num_frames = 30;
    } else if (test_mode == "stress" && argc <= 2) {
        num_frames = 1000;
    } else if (test_mode == "endurance" && argc <= 2) {
        num_frames = 3000;
    }
    
    std::cout << "[TEST CONFIGURATION]" << std::endl;
    std::cout << "  Mode: " << test_mode << std::endl;
    std::cout << "  Frames: " << num_frames << std::endl;
    std::cout << "  Expected duration: ~" << (num_frames / 30.0) << " seconds" << std::endl;
    std::cout << std::string(60, '=') << "\n" << std::endl;
    
    // Create and run camera interface
    CameraInterface camera;
    
    std::cout << "--- Test 1: Camera Initialization ---" << std::endl;
    if (!camera.initialize()) {
        std::cerr << "❌ Failed to initialize camera!" << std::endl;
        return 1;
    }
    std::cout << "✓ Camera initialized successfully\n" << std::endl;
    
    std::cout << "--- Test 2: Start Capture ---" << std::endl;
    if (!camera.startCapture()) {
        std::cerr << "❌ Failed to start capture!" << std::endl;
        return 1;
    }
    std::cout << "✓ Capture started successfully\n" << std::endl;
    
    std::cout << "--- Test 3: Frame Capture (" << test_mode << " mode) ---" << std::endl;
    camera.run(num_frames);
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "✓ All tests completed successfully!" << std::endl;
    
    return 0;
}
