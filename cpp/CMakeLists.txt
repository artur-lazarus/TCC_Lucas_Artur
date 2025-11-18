cmake_minimum_required(VERSION 3.16)
project(TCC VERSION 1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find Eigen (required by tracker)
find_package(Eigen3 REQUIRED)

# Find libcamera via pkg-config and threads
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBCAMERA REQUIRED IMPORTED_TARGET libcamera)
find_package(Threads REQUIRED)

# Add include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

# Collect library source files (excluding main files)
set(LIB_SOURCES
    ${CMAKE_SOURCE_DIR}/src/background.cpp
    ${CMAKE_SOURCE_DIR}/src/calibration.cpp
    ${CMAKE_SOURCE_DIR}/src/camera_interface.cpp
    ${CMAKE_SOURCE_DIR}/src/detection.cpp
    ${CMAKE_SOURCE_DIR}/src/tracker.cpp
)

# Build the main application executable
add_executable(tcc 
    ${CMAKE_SOURCE_DIR}/src/main_app.cpp
    ${LIB_SOURCES}
)

# Link libraries for main app
target_link_libraries(tcc PRIVATE
    PkgConfig::LIBCAMERA
    Threads::Threads
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

# Build the test executable
add_executable(tcc_camera_test 
    ${CMAKE_SOURCE_DIR}/src/test_main.cpp
    ${LIB_SOURCES}
)

# Link libraries for test
target_link_libraries(tcc_camera_test PRIVATE
    PkgConfig::LIBCAMERA
    Threads::Threads
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

# Compiler warnings
target_compile_options(tcc PRIVATE
    -Wall -Wextra -Wpedantic
)

target_compile_options(tcc_camera_test PRIVATE
    -Wall -Wextra -Wpedantic
)

# Print configuration info
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "libcamera cflags: ${LIBCAMERA_CFLAGS}")
message(STATUS "libcamera libs: ${LIBCAMERA_LIBRARIES}")
message(STATUS "Source files: ${LIB_SOURCES}")
