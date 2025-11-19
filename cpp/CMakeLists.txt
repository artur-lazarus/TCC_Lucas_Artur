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

# Set build type to Release if not specified
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -DNDEBUG")

# Add include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

# Collect all source files from src directory
file(GLOB_RECURSE SOURCES 
    ${CMAKE_SOURCE_DIR}/src/*.cpp
)

# Build the executable
add_executable(tcc ${SOURCES})

# Link libraries
target_link_libraries(tcc PRIVATE
    PkgConfig::LIBCAMERA
    Threads::Threads
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

# Compiler warnings
target_compile_options(tcc PRIVATE
    -Wall -Wextra -Wpedantic
)

# Find OpenMP
find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
    target_link_libraries(tcc PRIVATE OpenMP::OpenMP_CXX)
endif()

# Print configuration info
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "libcamera cflags: ${LIBCAMERA_CFLAGS}")
message(STATUS "libcamera libs: ${LIBCAMERA_LIBRARIES}")
message(STATUS "Source files: ${SOURCES}")
