cmake_minimum_required(VERSION 3.16)
project(TCC VERSION 1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find libcamera via pkg-config and threads
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBCAMERA REQUIRED IMPORTED_TARGET libcamera)
find_package(Threads REQUIRED)

# Add include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
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
)

# Compiler warnings
target_compile_options(tcc PRIVATE
    -Wall -Wextra -Wpedantic
)

# Print configuration info
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "libcamera cflags: ${LIBCAMERA_CFLAGS}")
message(STATUS "libcamera libs: ${LIBCAMERA_LIBRARIES}")
message(STATUS "Source files: ${SOURCES}")
