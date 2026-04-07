cmake_minimum_required(VERSION 3.20)

project(opencv-vision-tracker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)

add_executable(opencv-vision-tracker
    src/main.cpp
    src/FrameGrabber.cpp
    src/FrameQueue.cpp
    src/VisionProcessor.cpp
    src/KalmanTracker.cpp
    src/Utils.cpp
    src/detect/FaceDetector_OpenCVDNN.cpp
)

target_include_directories(opencv-vision-tracker PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(opencv-vision-tracker PRIVATE
    ${OpenCV_LIBS}
)

if (MSVC)
    target_compile_options(opencv-vision-tracker PRIVATE /W4)
else()
    target_compile_options(opencv-vision-tracker PRIVATE -Wall -Wextra -Wpedantic)
endif()