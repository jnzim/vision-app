cmake_minimum_required(VERSION 3.16)
project(opencv_vision_tracker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(opencv_vision_tracker
    src/main.cpp
    src/FrameGrabber.cpp
    src/FrameQueue.cpp
    src/VisionProcessor.cpp
    src/KalmanTracker.cpp
)

target_include_directories(opencv_vision_tracker PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(opencv_vision_tracker PRIVATE ${OpenCV_LIBS})