cmake_minimum_required(VERSION 3.20)

project(opencv_vision_tracker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(USE_TENSORRT "Enable TensorRT face backends (Jetson)" OFF)

# ---- OpenCV ----
find_package(OpenCV REQUIRED)

# ---- Face backend source selection ----
set(FACE_BACKEND_SOURCES
  src/detect/FaceDetector_OpenCVDNN.cpp
  src/embed/FaceEmbedder_OpenCVDNN.cpp
)

if (USE_TENSORRT)
  # Expected future files (not present yet per your repo state)
  set(TENSORRT_SOURCES
    src/detect/FaceDetector_TensorRT.cpp
    src/embed/FaceEmbedder_TensorRT.cpp
  )

  foreach(f IN LISTS TENSORRT_SOURCES)
    if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${f}")
      message(FATAL_ERROR
        "USE_TENSORRT=ON but missing ${f}. Add TensorRT sources before enabling.")
    endif()
  endforeach()

  set(FACE_BACKEND_SOURCES ${TENSORRT_SOURCES})
  add_compile_definitions(USE_TENSORRT=1)
else()
  add_compile_definitions(USE_OPENCV_DNN=1)
endif()

# ---- Main executable ----
add_executable(opencv-vision-tracker
  src/main.cpp
  src/FrameGrabber.cpp
  src/FrameQueue.cpp
  src/VisionProcessor.cpp
  src/KalmanTracker.cpp
  src/ThreadPool.cpp
  src/Utils.cpp

  ${FACE_BACKEND_SOURCES}
)

target_include_directories(opencv-vision-tracker PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(opencv-vision-tracker PRIVATE
  ${OpenCV_LIBS}
)

# Warnings (lightweight, cross-platform friendly)
if (MSVC)
  target_compile_options(opencv-vision-tracker PRIVATE /W4)
else()
  target_compile_options(opencv-vision-tracker PRIVATE -Wall -Wextra -Wpedantic)
endif()

# NOTE:
# TensorRT linking will be added later when TensorRT sources exist again.
# That commit will also add include paths + libs (nvinfer, nvonnxparser, cudart).
