# Camera Tracking Pipeline (C++ / OpenCV)

This is a live camera tracking pipeline written in modern C++. It serves as a small, practical test bed for vision processing, multithreaded frame handoff.

The goal is to keep the design clean, correct, and reasonably efficient without overcomplicating the architecture. This is a low-latency vision application running on a general-purpose operating system, not a hard real-time system.

## Key Features

- Camera capture using OpenCV
- Thread-safe frame handoff using a mutex and condition variable
- Clear, modular pipeline with separate stages for:
  - frame capture
  - vision processing
  - optional smoothing / state estimation
- Timestamped frames for basic latency measurement
- Live visualization with overlays
- Debug views at different stages such as raw, thresholded, and final output

## Architecture Overview

Rather than doing everything in a single loop, the pipeline is split into a few focused components. This makes the code easier to understand, easier to modify, and easier to extend.

## Components

### FrameGrabber
Handles camera input and timestamps frames when they are captured.

### FrameQueue
A thread-safe buffer that allows capture and processing to run somewhat independently.

### VisionProcessor
Runs on a separate thread and performs the main image processing steps, such as:

- grayscale conversion
- thresholding
- contour detection
- optional Kalman filtering

### Tracker (optional)
Smooths noisy detections and can provide simple prediction between measurements.

### Visualization
Displays intermediate processing stages, final results, and a few basic runtime statistics.

## Frame and Timestamping

Each frame is wrapped in a small struct:

```cpp
struct Frame {
    cv::Mat image;
    std::chrono::steady_clock::time_point timeStamp;
};