# Real-Time Camera Tracking Pipeline (C++ / OpenCV)

A real-time webcam processing and tracking pipeline written in modern C++.  
This project demonstrates clean system architecture, thread-safe frame handoff, and practical computer-vision techniques similar to those used in robotics and industrial automation systems.

The emphasis is on **clarity, correctness, and performance-aware design**, not just a minimal demo.

---

## Key Features
- Real-time camera capture using OpenCV
- Thread-safe frame handoff using mutexes and condition variables
- Modular processing pipeline:
  - frame acquisition
  - vision processing
  - optional state estimation / smoothing
- Timestamped frames for basic latency analysis
- Live visualization with detection overlays
- Multi-stage debug visualization (raw, thresholded, overlay)

---

## Architecture Overview
The application is structured as a small but realistic vision system rather than a monolithic example.

### Major Components

- **FrameGrabber**  
  Owns the camera interface and timestamps frames immediately after acquisition.

- **FrameQueue**  
  Thread-safe queue used to decouple capture and processing rates.

- **VisionProcessor**  
  Runs in a background thread and performs image processing:
  - grayscale conversion
  - thresholding
  - contour detection
  - optional state estimation via Kalman filtering

- **Tracker (optional)**  
  Applies smoothing / prediction to noisy measurements.

- **UI / Visualization**  
  Displays selected debug stages and reports runtime metrics.

---

## Frame & Timestamping

Each captured frame is wrapped in a lightweight structure:

```cpp
struct Frame {
    cv::Mat image;
    std::chrono::steady_clock::time_point timeStamp;
};
