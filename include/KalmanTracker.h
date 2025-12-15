#pragma once

#include <opencv2/opencv.hpp>
#include <chrono>

// Simple 2D constant-velocity Kalman filter.
// State x = [pos_x, pos_y, vel_x, vel_y]^T
// Measurement z = [pos_x, pos_y]^T
class KalmanTracker
{
public:
    KalmanTracker();

    // Update with a measurement taken at time t
    cv::Point2f update(const cv::Point2f& measured,
                       std::chrono::steady_clock::time_point t);

    // Predict state forward to time t (latency compensation / display time)
    cv::Point2f predictToTime(std::chrono::steady_clock::time_point t);

    void reset();

private:
    void predictTo(std::chrono::steady_clock::time_point t);

private:
    cv::KalmanFilter kf;
    cv::Mat measurement; // 2x1 vector [pos_x, pos_y]

    std::chrono::steady_clock::time_point lastTime{};
    bool hasState{false};
};
