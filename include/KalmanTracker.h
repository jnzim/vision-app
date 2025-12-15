#pragma once

#include <opencv2/opencv.hpp>

// Simple 2D constant-velocity Kalman filter.
// State x = [pos_x, pos_y, vel_x, vel_y]^T
// Measurement z = [pos_x, pos_y]^T
class KalmanTracker
{
public:
    // dt = time step between updates (seconds), e.g. 0.033 ~ 30 FPS
    explicit KalmanTracker(float dt = 0.033f);

    // Provide a measured position (x, y) and get a filtered position back.
    cv::Point2f update(const cv::Point2f& measured);

    // Get a predicted position without a measurement (e.g. if detection fails).
    cv::Point2f predictOnly();

private:
    cv::KalmanFilter kf;
    cv::Mat measurement; // 2x1 vector [pos_x, pos_y]
};
