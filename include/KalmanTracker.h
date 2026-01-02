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

    cv::Point2f update(const cv::Point2f& measured,
                       std::chrono::steady_clock::time_point t);

    cv::Point2f predictToTime(std::chrono::steady_clock::time_point t);

    // Model uncertainty knob: assumed acceleration standard deviation (px/s^2).
    // Lower => smaller Q => smoother (more model trust). Higher => more responsive.
    void setAccelerationNoiseStd(float sigmaA) { m_sigmaA = sigmaA; }
    float getAccelerationNoiseStd() const { return m_sigmaA; }

    void setMaxDt(float maxDt) { m_maxDt = maxDt; }
    float getMaxDt() const { return m_maxDt; }

    // The actual process noise covariance matrix Q used by the filter (changes with dt).
    const cv::Mat& getProcessNoiseCov() const { return kf.processNoiseCov; }

    // The actual measurement noise covariance matrix R.
    const cv::Mat& getMeasurementNoiseCov() const { return kf.measurementNoiseCov; }

    void reset();

private:
    void predictTo(std::chrono::steady_clock::time_point t);

private:
    cv::KalmanFilter kf;
    cv::Mat measurement;

    std::chrono::steady_clock::time_point lastTime{};
    bool hasState{false};

    // Tuning knobs (parameters that generate Q in predictTo()).
    float m_sigmaA = 30.0f; // px/s^2
    float m_maxDt  = 0.10f; // seconds
};
