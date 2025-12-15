#include "KalmanTracker.h"

KalmanTracker::KalmanTracker(float dt)
{
    // 4 state vars, 2 measurements, 0 control inputs, float precision.
    kf = cv::KalmanFilter(4, 2, 0, CV_32F);

    // ----- State transition matrix A -----
    // x_k = A * x_{k-1} + w
    //
    // [pos_x]   [1 0 dt 0] [pos_x]
    // [pos_y] = [0 1 0  dt] [pos_y]
    // [vel_x]   [0 0 1  0 ] [vel_x]
    // [vel_y]   [0 0 0  1 ] [vel_y]
    //
    cv::Mat A = (cv::Mat_<float>(4, 4) <<
        1, 0, dt, 0,
        0, 1, 0,  dt,
        0, 0, 1,  0,
        0, 0, 0,  1
    );
    kf.transitionMatrix = A;

    // ----- Measurement matrix H -----
    // We measure position only:
    //
    // [z_x]   [1 0 0 0] [pos_x]
    // [z_y] = [0 1 0 0] [pos_y]
    //                 [vel_x]
    //                 [vel_y]
    //
    kf.measurementMatrix = (cv::Mat_<float>(2, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0
    );

    // ----- Process noise covariance Q -----
    // How much we allow the state to "wander" each step.
    // Bigger -> more responsive, less smooth.
    setIdentity(kf.processNoiseCov, cv::Scalar(1e-2));

    // ----- Measurement noise covariance R -----
    // How noisy our measurements are.
    // Bigger -> trust model more, measurements less.
    setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));

    // ----- Initial estimate covariance P -----
    // How uncertain we are about the initial state.
    setIdentity(kf.errorCovPost, cv::Scalar(1));

    // ----- Initial state x0 -----
    // Start assuming object is at (0,0) with velocity (0,0).
    // You can set this to something else if you know it.
    kf.statePost = (cv::Mat_<float>(4, 1) <<
        0,  // pos_x
        0,  // pos_y
        0,  // vel_x
        0   // vel_y
    );

    // 2x1 measurement vector: [pos_x, pos_y]
    measurement = cv::Mat::zeros(2, 1, CV_32F);
}

// Full update with a measurement: predict -> correct.
cv::Point2f KalmanTracker::update(const cv::Point2f& measured)
{
    // 1) Predict where we think the state is now
    cv::Mat prediction = kf.predict();
    (void)prediction; // we don't use it directly here, but it's updating kf's internal state

    // 2) Fill in measurement vector
    measurement.at<float>(0) = measured.x;
    measurement.at<float>(1) = measured.y;

    // 3) Correct with actual measurement (Kalman magic happens inside)
    cv::Mat estimated = kf.correct(measurement);

    // estimated = [pos_x, pos_y, vel_x, vel_y]^T
    float est_x = estimated.at<float>(0);
    float est_y = estimated.at<float>(1);

    return cv::Point2f(est_x, est_y);
}

// Prediction-only step (e.g. if we didn't get a valid measurement this frame).
cv::Point2f KalmanTracker::predictOnly()
{
    cv::Mat prediction = kf.predict();
    float pred_x = prediction.at<float>(0);
    float pred_y = prediction.at<float>(1);
    return cv::Point2f(pred_x, pred_y);
}
