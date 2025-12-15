
#include <chrono>
#include "KalmanTracker.h"

KalmanTracker::KalmanTracker()
{
    kf = cv::KalmanFilter(4, 2, 0, CV_32F);

    kf.transitionMatrix = cv::Mat::eye(4, 4, CV_32F);

    kf.measurementMatrix = (cv::Mat_<float>(2, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0
    );

    setIdentity(kf.processNoiseCov,     cv::Scalar(1e-2));
    setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));
    setIdentity(kf.errorCovPost,        cv::Scalar(1));

    kf.statePost = (cv::Mat_<float>(4,1) << 0,0,0,0);

    measurement = cv::Mat::zeros(2, 1, CV_32F);
}



cv::Point2f KalmanTracker::update(const cv::Point2f& measured,
                                    std::chrono::steady_clock::time_point t)
{
    if (!hasState)
    {
        kf.statePost.at<float>(0) = measured.x;
        kf.statePost.at<float>(1) = measured.y;
        kf.statePost.at<float>(2) = 0.0f;
        kf.statePost.at<float>(3) = 0.0f;

        lastTime = t;
        hasState = true;
        return measured;
    }

    predictTo(t);

    measurement.at<float>(0) = measured.x;
    measurement.at<float>(1) = measured.y;

    cv::Mat estimated = kf.correct(measurement);
    return { estimated.at<float>(0), estimated.at<float>(1) };
}

cv::Point2f KalmanTracker::predictToTime(std::chrono::steady_clock::time_point t)
{
    if (!hasState)
        return {};

    predictTo(t);

    return {
        kf.statePost.at<float>(0),
        kf.statePost.at<float>(1)
    };
}

void KalmanTracker::predictTo(std::chrono::steady_clock::time_point t)
{
    float dt = std::chrono::duration<float>(t - lastTime).count();
    if (dt < 0.0f) dt = 0.0f;

    kf.transitionMatrix.at<float>(0,2) = dt;
    kf.transitionMatrix.at<float>(1,3) = dt;

    kf.predict();
    lastTime = t;
}

void KalmanTracker::reset()
{
    hasState = false;
    lastTime = std::chrono::steady_clock::time_point{};
}