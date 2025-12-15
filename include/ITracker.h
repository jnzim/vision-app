#pragma once
#include <opencv2/core.hpp>
#include <chrono>


struct TrackerEstimate
{
    cv::Point2f postion {};
    bool valid {false};
    std::chrono::steady_clock::time_point estimateTime;

};


class ITracker
{
public:
    virtual ~ITracker() = default;
    virtual TrackerEstimate


private:
};