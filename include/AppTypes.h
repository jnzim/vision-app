#include <chrono>
#include <opencv2/opencv.hpp>

struct Frame
{
    cv::Mat image;
    std::chrono::steady_clock::time_point timeStamp;
};

struct TrackingResult
{
    bool hasTarget  = false;
    bool isLost     = false;
    cv::Point2f position();
};