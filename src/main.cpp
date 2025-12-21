#include <iostream>
#include <random>
#include <opencv2/opencv.hpp>

#include "KalmanTracker.h"
#include "ThreadPool.h"
#include "TrackerSM.h"
#include "FrameQueue.h"
#include "FrameGrabber.h"
#include "VisionProcessor.h"

using namespace std;


int main()
{
    FrameQueue      queue;
    FrameQueue&     q = queue;
    FrameGrabber    grabber(q, 0);
    VisionProcessor processor(queue);

    grabber.start();
    processor.start();

    cv::namedWindow("Camera", cv::WINDOW_AUTOSIZE);

cv::Mat img;
Clock::time_point t_acq{};
DebugStage stage = DebugStage::OVERLAY;
double lastLatencyMs = 0.0;

while (true)
{
    if (processor.getLatestDebugImage(stage, img, t_acq))
    {
        cv::imshow("Debug", img);
    }

        int key = cv::waitKey(1);
        auto t_disp = Clock::now();

        if (!img.empty())
        {
            lastLatencyMs = std::chrono::duration<double, std::milli>(t_disp - t_acq).count();
        }

        if (key == 27) break;
        if (key == '1') stage = DebugStage::RAW;
        if (key == '2') stage = DebugStage::THRESHOLD;
        if (key == '3') stage = DebugStage::OVERLAY;
    }


    grabber.stop();
    processor.stop();
    queue.stop();

    return 0;
}
