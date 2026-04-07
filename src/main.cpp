#include <iostream>

#include <opencv2/opencv.hpp>
#include "KalmanTracker.h"

#include "FrameQueue.h"
#include "FrameGrabber.h"
#include "VisionProcessor.h"

using namespace std;

int main()
{
    FrameQueue queue;
    FrameQueue& q = queue;                  // shared queue between grabber and processor
    FrameGrabber grabber(q, 0);             // camera 0
    VisionProcessor processor(queue);

    grabber.start();
    processor.start();

    cv::namedWindow("Debug", cv::WINDOW_AUTOSIZE);

    cv::Mat img;
    Clock::time_point t_acq{};
    DebugStage stage = DebugStage::OVERLAY;
    double lastLatencyMs = 0.0;

    std::cerr << "OpenCV version: " << CV_VERSION << "\n";

    while (true)
    {
        if (processor.getLatestDebugImage(stage, img, t_acq))
        {
            cv::imshow("Debug", img);
        }

        const int key = cv::waitKey(1);
        const auto t_disp = Clock::now();

        // display latency from acquisition timestamp to UI update
        if (!img.empty())
        {
            lastLatencyMs =
                std::chrono::duration<double, std::milli>(t_disp - t_acq).count();
        }

        if (key == 27)
        {
            break;
        }
        else if (key == '1')
        {
            stage = DebugStage::RAW;
        }
        else if (key == '2')
        {
            stage = DebugStage::GRAY;
        }
        else if (key == '3')
        {
            stage = DebugStage::BLUR;
        }
        else if (key == '4')
        {
            stage = DebugStage::THRESHOLD;
        }
        else if (key == '5')
        {
            stage = DebugStage::OVERLAY;
        }
    }

    grabber.stop();
    processor.stop();
    queue.stop();

    return 0;
}