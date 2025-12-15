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
    while (true)
    {
        if (processor.getLatestDebugImage(img))
            cv::imshow("Debug", img);

        int key = cv::waitKey(1);
        if (key == 27) break;
    }

    grabber.stop();
    processor.stop();
    queue.stop();

    return 0;
}
