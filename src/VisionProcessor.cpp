#include "VisionProcessor.h"
#include <iostream>
#include <chrono>

VisionProcessor::VisionProcessor(FrameQueue& queue) : m_queue(queue), m_running(false)
{ }

VisionProcessor::~VisionProcessor()
{
    stop();
}

void VisionProcessor::start()
{
    if (m_running.load()) return;

    m_running.store(true);
    m_thread = std::thread(&VisionProcessor::run, this);
}

void VisionProcessor::stop()
{
    if (!m_running.load()) return;

    m_running.store(false);

    m_queue.stop(); // wake pop()

    if (m_thread.joinable())
        m_thread.join();
}

void VisionProcessor::run()
{
    try
    {
        while (m_running.load())
        {
            Frame frame;
            if (!m_queue.pop(frame))
                break;

            processFrame(frame);
        }
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "[VisionProcessor] OpenCV exception: " << e.what() << "\n";
        m_running.store(false);
        m_queue.stop();
    }
    catch (const std::exception& e)
    {
        std::cerr << "[VisionProcessor] std::exception: " << e.what() << "\n";
        m_running.store(false);
        m_queue.stop();
    }
    catch (...)
    {
        std::cerr << "[VisionProcessor] unknown exception\n";
        m_running.store(false);
        m_queue.stop();
    }
}


void VisionProcessor::processFrame (const Frame& f)
{
    if (f.image.empty()) return;

    cv::Mat display = f.image.clone();
    // draw overlays on display...
    publishDebugImage(display);

    cv::Mat gray;
    cv::cvtColor(f.image, gray, cv::COLOR_BGR2GRAY);

    cv::Mat binary;
    cv::threshold(gray, binary, 200, 255, cv:: THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (!contours.empty())
    {  
        int bestIdx    = 0;
        double bestArea = 0;
        for (int i = 0; i < contours.size(); i++)
        {
            double area = cv::contourArea(contours[i]);
            if (area > bestArea)
            {
                bestArea = area;
                bestIdx = i;
            }
        }
        cv::Moments m = cv::moments(contours[bestIdx]);

        if (m.m00 > 0.0) 
        {
            cv::Point2f center(
                static_cast<float>(m.m10 / m.m00),
                static_cast<float>(m.m01 / m.m00));
            
            std::cout << "Center: " << center.x << " " << center.y << std::endl;
        }


    }
}


void VisionProcessor::publishDebugImage(const cv::Mat& img)
{
    std::lock_guard<std::mutex> lock(m_dbgMutex);
    m_latestDbg = img.clone();   // deep copy into shared slot
}

bool VisionProcessor::getLatestDebugImage(cv::Mat& out)
{
    std::lock_guard<std::mutex> lock(m_dbgMutex);
    if (m_latestDbg.empty())
        return false;

    m_latestDbg.copyTo(out);     // deep copy out for UI thread
    return true;

}