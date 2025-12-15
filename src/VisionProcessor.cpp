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

void VisionProcessor::processFrame(const Frame& f)
{
    if (f.image.empty()) return;

    const auto tStart = std::chrono::steady_clock::now();

    // ----------------------------
    // 1) Detect measurement (centroid)
    // ----------------------------
    cv::Mat gray;
    cv::cvtColor(f.image, gray, cv::COLOR_BGR2GRAY);

    cv::Mat binary;
    cv::threshold(gray, binary, 200, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    bool hasMeas = false;
    cv::Point2f meas{};

    if (!contours.empty())
    {
        int bestIdx = 0;
        double bestArea = 0.0;

        for (int i = 0; i < static_cast<int>(contours.size()); i++)
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
            meas = cv::Point2f(
                static_cast<float>(m.m10 / m.m00),
                static_cast<float>(m.m01 / m.m00));
            hasMeas = true;
        }
    }

    // ----------------------------
    // 2) Kalman update/predict
    // ----------------------------
    const auto now = std::chrono::steady_clock::now();

    cv::Point2f pred{};
    if (hasMeas) pred = m_tracker.update(meas, now);
    else         pred = m_tracker.predictToTime(now);

    // ----------------------------
    // 3) Metrics: proc, cycle, fps
    // ----------------------------
    const double procMs =
        std::chrono::duration<double, std::milli>(now - tStart).count();

    double cycleMs = 0.0;
    if (m_lastFrameTime.time_since_epoch().count() != 0)
        cycleMs = std::chrono::duration<double, std::milli>(now - m_lastFrameTime).count();
    m_lastFrameTime = now;

    if (cycleMs > 0.0) m_fps = 1000.0 / cycleMs;

    // ----------------------------
    // 4) Draw overlays
    // ----------------------------
    cv::Mat display = f.image.clone();

    // Bigger dots
    if (hasMeas)
    {
        cv::circle(display, meas, 10, cv::Scalar(0,255,0), -1);
        cv::circle(display, meas, 10, cv::Scalar(0,0,0), 2);
    }

    // Avoid drawing junk (0,0) before init
    const bool predValidForDraw = hasMeas || (pred.x != 0.0f || pred.y != 0.0f);
    if (predValidForDraw)
    {
        cv::circle(display, pred, 8, cv::Scalar(255,0,0), -1);
        cv::circle(display, pred, 8, cv::Scalar(0,0,0), 2);
    }

    // Big readable text (shadow + main)
    const double fontScale = 1.0;
    const int textThickness = 3;
    const int shadowThickness = 5;

    std::ostringstream ss1;
    ss1 << std::fixed << std::setprecision(1)
        << "proc(ms): " << procMs
        << "  cycle(ms): " << cycleMs
        << "  fps: " << m_fps;

    cv::Point org1(15, 35);
    cv::putText(display, ss1.str(), org1,
                cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0,0,0), shadowThickness);
    cv::putText(display, ss1.str(), org1,
                cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255,255,255), textThickness);

    std::ostringstream ss2;
    ss2 << std::fixed << std::setprecision(1);

    if (hasMeas)
        ss2 << "meas: (" << meas.x << ", " << meas.y << ")  ";
    else
        ss2 << "meas: (n/a)  ";

    if (predValidForDraw)
        ss2 << "pred: (" << pred.x << ", " << pred.y << ")";
    else
        ss2 << "pred: (n/a)";

    cv::Point org2(15, 70);
    cv::putText(display, ss2.str(), org2,
                cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0,0,0), shadowThickness);
    cv::putText(display, ss2.str(), org2,
                cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255,255,255), textThickness);

    // ----------------------------
    // 5) Publish AFTER overlays
    // ----------------------------
    publishDebugImage(display);
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