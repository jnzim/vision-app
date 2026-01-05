// VisionProcessor.cpp

#include "VisionProcessor.h"

#include <cstddef>
#include <iostream>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <vector>
#include <vision/Interfaces/FaceBackendFactory.h>


VisionProcessor::VisionProcessor(FrameQueue& queue)
    : m_queue(queue), m_running(false)
{
    m_faceDetector = vision::CreateFaceDetector_OpenCVDNN();
    m_faceEmbedder = vision::CreateFaceEmbedder_OpenCVDNN();
}

VisionProcessor::~VisionProcessor()
{
    stop();
}

void VisionProcessor::start()
{
    if (m_running.load(std::memory_order_relaxed)) return;

    m_running.store(true, std::memory_order_relaxed);
    m_thread = std::thread(&VisionProcessor::run, this);
}

void VisionProcessor::stop()
{
    if (!m_running.load(std::memory_order_relaxed)) return;

    m_running.store(false, std::memory_order_relaxed);
    m_queue.stop(); // wake pop()

    if (m_thread.joinable())
        m_thread.join();
}

void VisionProcessor::run()
{
    try
    {
        while (m_running.load(std::memory_order_relaxed))
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
        m_running.store(false, std::memory_order_relaxed);
        m_queue.stop();
    }
    catch (const std::exception& e)
    {
        std::cerr << "[VisionProcessor] std::exception: " << e.what() << "\n";
        m_running.store(false, std::memory_order_relaxed);
        m_queue.stop();
    }
    catch (...)
    {
        std::cerr << "[VisionProcessor] unknown exception\n";
        m_running.store(false, std::memory_order_relaxed);
        m_queue.stop();
    }
}

void VisionProcessor::processFrame(const Frame& f)
{
    if (f.image.empty()) return;

    // =========================================================
    // 0) Acquisition cadence jitter
    // =========================================================
    if (m_lastAcqTime.time_since_epoch().count() != 0)
    {
        double acqCycleMs =
            std::chrono::duration<double, std::milli>(f.timeStamp - m_lastAcqTime).count();

        if (acqCycleMs > 0.0)
        {
            m_acqStats.add(acqCycleMs);
            acqJitterMs.store(m_acqStats.stddev(), std::memory_order_relaxed);
        }
    }
    m_lastAcqTime = f.timeStamp;

    const auto tStart = Clock::now();

    // =========================================================
    // 1) Detect measurement (centroid)
    // =========================================================
    cv::Mat gray;
    cv::cvtColor(f.image, gray, cv::COLOR_BGR2GRAY);

    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5,5), 0);

    cv::Mat binary;
    cv::threshold(blurred, binary, 35, 255, cv::THRESH_BINARY);

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

    // =========================================================
    // 2) Kalman update / predict (use acquisition timestamp)
    // =========================================================
    const auto tObs = f.timeStamp;

    cv::Point2f pred{};
    if (hasMeas) 
    {
        pred = m_tracker.update(meas, tObs);
    }
    else         
    {
        pred = m_tracker.predictToTime(tObs);
    }
    const bool predValid = hasMeas || (pred.x != 0.0f || pred.y != 0.0f);

    // =========================================================
    // 3) Spatial jitter (step-based)
    // =========================================================
    if (hasMeas)
    {
        if (m_hasPrevMeas)
        {
            m_rawStepStats.add(cv::norm(meas - m_prevMeas));
        }
        else
        {
            m_hasPrevMeas = true;
        }
        m_prevMeas = meas;
    }
    rawJitterPx.store(m_rawStepStats.stddev(), std::memory_order_relaxed);

    if (predValid)
    {
        if (m_hasPrevPred)
            m_kfStepStats.add(cv::norm(pred - m_prevPred));
        else
            m_hasPrevPred = true;

        m_prevPred = pred;
    }
    kalmanJitterPx.store(m_kfStepStats.stddev(), std::memory_order_relaxed);

    // =========================================================
    // 4) Timing metrics
    // =========================================================
    const auto tEnd = Clock::now();

    const double procMsLocal =
        std::chrono::duration<double, std::milli>(tEnd - tStart).count();

    procMs.store(procMsLocal, std::memory_order_relaxed);
    m_procStats.add(procMsLocal);
    procJitterMs.store(m_procStats.stddev(), std::memory_order_relaxed);

    double cycleMsLocal = 0.0;
    if (m_lastFrameTime.time_since_epoch().count() != 0)
    {
        cycleMsLocal =
            std::chrono::duration<double, std::milli>(tEnd - m_lastFrameTime).count();

        if (cycleMsLocal > 0.0)
            m_cycleStats.add(cycleMsLocal);
    }
    m_lastFrameTime = tEnd;

    cycleMs.store(cycleMsLocal, std::memory_order_relaxed);
    cycleJitterMs.store(m_cycleStats.stddev(), std::memory_order_relaxed);

    double fpsLocal = 0.0;
    if (cycleMsLocal > 0.0)
        fpsLocal = 1000.0 / cycleMsLocal;
    fps.store(fpsLocal, std::memory_order_relaxed);

    // =========================================================
    // 5) Draw overlays (BLACK TEXT ONLY)
    // =========================================================
    cv::Mat display = f.image.clone();

    if (hasMeas)
    {
        cv::circle(display, meas, 10, cv::Scalar(0, 255, 0), -1);
        cv::circle(display, meas, 10, cv::Scalar(0, 0, 0), 2);
    }

    if (predValid)
    {
        cv::circle(display, pred, 8, cv::Scalar(255, 0, 0), -1);
        cv::circle(display, pred, 8, cv::Scalar(0, 0, 0), 2);
    }

    const double fontScale = 1.0;
    const int textThickness = 5;

    std::ostringstream ss1;
    ss1 << std::fixed << std::setprecision(1)
        << "proc(ms): " << procMsLocal
        << "  cycle(ms): " << cycleMsLocal
        << "  fps: " << fpsLocal
        << "  jit(acq/cyc/proc ms): "
        << acqJitterMs.load(std::memory_order_relaxed) << "/"
        << cycleJitterMs.load(std::memory_order_relaxed) << "/"
        << procJitterMs.load(std::memory_order_relaxed);

    cv::putText(display, ss1.str(), {15, 35},
                cv::FONT_HERSHEY_SIMPLEX, fontScale,
                cv::Scalar(0, 255, 255), textThickness);

#include <cmath>

std::ostringstream ss2;
ss2 << std::fixed << std::setprecision(1);

ss2 << "meas: ";
if (hasMeas) {
    ss2 << "(" << std::setw(6) << meas.x << ", " << std::setw(6) << meas.y << ")";
} else {
    ss2 << "(  n/a,   n/a)";
}

ss2 << "  pred: ";
if (predValid) {
    ss2 << "(" << std::setw(6) << pred.x << ", " << std::setw(6) << pred.y << ")";
} else {
    ss2 << "(  n/a,   n/a)";
}


// d = innovation magnitude (prediction error)
//     Pixel distance between predicted state and measured centroid.
//     Indicates how "surprised" the model is by the measurement.
ss2 << "  d:";
if (hasMeas && predValid) {
    float dx = meas.x - pred.x;
    float dy = meas.y - pred.y;
    ss2 << std::setw(5) << std::sqrt(dx*dx + dy*dy) << "px";
} else {
    ss2 << "  n/a";
}


    cv::putText(display, ss2.str(), {15, 70},
                cv::FONT_HERSHEY_SIMPLEX, fontScale,
                cv::Scalar(0, 255, 255), textThickness);

    std::ostringstream ss3;
    ss3 << std::fixed << std::setprecision(2)
        << "jit(step px) raw/kf: "
        << rawJitterPx.load(std::memory_order_relaxed) << "/"
        << kalmanJitterPx.load(std::memory_order_relaxed);

    cv::putText(display, ss3.str(), {15, 105},
                cv::FONT_HERSHEY_SIMPLEX, fontScale,
                cv::Scalar(0, 255, 255), textThickness);

    // =========================================================
    // 6) Publish AFTER overlays
    // =========================================================
    publishDebugImage(DebugStage::RAW, f.image, f.timeStamp);
    publishDebugImage(DebugStage::GRAY, gray, f.timeStamp);
    publishDebugImage(DebugStage::BLUR, blurred, f.timeStamp);
    publishDebugImage(DebugStage::THRESHOLD, binary, f.timeStamp);
    publishDebugImage(DebugStage::OVERLAY, display, f.timeStamp);
}

void VisionProcessor::publishDebugImage(DebugStage stage,
                                        const cv::Mat& img,
                                        Clock::time_point t_acquired)
{
    std::lock_guard<std::mutex> lock(m_debugMutex);

    DebugPacket& pkt = m_latestDebugByStage[stage];
    pkt.t_acquired = t_acquired;
    img.copyTo(pkt.img);
}

bool VisionProcessor::getLatestDebugImage(DebugStage stage,
                                          cv::Mat& outImg,
                                          Clock::time_point& outAcquired)
{
    std::lock_guard<std::mutex> lock(m_debugMutex);

    auto it = m_latestDebugByStage.find(stage);
    if (it == m_latestDebugByStage.end() || it->second.img.empty())
        return false;

    it->second.img.copyTo(outImg);
    outAcquired = it->second.t_acquired;
    return true;
}
