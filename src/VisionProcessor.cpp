// VisionProcessor.cpp

#include "VisionProcessor.h"

#include <cstddef>
#include <iostream>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <vector>
#include <algorithm>   // max_element
#include <vision/Interfaces/FaceBackendFactory.h>
#include <math.h>

VisionProcessor::VisionProcessor(FrameQueue& queue)
    : m_queue(queue), m_running(false)
{
    m_faceDetector = vision::CreateFaceDetector_OpenCVDNN();
    // m_faceEmbedder = vision::CreateFaceEmbedder_OpenCVDNN();
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
    // Face detection cache / throttle / ROI state
    // =========================================================
    static int frameCounter = 0;
    static std::vector<vision::Detection> lastFaceDets;
    static double lastFaceMs = 0.0;

    static cv::Point2f lastPredForROI{0.f, 0.f};
    static bool lastPredValidForROI = false;

    const bool runFaceDetect = (++frameCounter % 5) == 0; // every 5th frame

    // =========================================================
    // 1) Detect measurement (blob centroid fallback)
    // =========================================================
    cv::Mat gray;
    cv::cvtColor(f.image, gray, cv::COLOR_BGR2GRAY);

    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5,5), 0);

    cv::Mat binary;
    cv::threshold(blurred, binary, 35, 255, cv::THRESH_BINARY);

    bool hasMeas = false;
    cv::Point2f meas{};


    // =========================================================
    // 1b) Face detection (throttled) + ROI crop near last prediction
    // =========================================================
    if (m_faceDetector)
    {
        try
        {
            if (runFaceDetect)
            {
                // Decide ROI:
                // - If we have a previous prediction, crop around it.
                // - Otherwise detect on full frame (bootstrap).
                cv::Rect roiRect(0, 0, f.image.cols, f.image.rows);

                if (lastPredValidForROI)
                {
                    // ROI size: tune this. 640 is a good starting point for 1080p.
                    const int roiW = 640;
                    const int roiH = 640;

                    int cx = static_cast<int>(std::round(lastPredForROI.x));
                    int cy = static_cast<int>(std::round(lastPredForROI.y));

                    roiRect = cv::Rect(cx - roiW / 2, cy - roiH / 2, roiW, roiH);
                    roiRect &= cv::Rect(0, 0, f.image.cols, f.image.rows); // clamp
                }

                // Run detection on ROI
                const auto t0 = Clock::now();

                cv::Mat roiImg = f.image(roiRect); // view (no copy)
                auto detsRoi = m_faceDetector->detect(roiImg);

                const auto t1 = Clock::now();
                lastFaceMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

                // Translate detections back into full-frame coordinates
                lastFaceDets.clear();
                lastFaceDets.reserve(detsRoi.size());

                for (auto d : detsRoi)
                {
                    d.box.x += roiRect.x;
                    d.box.y += roiRect.y;
                    lastFaceDets.push_back(d);
                }
            }

            // Prefer face measurement if available
            if (!lastFaceDets.empty())
            {
                const auto bestIt = std::max_element(
                    lastFaceDets.begin(), lastFaceDets.end(),
                    [](const vision::Detection& a, const vision::Detection& b)
                    {
                        return a.score < b.score;
                    });

                const auto& box = bestIt->box;
                meas = cv::Point2f(
                    box.x + 0.5f * box.width,
                    box.y + 0.5f * box.height);
                hasMeas = true;
            }
        }
        catch (const cv::Exception& e)
        {
            static int errCount = 0;
            if (errCount++ < 5)
                std::cerr << "[FaceDetect] OpenCV exception: " << e.what() << "\n";
        }
        catch (const std::exception& e)
        {
            static int errCount = 0;
            if (errCount++ < 5)
                std::cerr << "[FaceDetect] exception: " << e.what() << "\n";
        }
    }

    // =========================================================
    // 2) Kalman update / predict (use acquisition timestamp)
    // =========================================================
    const auto tObs = f.timeStamp;

    cv::Point2f pred{};
    if (hasMeas)
        pred = m_tracker.update(meas, tObs);
    else
        pred = m_tracker.predictToTime(tObs);

    const bool predValid = hasMeas || (pred.x != 0.0f || pred.y != 0.0f);

    // =========================================================
    // 3) Spatial jitter (step-based)
    // =========================================================
    if (hasMeas)
    {
        if (m_hasPrevMeas)
            m_rawStepStats.add(cv::norm(meas - m_prevMeas));
        else
            m_hasPrevMeas = true;

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
    // 5a) Draw overlays
    // =========================================================
    cv::Mat display = f.image.clone();

    // 5b) Draw face boxes + “proof” overlay
    if (m_faceDetector)
    {
        cv::putText(display,
                    cv::format("faces: %zu  face(ms): %.1f", lastFaceDets.size(), lastFaceMs),
                    {15, 140},
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.8,
                    cv::Scalar(0, 255, 0),
                    2);

        for (const auto& d : lastFaceDets)
            cv::rectangle(display, d.box, cv::Scalar(0, 255, 0), 2);
    }

    // Draw meas/pred points
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

    cv::putText(display, cv::format("KF update: %s", hasMeas ? "YES" : "NO"),
                {15, 200},
            cv::FONT_HERSHEY_SIMPLEX,
            0.7,
                hasMeas ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255),
            2);

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
