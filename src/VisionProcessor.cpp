#include "VisionProcessor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

VisionProcessor::VisionProcessor(FrameQueue& queue)
    : m_running(false), m_queue(queue)
{
}

VisionProcessor::~VisionProcessor()
{
    stop();
}

void VisionProcessor::start()
{
    if (m_running.load(std::memory_order_relaxed))
    {
        return;
    }

    m_running.store(true, std::memory_order_relaxed);
    m_thread = std::thread(&VisionProcessor::run, this);
}

void VisionProcessor::stop()
{
    if (!m_running.load(std::memory_order_relaxed))
    {
        return;
    }

    m_running.store(false, std::memory_order_relaxed);
    m_queue.stop(); // unblock pop()

    if (m_thread.joinable())
    {
        m_thread.join();
    }
}

void VisionProcessor::run()
{
    try
    {
        while (m_running.load(std::memory_order_relaxed))
        {
            Frame frame;

            if (!m_queue.pop(frame))
            {
                break;
            }

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
    if (f.image.empty())
    {
        return;
    }

    // acquisition timing jitter (camera cadence stability)
    if (m_lastAcqTime.time_since_epoch().count() != 0)
    {
        const double acqCycleMs =
            std::chrono::duration<double, std::milli>(f.timeStamp - m_lastAcqTime).count();

        if (acqCycleMs > 0.0)
        {
            m_acqStats.add(acqCycleMs);
            acqJitterMs.store(m_acqStats.stddev(), std::memory_order_relaxed);
        }
    }

    m_lastAcqTime = f.timeStamp;

    const auto tStart = Clock::now();

    static int frameCounter = 0;
    static cv::Point2f lastPredForROI{0.f, 0.f};
    static bool lastPredValidForROI = false;

    // detector every 5 frames to keep CPU under control
    const bool runFaceDetect = ((++frameCounter % 5) == 0);

    cv::Mat gray;
    cv::cvtColor(f.image, gray, cv::COLOR_BGR2GRAY);

    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

    cv::Mat binary;
    cv::threshold(blurred, binary, 35, 255, cv::THRESH_BINARY);

    bool hasMeas = false;
    cv::Point2f meas{};

    std::vector<vision::Detection> dets;
    double faceMs = 0.0;

    try
    {
        if (runFaceDetect)
        {
            cv::Rect roiRect(0, 0, f.image.cols, f.image.rows);

            // crop around last prediction if tracker is locked on
            if (lastPredValidForROI)
            {
                const int roiW = 640;
                const int roiH = 640;

                const int cx = static_cast<int>(std::lround(lastPredForROI.x));
                const int cy = static_cast<int>(std::lround(lastPredForROI.y));

                // keep only the pixels that are in both the image and the ROI
                // ignore ROI hanging off the image
                roiRect = cv::Rect(cx - roiW / 2, cy - roiH / 2, roiW, roiH);
                roiRect &= cv::Rect(0, 0, f.image.cols, f.image.rows);
                // dont' bother with tiny ROI
                if ((roiRect.width < 64) || (roiRect.height < 64))
                {
                    roiRect = cv::Rect(0, 0, f.image.cols, f.image.rows);
                }
            }

            const auto t0 = Clock::now();

            cv::Mat roiImg  = f.image(roiRect);
            auto detsRoi    = m_faceDetector.detect(roiImg);
            const auto t1   = Clock::now();
            faceMs          = std::chrono::duration<double, std::milli>(t1 - t0).count();

            dets.reserve(detsRoi.size());

            for (auto det : detsRoi)
            {
                det.box.x += roiRect.x;
                det.box.y += roiRect.y;
                dets.push_back(det);
            }
        }

        if (!dets.empty())
        {
            // get the best scored detection
            const auto bestIt = std::max_element(
                dets.begin(),
                dets.end(),
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
        {
            std::cerr << "[FaceDetect] OpenCV exception: " << e.what() << "\n";
        }
    }
    catch (const std::exception& e)
    {
        static int errCount = 0;

        if (errCount++ < 5)
        {
            std::cerr << "[FaceDetect] exception: " << e.what() << "\n";
        }
    }

    const auto tObs = f.timeStamp;

    cv::Point2f pred{};

    // Kalman: update if measurement, otherwise predict forward
    if (hasMeas)
    {
        pred = m_tracker.update(meas, tObs);
    }
    else
    {
        pred = m_tracker.predictToTime(tObs);
    }

    const bool predValid = hasMeas || ((pred.x != 0.0f) || (pred.y != 0.0f));

    // feed prediction back into ROI logic
    lastPredValidForROI = predValid;

    if (predValid)
    {
        lastPredForROI = pred;
    }

    // raw detector jitter - pixel noise
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

    // filtered jitter (should be smoother)
    if (predValid)
    {
        if (m_hasPrevPred)
        {
            m_kfStepStats.add(cv::norm(pred - m_prevPred));
        }
        else
        {
            m_hasPrevPred = true;
        }

        m_prevPred = pred;
    }

    kalmanJitterPx.store(m_kfStepStats.stddev(), std::memory_order_relaxed);

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
        {
            m_cycleStats.add(cycleMsLocal);
        }
    }

    m_lastFrameTime = tEnd;

    cycleMs.store(cycleMsLocal, std::memory_order_relaxed);
    cycleJitterMs.store(m_cycleStats.stddev(), std::memory_order_relaxed);

    double fpsLocal = 0.0;

    if (cycleMsLocal > 0.0)
    {
        fpsLocal = 1000.0 / cycleMsLocal;
    }
    else
    {
        fpsLocal = 0.0;
    }

    fps.store(fpsLocal, std::memory_order_relaxed);

    cv::Mat display = f.image.clone();

    // publish debug images (latest only)
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