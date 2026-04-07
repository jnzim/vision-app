#ifndef VISIONPROCESSOR_H
#define VISIONPROCESSOR_H

#include <thread>
#include <atomic>
#include <chrono>
#include <unordered_map>
#include <mutex>
#include <cstddef>

#include <opencv2/opencv.hpp>

#include "FrameQueue.h"
#include "KalmanTracker.h"
#include "RollingStats.h"
#include "FaceDetector_OpenCVDNN.h"

using Clock = std::chrono::steady_clock;

struct DebugPacket
{
    cv::Mat img;
    Clock::time_point t_acquired{};
};

enum class DebugStage
{
    RAW         = 0,
    GRAY        = 1,
    BLUR        = 2,
    THRESHOLD   = 3,
    OVERLAY     = 4,
    KALMAN      = 5,
    NUM_STAGES  = 6,
};

struct DebugStageHash
{
    std::size_t operator()(DebugStage s) const noexcept
    {
        return static_cast<std::size_t>(s);
    }
};

class VisionProcessor
{
public:
    VisionProcessor(FrameQueue& queue);
    ~VisionProcessor();

    void start();
    void stop();

    bool getLatestDebugImage(DebugStage stage,
                             cv::Mat& outImg,
                             Clock::time_point& outAcquired);

    std::atomic<double> procMs{0.0};
    std::atomic<double> cycleMs{0.0};
    std::atomic<double> fps{0.0};

    std::atomic<double> procJitterMs{0.0};
    std::atomic<double> cycleJitterMs{0.0};
    std::atomic<double> acqJitterMs{0.0};

    std::atomic<double> rawJitterPx{0.0};
    std::atomic<double> kalmanJitterPx{0.0};

private:
    void run();
    void processFrame(const Frame& frame);
    void publishDebugImage(DebugStage stage,
                           const cv::Mat& img,
                           Clock::time_point t_acquired);

private:
    RollingStats m_procStats{120};
    RollingStats m_cycleStats{120};
    RollingStats m_acqStats{120};
    RollingStats m_rawStepStats{120};
    RollingStats m_kfStepStats{120};

    Clock::time_point m_lastAcqTime{};
    Clock::time_point m_lastFrameTime{};

    bool m_hasPrevMeas = false;
    bool m_hasPrevPred = false;
    cv::Point2f m_prevMeas{};
    cv::Point2f m_prevPred{};

    std::atomic<bool> m_running{false};
    FrameQueue& m_queue;
    std::thread m_thread;
    mutable std::mutex m_debugMutex;
    std::unordered_map<DebugStage, DebugPacket, DebugStageHash> m_latestDebugByStage;

    KalmanTracker m_tracker;
    vision::FaceDetector_OpenCVDNN m_faceDetector;
};

#endif