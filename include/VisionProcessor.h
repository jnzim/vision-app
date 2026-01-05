#ifndef VISIONPROCESSOR_H
#define VISIONPROCESSOR_H

#include <thread>
#include <atomic>
#include <chrono>
#include <string>
#include <unordered_map>
#include <mutex>
#include <cstddef>
#include <opencv2/opencv.hpp>
#include "FrameGrabber.h"
#include "FrameQueue.h"
#include "KalmanTracker.h"
#include "RollingStats.h"
#include <memory>
#include <vision/Interfaces/IFaceDetector.h>
#include <vision/Interfaces/IFaceEmbedder.h>

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


// VisionProcessor
// Runs a background thread that pulls frames from FrameQueue
// and runs your vision pipeline (blob, Kalman, etc.)
class VisionProcessor
{

public:
    // ctor: keep a reference to the shared queue
    VisionProcessor(FrameQueue& queue);
    ~VisionProcessor();
    void start();
    void stop();
    
    bool getLatestDebugImage(DebugStage stage,
                            cv::Mat& outImg,
                            Clock::time_point& outAcquired);
public:


    // =========================================================
    // PUBLIC READ-ONLY METRICS (written by vision thread)
    // =========================================================

    // --- Timing (milliseconds) ---
    std::atomic<double> procMs{0.0};          // last frame
    std::atomic<double> cycleMs{0.0};         // last processed cycle
    std::atomic<double> fps{0.0};             

    std::atomic<double> procJitterMs{0.0};    // σ(procMs)
    std::atomic<double> cycleJitterMs{0.0};   // σ(processed cadence)
    std::atomic<double> acqJitterMs{0.0};     // σ(camera cadence)

    // --- Spatial jitter (pixels) ---
    std::atomic<double> rawJitterPx{0.0};     // σ(|meas[i]-meas[i-1]|)
    std::atomic<double> kalmanJitterPx{0.0};  // σ(|pred[i]-pred[i-1]|)

private:
    void run();
    void processFrame(const Frame& frame);
    void publishDebugImage( DebugStage stage,
                            const cv::Mat& img,
                            Clock::time_point t_acquired);

private:

    // =========================================================
    // PRIVATE ROLLING STATISTICS (implementation detail)
    // =========================================================
    RollingStats m_procStats{120};
    RollingStats m_cycleStats{120};
    RollingStats m_acqStats{120};
    RollingStats m_rawStepStats{120};
    RollingStats m_kfStepStats{120};

    // =========================================================
    // PRIVATE TIMESTAMP STATE
    // =========================================================
    Clock::time_point m_lastAcqTime{};
    Clock::time_point m_lastFrameTime{};

    // =========================================================
    // PRIVATE POSITION HISTORY
    // =========================================================
    bool m_hasPrevMeas = false;
    bool m_hasPrevPred = false;
    cv::Point2f m_prevMeas{};
    cv::Point2f m_prevPred{};

    std::atomic<bool>   m_running{false};
    FrameQueue&         m_queue;
    std::thread         m_thread;
    mutable std::mutex  m_debugMutex;
    std::unordered_map<DebugStage, DebugPacket, DebugStageHash> m_latestDebugByStage;

    KalmanTracker       m_tracker;

    std::unique_ptr<vision::IFaceDetector> m_faceDetector;
    std::unique_ptr<vision::IFaceEmbedder> m_faceEmbedder;


};

#endif