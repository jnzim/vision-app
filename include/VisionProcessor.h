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


using Clock = std::chrono::steady_clock;

struct DebugPacket
{
    cv::Mat img;
    Clock::time_point t_acquired{};
};

enum class DebugStage
{
    RAW         = 0,
    THRESHOLD   = 1,
    OVERLAY     = 2,
    KALMAN      = 3,
    NUM_STAGES  = 4,
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

private:
    void run();
    void processFrame(const Frame& frame);
    void publishDebugImage( DebugStage stage,
                            const cv::Mat& img,
                            Clock::time_point t_acquired);

private:


    std::atomic<bool>   m_running{false};
    FrameQueue&         m_queue;
    std::thread         m_thread;
    mutable std::mutex  m_debugMutex;
    std::unordered_map<DebugStage, DebugPacket, DebugStageHash> m_latestDebugByStage;

    KalmanTracker       m_tracker;
    std::chrono::steady_clock::time_point m_lastFrameTime{};
    double m_fps{0.0};


};

#endif