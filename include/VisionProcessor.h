#ifndef VISIONPROCESSOR_H
#define VISIONPROCESSOR_H

#include <thread>
#include <atomic>
#include <opencv2/opencv.hpp>
#include "FrameGrabber.h"
#include "FrameQueue.h"

#pragma once

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
    bool getLatestDebugImage(cv::Mat& out);

private:
    void run();
    void processFrame(const Frame& frame);
    void publishDebugImage(const cv::Mat& img);
    std::atomic<bool>   m_running;
    FrameQueue&         m_queue;
    std::thread         m_thread;
    std::mutex          m_dbgMutex;
    cv::Mat             m_latestDbg;   // owned by VisionProcessor

};

#endif