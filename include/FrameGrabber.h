#ifndef FRAMEGRABBER_H
#define FRAMEGRABBER_H

#pragma once

#include <thread>
#include "FrameQueue.h"
#include <atomic>
#include <thread>


class FrameGrabber
{
public:
     FrameGrabber(FrameQueue& queue, int cameraIndex = 0);
    ~FrameGrabber();
    void start();
    void stop();

private:
    void run();
    FrameQueue&         m_queue;
    cv::VideoCapture    m_cap;
    int                 m_cameraIndex;             // which camera to open (0 = default)
    std::atomic<bool>   m_running{false};
    std::thread         m_thread;

};

#endif