#include "FrameGrabber.h"

FrameGrabber::FrameGrabber(FrameQueue& queue, int cameraIndex) :   
m_queue(queue), m_cameraIndex(cameraIndex), m_running(false)
{
}

FrameGrabber::~FrameGrabber()
{
    stop();
}

void FrameGrabber::start()
{
    m_running = true;
    // spin up the worker thread
    m_thread = std::thread(&FrameGrabber::run, this);
}

void FrameGrabber::stop()
{
    m_running = false;
    if (m_thread.joinable())
    {
        m_thread.join();
    }
}

void FrameGrabber::run()
{
    try
    {
        cv::VideoCapture cap(m_cameraIndex); 

        if (!cap.isOpened())
        {
            std::cerr << "ERROR: Could not open camera " << m_cameraIndex << "\n";
            return;
        }

        // main grab loop, runs until stop() flips m_running
        while (m_running)
        {
            Frame frame;

            cv::Mat img;
            // OpenCV blocks here, so we're not burning CPU waiting
            cap >> img;   

            if (img.empty())
            {
                continue; // camera glitch, just skip and keep going
            }

            // move image into frame (no copy)
            frame.image     = std::move(img);
            frame.timeStamp = std::chrono::steady_clock::now();

            // push frame to queue for the vision thread
            m_queue.push(std::move(frame));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "[FrameGrabber] EXCEPTION: " << e.what() << "\n";
        m_queue.stop();          // wake consumer so shutdown doesn't hang
    }
    catch (...)
    {
        std::cerr << "[FrameGrabber] UNKNOWN EXCEPTION\n";
        m_queue.stop();
    }
}