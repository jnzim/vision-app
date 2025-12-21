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

        // Main grab loop: runs until stop() is called
        while (m_running)
        {
            Frame frame;

            cv::Mat img;
            // OpenCV will block, using 0 CPU while waiting
            cap >> img;   

            if (img.empty())
            {
                continue; // camera glitch? skip
            }

            // just moves a pointer 
            frame.image = std::move(img);
            frame.timeStamp = std::chrono::steady_clock::now();

            // Push frame into the queue for processing
            m_queue.push(std::move(frame));
        }
    }
       catch (const std::exception& e)
    {
        std::cerr << "[FrameGrabber] EXCEPTION: " << e.what() << "\n";
        m_queue.stop();          // wake consumers so app can exit cleanly
    }
    catch (...)
    {
        std::cerr << "[FrameGrabber] UNKNOWN EXCEPTION\n";
        m_queue.stop();
    }
}