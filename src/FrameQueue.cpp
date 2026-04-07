#include <mutex>
#include "FrameQueue.h"

FrameQueue::FrameQueue() : m_stopping(false)
{}

FrameQueue::~FrameQueue()
{
    stop();
}

/*
void FrameQueue::push(Frame&& frame)
{
    {
        // Lock to protect queue access
        std::lock_guard<std::mutex> lock(m_mutex);

        // Transfer ownership of the frame into the queue
        m_queue.push(std::move(frame));
    }

    // Notify consumer thread that a new frame is available
    m_cv.notify_one();
}
*/

void FrameQueue::push(Frame&& frame)
{
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        // Real-time behavior: only keep the latest frame
        // Prevents latency buildup if processing falls behind
        while (!m_queue.empty())
            m_queue.pop();

        // Move new frame into the queue
        m_queue.push(std::move(frame));
    }

    // Wake up any waiting consumer
    m_cv.notify_one();
}

bool FrameQueue::pop(Frame& out)
{
    std::unique_lock<std::mutex> lock(m_mutex);

    // Wait until:
    // 1) A frame is available, or
    // 2) Stop has been requested
    //
    // wait() releases the mutex while sleeping and reacquires it on wake
    m_cv.wait(lock, [&]{ return !m_queue.empty() || m_stopping; });

    // If stopping and no work left, exit cleanly
    if (m_stopping && m_queue.empty())
    {
        return false;
    }

    // Move frame out of queue (no copies)
    out = std::move(m_queue.front());
    m_queue.pop();

    return true;
}

void FrameQueue::stop()
{
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        // Signal all threads to stop waiting
        m_stopping = true;
    }

    // Wake up any blocked threads so they can exit
    m_cv.notify_all();
}