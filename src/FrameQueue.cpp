 #include <mutex>
 #include "FrameQueue.h"

 FrameQueue::FrameQueue() : m_stopping(false)
 {}
FrameQueue::~FrameQueue()
{
    stop();
}

void FrameQueue::push(Frame&& frame)
{
    {
        // lock the queue to prevent race condion / coruption
        std::lock_guard<std::mutex>lock(m_mutex);
        //hand over the image pointer + timestamp.
        m_queue.push(std::move(frame));
    }
    // let the processor thread know there's a new frame -
    m_cv.notify_one();
}

bool FrameQueue::pop(Frame& out)
{
    std::unique_lock<std::mutex>lock(m_mutex);
    // wait here unill there's a new frame in the queue.
    m_cv.wait(lock, [&]{ return !m_queue.empty()|| m_stopping; });
        
    if(m_stopping && m_queue.empty())
    {
        return false;
    }
    out = std::move(m_queue.front());
    m_queue.pop();
    return true;   // <-- MUST return a bool here
}

void FrameQueue::stop()
{
    {
        std::lock_guard<std::mutex>lock(m_mutex);
        m_stopping = true;
    }
    m_cv.notify_one();
}