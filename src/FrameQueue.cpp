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
    
    // If the queue is empty, pop() waits on the condition_variable.
    // During wait(), the mutex is released and this thread sleeps.
    //
    // When push() adds a frame, it notifies the condition_variable.
    // The waiting thread wakes, re-acquires the mutex, then:
    //   1) moves the front frame into an output variable,
    //   2) pops it off the queue,
    //   3) returns to the caller.
    //
    // pop() should return by value or fill an output parameter — not a reference
    // to an element in the queue, because that element is removed.
    m_cv.wait(lock, [&]{ return !m_queue.empty() || m_stopping; });
        
    if(m_stopping && m_queue.empty())
    {
        return false;
    }
    
    out = std::move(m_queue.front());
    m_queue.pop();
    return true;   
}

void FrameQueue::stop()
{
    {
        std::lock_guard<std::mutex>lock(m_mutex);
        m_stopping = true;
    }
    m_cv.notify_one();
}