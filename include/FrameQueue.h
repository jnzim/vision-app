#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include "AppTypes.h"

class FrameQueue
{
public:
    FrameQueue();
    ~FrameQueue();
    void push(Frame&& );
    bool pop(Frame&);
    void stop();

private:
    std::queue<Frame>       m_queue;
    std::mutex              m_mutex;
    std::condition_variable m_cv;
    bool                    m_stopping;

    
};

