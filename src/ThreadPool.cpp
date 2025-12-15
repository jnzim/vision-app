#include "ThreadPool.h"

ThreadPool::ThreadPool(std::size_t _numThreads) : stop(false)
{
    for (size_t i = 0; i < _numThreads; ++i)
    {
        workers.emplace_back(&ThreadPool::workerLoop, this);
    }
}

ThreadPool::~ThreadPool()
{
    {
        std::unique_lock<std::mutex> lock(m);
        stop = true;
    }
    cv.notify_all();

    for(auto &t : workers)
    {
        if(t.joinable())
        {
            t.join();
        }
    }
}
    
void ThreadPool::workerLoop()
{
    while (true)
    {
        // create a job variable
        std::function<void()> job; 

        {
            // lock it up
            std::unique_lock<std::mutex> lock(m);

            // Wait until there is work, or we are stopping
            cv.wait(lock, [this] 
            {
                return !jobQueue.empty() || stop;
            });

            if (stop && jobQueue.empty()) {return;}

            job = std::move(jobQueue.front());
            jobQueue.pop();
        }

        // Run the job outside the lock
        job();
    }
}


// Add a job to the queue
void ThreadPool::enqueue(std::function<void()> job)
{
    {
        std::lock_guard<std::mutex> lock(m);
        jobQueue.push(std::move(job));
    }
    cv.notify_one(); // wake one worker
}
