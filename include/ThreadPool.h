#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <iostream>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <chrono>

class ThreadPool
{

private:
    std::vector<std::thread>                workers;
    std::queue<std::function<void()>>       jobQueue;
    std::mutex                              m;
    std::condition_variable                 cv;
    std::atomic<bool>                       stop;

    void workerLoop();

public:

    explicit ThreadPool(size_t numThreads);
    ~ThreadPool();

    void enqueue(std::function<void()> job);

        
};


#endif