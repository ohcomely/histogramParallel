#include "CPUParallelism/ThreadPool.h"
#include "UtilityFunctions.h"
#include <algorithm>
#include <stdexcept>
#include <system_error>
#include <cerrno>

using namespace std;
using namespace Utils::CPUParallelism;

ThreadPool::ThreadPool(size_t numberOfThreads, uint64_t affinityMask, size_t priority)
  : numberOfThreads_{numberOfThreads}
  , done_{false}
  , threads_{make_unique<thread[]>(max<size_t>(1, numberOfThreads))} // avoid a bad_alloc & throw an invalid_argument be used below
  , joiner_{threads_.get(), numberOfThreads}
  , barrier_{numberOfThreads + 1} // to be used for the parallelFor() construct (numberOfThreads + 1 main driving thread)
{
  if (numberOfThreads_ == 0)
  {
    throw invalid_argument("ThreadPool constructor: number of threads cannot be zero.");
  }

  try
  {
    for (size_t threadIdx = 0; threadIdx < numberOfThreads_; ++threadIdx)
    {
      threads_[threadIdx] = thread(&ThreadPool::workerThread, this);

      if (affinityMask != AFFINITY_MASK_NONE)
      {
        if (threadIdx < numberOfHardwareThreads())
        {
          Utils::CPUParallelism::setAffinity(threads_[threadIdx], threadIdx, affinityMask);
        }
        else
        {
          DebugConsole_consoleOutLine("ThreadPool constructor warning: enabling thread affinity skipped for a non-hardware threadIdx: ", threadIdx);
        }
      }

      if (priority != PRIORITY_NONE && priority > PRIORITY_NONE && priority <= PRIORITY_MAX) // range (0, 99], show PRIORITY_NONE check explicitly
      {
        Utils::CPUParallelism::setPriority(threads_[threadIdx], threadIdx, priority);
      }
    }
  }
  catch (system_error& e)
  {
    done_ = true;
    unique_lock<mutex> lock(mutex_);
    conditionVariable_.notify_all();
    throw system_error(e.code(), "ThreadPool constructor: exception '" + string(e.what()) + "' caught, thread pool now terminating.");
  }
  catch (...)
  {
    done_ = true;
    unique_lock<mutex> lock(mutex_);
    conditionVariable_.notify_all();
    throw system_error(ENOTSUP, system_category(), "ThreadPool constructor: unknown exception caught, thread pool now terminating.");
  }
}

ThreadPool::~ThreadPool() noexcept
{
  done_ = true;
  unique_lock<mutex> lock(mutex_);
  conditionVariable_.notify_all();
}

void ThreadPool::checkThreadIdx(size_t threadIdx)
{
  // make sure the threadIdx is within the bounds of [0, numberOfThreads - 1]
  if (threadIdx >= numberOfThreads_)
  {
    throw invalid_argument("ThreadBarrier::checkThreadIdx(): threadIdx >= numberOfThreads.");
  }
}

void ThreadPool::setAffinity(size_t threadIdx, uint64_t affinityMask)
{
  checkThreadIdx(threadIdx);
  Utils::CPUParallelism::setAffinity(threads_[threadIdx], threadIdx, affinityMask);
}

bool ThreadPool::getAffinity(size_t threadIdx)
{
  checkThreadIdx(threadIdx);
  return Utils::CPUParallelism::getAffinity(threads_[threadIdx], threadIdx);
}

void ThreadPool::setPriority(size_t threadIdx, size_t priority)
{
  checkThreadIdx(threadIdx);
  return Utils::CPUParallelism::setPriority(threads_[threadIdx], threadIdx, priority);
}

size_t ThreadPool::getPriority(size_t threadIdx)
{
  checkThreadIdx(threadIdx);
  return Utils::CPUParallelism::getPriority(threads_[threadIdx], threadIdx);
}

void ThreadPool::runPendingTask()
{
  function<void()> task;
  if (workQueue_.tryPop(task))
  {
    task();
  }
  else
  {
    unique_lock<mutex> lock(mutex_);
    conditionVariable_.wait(lock, [this] { return !workQueue_.empty() || done_.load(); });
  }
}

void ThreadPool::workerThread()
{
  while (!done_.load())
  {
    runPendingTask();
  }
}