#include "CPUParallelism/ThreadBarrier.h"
#include "UtilityFunctions.h"
#include <stdexcept>

using namespace std;
using namespace Utils::CPUParallelism;

ThreadBarrier::ThreadBarrier(size_t threadCount)
  : threadCount_{threadCount}
#ifdef _WIN32
  , threshold_{threadCount}
  , generation_{0}
#endif
{
  if (threadCount_ == 0)
  {
    throw invalid_argument("ThreadBarrier constructor: thread count cannot be zero.");
  }

#ifndef _WIN32
  const int errorCode = pthread_barrier_init(&barrier_, nullptr, threadCount_);
  if (errorCode != 0)
  {
    DebugConsole_consoleOutLine("Error calling pthread_barrier_init() with errorCode: ", errorCode);
  }
#endif
}

ThreadBarrier::~ThreadBarrier() noexcept
{
#ifndef _WIN32
  const  int errorCode = pthread_barrier_destroy(&barrier_);
  if (errorCode != 0)
  {
    DebugConsole_consoleOutLine("Error calling pthread_barrier_destroy() with errorCode: ", errorCode);
  }
#endif
}

void ThreadBarrier::wait()
{
#ifndef _WIN32
  const int errorCode = pthread_barrier_wait(&barrier_);
  if (errorCode != 0 && errorCode != PTHREAD_BARRIER_SERIAL_THREAD) // value returned by 'pthread_barrier_wait' for one of the threads after the required number of threads have called this function
  {
    DebugConsole_consoleOutLine("Error calling pthread_barrier_wait() with errorCode: ", errorCode);
  }
#else
  unique_lock<mutex> lock(mutex_);
  size_t thisGeneration = generation_;

  if (--threadCount_ == 0)
  {
    ++generation_;
    threadCount_ = threshold_;
    conditionVariable_.notify_all();
    return;
  }

  conditionVariable_.wait(lock, [thisGeneration, this] { return thisGeneration != generation_; });
#endif
}