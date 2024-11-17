#include "CPUParallelism/ThreadJoiner.h"

using namespace std;
using namespace Utils::CPUParallelism;

ThreadJoiner::ThreadJoiner(thread* __restrict threads, size_t numberOfThreads) noexcept
  : threads_{threads}
  , numberOfThreads_{numberOfThreads}
{
}

ThreadJoiner::~ThreadJoiner() noexcept
{
  for (size_t i = 0; i < numberOfThreads_; ++i)
  {
    if (threads_[i].joinable())
    {
      threads_[i].join();
    }
  }
}