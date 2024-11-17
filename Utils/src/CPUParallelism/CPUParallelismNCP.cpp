#include "CPUParallelism/CPUParallelismNCP.h"
#include "CPUParallelism/ThreadBarrier.h"
#include "CPUParallelism/ThreadPool.h"
#include "AccurateTimers.h"
#include "UtilityFunctions.h"
#include <algorithm>

using namespace std;
using namespace Utils::CPUParallelism;
using AccurateTimer = Utils::AccurateTimers::AccurateCPUTimer;

namespace // anonymous namespace used instead of deprecated 'static' keyword used for cpp variable locality
{
  // N-CP related variables
  const size_t NUMBER_OF_HARDWARE_THREADS = max<size_t>(1, thread::hardware_concurrency()); // make sure the thread result is at least 1

  // enforce inlining via a template function implementation to pass the kernelFunction directly
  template <typename FunctionType>
  inline void parallelForKernel(ThreadBarrier& threadBarrier, size_t numberOfThreads, size_t threadIdx, size_t indexStart, size_t indexEnd, const FunctionType& kernelFunction)
  {
    threadBarrier.wait(); // wait for all NCP threads to be ready

    try
    {
      const size_t totalIterationsSize = (indexEnd - indexStart);
      if (numberOfThreads > totalIterationsSize) // numberOfThreads more than totalIterationsSize, use task parallelism
      {
        // conduct task parallelism
        const size_t index = (threadIdx + indexStart);
        // execute kernelFunction
        if (index < indexEnd)
        {
          kernelFunction(index, threadIdx);
        }
      }
      else // totalIterationsSize more than (or equal) numberOfThreads, use data parallelism
      {
        // conduct data parallelism
        const size_t totalIterationsPerProcess = (totalIterationsSize / numberOfThreads);
        const size_t startPosition             = (threadIdx * totalIterationsPerProcess) + ((threadIdx == 0) ? indexStart : 0);
        const size_t extraIterations           = (threadIdx == (numberOfThreads - 1)) ? (totalIterationsSize % numberOfThreads) + (indexEnd - totalIterationsSize) : 0;
        const size_t endPosition               = ((threadIdx + 1) * totalIterationsPerProcess) + extraIterations;
        // execute kernelFunction
        for (size_t index = startPosition; index < endPosition; ++index)
        {
          kernelFunction(index, threadIdx);
        }
      }
    }
    catch (...)
    {
      DebugConsole_consoleOutLine("Problem with the N-Core thread with threadIdx ", threadIdx, " in processNCPKernel()!");
    }

    threadBarrier.wait(); // wait for all NCP threads to finish
  }

  template <typename FunctionType>
  inline void executeParallelFor(size_t indexStart, size_t indexEnd, const FunctionType& kernelFunction, ThreadPool& threadPool)
  {
    if (indexEnd > 0 && indexEnd > indexStart)
    {
      size_t numberOfThreads       = threadPool.getNumberOfThreads();
      ThreadBarrier& threadBarrier = threadPool.getBarrier();
      for (size_t threadIdx = 0; threadIdx < numberOfThreads; ++threadIdx)
      {
        // use a lambda instead of bind() as Scott Meyers suggests in his C++11/14 book (can be faster with inlining)
        threadPool.submit([&threadBarrier, numberOfThreads, threadIdx, indexStart, indexEnd, &kernelFunction]
        {
          parallelForKernel(threadBarrier, numberOfThreads, threadIdx, indexStart, indexEnd, kernelFunction);
        }); // submit function call in thread pool
      }

#ifdef GPU_FRAMEWORK_PROFILE_NCP_PARALLEL_FOR
      AccurateTimer timer;
      timer.startTimer();
      threadBarrier.wait(); // wait for all NCP threads to be ready
      threadBarrier.wait(); // wait for all NCP threads to finish
      timer.stopTimer();
      DebugConsole_consoleOutLine("Total parallelFor N-CP run time: ", timer.getElapsedTimeInMilliSecs(), " msecs.");
#else
      threadBarrier.wait(); // wait for all NCP threads to be ready
      threadBarrier.wait(); // wait for all NCP threads to finish
#endif // GPU_FRAMEWORK_PROFILE_NCP_PARALLEL_FOR
    }
    else
    {
      DebugConsole_consoleOutLine("\nWarning: problem detected with indexStart & indexEnd.\nindexStart is '", indexStart, "' and indexEnd is '", indexEnd, "',\nwhich is not allowed as indexStart > indexEnd.\nNow skipping the parallelFor() call.\n");
    }
  }
}

size_t Utils::CPUParallelism::numberOfHardwareThreads()
{
  return NUMBER_OF_HARDWARE_THREADS;
}

void Utils::CPUParallelism::threadSleep(size_t millisecs)
{
  this_thread::sleep_for(chrono::milliseconds(millisecs));
}

void Utils::CPUParallelism::parallelFor(size_t indexEnd, const FunctionView<void(size_t)>& kernelFunction, size_t numberOfThreads, uint64_t affinityMask, size_t priority)
{
  ThreadPool threadPool(numberOfThreads, affinityMask, priority); // init all N-CP threads in a local thread pool using the RAII idiom
  executeParallelFor(0, indexEnd, [&kernelFunction](size_t i, size_t){ kernelFunction(i); }, threadPool);
}

void Utils::CPUParallelism::parallelFor(size_t indexStart, size_t indexEnd, const FunctionView<void(size_t)>& kernelFunction, size_t numberOfThreads, uint64_t affinityMask, size_t priority)
{
  ThreadPool threadPool(numberOfThreads, affinityMask, priority); // init all N-CP threads in a local thread pool using the RAII idiom
  executeParallelFor(indexStart, indexEnd, [&kernelFunction](size_t i, size_t){ kernelFunction(i); }, threadPool);
}

void Utils::CPUParallelism::parallelFor(size_t indexEnd, const FunctionView<void(size_t)>& kernelFunction, ThreadPool& threadPool)
{
  executeParallelFor(0, indexEnd, [&kernelFunction](size_t i, size_t) { kernelFunction(i); }, threadPool);
}

void Utils::CPUParallelism::parallelFor(size_t indexStart, size_t indexEnd, const FunctionView<void(size_t)>& kernelFunction, ThreadPool& threadPool)
{
  executeParallelFor(indexStart, indexEnd, [&kernelFunction](size_t i, size_t) { kernelFunction(i); }, threadPool);
}

void Utils::CPUParallelism::parallelForThreadLocal(size_t indexEnd, const FunctionView<void(size_t, size_t)>& kernelFunction, size_t numberOfThreads, uint64_t affinityMask, size_t priority)
{
  ThreadPool threadPool(numberOfThreads, affinityMask, priority); // init all N-CP threads in a local thread pool using the RAII idiom
  executeParallelFor(0, indexEnd, kernelFunction, threadPool);
}

void Utils::CPUParallelism::parallelForThreadLocal(size_t indexStart, size_t indexEnd, const FunctionView<void(size_t, size_t)>& kernelFunction, size_t numberOfThreads, uint64_t affinityMask, size_t priority)
{
  ThreadPool threadPool(numberOfThreads, affinityMask, priority); // init all N-CP threads in a local thread pool using the RAII idiom
  executeParallelFor(indexStart, indexEnd, kernelFunction, threadPool);
}

void Utils::CPUParallelism::parallelForThreadLocal(size_t indexEnd, const FunctionView<void(size_t, size_t)>& kernelFunction, ThreadPool& threadPool)
{
  executeParallelFor(0, indexEnd, kernelFunction, threadPool);
}

void Utils::CPUParallelism::parallelForThreadLocal(size_t indexStart, size_t indexEnd, const FunctionView<void(size_t, size_t)>& kernelFunction, ThreadPool& threadPool)
{
  executeParallelFor(indexStart, indexEnd, kernelFunction, threadPool);
}