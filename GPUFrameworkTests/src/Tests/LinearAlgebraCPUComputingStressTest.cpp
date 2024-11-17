#include "Tests/LinearAlgebraCPUComputingStressTest.h"
#include "Randomizers.h"
#include "CPUParallelism/CPUParallelismNCP.h"
#include "UtilityFunctions.h"
#include <algorithm>
#include <limits>

using namespace std;
using namespace Utils;
using namespace Utils::CPUParallelism;
using namespace Utils::AccurateTimers;
using AccurateRandomizer = Utils::Randomizers::RandomRNGWELL512;

namespace // anonymous namespace used instead of deprecated 'static' keyword used for cpp variable locality
{
  constexpr size_t NUMBER_OF_RANDOMIZER_ITERATIONS = 10;

  inline int32_t randomizer(AccurateRandomizer& random)
  {
    int32_t randomValue = 0;
    for (size_t i = 0; i < NUMBER_OF_RANDOMIZER_ITERATIONS; ++i)
    {
      randomValue += int32_t(random() * numeric_limits<int32_t>::max() / NUMBER_OF_RANDOMIZER_ITERATIONS);
    }
    return randomValue;
  }

  inline void linearAlgebraFunction(size_t index, const int32_t* __restrict a, const int32_t* __restrict b, int32_t* __restrict c, AccurateRandomizer& random)
  {
    c[index] += a[index] + b[index] + randomizer(random);
  }

  inline void linearAlgebraKernelCPU(size_t index, const int32_t* __restrict a, const int32_t* __restrict b, int32_t* __restrict c, size_t iterations)
  {
    AccurateRandomizer random;
    for (size_t i = 0; i < iterations; ++i)
    {
      linearAlgebraFunction(index, a, b, c, random);
    }
  }
}

LinearAlgebraCPUComputingStressTest::LinearAlgebraCPUComputingStressTest(size_t arraySize, size_t numberOfCPUKernelIterations) noexcept
  : arraySize_(max<size_t>(1, arraySize * arraySize))
  , numberOfCPUKernelIterations_(max<size_t>(1, numberOfCPUKernelIterations))
  , arrayA_(make_unique<int32_t[]>(arraySize_))
  , arrayB_(make_unique<int32_t[]>(arraySize_))
  , arrayC_(make_unique<int32_t[]>(arraySize_))
{
}

void LinearAlgebraCPUComputingStressTest::resetTests()
{
    cpuTimer_.startTimer();
    parallelFor(0, arraySize_, [&](size_t i)
    {
      arrayA_[i] = -int32_t(i);
      arrayB_[i] =  int32_t(i * i);
    });
    DebugConsole_consoleOutLine("LinearAlgebraCPUComputingStressTest reset time taken: ", cpuTimer_.getElapsedTimeInMilliSecs(), " ms.\n");
    totalTimeTakenInMs_ += cpuTimer_.getElapsedTimeInMilliSecs();
}

bool LinearAlgebraCPUComputingStressTest::conductTests()
{
    cpuTimer_.startTimer();
    // run the CPU kernel
    parallelFor(0, arraySize_, [&](size_t index)
    {
      linearAlgebraKernelCPU(index, arrayA_.get(), arrayB_.get(), arrayC_.get(), numberOfCPUKernelIterations_);
    });
    DebugConsole_consoleOutLine("LinearAlgebraCPUComputingStressTest compute time taken: ", cpuTimer_.getElapsedTimeInMilliSecs(), " ms.\n");
    totalTimeTakenInMs_ += cpuTimer_.getElapsedTimeInMilliSecs();

    return true;
}

void LinearAlgebraCPUComputingStressTest::reportTestResults()
{
    cpuTimer_.startTimer();
    arrayA_.reset(nullptr); // release & delete host memory so as to avoid temporary memory consumption
    arrayB_.reset(nullptr); // release & delete host memory so as to avoid temporary memory consumption
    arrayC_.reset(nullptr); // release & delete host memory so as to avoid temporary memory consumption
    DebugConsole_consoleOutLine("LinearAlgebraCPUComputingStressTest de-allocation time taken: ", cpuTimer_.getElapsedTimeInMilliSecs(), " ms.\n");
    totalTimeTakenInMs_ += cpuTimer_.getElapsedTimeInMilliSecs();

    DebugConsole_consoleOutLine("LinearAlgebraCPUComputingStressTest total time taken: ", totalTimeTakenInMs_, " ms.\n");
}