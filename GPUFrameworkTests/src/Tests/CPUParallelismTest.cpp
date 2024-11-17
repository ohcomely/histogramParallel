#include "Tests/CPUParallelismTest.h"
#include "CPUParallelism/ThreadPool.h"
#include "AccurateTimers.h"
#include "Randomizers.h"
#include "UtilityFunctions.h"
#include <cstdlib>
#include <algorithm>
#include <memory>

using namespace std;
using namespace Utils::CPUParallelism;
using namespace Utils::UtilityFunctions;
using AccurateRandomizer = Utils::Randomizers::RandomRNGWELL512;
using AccurateTimer      = Utils::AccurateTimers::AccurateCPUTimer;

namespace // anonymous namespace used instead of deprecated 'static' keyword used for cpp variable locality
{
  const bool USE_PARALLEL_FOR_TESTING = false;

  /** @brief Union struct for a packed i10i10i10i2 normal.
  *
  * @author Thanos Theo, 2009-2018
  * @version 14.0.0.0
  */
  union i10i10i10i2
  {
    struct
    {
      int32_t x : 10;
      int32_t y : 10;
      int32_t z : 10;
      int32_t w : 2;
    } data;
    uint32_t pack;
  };

  inline uint32_t packGradientToUInt(int32_t x, int32_t y, int32_t z)
  {
    const i10i10i10i2 value{ { int32_t(round(MathFunctions::reintervalClamped(x, -1000, 1000, 0, 1023))),
                               int32_t(round(MathFunctions::reintervalClamped(y, -1000, 1000, 0, 1023))),
                               int32_t(round(MathFunctions::reintervalClamped(z, -1000, 1000, 0, 1023))),
                               0
                           } };
    return value.pack;
  }

  inline void calculateGradientDataKernel(const uint32_t* __restrict ctDataArray, size_t w, size_t h, uint32_t* __restrict gradientDataArray, size_t i)
  {
    const size_t wh = w * h;
    const size_t z = i;
    {
      for (size_t y = 1; y < h - 1; ++y)
      {
        const size_t yw = y * w;
        uint32_t* gradientData = gradientDataArray + (z * wh + y * w + 1);
        for (size_t x = 1; x < w - 1; ++x)
        {
          const int32_t vx = (ctDataArray[ z      * wh +  yw         + (x + 1)] - ctDataArray[ z      * wh +  yw         + (x - 1)]) >> 1;
          const int32_t vy = (ctDataArray[ z      * wh + (y + 1) * w +  x     ] - ctDataArray[ z      * wh + (y - 1) * w +  x     ]) >> 1;
          const int32_t vz = (ctDataArray[(z + 1) * wh +  yw         +  x     ] - ctDataArray[(z - 1) * wh +  yw         +  x     ]) >> 1;
          *gradientData++ = packGradientToUInt(vx, vy, vz);
        }
      }
    }
  }
}

CPUParallelismTest::CPUParallelismTest(size_t dimensions, size_t numberOfThreads, bool useRandomness) noexcept
  : dimensionX_(dimensions)
  , dimensionY_(dimensions)
  , dimensionZ_(dimensions)
  , numberOfThreads_(numberOfThreads)
  , useRandomness_(useRandomness)
{
}

CPUParallelismTest::CPUParallelismTest(tuple<size_t, size_t, size_t> dimensionsXYZ, size_t numberOfThreads, bool useRandomness) noexcept
  : dimensionX_(get<0>(dimensionsXYZ))
  , dimensionY_(get<1>(dimensionsXYZ))
  , dimensionZ_(get<2>(dimensionsXYZ))
  , numberOfThreads_(numberOfThreads)
  , useRandomness_(useRandomness)
{
}

void CPUParallelismTest::resetTests()
{
  testIterations_             = 0;
  meanTimeCounterRandomizer_  = 0.0;
  meanTimeCounterSingleCore_  = 0.0;
  meanTimeCounterNCP_         = 0.0;
}

bool CPUParallelismTest::conductTests()
{
  /*----------Initialize all relevant variables----------*/
  AccurateRandomizer random;
  AccurateTimer timer;
  ThreadPool threadPool(numberOfThreads_);
  size_t indexStart      = 1;
  size_t indexEnd        = dimensionZ_ - 1;
  size_t arraySize       = max<size_t>(1, dimensionX_ * dimensionY_ * dimensionZ_);
  if (useRandomness_ && (4 * numberOfThreads_) < (indexEnd - indexStart))
  {
    indexStart +=                          (random.getRandomInteger() % (2 * numberOfThreads_));
    indexEnd   -= (2 * numberOfThreads_) - (random.getRandomInteger() % (2 * numberOfThreads_));
  }



  DebugConsole_consoleOutLine("Test iteration ",             (testIterations_ + 1), " results:");
  DebugConsole_consoleOutLine("Number of Hardware Threads: ", numberOfThreads_);
  DebugConsole_consoleOutLine("Current Dimensions: ",         dimensionX_, "x", dimensionY_, "x", dimensionZ_);
  DebugConsole_consoleOutLine("UseRandomness: ",              StringAuxiliaryFunctions::toString<bool>(useRandomness_));
  DebugConsole_consoleOutLine("IndexStart: ",                 indexStart);
  DebugConsole_consoleOutLine("IndexEnd: ",                   indexEnd);
  DebugConsole_consoleOutLine("ArraySize: ",                  arraySize);



  /*----------Degenerate parallelFor() case where indexStart > indexEnd: should never reach the exit() call----------*/
  parallelFor(indexEnd, indexStart, [&](size_t)
  {
    exit(EXIT_FAILURE);
  }, threadPool);



  /*----------Populate with random integer density values----------*/
  auto ctData = unique_ptr<uint32_t[]>(new uint32_t[arraySize]); // avoid enforcing the default constructor through the make_unique() call for the primitive uint32_t (make_unique() is using the C++03 array initialization syntax)
  timer.startTimer();
  if (USE_PARALLEL_FOR_TESTING)
  {
    parallelFor(0, arraySize, [&](size_t i)
    {
      ctData[i] = useRandomness_ ? uint32_t(random.getRandomInteger()) : uint32_t(i);
    }, threadPool);
  }
  else // !USE_PARALLEL_FOR_TESTING
  {
    for (size_t i = 0; i < arraySize; ++i)
    {
      ctData[i] = useRandomness_ ? uint32_t(random.getRandomInteger()) : uint32_t(i);
    }
  }
  timer.stopTimer();
  DebugConsole_consoleOutLine("Populate with random integer density values run time: ", timer.getElapsedTimeInMilliSecs(), " msecs.");
  meanTimeCounterRandomizer_ += timer.getElapsedTimeInMilliSecs();



  /*----------Run in Single Core mode----------*/
  auto gradientDataSingleCore = make_unique<uint32_t[]>(arraySize); // uint32_t values initialize to zero (needed since our loops are for [1, dimXYZ -1], we miss 0 & dimXYZ borderline cases)

  timer.startTimer();
  // Done explicitly just to show the start-end of positions for the single-core case
  for (size_t i = indexStart; i < indexEnd; ++i)
  {
    calculateGradientDataKernel(ctData.get(), dimensionX_, dimensionY_, gradientDataSingleCore.get(), i);
  }
  timer.stopTimer();
  DebugConsole_consoleOutLine("Total calculateGradientData Single Core run time: ", timer.getElapsedTimeInMilliSecs(), " msecs.");
  meanTimeCounterSingleCore_ += timer.getElapsedTimeInMilliSecs();



  /*----------Run in N-CP mode----------*/
  auto gradientDataNCP = make_unique<uint32_t[]>(arraySize); // uint32_t values initialize to zero (needed since our loops are for [1, dimXYZ -1], we miss 0 & dimXYZ borderline cases)
  timer.startTimer();
  parallelFor(indexStart, indexEnd, [&](size_t i)
  {
    calculateGradientDataKernel(ctData.get(), dimensionX_, dimensionY_, gradientDataNCP.get(), i);
  }, threadPool);
  timer.stopTimer();
  DebugConsole_consoleOutLine("Total calculateGradientData N-CP run time: ", timer.getElapsedTimeInMilliSecs(), " msecs.");
  meanTimeCounterNCP_ += timer.getElapsedTimeInMilliSecs();



  /*----------Check if results are equal between Single Core & N-CP modes----------*/
  timer.startTimer();
  bool resultTestsFlag = true;
  if (USE_PARALLEL_FOR_TESTING)
  {
    parallelFor(0, arraySize, [&](size_t i)
    {
      if (gradientDataSingleCore[i] != gradientDataNCP[i])
      {
        DebugConsole_consoleOutLine("N-CP Unit Tests in CPUParallelismTest::conductTests() failed at index: ", i, " with values: ", gradientDataSingleCore[i], " != ", gradientDataNCP[i]);
        resultTestsFlag = false;
      }
    }, threadPool);
  }
  else // !USE_PARALLEL_FOR_TESTING
  {
    for (size_t i = 0; i < arraySize; ++i)
    {
      if (gradientDataSingleCore[i] != gradientDataNCP[i])
      {
        DebugConsole_consoleOutLine("N-CP Unit Tests in CPUParallelismTest::conductTests() failed at index: ", i, " with values: ", gradientDataSingleCore[i], " != ", gradientDataNCP[i]);
        resultTestsFlag = false;
      }
    }
  }
  gradientDataSingleCore.reset(nullptr); // release & delete memory for single core so as to avoid temporary memory consumption
  gradientDataNCP.reset(nullptr);        // release & delete memory for N-CP so as to avoid temporary memory consumption
  timer.stopTimer();
  DebugConsole_consoleOutLine("Total comparison between Single Core & N-CP run time: ", timer.getElapsedTimeInMilliSecs(), " msecs.");



  DebugConsole_consoleOutLine("Test iteration ", (testIterations_ + 1), " in CPUParallelismTest::conductTests() passed successfully!");
  ++testIterations_; // increase test iterations counter

  return resultTestsFlag;
}

void CPUParallelismTest::reportTestResults()
{
  DebugConsole_consoleOutLine("Mean test execution times for ",        testIterations_,                                       " test iterations:");
  DebugConsole_consoleOutLine("MeanTimeCounterRandomizer: ",          (meanTimeCounterRandomizer_ / double(testIterations_)), " msecs.");
  DebugConsole_consoleOutLine("MeanTimeCounterSingleCore: ",          (meanTimeCounterSingleCore_ / double(testIterations_)), " msecs.");
  DebugConsole_consoleOutLine("MeanTimeCounterNCP: ",                 (meanTimeCounterNCP_ / double(testIterations_)),        " msecs.");
  DebugConsole_consoleOutLine("N-CP vs Single Core execution time: ", (meanTimeCounterSingleCore_ / meanTimeCounterNCP_),     " times faster.");
}