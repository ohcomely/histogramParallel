#include "HostUnitTests.h"
#include "AccurateTimers.h"
#include "CPUParallelism/CPUParallelismNCP.h"
#include "CPUParallelism/CPUParallelismUtilityFunctions.h"
#include "CPUParallelism/ThreadPool.h"
#include "Randomizers.h"
#include "SIMDVectorizations.h"
#include "Tests/CPUParallelismTest.h"
#include "Tests/ColorHistogramTest.h"
#include "UtilityFunctions.h"
#include "lodepng.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace std;
using namespace Tests;
using namespace Utils;
using namespace Utils::AccurateTimers;
using namespace Utils::CPUParallelism;
using namespace Utils::Randomizers;
using namespace Utils::SIMDVectorizations;
using namespace Utils::UnitTests;
using namespace Utils::UtilityFunctions;

void HostGoogleTest01__UTILS_Classes::executeTest() {
  AccurateCPUTimer timer;
  timer.startTimer();
  EXPECT_GT(AccurateCPUTimer::getMillisecondsTimeSinceEpoch(), 0u);
  timer.stopTimer();
  EXPECT_GE(timer.getElapsedTimeInNanoSecs(), 0.0);
  EXPECT_GE(timer.getMeanTimeInNanoSecs(), 0.0);
}

TEST(HostGoogleTest01__UTILS_Classes, AccurateCPUTimer) {
  HostGoogleTest01__UTILS_Classes::executeTest();
}

void HostGoogleTest02__UTILS_Classes::executeTest() {
  RandomRNGWELL512 random;
  EXPECT_LE(random(), 1.0);
}

TEST(HostGoogleTest02__UTILS_Classes, RandomRNGWELL512) {
  HostGoogleTest02__UTILS_Classes::executeTest();
}

void HostGoogleTest03__UTILS_Classes::executeTest() {
#ifdef _WIN32
  reportCPUCapabilities();
#endif // _WIN32

#ifdef __aarch64__
  EXPECT_TRUE(isSupportedNEON());
#else
  EXPECT_TRUE(isSupportedSSE3());
  EXPECT_TRUE(isSupportedSSE41());
  EXPECT_TRUE(isSupportedSSE42());
  EXPECT_TRUE(isSupportedAVX());
  EXPECT_TRUE(isSupportedAVX2());
#endif // __aarch64__
}

TEST(HostGoogleTest03__UTILS_Classes, SIMDVectorizations) {
  // HostGoogleTest03__UTILS_Classes::executeTest();
}

void HostGoogleTest04__UTILS_Classes::executeTest() {
  // EXPECT_EQ(BitManipulationFunctions::getNextPowerOfTwo(17), 32u);
}

TEST(HostGoogleTest04__UTILS_Classes, BitManipulationFunctions) {
  // HostGoogleTest04__UTILS_Classes::executeTest();
}

void HostGoogleTest05__UTILS_CPUParallelism_Classes::executeTest() {
  const size_t parallelForSize = 100;
  parallelFor(0, parallelForSize, [&](size_t i) {
    RandomRNGWELL512 random;
    threadSleep(size_t(250 * random()));
    EXPECT_LT(i, parallelForSize);
  });
}

TEST(HostGoogleTest05__UTILS_CPUParallelism_Classes, parallelFor) {
  // HostGoogleTest05__UTILS_CPUParallelism_Classes::executeTest();
}

void HostGoogleTest06__UTILS_CPUParallelism_Classes::executeTest() {
  const size_t minDimensionSize = 64;
  const size_t maxDimensionSize = 512;
  const size_t iterationSize = 3;
  for (size_t i = minDimensionSize; i <= maxDimensionSize;
       i <<= 1) // implies 'i *= 2'
  {
    CPUParallelismTest cpuParallelismTest(i);
    cpuParallelismTest.resetTests();
    for (size_t iteration = 0; iteration < iterationSize;
         ++iteration) // run for 3 iterations
    {
      EXPECT_TRUE(cpuParallelismTest.conductTests());
    }
    cpuParallelismTest.reportTestResults();
  }
}

TEST(HostGoogleTest06__UTILS_CPUParallelism_Classes, CPUParallelismTest) {
  // HostGoogleTest06__UTILS_CPUParallelism_Classes::executeTest();
}

void HostGoogleTest07__Lodepng_Classes::executeTest() {
  vector<uint8_t> alpsPng;
  const string currentExecutablePath =
      StdReadWriteFileFunctions::getCurrentPath();
  const string alpsPngFilename =
      string(currentExecutablePath + "/" + "Assets" + "/" + "alps.png");
  uint32_t imageWidth = 0;
  uint32_t imageHeight = 0;
  uint32_t lodepngError =
      lodepng::decode(alpsPng, imageWidth, imageHeight, alpsPngFilename,
                      LodePNGColorType::LCT_RGB);
  if (lodepngError) // check if the png image was loaded successfully (ie: the
                    // file actually exists & loading succeeded)
  {
    DebugConsole_consoleOutLine("Lodepng decoder error ", lodepngError, ": ",
                                lodepng_error_text(lodepngError),
                                " for file: ", alpsPngFilename,
                                ".\nNow aborting decoding the file.");
    EXPECT_TRUE(false);
    return;
  }

  // save the alpsPng to Images png directory
  const string imagesPath = string(currentExecutablePath + "/" + "Images");
  if (!StdReadWriteFileFunctions::pathExists(imagesPath)) {
    StdReadWriteFileFunctions::createDirectory(imagesPath);
  }
  const string alpsTestPngFilename = string(imagesPath + "/" + "alpsTest.png");
  lodepngError = lodepng::encode(alpsTestPngFilename, alpsPng, imageWidth,
                                 imageHeight, LodePNGColorType::LCT_RGB);
  if (lodepngError) // check if the png image was loaded successfully (ie:
                    // saving succeeded)
  {
    DebugConsole_consoleOutLine("Lodepng encoder error ", lodepngError, ": ",
                                lodepng_error_text(lodepngError),
                                " for file: ", alpsTestPngFilename,
                                ".\nNow aborting encoding the file.");
    EXPECT_TRUE(false);
  }
  EXPECT_TRUE(StdReadWriteFileFunctions::removeDirectory(imagesPath) !=
              numeric_limits<uintmax_t>::max());
}

TEST(HostGoogleTest07__Lodepng_Classes, Lodepng) {
  // HostGoogleTest07__Lodepng_Classes::executeTest();
}

void HostGoogleTest08__UTILS_Classes::executeTest() {
  constexpr size_t ARRAY1_SIZE = 10;
  constexpr size_t ARRAY2_SIZE = 1000000;

  const size_t numberOfThreads = numberOfHardwareThreads();
  // initialize thread pool with default parameters
  ThreadPool threadPool(numberOfThreads, AFFINITY_MASK_ALL, PRIORITY_NONE);

  // float32 -> uint32 representation case
  {
    const float originalFloat32Value1 = 2.25f;
    const uint32_t uint32Value1 =
        MathFunctions::asUint32(originalFloat32Value1);
    const float float32Value1 = MathFunctions::asFloat32(uint32Value1);
    const bool float32Result1 =
        MathFunctions::equal(originalFloat32Value1, float32Value1);
    EXPECT_TRUE(float32Result1);
    if (!float32Result1) {
      return;
    }

    const float originalFloat32Value2 = -333.45f;
    const uint32_t uint32Value2 =
        MathFunctions::float32Flip(originalFloat32Value2);
    const float float32Value2 = MathFunctions::float32Unflip(uint32Value2);
    const bool float32Result2 =
        MathFunctions::equal(originalFloat32Value2, float32Value2);
    EXPECT_TRUE(float32Result2);
    if (!float32Result2) {
      return;
    }

    array<float, ARRAY1_SIZE> float32Values1 = {
        {8687325.5535f, 242.24835f, 1.67857f, -0.7749938f, 3.334463f,
         -45675.768f, -141.64555f, 145.89f, 3.14f, -3.14f}};
    array<uint32_t, ARRAY1_SIZE> uint32Values1 = {{0}};
    for (size_t i = 0; i < ARRAY1_SIZE; ++i) {
      uint32Values1[i] = MathFunctions::float32Flip(float32Values1[i]);
    }
    StdAuxiliaryFunctions::insertionSort<ARRAY1_SIZE>(
        uint32Values1.data()); // sort an array using insertion sort with a
                               // constant small size of N
    float minFloat32Value1 = numeric_limits<float>::max();
    float maxFloat32Value1 = numeric_limits<float>::min();
    for (size_t i = 0; i < ARRAY1_SIZE; ++i) {
      float32Values1[i] = MathFunctions::float32Unflip(uint32Values1[i]);
      if (minFloat32Value1 > float32Values1[i]) {
        minFloat32Value1 = float32Values1[i]; // global min
      }
      if (maxFloat32Value1 < float32Values1[i]) {
        maxFloat32Value1 = float32Values1[i]; // global max
      }
    }
    const bool float32Result3 = MathFunctions::equal(
        minFloat32Value1,
        float32Values1[0]); // global min should match the first element of the
                            // sorted array
    EXPECT_TRUE(float32Result3);
    if (!float32Result3) {
      return;
    }
    const bool float32Result4 = MathFunctions::equal(
        maxFloat32Value1,
        float32Values1[ARRAY1_SIZE - 1]); // global max should match the last
                                          // element of the sorted array
    EXPECT_TRUE(float32Result4);
    if (!float32Result4) {
      return;
    }

    auto float32Values2 = unique_ptr<float[]>(new float[ARRAY2_SIZE]);
    auto uint32Values2 = unique_ptr<uint32_t[]>(new uint32_t[ARRAY2_SIZE]);
    const float maxFloat32Value = numeric_limits<float>::max();
    RandomRNGWELL512 random;
    for (size_t i = 0; i < ARRAY2_SIZE; ++i) {
      uint32Values2[i] = MathFunctions::float32Flip(
          maxFloat32Value * float(random()) - maxFloat32Value / 2.0f);
    }
    sort(uint32Values2.get(), uint32Values2.get() + ARRAY2_SIZE);
    float minFloat32Value2 = numeric_limits<float>::max();
    float maxFloat32Value2 = numeric_limits<float>::min();
    for (size_t i = 0; i < ARRAY2_SIZE; ++i) {
      float32Values2[i] = MathFunctions::float32Unflip(uint32Values2[i]);
      if (minFloat32Value2 > float32Values2[i]) {
        minFloat32Value2 = float32Values2[i]; // global min
      }
      if (maxFloat32Value2 < float32Values2[i]) {
        maxFloat32Value2 = float32Values2[i]; // global max
      }
    }
    const bool float32Result5 = MathFunctions::equal(
        minFloat32Value2,
        float32Values2[0]); // global min should match the first element of the
                            // sorted array
    EXPECT_TRUE(float32Result5);
    if (!float32Result5) {
      return;
    }
    const bool float32Result6 = MathFunctions::equal(
        maxFloat32Value2,
        float32Values2[ARRAY2_SIZE - 1]); // global max should match the last
                                          // element of the sorted array
    EXPECT_TRUE(float32Result6);
    if (!float32Result6) {
      return;
    }

    atomic<uint32_t> atomicMinUint32(numeric_limits<uint32_t>::max());
    atomic<uint32_t> atomicMaxUint32(numeric_limits<uint32_t>::min());
    parallelFor(
        0, ARRAY2_SIZE,
        [&](size_t i) {
          CPUParallelismUtilityFunctions::atomicMin(
              atomicMinUint32, uint32Values2[i]); // global atomic min
          CPUParallelismUtilityFunctions::atomicMax(
              atomicMaxUint32, uint32Values2[i]); // global atomic max
        },
        threadPool);
    const bool float32Result7 = MathFunctions::equal(
        MathFunctions::float32Unflip(atomicMinUint32.load()),
        float32Values2[0]); // global atomic min should match the first element
                            // of the sorted array
    EXPECT_TRUE(float32Result7);
    if (!float32Result7) {
      return;
    }
    const bool float32Result8 = MathFunctions::equal(
        MathFunctions::float32Unflip(atomicMaxUint32.load()),
        float32Values2[ARRAY2_SIZE - 1]); // global atomic max should match the
                                          // last element of the sorted array
    EXPECT_TRUE(float32Result8);
    if (!float32Result8) {
      return;
    }
  }

  // float64 -> uint64 representation case
  {
    const double originalFloat64Value1 = 2465884654.245664368686543655;
    const uint64_t uint64Value1 =
        MathFunctions::asUint64(originalFloat64Value1);
    const double float64Value1 = MathFunctions::asFloat64(uint64Value1);
    const bool float64Result1 =
        MathFunctions::equal(originalFloat64Value1, float64Value1);
    EXPECT_TRUE(float64Result1);
    if (!float64Result1) {
      return;
    }

    const double originalFloat64Value2 = -46456546221344.44233567424234743245;
    const uint64_t uint64Value2 =
        MathFunctions::float64Flip(originalFloat64Value2);
    const double float64Value2 = MathFunctions::float64Unflip(uint64Value2);
    const bool float64Result2 =
        MathFunctions::equal(originalFloat64Value2, float64Value2);
    EXPECT_TRUE(float64Result2);
    if (!float64Result2) {
      return;
    }

    array<double, ARRAY1_SIZE> float64Values1 = {
        {868242342347325.553534534535, 242.27482143432435, 1.677567567567567857,
         -0.77497567567567567938, 3.3357567756764463, -4565756775.7171768,
         -14127471.6455646455, 1411671715.8535345349, 3.143534534534,
         -3.145235435435}};
    array<uint64_t, ARRAY1_SIZE> uint64Values1 = {{0}};
    for (size_t i = 0; i < ARRAY1_SIZE; ++i) {
      uint64Values1[i] = MathFunctions::float64Flip(float64Values1[i]);
    }
    StdAuxiliaryFunctions::insertionSort<ARRAY1_SIZE>(
        uint64Values1.data()); // sort an array using insertion sort with a
                               // constant small size of N
    double minFloat64Value1 = numeric_limits<double>::max();
    double maxFloat64Value1 = numeric_limits<double>::min();
    for (size_t i = 0; i < ARRAY1_SIZE; ++i) {
      float64Values1[i] = MathFunctions::float64Unflip(uint64Values1[i]);
      if (minFloat64Value1 > float64Values1[i]) {
        minFloat64Value1 = float64Values1[i]; // global min
      }
      if (maxFloat64Value1 < float64Values1[i]) {
        maxFloat64Value1 = float64Values1[i]; // global max
      }
    }
    const bool float64Result3 = MathFunctions::equal(
        minFloat64Value1,
        float64Values1[0]); // global min should match the first element of the
                            // sorted array
    EXPECT_TRUE(float64Result3);
    if (!float64Result3) {
      return;
    }
    const bool float64Result4 = MathFunctions::equal(
        maxFloat64Value1,
        float64Values1[ARRAY1_SIZE - 1]); // global max should match the last
                                          // element of the sorted array
    EXPECT_TRUE(float64Result4);
    if (!float64Result4) {
      return;
    }

    auto float64Values2 = unique_ptr<double[]>(new double[ARRAY2_SIZE]);
    auto uint64Values2 = unique_ptr<uint64_t[]>(new uint64_t[ARRAY2_SIZE]);
    const double maxFloat64Value = numeric_limits<double>::max();
    RandomRNGWELL512 random;
    for (size_t i = 0; i < ARRAY2_SIZE; ++i) {
      uint64Values2[i] = MathFunctions::float64Flip(maxFloat64Value * random() -
                                                    maxFloat64Value / 2.0);
    }
    sort(uint64Values2.get(), uint64Values2.get() + ARRAY2_SIZE);
    double minFloat64Value2 = numeric_limits<double>::max();
    double maxFloat64Value2 = numeric_limits<double>::min();
    for (size_t i = 0; i < ARRAY2_SIZE; ++i) {
      float64Values2[i] = MathFunctions::float64Unflip(uint64Values2[i]);
      if (minFloat64Value2 > float64Values2[i]) {
        minFloat64Value2 = float64Values2[i]; // global min
      }
      if (maxFloat64Value2 < float64Values2[i]) {
        maxFloat64Value2 = float64Values2[i]; // global max
      }
    }
    const bool float64Result5 = MathFunctions::equal(
        minFloat64Value2,
        float64Values2[0]); // global min should match the first element of the
                            // sorted array
    EXPECT_TRUE(float64Result5);
    if (!float64Result5) {
      return;
    }
    const bool float64Result6 = MathFunctions::equal(
        maxFloat64Value2,
        float64Values2[ARRAY2_SIZE - 1]); // global max should match the last
                                          // element of the sorted array
    EXPECT_TRUE(float64Result6);
    if (!float64Result6) {
      return;
    }

    atomic<uint64_t> atomicMinUint64(numeric_limits<uint64_t>::max());
    atomic<uint64_t> atomicMaxUint64(numeric_limits<uint64_t>::min());
    parallelFor(
        0, ARRAY2_SIZE,
        [&](size_t i) {
          CPUParallelismUtilityFunctions::atomicMin(
              atomicMinUint64, uint64Values2[i]); // global atomic min
          CPUParallelismUtilityFunctions::atomicMax(
              atomicMaxUint64, uint64Values2[i]); // global atomic max
        },
        threadPool);
    const bool float64Result7 = MathFunctions::equal(
        MathFunctions::float64Unflip(atomicMinUint64.load()),
        float64Values2[0]); // global atomic min should match the first element
                            // of the sorted array
    EXPECT_TRUE(float64Result7);
    if (!float64Result7) {
      return;
    }
    const bool float64Result8 = MathFunctions::equal(
        MathFunctions::float64Unflip(atomicMaxUint64.load()),
        float64Values2[ARRAY2_SIZE - 1]); // global atomic max should match the
                                          // last element of the sorted array
    EXPECT_TRUE(float64Result8);
    if (!float64Result8) {
      return;
    }
  }
}

TEST(HostGoogleTest08__UTILS_Classes, MathFunctions) {
  // HostGoogleTest08__UTILS_Classes::executeTest();
}

void HostGoogleTest09__UTILS_CPUParallelism_Classes::executeTest() {
  const size_t numberOfElements =
      100 * 1024 * 1024; // 100Mb indices x4 bytes for floats
  const auto checkArrayPtr = make_unique<double[]>(numberOfElements);
  // initialize thread local storage data
  const size_t numberOfThreads = numberOfHardwareThreads();
  const auto threadLocalStorage =
      make_unique<RandomRNGWELL512[]>(numberOfThreads);
  // initialize thread pool with default parameters
  ThreadPool threadPool(numberOfThreads, AFFINITY_MASK_ALL, PRIORITY_NONE);
  // first check the setup of the ThreadPool above with the
  // parallelForThreadLocal() call below
  parallelForThreadLocal(
      0, 64 * numberOfThreads,
      [&](size_t, size_t threadIdx) // size_t i not used
      {
        EXPECT_TRUE(threadPool.getAffinity(threadIdx));
#ifndef _WIN32
        // on Linux, not assigning any priority to the ThreadPool returns
        // PRIORITY_NONE for the threads (ie not set)
        EXPECT_TRUE(threadPool.getPriority(threadIdx) == PRIORITY_NONE);
#else
        // on Windows, not assigning any priority to the ThreadPool returns
        // PRIORITY_NORMAL for the threads (ie set to default)
        EXPECT_TRUE(threadPool.getPriority(threadIdx) == PRIORITY_NORMAL);
#endif
        threadSleep(10); // give chance for things to possibly chance in thread
                         // scheduler
      },
      threadPool);
  // 1st run with parallelFor() & given thread pool
  parallelForThreadLocal(
      0, numberOfElements,
      [&](size_t i, size_t threadIdx) {
        checkArrayPtr[i] =
            threadLocalStorage[threadIdx]() + numeric_limits<double>::epsilon();
      },
      threadPool);
  bool result = true;
  for (size_t i = 0; i < numberOfElements; ++i) {
    if (checkArrayPtr[i] < 0.0) {
      result = false;
      break;
    }
  }
  EXPECT_TRUE(result);
  // 2nd run with parallelFor() & given thread pool
  parallelForThreadLocal(
      0, numberOfElements,
      [&](size_t i, size_t threadIdx) {
        checkArrayPtr[i] = -(threadLocalStorage[threadIdx]() +
                             numeric_limits<double>::epsilon());
      },
      threadPool);
  for (size_t i = 0; i < numberOfElements; ++i) {
    if (checkArrayPtr[i] > 0.0) {
      result = false;
      break;
    }
  }
  EXPECT_TRUE(result);
}

TEST(HostGoogleTest09__UTILS_CPUParallelism_Classes,
     parallelForThreadLocalThreadPool) {
  // HostGoogleTest09__UTILS_CPUParallelism_Classes::executeTest();
}

////////////////////////////////////////////////

void HostGoogleTest10__ColorHistogram::executeTest() {
  // Load image data
  vector<unsigned char> imageData;
  unsigned width, height;
  const string currentPath = StdReadWriteFileFunctions::getCurrentPath();
  // const string inputFile =
  //     string(currentPath + "/" + "Assets" + "/" + "alps.png");

  const string inputFile =
      string(currentPath + "/" + "Assets" + "/" + "sample.png");

  unsigned error =
      lodepng::decode(imageData, width, height, inputFile, LCT_RGB);
  EXPECT_EQ(error, 0u) << "Error loading image: " << lodepng_error_text(error);
  if (error)
    return;

  // Print image info
  const size_t imageSizeBytes = width * height * 3; // 3 channels (RGB)
  DebugConsole_consoleOutLine("\n=== Color Histogram Benchmark ===");
  DebugConsole_consoleOutLine("Image size: ", width, "x", height, " (",
                              imageSizeBytes / 1024.0 / 1024.0, " MB)");
  DebugConsole_consoleOutLine("Number of pixels: ", width * height);
  DebugConsole_consoleOutLine("Hardware threads available: ",
                              numberOfHardwareThreads());
  DebugConsole_consoleOutLine("");

  // Create test instance
  ColorHistogramTest histTest;

  // Initialize
  histTest.initializeFromImage(imageData.data(), width, height);

  // Run single core baseline
  DebugConsole_consoleOutLine("=== Single Core Test ===");
  histTest.computeSingleCore();
  // const auto &baselineHistogram = histTest.getHistogram();
  const double baselineTime = histTest.getTotalTime();

  // Print baseline stats
  DebugConsole_consoleOutLine("Single core time: ", baselineTime, " ms");
  DebugConsole_consoleOutLine(
      "Processing speed: ",
      (width * height) / (baselineTime / 1000.0) / 1000000.0, " MP/s");
  DebugConsole_consoleOutLine("");

  DebugConsole_consoleOutLine("=== Multi Core Test ===");
  ColorHistogramTest histTest2;

  // Initialize
  histTest2.initializeFromImage(imageData.data(), width, height);
  // Run multi core baseline
  histTest2.computeParallel();
  // const auto &baselineHistogram = histTest.getHistogram();
  const double baselineTimeMulti = histTest2.getTotalTime();
  DebugConsole_consoleOutLine("Multi core time: ", baselineTimeMulti, " ms");
  DebugConsole_consoleOutLine(
      "Processing speed: ",
      (width * height) / (baselineTimeMulti / 1000.0) / 1000000.0, " MP/s");
  

  // Save results
  const string outputDir = string(currentPath + "/" + "Images");
  if (!StdReadWriteFileFunctions::pathExists(outputDir)) {
    StdReadWriteFileFunctions::createDirectory(outputDir);
  }
  const string outputFile = string(outputDir + "/" + "histogram.csv");
  const string outputFileMulti = string(outputDir + "/" + "histogramMulti.csv");
  histTest.saveHistogramCSV(outputFile.c_str());
  histTest2.saveHistogramCSV(outputFileMulti.c_str());
  DebugConsole_consoleOutLine("\nHistogram data saved to: ", outputFile);
  DebugConsole_consoleOutLine("\nHistogram data saved to: ", outputFileMulti);
}

TEST(HostGoogleTest10__ColorHistogram, SingleAndMultiCore) {
  HostGoogleTest10__ColorHistogram::executeTest();
}

////////////////////////////////////////

// The main entry point of the HostUnitTests executable.
int main(int argc, char *argv[]) {
#ifdef GPU_FRAMEWORK_DEBUG
  DebugConsole::setUseLogFile(true);
  DebugConsole::setLogFileName("HostUnitTests.log");
#endif // GPU_FRAMEWORK_DEBUG

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}