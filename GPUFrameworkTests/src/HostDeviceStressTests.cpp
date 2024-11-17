#include "HostDeviceStressTests.h"
#include "Tests/CUDALinearAlgebraGPUComputingStressTest.h"
#include "CUDADriverInfo.h"
#include "CUDAProcessMemoryPool.h"
#include "CPUParallelism/CPUParallelismNCP.h"
#include "CPUParallelism/ThreadBarrier.h"
#include "UtilityFunctions.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdio>

using namespace std;
using namespace Tests;
using namespace Utils;
using namespace UtilsCUDA;
using namespace Utils::CPUParallelism;
using namespace Utils::UtilityFunctions;

namespace // anonymous namespace used instead of deprecated 'static' keyword used for cpp variable locality
{
  CUDALinearAlgebraGPUComputingStressTest::RunTypes runType = CUDALinearAlgebraGPUComputingStressTest::RunTypes::RUN_BOTH;
  size_t gpu_number     = 0;
  size_t cpu_iterations = 160;
  bool max_memory       = false;
  bool use_uva          = false;
  bool use_counter      = false;

  /*
  *  Displays the program usage information.
  */
  inline void giveUsage(const char* cmd, int exitCode = EXIT_FAILURE)
  {
    cout << "\nUsage: " << cmd << " [-h][-help][-cpu][-gpu][-gpu_number][-cpu_iterations][-max_memory][-use_uva][-use_counter]\n";
    cout << "-------------------------------------------------------------------------------\n";
    cout << "   -h or -help          this help text\n";
    cout << "   -cpu                 run CPU stress test only (mutually exclusive with -gpu)\n";
    cout << "   -gpu                 run GPU stress test only (mutually exclusive with -cpu)\n";
    cout << "   -gpu_number #n       uses a given number of GPUs (default is to use all)    \n";
    cout << "   -cpu_iterations #n   uses a given number of CPU iterations (default is 160) \n";
    cout << "   -max_memory          run stress test with max memory iterations             \n";
    cout << "   -use_uva             uses UVA (Unified Virtual Addressing) ie Unified Memory\n";
    cout << "   -use_counter         uses a counter informing about GPU vs. CPU executions\n\n";

    exit(exitCode);
  }

  inline array<size_t, CUDAProcessMemoryPool::MAX_DEVICES> getDeviceBytesToAllocatePerDevice(size_t bytesToAllocate, size_t deviceCount)
  {
    array<size_t, CUDAProcessMemoryPool::MAX_DEVICES> deviceBytesToAllocatePerDevice = { { 0 } };
    for (size_t device = 0; device < deviceCount; ++device)
    {
      deviceBytesToAllocatePerDevice[device] = bytesToAllocate;
    }

    return deviceBytesToAllocatePerDevice;
  }

  inline bitset<CUDAProcessMemoryPool::MAX_DEVICES> getUnifiedMemoryFlags(size_t deviceCount)
  {
    bitset<CUDAProcessMemoryPool::MAX_DEVICES> unifiedMemoryFlags;
    for (size_t device = 0; device < deviceCount; ++device)
    {
      unifiedMemoryFlags[device] = 1;
    }

    return unifiedMemoryFlags;
  }

  inline void startMemoryPool(const CUDADriverInfo& cudaDriverInfo, CUDAProcessMemoryPool& cudaProcessMemoryPool, int device, size_t numberOfGPUs, size_t arraySize, bool useUnifiedMemory)
  {
    const size_t hostBytesToAllocate = (3 * arraySize + 5 + 8 * cudaDriverInfo.getTextureAlignment(device)) * sizeof(int32_t); // Note: allocation is in bytes, plus some padding space
    const array<size_t, CUDAProcessMemoryPool::MAX_DEVICES> deviceBytesToAllocatePerDevice = getDeviceBytesToAllocatePerDevice(hostBytesToAllocate, numberOfGPUs);
    if (useUnifiedMemory)
    {
      const bitset<CUDAProcessMemoryPool::MAX_DEVICES> unifiedMemoryFlags = getUnifiedMemoryFlags(numberOfGPUs);
      // use the Host/Device Memory Pool for allocation of host/device memory for the given device
      cudaProcessMemoryPool.allocateDeviceMemoryPool(deviceBytesToAllocatePerDevice, unifiedMemoryFlags);
    }
    else
    {
      const size_t extraPaddingPerGPU = numberOfGPUs ? cudaDriverInfo.getTextureAlignment(device) * sizeof(int32_t) : 0;
      // use the Host/Device Memory Pool for allocation of host/device memory for the given device
      cudaProcessMemoryPool.allocateHostDeviceMemoryPool(numberOfGPUs * (hostBytesToAllocate + extraPaddingPerGPU), deviceBytesToAllocatePerDevice);
    }
  }

  inline void stopMemoryPool(CUDAProcessMemoryPool& cudaProcessMemoryPool, bool useUnifiedMemory)
  {
    if (useUnifiedMemory)
    {
      // use the Device Memory pool for de-allocation of host memory
      cudaProcessMemoryPool.freeDeviceMemoryPool();
    }
    else
    {
      // use the Host/Device Memory Pool for de-allocation of host/device memory
      cudaProcessMemoryPool.freeHostDeviceMemoryPool();
    }
  }
}

void HostDeviceStressGoogleTest01::executeTest()
{
  // create below the CUDA driver info for testing the GPU(s) with optional CUDA profiling enabled
  const CUDADriverInfo cudaDriverInfo(cudaDeviceScheduleAuto, true);
  CUDAProcessMemoryPool cudaProcessMemoryPool(cudaDriverInfo, true); // use default allocations for all devices
  const size_t numberOfGPUs       = (gpu_number > 0) ? min<size_t>(gpu_number, size_t(cudaDriverInfo.getDeviceCount())) : size_t(cudaDriverInfo.getDeviceCount()); // fail-safe checks for numberOfGPUs
  const size_t numberOfCPUThreads = numberOfHardwareThreads() / numberOfGPUs; // Note: we assume that the CPU threads are always >= numberOfGPUs
  ThreadBarrier threadBarrier{numberOfGPUs};

  // use N number of GPUs for the stress test
  parallelFor(0, numberOfGPUs, [&](size_t device)
  {
    bool testResult = true;
    size_t index    = 0;
    const size_t minDimensionSize = 64;
    const size_t maxDimensionSize = max_memory ? (((cudaDriverInfo.getTotalGlobalMemory(int(device)) >> 20) <= size_t(1 << 12)) ? 8192 : 16384) // 8192 for <= 4Gb ((1 << 12) -> 2048Mb), 16384 for <= 8Gb
                                               : 2048;
    const bool useUnifiedMemory   = use_uva && cudaDriverInfo.hasUnifiedMemory(int(device)) && cudaDriverInfo.getConcurrentManagedAccess(int(device));
    // check for last GPU run and optionally add more CPU threads to it
    const size_t extraCPUThreads  = ((device == numberOfGPUs - 1)) ? (numberOfHardwareThreads() % numberOfGPUs) : 0;
    for (size_t dimensionSize = minDimensionSize; testResult && (dimensionSize <= maxDimensionSize); dimensionSize <<= 1) // implies 'i *= 2'
    {
      // memory pool allocation is driven from device 0
      if (device == 0) startMemoryPool(cudaDriverInfo, cudaProcessMemoryPool, int(device), numberOfGPUs, dimensionSize * dimensionSize, useUnifiedMemory);
      // wait for all GPU stress test threads to be ready for this iteration
      threadBarrier.wait();

      ++index;
      DebugConsole_consoleOutLine("Running iteration '", index, "' for device '", device, "':");
      CUDALinearAlgebraGPUComputingStressTest cudaLinearAlgebraGPUComputingStressTest(cudaDriverInfo, cudaProcessMemoryPool, int(device), dimensionSize, runType, numberOfCPUThreads + extraCPUThreads, cpu_iterations, use_uva, use_counter);
      cudaLinearAlgebraGPUComputingStressTest.initializeGPUMemory();
      threadBarrier.wait(); // run in lock-step
      cudaLinearAlgebraGPUComputingStressTest.performGPUComputing();
      threadBarrier.wait(); // run in lock-step
      cudaLinearAlgebraGPUComputingStressTest.retrieveGPUResults();
      threadBarrier.wait(); // run in lock-step
      testResult = cudaLinearAlgebraGPUComputingStressTest.verifyComputingResults();
      threadBarrier.wait(); // run in lock-step
      cudaLinearAlgebraGPUComputingStressTest.releaseGPUComputingResources();
      threadBarrier.wait(); // run in lock-step

      // memory pool de-allocation is driven from device 0
      if (device == 0) stopMemoryPool(cudaProcessMemoryPool, useUnifiedMemory);
      // wait for all GPU stress test threads to be ready for the next iteration
      threadBarrier.wait();
    }
    // we do an ASSERT_TRUE to gracefully halt execution of other threads (ie GPU driving threads)
    ASSERT_TRUE(testResult);
  }, numberOfGPUs, AFFINITY_MASK_NONE); // let stress test use the underlying OS scheduler for thread execution
}

TEST(HostDeviceStressGoogleTest01, CUDALinearAlgebraGPUComputingStressTest)
{
  HostDeviceStressGoogleTest01::executeTest();
}

// The main entry point of the DeviceStressTests executable.
int main(int argc, char* argv[])
{
#ifdef GPU_FRAMEWORK_DEBUG
  DebugConsole::setUseLogFile(true);
  DebugConsole::setLogFileName("HostDeviceStressTests.log");
#endif // GPU_FRAMEWORK_DEBUG

  for (int i = 1; i < argc; ++i)
  {
    const string currentOption = StringAuxiliaryFunctions::toLowerCase(string(argv[i]));
    if ((currentOption == "-h") || (currentOption == "-help"))
    {
      giveUsage(argv[0], EXIT_SUCCESS);
    }
    else if (currentOption == "-cpu")
    {
      if (runType != CUDALinearAlgebraGPUComputingStressTest::RunTypes::RUN_BOTH)
      {
        DebugConsole_consoleOutLine("Command-line error:\n  Override argument cpu option mutually exclusive with gpu option.");
        giveUsage(argv[0]);
      }
      runType = CUDALinearAlgebraGPUComputingStressTest::RunTypes::RUN_CPU;
      DebugConsole_consoleOutLine("Command-line override argument cpu stress test detected.");
    }
    else if (currentOption == "-gpu")
    {
      if (runType != CUDALinearAlgebraGPUComputingStressTest::RunTypes::RUN_BOTH)
      {
        DebugConsole_consoleOutLine("Command-line error:\n  Override argument gpu option mutually exclusive with cpu option.");
        giveUsage(argv[0]);
      }
      runType = CUDALinearAlgebraGPUComputingStressTest::RunTypes::RUN_GPU;
      DebugConsole_consoleOutLine("Command-line override argument gpu stress test detected.");
    }
    else if (currentOption == "-gpu_number")
    {
      if ((i + 1) >= argc) giveUsage(argv[0]);
      gpu_number = StringAuxiliaryFunctions::fromString<size_t>(argv[i + 1]);
      if (gpu_number < 1)
      {
        DebugConsole_consoleOutLine("Command-line argument gpu_number should be > 0.");
        giveUsage(argv[0]);
      }
      ++i; // increment counter to skip the i + 1 option
      DebugConsole_consoleOutLine("Command-line argument gpu_number detected: ", gpu_number);
    }
    else if (currentOption == "-cpu_iterations")
    {
      if ((i + 1) >= argc) giveUsage(argv[0]);
      cpu_iterations = StringAuxiliaryFunctions::fromString<size_t>(argv[i + 1]);
      if (cpu_iterations < 1)
      {
        DebugConsole_consoleOutLine("Command-line argument cpu_iterations should be > 0.");
        giveUsage(argv[0]);
      }
      ++i; // increment counter to skip the i + 1 option
      DebugConsole_consoleOutLine("Command-line argument cpu_iterations detected: ", cpu_iterations);
    }
    else if (currentOption == "-max_memory")
    {
      max_memory = true;
      DebugConsole_consoleOutLine("Command-line argument max_memory detected: ", StringAuxiliaryFunctions::toString<bool>(max_memory));
    }
    else if (currentOption == "-use_uva")
    {
      use_uva = true;
      DebugConsole_consoleOutLine("Command-line argument use_uva detected: ", StringAuxiliaryFunctions::toString<bool>(use_uva));
    }
    else if (currentOption == "-use_counter")
    {
      use_counter = true;
      DebugConsole_consoleOutLine("Command-line argument use_counter detected: ", StringAuxiliaryFunctions::toString<bool>(use_counter));
    }
    else
    {
      giveUsage(argv[0]);
    }
  }

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}