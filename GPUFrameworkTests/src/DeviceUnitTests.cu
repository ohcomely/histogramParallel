#include "AccurateTimers.h"
#include "CPUParallelism/CPUParallelismNCP.h"
#include "CPUParallelism/CPUParallelismUtilityFunctions.h"
#include "CPUParallelism/ThreadPool.h"
#include "CUDADriverInfo.h"
#include "CUDAEventTimer.h"
#include "CUDAKernelLauncher.h"
#include "CUDAMemoryHandlers.h"
#include "CUDAMemoryPool.h"
#include "CUDAMemoryRegistry.h"
#include "CUDAMemoryWrappers.h"
#include "CUDAParallelFor.h"
#include "CUDAQueue.h"
#include "CUDAStreamsHandler.h"
#include "CUDAUtilityDeviceFunctions.h"
#include "CUDAUtilityFunctions.h"
#include "DeviceUnitTests.h"
#include "MathConstants.h"
#include "Randomizers.h"
#include "Tests/CUDALinearAlgebraGPUComputingTest.h"
// #include "Tests/ColorHistogramGPUTest.h"
#include "UnitTests.h"
#include "UtilityFunctions.h"
#include <algorithm>
#include <array>
#include <future>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>

using namespace std;
using namespace Tests;
using namespace UtilsCUDA;
using namespace UtilsCUDA::CUDAParallelFor;
using namespace Utils;
using namespace Utils::CPUParallelism;
using namespace Utils::UnitTests;
using namespace Utils::AccurateTimers;
using namespace Utils::Randomizers;
using namespace Utils::UtilityFunctions;

namespace // anonymous namespace used instead of deprecated 'static' keyword
          // used for cpp variable locality
{
inline void CUDART_CB testStreamCallback(cudaStream_t stream,
                                         cudaError_t status, void *data) {
  CUDAError_checkCUDAError(status);
  DebugConsole_consoleOutLine("testStreamCallback firing for stream: '", stream,
                              "' and data: '", data, "', all ok!\n");
}

template <typename T, size_t N, bool COND = false>
inline void checkArray(const T *__restrict hostArray, T value = 0,
                       enable_if_t<is_integral<T>::value> * = nullptr) {
  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(hostArray[i], COND ? value : T(i));
  }
}

template <typename T, size_t N, bool COND = false>
inline void checkArray(const T *__restrict hostArray, T value = 0,
                       enable_if_t<is_floating_point<T>::value> * = nullptr) {
  for (size_t i = 0; i < N; ++i) {
    EXPECT_FLOAT_EQ(hostArray[i], COND ? value : T(i));
  }
}

#if __CUDA_ARCH__ >=                                                           \
    500 // Maxwell GPUs onwards for good atomicMin/Max 64bit support
__forceinline__ __device__ void
setAtomicMinMaxUint64(size_t index, uint64_t *__restrict atomicMinMaxUint64,
                      const uint64_t *__restrict uint64Values) {
  // Note: below are CUDA-needed casts to bridge the uint64_t -> unsigned long
  // long int issue in the atomicMinMax CUDA API
  unsigned long long int uint64Value =
      (unsigned long long int)uint64Values[index];
  unsigned long long int *atomicMinMaxUint64Addr =
      (unsigned long long int *)atomicMinMaxUint64;
  atomicMin(&atomicMinMaxUint64Addr[0], uint64Value); // global atomic min
  atomicMax(&atomicMinMaxUint64Addr[1], uint64Value); // global atomic max
}
#endif // __CUDA_ARCH__

__global__ void kernelTestRun(size_t arraySize, float *__restrict__ target,
                              const float *__restrict__ a,
                              const size_t *__restrict__ b) {
  size_t index = size_t(CUDAUtilityDeviceFunctions::globalLinearIndex());
  if (index < arraySize) {
    target[index] = a[index] + b[index];
  }
}

inline void testCUDAMemoryPool(CUDAMemoryPool &cudaMemoryPool, size_t arraySize,
                               int device = 0) {
  HostMemory<float> hostHandler;
  DeviceMemory<float> deviceHandler;
  HostDeviceMemory<float> hostDeviceHandler;

  EXPECT_TRUE(cudaMemoryPool.addToHostMemoryPool(hostHandler, arraySize / 2));
  EXPECT_TRUE(cudaMemoryPool.addToDeviceMemoryPool(deviceHandler, arraySize / 2,
                                                   device));
  EXPECT_TRUE(cudaMemoryPool.addToHostDeviceMemoryPool(hostDeviceHandler,
                                                       arraySize, device));

  // use the Host Memory Pool for allocation of host memory
  cudaMemoryPool.allocateHostMemoryPool(
      "Test"); // Note: internally known total size

  // use the Device Memory Pool for allocation of device memory
  cudaMemoryPool.allocateDeviceMemoryPool(
      "Test"); // Note: internally known total size

  EXPECT_TRUE(hostHandler.host() != nullptr);
  EXPECT_TRUE(hostHandler.isMemoryPoolMode());

  EXPECT_TRUE(deviceHandler.device() != nullptr);
  EXPECT_TRUE(deviceHandler.isMemoryPoolMode());

  EXPECT_TRUE(hostDeviceHandler.host() != nullptr);
  EXPECT_TRUE(hostDeviceHandler.device() != nullptr);
  EXPECT_TRUE(hostDeviceHandler.isMemoryPoolMode());

  // use the Host/Device Memory Pool for de-allocation of host/device memory
  cudaMemoryPool.freeHostDeviceMemoryPool();

  EXPECT_TRUE(cudaMemoryPool.getHostMemoryPoolSize() == 0);
  EXPECT_TRUE(cudaMemoryPool.getDeviceMemoryPoolSize() == 0);
}

inline void
testCUDAProcessMemoryPool(const CUDADriverInfo &cudaDriverInfo,
                          CUDAProcessMemoryPool &cudaProcessMemoryPool,
                          size_t arraySize, int device = 0) {
  HostMemory<float> hostHandler;
  DeviceMemory<float> deviceHandler;
  HostDeviceMemory<float> hostDeviceHandler;

  EXPECT_FALSE(cudaProcessMemoryPool.reserve(hostHandler, arraySize / 2));
  EXPECT_FALSE(
      cudaProcessMemoryPool.reserve(deviceHandler, arraySize / 2, device));

  // use the Host Memory Pool for allocation of host memory
  cudaProcessMemoryPool.allocateHostMemoryPool(
      (arraySize + arraySize / 2 +
       3 * cudaDriverInfo.getTextureAlignment(device)) *
      sizeof(float)); // Note: allocation is in bytes, plus some padding space

  // use the Device Memory Pool for allocation of device memory
  cudaProcessMemoryPool.allocateDeviceMemoryPool(
      {(arraySize + arraySize / 2 +
        3 * cudaDriverInfo.getTextureAlignment(device)) *
       sizeof(float)}); // Note: allocation is in bytes, plus some padding space

  EXPECT_TRUE(cudaProcessMemoryPool.reserve(hostHandler, arraySize / 2));
  EXPECT_TRUE(
      cudaProcessMemoryPool.reserve(deviceHandler, arraySize / 2, device));
  EXPECT_TRUE(
      cudaProcessMemoryPool.reserve(hostDeviceHandler, arraySize, device));

  cudaProcessMemoryPool.reportHostDeviceMemoryPoolInformation("Test");

  EXPECT_TRUE(hostHandler.host() != nullptr);
  EXPECT_TRUE(hostHandler.isMemoryPoolMode());

  EXPECT_TRUE(deviceHandler.device() != nullptr);
  EXPECT_TRUE(deviceHandler.isMemoryPoolMode());

  EXPECT_TRUE(hostDeviceHandler.host() != nullptr);
  EXPECT_TRUE(hostDeviceHandler.device() != nullptr);
  EXPECT_TRUE(hostDeviceHandler.isMemoryPoolMode());

  // use the Host/Device Memory Pool for de-allocation of host/device memory
  cudaProcessMemoryPool.freeHostDeviceMemoryPool();

  EXPECT_TRUE(cudaProcessMemoryPool.getHostMemoryPoolSize() == 0);
  EXPECT_TRUE(cudaProcessMemoryPool.getDeviceMemoryPoolSize() == 0);
}

__global__ void kernelVerifyOldMemset(uint8_t *__restrict p, size_t offset,
                                      int value, size_t step, size_t length) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (offset + tid * step >= length) {
    return;
  }

  size_t localLength = min((tid + 1) * step, length) - tid * step;
  ::memset(p + offset + tid * step, value, localLength);
}

__global__ void kernelVerifyFastMemset(uint8_t *__restrict p, size_t offset,
                                       int value, size_t step, size_t length) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (offset + tid * step >= length) {
    return;
  }

  size_t localLength = min((tid + 1) * step, length) - tid * step;
  CUDAUtilityFunctions::memset(p + offset + tid * step, value, localLength);
}

inline bool testFastMemset(size_t from, size_t to, size_t length) {
  const auto origin = make_unique<uint8_t[]>(
      length); // make_unique also initializes the heap array to 0
  // memset(origin.get(), 0, sizeof(origin));
  memset(origin.get() + from, 0x01, to - from);

  HostDeviceMemory<uint8_t> memory(length);
  memory.memset(0, false);
  CUDAParallelFor::launchCUDAParallelFor(
      1,
      [] __device__(size_t, uint8_t *__restrict p, size_t length) {
        CUDAUtilityFunctions::memset(p, 0x01, length);
      },
      memory.device() + from, to - from);
  memory.copyDeviceToHost();

  return equal(origin.get(), origin.get() + length, memory.host());
}

inline bool verifyFastMemset(CUDAEventTimer &timer, size_t offset,
                             size_t size) {
  if (offset >= size) {
    DebugConsole_consoleOutLine("Skip test for 'offset = ", offset,
                                "', 'size = ", size, "'.");
    return true;
  }

  const size_t padding = 16;
  const size_t memorySize = size + offset + padding;

  HostDeviceMemory<uint8_t> memory0(memorySize);
  HostDeviceMemory<uint8_t> memory1(memorySize);
  memory0.memset(0xCC, false);
  memory1.memset(0xCC, false);

  const uint32_t chunk = 1024;
  uint32_t chunks = uint32_t((size + chunk - 1) / chunk);
  const uint32_t threads = 256;
  uint32_t blocks = uint32_t((chunks + threads - 1) / threads);

  timer.startTimer();
  KernelLauncher::create().setGrid(blocks).setBlock(threads).run(
      kernelVerifyOldMemset, memory0.device(), offset, 0x0A, chunk, size);
  timer.stopTimer();
  double oldTime = timer.getDecimalElapsedTimeInMilliSecs();
  memory0.copyDeviceToHost();

  timer.startTimer();
  KernelLauncher::create().setGrid(blocks).setBlock(threads).run(
      kernelVerifyFastMemset, memory1.device(), offset, 0x0A, chunk, size);
  timer.stopTimer();
  double newTime = timer.getDecimalElapsedTimeInMilliSecs();
  memory1.copyDeviceToHost();

  // print timer results between old & new memset
  DebugConsole_consoleOutLine("memset() for ", offset, "\t", oldTime, " ms\t",
                              newTime, " ms\t",
                              (newTime < oldTime ? "FASTER " : "SLOWER "),
                              size_t(oldTime * 100.0 / newTime), "%");

  return equal(memory0.host(), memory0.host() + memorySize, memory1.host());
}
} // namespace

void DeviceGoogleTest01__UTILS_CUDA_Classes::executeTest() {
  // create below the CUDA driver info for testing the GPU(s) with optional CUDA
  // profiling enabled
  const CUDADriverInfo cudaDriverInfo(cudaDeviceScheduleAuto, true);

  // at least one GPU needed for the  computations
  EXPECT_GE(cudaDriverInfo.getDeviceCount(), 1);
  for (int i = 0; i < cudaDriverInfo.getDeviceCount(); ++i) {
    // minimum compute capability expected of at least a Kepler GPU of Compute
    // Capability 3.0
    EXPECT_TRUE(
        cudaDriverInfo.isAtLeastGPUType(CUDADriverInfo::GPUTypes::KEPLER, i));
    // we need a decent memory controller for GPU I/O: 256 bits
    EXPECT_GE(cudaDriverInfo.getMemoryBusWidth(i), 256);
    // main VRAM (TotalGlobalMemory) >= 4096Mb (4Gb)
    EXPECT_GE(size_t(cudaDriverInfo.getTotalGlobalMemory(i) >> 20),
              size_t(1 << 12));
  }
}

TEST(DeviceGoogleTest01__UTILS_CUDA_Classes, CUDADriverInfo) {
  DeviceGoogleTest01__UTILS_CUDA_Classes::executeTest();
}

void DeviceGoogleTest02__UTILS_CUDA_Classes::executeTest() {
  // create below the CUDA driver info for testing the GPU(s) with optional CUDA
  // profiling enabled
  const CUDADriverInfo cudaDriverInfo(cudaDeviceScheduleAuto, true);

  for (int i = 0; i < cudaDriverInfo.getDeviceCount(); ++i) {
    bool testResult = true;
    const size_t minDimensionSize = 64;
    const size_t maxDimensionSize =
        ((cudaDriverInfo.getTotalGlobalMemory(i) >> 20) <= size_t(1 << 12))
            ? 4096
            : 8192; // 4096 for <= 4Gb ((1 << 12) -> 2048Mb), 8192 for <= 8Gb,
                    // should be 16384 for large VRAM/GPU systems
    // run core GPU tests with default modes on (Unified Memory on supported
    // platforms, Pascal GPU architecture & onwards)
    for (size_t dimensionSize = minDimensionSize;
         testResult && dimensionSize <= maxDimensionSize;
         dimensionSize <<= 1) // implies 'i *= 2'
    {
      // run a mock-up Linear Algebra calculation to test the CUDA code path
      // part
      CUDALinearAlgebraGPUComputingTest cudaLinearAlgebraGPUComputingTest(
          cudaDriverInfo, i, true, dimensionSize);
      cudaLinearAlgebraGPUComputingTest.initializeGPUMemory();
      cudaLinearAlgebraGPUComputingTest.performGPUComputing();
      cudaLinearAlgebraGPUComputingTest.retrieveGPUResults();
      testResult = cudaLinearAlgebraGPUComputingTest.verifyComputingResults();
      cudaLinearAlgebraGPUComputingTest.releaseGPUComputingResources();
    }
    EXPECT_TRUE(testResult);

    // if Unified Memory is supported (best performance on Pascal GPU
    // architecture & onwards), also run the test without it to check CUDA async
    // commands
    if (cudaDriverInfo.hasUnifiedMemory(i)) {
      for (size_t dimensionSize = minDimensionSize;
           testResult && dimensionSize <= maxDimensionSize;
           dimensionSize <<= 1) // implies 'i *= 2'
      {
        // run a mock-up Linear Algebra calculation to test the CUDA code path
        // part
        CUDALinearAlgebraGPUComputingTest cudaLinearAlgebraGPUComputingTest(
            cudaDriverInfo, i, false, dimensionSize);
        cudaLinearAlgebraGPUComputingTest.initializeGPUMemory();
        cudaLinearAlgebraGPUComputingTest.performGPUComputing();
        cudaLinearAlgebraGPUComputingTest.retrieveGPUResults();
        testResult = cudaLinearAlgebraGPUComputingTest.verifyComputingResults();
        cudaLinearAlgebraGPUComputingTest.releaseGPUComputingResources();
      }
      EXPECT_TRUE(testResult);
    }
  }
}

TEST(DeviceGoogleTest02__UTILS_CUDA_Classes,
     CUDALinearAlgebraGPUComputingTest) {
  DeviceGoogleTest02__UTILS_CUDA_Classes::executeTest();
}

void DeviceGoogleTest03__UTILS_CUDA_Classes::executeTest() {
  // create below the CUDA driver info for testing the GPU(s) with optional CUDA
  // profiling enabled
  const CUDADriverInfo cudaDriverInfo(cudaDeviceScheduleAuto, true);
  CUDAMemoryRegistry cudaMemoryRegistry;

  // explicit CUDAMemoryRegistry test with host/device memory
  // register/allocation respectively
  constexpr size_t STACK_ARRAY_SIZE = 10000;
  constexpr size_t HEAP_ARRAY_SIZE = 1000000;
  array<float, STACK_ARRAY_SIZE> stackArray{};
  unique_ptr<float[]> heapArrayA =
      unique_ptr<float[]>(new float[HEAP_ARRAY_SIZE]);
  unique_ptr<float[]> heapArrayB =
      unique_ptr<float[]>(new float[HEAP_ARRAY_SIZE]);

  cudaMemoryRegistry.addToMemoryRegistry<float>("stackArray", stackArray.data(),
                                                STACK_ARRAY_SIZE);
  cudaMemoryRegistry.addToMemoryRegistry<float>("heapArrayA", heapArrayA.get(),
                                                HEAP_ARRAY_SIZE);
  cudaMemoryRegistry.addToMemoryRegistry<float>("heapArrayB", heapArrayB.get(),
                                                HEAP_ARRAY_SIZE);

  EXPECT_FALSE(cudaMemoryRegistry.getPtrFromMemoryRegistry<float>("array"));

  // use the Host Memory Registry for registering (pinning) the host memory with
  // CUDA
  cudaMemoryRegistry.registerMemoryRegistry("Test");

  EXPECT_FALSE(cudaMemoryRegistry.addToMemoryRegistry<float>("array", nullptr,
                                                             HEAP_ARRAY_SIZE));

  auto *stackArrayPtr =
      cudaMemoryRegistry.getPtrFromMemoryRegistry<float>("stackArray");
  auto *heapArrayAPtr =
      cudaMemoryRegistry.getPtrFromMemoryRegistry<float>("heapArrayA");
  auto *heapArrayBPtr =
      cudaMemoryRegistry.getPtrFromMemoryRegistry<float>("heapArrayB");
  auto *arrayPtr = cudaMemoryRegistry.getPtrFromMemoryRegistry<float>("array");
  EXPECT_TRUE(stackArrayPtr != nullptr);
  EXPECT_TRUE(heapArrayAPtr != nullptr);
  EXPECT_TRUE(heapArrayBPtr != nullptr);
  EXPECT_TRUE(arrayPtr == nullptr);

  // use the Host Memory Registry for unregistering (unpinning) the host memory
  // with CUDA
  cudaMemoryRegistry.unregisterMemoryRegistry();

  EXPECT_TRUE(cudaMemoryRegistry.getMemoryRegistrySize() == 0);
}

TEST(DeviceGoogleTest03__UTILS_CUDA_Classes, CUDAMemoryRegistry) {
  DeviceGoogleTest03__UTILS_CUDA_Classes::executeTest();
}

void DeviceGoogleTest04__UTILS_CUDA_Classes::executeTest() {
  // create below the CUDA driver info for testing the GPU(s) with optional CUDA
  // profiling enabled
  const CUDADriverInfo cudaDriverInfo(cudaDeviceScheduleAuto, true);

  const size_t numberOfElements =
      100 * 1024 * 1024; // 100Mb indices x4 bytes for floats
  const int device = 0;
  const size_t numberOfCUDAStreams = 1;

  // test the memory handlers (swap functionality)
  HostMemory<float> hostHandlerSwap1;
  HostMemory<float> hostHandlerSwap2;
  DeviceMemory<float> deviceHandlerSwap1;
  DeviceMemory<float> deviceHandlerSwap2;
  HostDeviceMemory<float> hostDeviceHandlerSwap1;
  HostDeviceMemory<float> hostDeviceHandlerSwap2;
  // Note: destructor of the futures will allocate in async mode
  {
    auto future1 = hostHandlerSwap1.allocateAsync(numberOfElements / 2);
    auto future2 = hostHandlerSwap2.allocateAsync(numberOfElements / 4);
    auto future3 =
        deviceHandlerSwap1.allocateAsync(numberOfElements / 2, device);
    auto future4 =
        deviceHandlerSwap2.allocateAsync(numberOfElements / 4, device);
    auto future5 =
        hostDeviceHandlerSwap1.allocateAsync(numberOfElements / 2, device);
    auto future6 =
        hostDeviceHandlerSwap2.allocateAsync(numberOfElements / 4, device);
  }
  // test HostMemory swap functionality
  EXPECT_TRUE(hostHandlerSwap1.getNumberOfElements() == numberOfElements / 2);
  EXPECT_TRUE(hostHandlerSwap2.getNumberOfElements() == numberOfElements / 4);
  auto hostHandlerSwap1Ptr = hostHandlerSwap1.host();
  auto hostHandlerSwap2Ptr = hostHandlerSwap2.host();
  hostHandlerSwap1.swap(hostHandlerSwap2);
  EXPECT_TRUE(hostHandlerSwap1.host() == hostHandlerSwap2Ptr);
  EXPECT_TRUE(hostHandlerSwap2.host() == hostHandlerSwap1Ptr);
  EXPECT_TRUE(hostHandlerSwap1.getNumberOfElements() == numberOfElements / 4);
  EXPECT_TRUE(hostHandlerSwap2.getNumberOfElements() == numberOfElements / 2);
  // test DeviceMemory swap functionality
  EXPECT_TRUE(deviceHandlerSwap1.getNumberOfElements() == numberOfElements / 2);
  EXPECT_TRUE(deviceHandlerSwap2.getNumberOfElements() == numberOfElements / 4);
  auto deviceHandlerSwap1Ptr = deviceHandlerSwap1.device();
  auto deviceHandlerSwap2Ptr = deviceHandlerSwap2.device();
  deviceHandlerSwap1.swap(deviceHandlerSwap2);
  EXPECT_TRUE(deviceHandlerSwap1.device() == deviceHandlerSwap2Ptr);
  EXPECT_TRUE(deviceHandlerSwap2.device() == deviceHandlerSwap1Ptr);
  EXPECT_TRUE(deviceHandlerSwap1.getNumberOfElements() == numberOfElements / 4);
  EXPECT_TRUE(deviceHandlerSwap2.getNumberOfElements() == numberOfElements / 2);
  // test HostDeviceMemory swap functionality
  EXPECT_TRUE(hostDeviceHandlerSwap1.getNumberOfElements() ==
              numberOfElements / 2);
  EXPECT_TRUE(hostDeviceHandlerSwap2.getNumberOfElements() ==
              numberOfElements / 4);
  auto hostDeviceHandlerSwap1HostPtr = hostDeviceHandlerSwap1.host();
  auto hostDeviceHandlerSwap2HostPtr = hostDeviceHandlerSwap2.host();
  auto hostDeviceHandlerSwap1DevicePtr = hostDeviceHandlerSwap1.device();
  auto hostDeviceHandlerSwap2DevicePtr = hostDeviceHandlerSwap2.device();
  hostDeviceHandlerSwap1.swap(hostDeviceHandlerSwap2);
  EXPECT_TRUE(hostDeviceHandlerSwap1.host() == hostDeviceHandlerSwap2HostPtr);
  EXPECT_TRUE(hostDeviceHandlerSwap2.host() == hostDeviceHandlerSwap1HostPtr);
  EXPECT_TRUE(hostDeviceHandlerSwap1.device() ==
              hostDeviceHandlerSwap2DevicePtr);
  EXPECT_TRUE(hostDeviceHandlerSwap2.device() ==
              hostDeviceHandlerSwap1DevicePtr);
  EXPECT_TRUE(hostDeviceHandlerSwap1.getNumberOfElements() ==
              numberOfElements / 4);
  EXPECT_TRUE(hostDeviceHandlerSwap2.getNumberOfElements() ==
              numberOfElements / 2);
  // Note: destructor of the futures will reset in async mode
  {
    auto future1 = hostHandlerSwap1.resetAsync();
    auto future2 = hostHandlerSwap2.resetAsync();
    auto future3 = deviceHandlerSwap1.resetAsync();
    auto future4 = deviceHandlerSwap2.resetAsync();
    auto future5 = hostDeviceHandlerSwap1.resetAsync();
    auto future6 = hostDeviceHandlerSwap2.resetAsync();
  }

  // test the usage of the HostDeviceMemory handler
  const CUDAStreamsHandler cudaStreamsHandler(cudaDriverInfo, device,
                                              numberOfCUDAStreams);
  cudaStreamsHandler.addCallback(0, testStreamCallback, nullptr);
  HostDeviceMemory<float> cudaMemoryHandler;
  auto future = cudaMemoryHandler.allocateAsync(numberOfElements);
  const auto checkHostPtr = make_unique<float[]>(numberOfElements);
  // initialize thread local storage data
  const size_t threadLocalSize = numberOfHardwareThreads();
  const auto threadLocalStorage = make_unique<uint32_t[]>(threadLocalSize);
  for (size_t i = 0; i < threadLocalSize; ++i) {
    const uint64_t time = AccurateCPUTimer::getNanosecondsTimeSinceEpoch();
    threadLocalStorage[i] =
        CUDAUtilityFunctions::seedGenerator<16>(uint32_t(time), uint32_t(i));
  }
  // wait until the reallyAsync() upload above is done
  future.wait();
  const auto hostPtr = cudaMemoryHandler.host();
  parallelForThreadLocal(
      0, numberOfElements,
      [&](size_t i, size_t threadIdx) {
        checkHostPtr[i] = hostPtr[i] =
            CUDAUtilityFunctions::rand1f(threadLocalStorage[threadIdx]);
      },
      threadLocalSize);
  cudaMemoryHandler.copyHostToDeviceAsync(cudaStreamsHandler[0]);
  cudaMemoryHandler.copyDeviceToHost(
      cudaStreamsHandler[0]); // note the non-async function usage here
  // test results
  bool result = true;
  for (size_t i = 0; i < numberOfElements; ++i) {
    if (checkHostPtr[i] != hostPtr[i]) {
      result = false;
      break;
    }
  }

  EXPECT_TRUE(result);
}

TEST(DeviceGoogleTest04__UTILS_CUDA_Classes, CUDAMemoryHandlers) {
  DeviceGoogleTest04__UTILS_CUDA_Classes::executeTest();
}

void DeviceGoogleTest05__UTILS_CUDA_Classes::executeTest() {
  // create below the CUDA driver info for testing the GPU(s) with optional CUDA
  // profiling enabled
  const CUDADriverInfo cudaDriverInfo(cudaDeviceScheduleAuto, true);

  // Part 1 Test: HostMemory & ArraySpan
  {
    HostMemory<size_t> memory(10);
    Span<size_t> span(memory.host(), 10);
    // we pass that to a kernel and change it (each thread writes its index)
    parallelFor(0, 10, [&](size_t index) { span[index] = index; });
    checkArray<size_t, 10>(memory.host());
  }

  // Part 2 Test: RawDeviceMemory
  {
    HostDeviceMemory<size_t> memory(10);
    // we create a raw memory view on it
    RawDeviceMemory<size_t> rawMemory(memory.device());

    // test RawDeviceMemory copy constructor
    {
      RawDeviceMemory<size_t> rawMemoryCpy = rawMemory;
      RawDeviceMemory<const size_t> rawMemoryConst = rawMemory;
      RawDeviceMemory<const size_t> rawMemoryConstCpy = rawMemoryConst;
      rawMemoryConstCpy = rawMemoryConst;
      // rawMemoryCpy = rawMemoryConst; // this will correctly not compile as it
      // discards the const from the pointer
    }

    // we pass that to a kernel and change it (each thread writes its index)
    CUDAParallelFor::launchCUDAParallelFor(
        10,
        [] __device__(size_t index, const RawDeviceMemory<size_t> &memory) {
          memory[index] = index;
        },
        rawMemory);
    memory.copyDeviceToHost();
    checkArray<size_t, 10>(memory.host());
  }

  // Part 3 Test: ArraySpan
  {
    HostDeviceMemory<size_t> memory(10);
    Span<size_t> span(memory.device(), 10);
    // we pass that to a kernel and change it (each thread writes its index)
    CUDAParallelFor::launchCUDAParallelFor(
        10,
        [] __device__(size_t index, const Span<size_t> &memory) {
          memory[index] = index;
        },
        span);
    memory.copyDeviceToHost();
    checkArray<size_t, 10>(memory.host());
  }

  // Part 4 Test: ArraySpanRangeBasedFor
  {
    HostDeviceMemory<size_t> memory(10);
    Span<size_t, 10> span(memory.device());
    // we pass that to a kernel and change it (each thread writes its index)
    CUDAParallelFor::launchCUDAParallelFor(
        10,
        [] __device__(size_t, const Span<size_t, 10> &memory) {
          size_t idx = 0;
          for (auto &i : memory) {
            i = idx++;
          }
        },
        span);
    memory.copyDeviceToHost();
    checkArray<size_t, 10>(memory.host());
  }

  // Part 5 Test: SubSpan
  {
    HostDeviceMemory<size_t> memory(20);
    // we create a span memory view on it
    Span<size_t> span(memory.device(), 20);
    auto subSpan = span.subSpan(10, 10);
    // we pass that to a kernel and change it (each thread writes its index)
    CUDAParallelFor::launchCUDAParallelFor(
        10,
        [] __device__(size_t index, const Span<size_t> &memory) {
          memory[index] = index;
        },
        subSpan);
    memory.copyDeviceToHost();
    checkArray<size_t, 10>(memory.host() + 10);
  }

  // Part 6 Test: Conversion to Const
  {
    HostDeviceMemory<size_t> memory(20);
    // we create a span memory view on it

    RawDeviceMemory<size_t> rawMemory(memory.device());
    Span<size_t> span(memory.device(), 20);

    RawDeviceMemory<const size_t> rawMemoryConst(rawMemory);
    EXPECT_EQ(rawMemory.data(), rawMemoryConst.data());

    RawDeviceMemory<const size_t> rawMemoryConst2 = rawMemory;
    EXPECT_EQ(rawMemory.data(), rawMemoryConst2.data());

    Span<const size_t> constSpan = span;
    Span<const size_t> constSpan2(span);

    EXPECT_EQ(span.data(), constSpan.data());
    EXPECT_EQ(span.size(), constSpan.size());
    EXPECT_EQ(span.data(), constSpan2.data());
    EXPECT_EQ(span.size(), constSpan2.size());
  }

  // static tests that pass as soon as the test compiles!
  static_assert(sizeof(Span<size_t, 10>) == sizeof(size_t *),
                "A fixed size span should not have any memory overhead");
  static_assert(
      sizeof(Span<size_t>) == sizeof(size_t *) + sizeof(size_t),
      "A dynamically sized array needs also to save the size of the span");

  // Part 7 Test: Fixed size array spans
  {
    HostDeviceMemory<size_t> memory(20);
    Span<size_t, 10> span; // tests whether the default constructor works
    span = memory.deviceSpan<10>(); // tests whether 1st copy assignment works +
                                    // fixed size access to deviceSpan()

    // this tests the implicit conversion to the dynamic case
    CUDAParallelFor::launchCUDAParallelFor(
        10,
        [] __device__(size_t index, const Span<size_t, 10> &memory) {
          memory[index] = index;
        },
        span);
    memory.copyDeviceToHost();
    checkArray<size_t, 10>(memory.host());

    // this tests the fixed-size subspan function
    CUDAParallelFor::launchCUDAParallelFor(
        5,
        [] __device__(size_t index, const Span<size_t, 5> &memory) {
          memory[index] = index;
        },
        span.subSpan<5>(5));
    memory.copyDeviceToHost();
    checkArray<size_t, 5>(memory.host() + 5);
  }

  // Part 8 Test: Spans on the host side
  {
    array<size_t, 10> data{};
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] = i;
    }

    Span<size_t> span(data.data(), data.size());
    for (size_t i = 0; i < data.size(); ++i) {
      EXPECT_EQ(data.at(i), span[i]);
    }

    // create fixed-size subspan
    auto subSpan = span.subSpan<5>(5);

    EXPECT_EQ(subSpan[0], data[5]);
    EXPECT_EQ(subSpan[4], data[9]);
  }

#ifndef NDEBUG
  // Note: only run this test in debug as in release the assertion is commented
  // out and will not fire Part 9 Test: ArraySubSpanOutOfBounds
  {
    HostDeviceMemory<size_t> memory(10);
    // we create a span memory view on it
    Span<size_t> span(memory.device(), 10);
    EXPECT_DEATH(span.subSpan(5, 10), ".*");
  }
#endif // NDEBUG
}

TEST(DeviceGoogleTest05__UTILS_CUDA_Classes, CUDAMemoryWrappers) {
  DeviceGoogleTest05__UTILS_CUDA_Classes::executeTest();
}

void DeviceGoogleTest06__UTILS_CUDA_Classes::executeTest() {
  // create below the CUDA driver info for testing the GPU(s) with optional CUDA
  // profiling enabled
  const CUDADriverInfo cudaDriverInfo(cudaDeviceScheduleAuto, true);

  constexpr size_t ARRAY1_SIZE = 10;
  constexpr size_t ARRAY2_SIZE = 10000000;

  // CUDA Tuning Guide Note: As was already the recommended best practice,
  // signed arithmetic should be preferred over unsigned arithmetic wherever
  // possible for best throughput on SMM. The C language standard places more
  // restrictions on overflow behavior for unsigned math, limiting compiler
  // optimization opportunities. As a result, the float32/64 -> uint32/64
  // conversions fail on pre-Turing GPU hardware
  if (cudaDriverInfo.isAtLeastGPUType(CUDADriverInfo::GPUTypes::TURING,
                                      0)) // 1st GPU only runs this test
  {
    // float32 -> uint32 ARRAY1_SIZE representation case
    {
      array<float, ARRAY1_SIZE> hostFloat32Values = {
          {8687325.5535f, 242.24835f, 1.67857f, -0.7749938f, 3.334463f,
           -45675.768f, -141.64555f, 145.89f, 3.14f, -3.14f}};
      DeviceMemory<float> float32Values(ARRAY1_SIZE);
      float32Values.copyHostToDevice(hostFloat32Values.data());
      HostDeviceMemory<uint32_t> uint32Values(ARRAY1_SIZE);
      CUDAParallelFor::launchCUDAParallelFor(
          ARRAY1_SIZE,
          [] __device__(size_t index, const float *__restrict float32Values,
                        uint32_t *__restrict uint32Values) {
            uint32Values[index] =
                CUDAUtilityFunctions::float32Flip(float32Values[index]);
          },
          float32Values.device(), uint32Values.device());
      uint32Values.copyDeviceToHost();
      CUDAParallelFor::launchCUDAParallelFor(
          ARRAY1_SIZE,
          [] __device__(size_t index, const uint32_t *__restrict uint32Values,
                        float *__restrict float32Values) {
            float32Values[index] =
                CUDAUtilityFunctions::float32Unflip(uint32Values[index]);
          },
          uint32Values.device(), float32Values.device());
      float32Values.copyDeviceToHost(hostFloat32Values.data());
      for (size_t i = 0; i < ARRAY1_SIZE; ++i) {
        const bool float32Result =
            MathFunctions::equal(MathFunctions::float32Unflip(uint32Values[i]),
                                 hostFloat32Values[i]);
        EXPECT_TRUE(float32Result);
        if (!float32Result) {
          DebugConsole_consoleOutLine(
              "Failed with float32 numbers not being equal: ",
              MathFunctions::float32Unflip(uint32Values[i]), " & ",
              hostFloat32Values[i]);
          return;
        }
      }

      StdAuxiliaryFunctions::insertionSort<ARRAY1_SIZE>(
          uint32Values.host()); // sort an array using insertion sort with a
                                // constant small size of N
      float minFloat32Value = numeric_limits<float>::max();
      float maxFloat32Value = numeric_limits<float>::min();
      for (size_t i = 0; i < ARRAY1_SIZE; ++i) {
        hostFloat32Values[i] = MathFunctions::float32Unflip(uint32Values[i]);
        if (minFloat32Value > hostFloat32Values[i]) {
          minFloat32Value = hostFloat32Values[i]; // global min
        }
        if (maxFloat32Value < hostFloat32Values[i]) {
          maxFloat32Value = hostFloat32Values[i]; // global max
        }
      }
      const bool float32Result1 = MathFunctions::equal(
          minFloat32Value,
          hostFloat32Values[0]); // global min should match the first element of
                                 // the sorted array
      EXPECT_TRUE(float32Result1);
      if (!float32Result1) {
        return;
      }
      const bool float32Result2 = MathFunctions::equal(
          maxFloat32Value,
          hostFloat32Values[ARRAY1_SIZE -
                            1]); // global max should match the last element of
                                 // the sorted array
      EXPECT_TRUE(float32Result2);
      if (!float32Result2) {
        return;
      }
    }

    // float32 -> uint32 ARRAY2_SIZE representation case
    {
      auto float32Values = unique_ptr<float[]>(new float[ARRAY2_SIZE]);
      HostDeviceMemory<uint32_t> uint32Values(ARRAY2_SIZE);
      CUDAParallelFor::launchCUDAParallelFor(
          ARRAY2_SIZE,
          [] __device__(size_t index, uint32_t *__restrict uint32Values) {
            const float maxFloat32Value = numeric_limits<float>::max();
            uint32_t seed =
                CUDAUtilityFunctions::seedGenerator(index, index * index);
            const float randomValue = CUDAUtilityFunctions::rand1f(seed);
            uint32Values[index] = CUDAUtilityFunctions::float32Flip(
                maxFloat32Value * randomValue - maxFloat32Value / 2.0f);
          },
          uint32Values.device());
      uint32Values.copyDeviceToHost();
      sort(uint32Values.host(), uint32Values.host() + ARRAY2_SIZE);
      float minFloat32Value = numeric_limits<float>::max();
      float maxFloat32Value = numeric_limits<float>::min();
      for (size_t i = 0; i < ARRAY2_SIZE; ++i) {
        float32Values[i] = MathFunctions::float32Unflip(uint32Values[i]);
        if (minFloat32Value > float32Values[i]) {
          minFloat32Value = float32Values[i]; // global min
        }
        if (maxFloat32Value < float32Values[i]) {
          maxFloat32Value = float32Values[i]; // global max
        }
      }
      const bool float32Result1 = MathFunctions::equal(
          minFloat32Value,
          float32Values[0]); // global min should match the first element of the
                             // sorted array
      EXPECT_TRUE(float32Result1);
      if (!float32Result1) {
        return;
      }
      const bool float32Result2 = MathFunctions::equal(
          maxFloat32Value,
          float32Values[ARRAY2_SIZE - 1]); // global max should match the last
                                           // element of the sorted array
      EXPECT_TRUE(float32Result2);
      if (!float32Result2) {
        return;
      }

      // check global min/max via the CUDA kernel below
      array<uint32_t, 2> hostAtomicMinMaxUint32 = {
          {numeric_limits<uint32_t>::max(), numeric_limits<uint32_t>::min()}};
      DeviceMemory<uint32_t> atomicMinMaxUint32Memory(2);
      atomicMinMaxUint32Memory.copyHostToDevice(hostAtomicMinMaxUint32.data());
      CUDAParallelFor::launchCUDAParallelFor(
          ARRAY2_SIZE,
          [] __device__(size_t index, uint32_t *__restrict atomicMinMaxUint32,
                        const uint32_t *__restrict uint32Values) {
            atomicMin(&atomicMinMaxUint32[0],
                      uint32Values[index]); // global atomic min
            atomicMax(&atomicMinMaxUint32[1],
                      uint32Values[index]); // global atomic max
          },
          atomicMinMaxUint32Memory.device(), uint32Values.device());
      atomicMinMaxUint32Memory.copyDeviceToHost(hostAtomicMinMaxUint32.data());
      const bool float32Result3 = MathFunctions::equal(
          MathFunctions::float32Unflip(hostAtomicMinMaxUint32[0]),
          float32Values[0]); // global atomic min should match the first element
                             // of the sorted array
      EXPECT_TRUE(float32Result3);
      if (!float32Result3) {
        return;
      }
      const bool float32Result4 = MathFunctions::equal(
          MathFunctions::float32Unflip(hostAtomicMinMaxUint32[1]),
          float32Values[ARRAY2_SIZE - 1]); // global atomic max should match the
                                           // last element of the sorted array
      EXPECT_TRUE(float32Result4);
      if (!float32Result4) {
        return;
      }
    }

    // float64 -> uint64 ARRAY1_SIZE representation case
    {
      array<double, ARRAY1_SIZE> hostFloat64Values = {
          {868242342347325.553534534535, 242.27482143432435,
           1.677567567567567857, -0.77497567567567567938, 3.3357567756764463,
           -4565756775.7171768, -14127471.6455646455, 1411671715.8535345349,
           3.143534534534, -3.145235435435}};
      DeviceMemory<double> float64Values(ARRAY1_SIZE);
      float64Values.copyHostToDevice(hostFloat64Values.data());
      HostDeviceMemory<uint64_t> uint64Values(ARRAY1_SIZE);
      CUDAParallelFor::launchCUDAParallelFor(
          ARRAY1_SIZE,
          [] __device__(size_t index, const double *__restrict float64Values,
                        uint64_t *__restrict uint64Values) {
            uint64Values[index] =
                CUDAUtilityFunctions::float64Flip(float64Values[index]);
          },
          float64Values.device(), uint64Values.device());
      uint64Values.copyDeviceToHost();
      CUDAParallelFor::launchCUDAParallelFor(
          ARRAY1_SIZE,
          [] __device__(size_t index, const uint64_t *__restrict uint64Values,
                        double *__restrict float64Values) {
            float64Values[index] =
                CUDAUtilityFunctions::float64Unflip(uint64Values[index]);
          },
          uint64Values.device(), float64Values.device());
      float64Values.copyDeviceToHost(hostFloat64Values.data());
      for (size_t i = 0; i < ARRAY1_SIZE; ++i) {
        const bool float64Result =
            MathFunctions::equal(MathFunctions::float64Unflip(uint64Values[i]),
                                 hostFloat64Values[i]);
        EXPECT_TRUE(float64Result);
        if (!float64Result) {
          DebugConsole_consoleOutLine(
              "Failed with float64 numbers not being equal: ",
              MathFunctions::float64Unflip(uint64Values[i]), " & ",
              hostFloat64Values[i]);
          return;
        }
      }
      StdAuxiliaryFunctions::insertionSort<ARRAY1_SIZE>(
          uint64Values.host()); // sort an array using insertion sort with a
                                // constant small size of N
      double minFloat64Value = numeric_limits<double>::max();
      double maxFloat64Value = numeric_limits<double>::min();
      for (size_t i = 0; i < ARRAY1_SIZE; ++i) {
        hostFloat64Values[i] = MathFunctions::float64Unflip(uint64Values[i]);
        if (minFloat64Value > hostFloat64Values[i]) {
          minFloat64Value = hostFloat64Values[i]; // global min
        }
        if (maxFloat64Value < hostFloat64Values[i]) {
          maxFloat64Value = hostFloat64Values[i]; // global max
        }
      }
      const bool float64Result1 = MathFunctions::equal(
          minFloat64Value,
          hostFloat64Values[0]); // global min should match the first element of
                                 // the sorted array
      EXPECT_TRUE(float64Result1);
      if (!float64Result1) {
        return;
      }
      const bool float64Result2 = MathFunctions::equal(
          maxFloat64Value,
          hostFloat64Values[ARRAY1_SIZE -
                            1]); // global max should match the last element of
                                 // the sorted array
      EXPECT_TRUE(float64Result2);
      if (!float64Result2) {
        return;
      }
    }

#if __CUDA_ARCH__ >=                                                           \
    500 // Maxwell GPUs onwards for good atomicMin/Max 64bit support
    // float64 -> uint64 ARRAY2_SIZE representation case
    {
      auto float64Values = unique_ptr<double[]>(new double[ARRAY2_SIZE]);
      HostDeviceMemory<uint64_t> uint64Values(ARRAY2_SIZE);
      CUDAParallelFor::launchCUDAParallelFor(
          ARRAY2_SIZE,
          [] __device__(size_t index, uint64_t *__restrict uint64Values) {
            const double maxFloat64Value = numeric_limits<double>::max();
            uint32_t seed =
                CUDAUtilityFunctions::seedGenerator(index, index * index);
            const double randomValue =
                double(CUDAUtilityFunctions::rand1f(seed));
            uint64Values[index] = CUDAUtilityFunctions::float64Flip(
                maxFloat64Value * randomValue - maxFloat64Value / 2.0f);
          },
          uint64Values.device());
      uint64Values.copyDeviceToHost();
      sort(uint64Values.host(), uint64Values.host() + ARRAY2_SIZE);
      double minFloat64Value = numeric_limits<double>::max();
      double maxFloat64Value = numeric_limits<double>::min();
      for (size_t i = 0; i < ARRAY2_SIZE; ++i) {
        float64Values[i] = MathFunctions::float64Unflip(uint64Values[i]);
        if (minFloat64Value > float64Values[i]) {
          minFloat64Value = float64Values[i]; // global min
        }
        if (maxFloat64Value < float64Values[i]) {
          maxFloat64Value = float64Values[i]; // global max
        }
      }
      const bool float64Result1 = MathFunctions::equal(
          minFloat64Value,
          float64Values[0]); // global min should match the first element of the
                             // sorted array
      EXPECT_TRUE(float64Result1);
      if (!float64Result1) {
        return;
      }
      const bool float64Result2 = MathFunctions::equal(
          maxFloat64Value,
          float64Values[ARRAY2_SIZE - 1]); // global max should match the last
                                           // element of the sorted array
      EXPECT_TRUE(float64Result2);
      if (!float64Result2) {
        return;
      }

      // check global min/max via the CUDA kernel below
      array<uint64_t, 2> hostAtomicMinMaxUint64 = {
          {numeric_limits<uint64_t>::max(), numeric_limits<uint64_t>::min()}};
      DeviceMemory<uint64_t> atomicMinMaxUint64Values(2);
      atomicMinMaxUint64Values.copyHostToDevice(hostAtomicMinMaxUint64.data());
      CUDAParallelFor::launchCUDAParallelFor(
          ARRAY2_SIZE,
          [] __device__(size_t index,
                        uint64_t *__restrict atomicMinMaxUint64Values,
                        const uint64_t *__restrict uint64Values) {
            setAtomicMinMaxUint64(index, atomicMinMaxUint64Values,
                                  uint64Values);
          },
          atomicMinMaxUint64Values.device(), uint64Values.device());
      atomicMinMaxUint64Values.copyDeviceToHost(hostAtomicMinMaxUint64.data());
      const bool float64Result3 = MathFunctions::equal(
          MathFunctions::float64Unflip(hostAtomicMinMaxUint64[0]),
          float64Values[0]); // global atomic min should match the first element
                             // of the sorted array
      EXPECT_TRUE(float64Result3);
      if (!float64Result3) {
        return;
      }
      const bool float64Result4 = MathFunctions::equal(
          MathFunctions::float64Unflip(hostAtomicMinMaxUint64[1]),
          float64Values[ARRAY2_SIZE - 1]); // global atomic max should match the
                                           // last element of the sorted array
      EXPECT_TRUE(float64Result4);
      if (!float64Result4) {
        return;
      }
    }
#endif // __CUDA_ARCH__
  }
}

TEST(DeviceGoogleTest06__UTILS_CUDA_Classes, CUDAUtilityFunctions) {
  DeviceGoogleTest06__UTILS_CUDA_Classes::executeTest();
}

void DeviceGoogleTest07__UTILS_CUDA_Classes::executeTest() {
  // create below the CUDA driver info for testing the GPU(s) with optional CUDA
  // profiling enabled
  const CUDADriverInfo cudaDriverInfo(cudaDeviceScheduleAuto, true);

  const size_t numberOfElements =
      30 * 1024 * 1024; // 30Mb indices x4 bytes for floats
  const int device = 0;
  const size_t numberOfCUDAStreams = 1;

  const CUDAStreamsHandler cudaStreamsHandler(cudaDriverInfo, device,
                                              numberOfCUDAStreams);
  cudaStreamsHandler.addCallback(0, testStreamCallback, nullptr);
  const size_t threadLocalSize = numberOfHardwareThreads();
  // initialize thread pool with default parameters
  ThreadPool threadPool(threadLocalSize, AFFINITY_MASK_ALL, PRIORITY_NONE);

  // initialize thread local storage data
  const auto threadLocalStorage =
      make_unique<RandomRNGWELL512[]>(threadLocalSize);

  // float32 representation case
  {
    HostDeviceMemory<float> cudaMemoryHandler;
    auto future1 = cudaMemoryHandler.allocateAsync(numberOfElements, device);
    const auto checkHostPtr = make_unique<float[]>(numberOfElements);
    // wait until the reallyAsync() upload above is done
    future1.wait();
    const auto hostPtr = cudaMemoryHandler.host();
    parallelForThreadLocal(
        0, numberOfElements,
        [&](size_t i, size_t threadIdx) {
          checkHostPtr[i] = hostPtr[i] =
              2.0f * float(threadLocalStorage[threadIdx]()) -
              1.0f; // -1.0f - +1.0f range
        },
        threadPool);
    cudaMemoryHandler.copyHostToDeviceAsync(cudaStreamsHandler[0]);
    // run CPU atomicMinMaxFloat32 kernel
    atomic<float> atomicMinFloat32(numeric_limits<float>::max());
    atomic<float> atomicMaxFloat32(numeric_limits<float>::min());
    parallelFor(
        0, numberOfElements,
        [&](size_t i) {
          // below a na�ve but thread-safe way of finding min/max in parallel
          CPUParallelismUtilityFunctions::atomicMin(
              atomicMinFloat32, checkHostPtr[i]); // global atomic min
          CPUParallelismUtilityFunctions::atomicMax(
              atomicMaxFloat32, checkHostPtr[i]); // global atomic max
        },
        threadPool);
    // run GPU atomicMinMaxFloat32 kernel
    array<float, 2> hostAtomicMinMaxFloat32 = {
        {numeric_limits<float>::max(), numeric_limits<float>::min()}};
    DeviceMemory<float> atomicMinMaxFloat32Values(2);
    atomicMinMaxFloat32Values.copyHostToDeviceAsync(
        hostAtomicMinMaxFloat32.data(), cudaStreamsHandler[0]);
    {
      ProfileGPUTimer profileGPUTimer(
          "parallelForCUDAAtomicMinMaxFloat32<<<>>> kernel time taken:", device,
          cudaStreamsHandler[0]);
      CUDAParallelFor::launchCUDAParallelForInStream(
          numberOfElements, 0, cudaStreamsHandler[0],
          [] __device__(size_t index,
                        float *__restrict atomicMinMaxFloat32Values,
                        const float *__restrict float32Values) {
            CUDAUtilityDeviceFunctions::atomicMin(
                &atomicMinMaxFloat32Values[0],
                float32Values[index]); // global atomic min (works with
                                       // shared-memory for Maxwell GPUs &
                                       // onwards which have native
                                       // shared-memory atomics support)
            CUDAUtilityDeviceFunctions::atomicMax(
                &atomicMinMaxFloat32Values[1],
                float32Values[index]); // global atomic max (works with
                                       // shared-memory for Maxwell GPUs &
                                       // onwards which have native
                                       // shared-memory atomics support)
          },
          atomicMinMaxFloat32Values.device(), cudaMemoryHandler.device());
    }
    // note the non-async function usage below
    atomicMinMaxFloat32Values.copyDeviceToHost(hostAtomicMinMaxFloat32.data(),
                                               cudaStreamsHandler[0]);
    cudaMemoryHandler.copyDeviceToHost(cudaStreamsHandler[0]);
    // test results
    const bool float32MinResult = MathFunctions::equal(
        atomicMinFloat32.load(), hostAtomicMinMaxFloat32[0]);
    const bool float32MaxResult = MathFunctions::equal(
        atomicMaxFloat32.load(), hostAtomicMinMaxFloat32[1]);
    EXPECT_TRUE(float32MinResult && float32MaxResult);

    DebugConsole_consoleOutLine("Float32 CPU-vs-GPU verification succeeded: ",
                                StringAuxiliaryFunctions::toString<bool>(
                                    float32MinResult && float32MaxResult));
  }

  // float64 representation case
  {
    HostDeviceMemory<double> cudaMemoryHandler;
    auto future1 = cudaMemoryHandler.allocateAsync(numberOfElements, device);
    const auto checkHostPtr = make_unique<double[]>(numberOfElements);
    // wait until the reallyAsync() upload above is done
    future1.wait();
    const auto hostPtr = cudaMemoryHandler.host();
    parallelForThreadLocal(
        0, numberOfElements,
        [&](size_t i, size_t threadIdx) {
          checkHostPtr[i] = hostPtr[i] =
              2.0 * threadLocalStorage[threadIdx]() - 1.0; // -1.0 - +1.0 range
        },
        threadPool);
    cudaMemoryHandler.copyHostToDeviceAsync(cudaStreamsHandler[0]);
    // run CPU atomicMinMaxFloat64 kernel
    atomic<double> atomicMinFloat64(numeric_limits<double>::max());
    atomic<double> atomicMaxFloat64(numeric_limits<double>::min());
    parallelFor(
        0, numberOfElements,
        [&](size_t i) {
          // below a na�ve but thread-safe way of finding min/max in parallel
          CPUParallelismUtilityFunctions::atomicMin(
              atomicMinFloat64, checkHostPtr[i]); // global atomic min
          CPUParallelismUtilityFunctions::atomicMax(
              atomicMaxFloat64, checkHostPtr[i]); // global atomic max
        },
        threadPool);
    // run GPU atomicMinMaxFloat64 kernel
    array<double, 2> hostAtomicMinMaxFloat64 = {
        {numeric_limits<double>::max(), numeric_limits<double>::min()}};
    DeviceMemory<double> atomicMinMaxFloat64Values(2);
    atomicMinMaxFloat64Values.copyHostToDeviceAsync(
        hostAtomicMinMaxFloat64.data(), cudaStreamsHandler[0]);
    {
      ProfileGPUTimer profileGPUTimer(
          "parallelForCUDAAtomicMinMaxFloat64<<<>>> kernel time taken:", device,
          cudaStreamsHandler[0]);
      CUDAParallelFor::launchCUDAParallelForInStream(
          numberOfElements, 0, cudaStreamsHandler[0],
          [] __device__(size_t index,
                        double *__restrict atomicMinMaxFloat64Values,
                        const double *__restrict float64Values) {
            CUDAUtilityDeviceFunctions::atomicMin(
                &atomicMinMaxFloat64Values[0],
                float64Values[index]); // global atomic min (works with
                                       // shared-memory for Maxwell GPUs &
                                       // onwards which have native
                                       // shared-memory atomics support)
            CUDAUtilityDeviceFunctions::atomicMax(
                &atomicMinMaxFloat64Values[1],
                float64Values[index]); // global atomic max (works with
                                       // shared-memory for Maxwell GPUs &
                                       // onwards which have native
                                       // shared-memory atomics support)
          },
          atomicMinMaxFloat64Values.device(), cudaMemoryHandler.device());
    }
    // note the non-async function usage below
    atomicMinMaxFloat64Values.copyDeviceToHost(hostAtomicMinMaxFloat64.data(),
                                               cudaStreamsHandler[0]);
    cudaMemoryHandler.copyDeviceToHost(cudaStreamsHandler[0]);
    // test results
    const bool float64MinResult = MathFunctions::equal(
        atomicMinFloat64.load(), hostAtomicMinMaxFloat64[0]);
    const bool float64MaxResult = MathFunctions::equal(
        atomicMaxFloat64.load(), hostAtomicMinMaxFloat64[1]);
    EXPECT_TRUE(float64MinResult && float64MaxResult);

    DebugConsole_consoleOutLine("Float64 CPU-vs-GPU verification succeeded: ",
                                StringAuxiliaryFunctions::toString<bool>(
                                    float64MinResult && float64MaxResult));
  }

  // float32 -> uint32 representation case
  {
    HostDeviceMemory<uint32_t> cudaMemoryHandler;
    auto future1 = cudaMemoryHandler.allocateAsync(numberOfElements);
    const auto checkHostPtr = make_unique<uint32_t[]>(numberOfElements);
    // wait until the reallyAsync() upload above is done
    future1.wait();
    const auto hostPtr = cudaMemoryHandler.host();
    parallelForThreadLocal(
        0, numberOfElements,
        [&](size_t i, size_t threadIdx) {
          checkHostPtr[i] = hostPtr[i] = CUDAUtilityFunctions::float32Flip(
              2.0f * float(threadLocalStorage[threadIdx]()) -
              1.0f); // -1.0f - +1.0f range
        },
        threadPool);
    cudaMemoryHandler.copyHostToDeviceAsync(cudaStreamsHandler[0]);
    // run CPU atomicMinMaxFloat32 kernel
    atomic<uint32_t> atomicMinUint32(numeric_limits<uint32_t>::max());
    atomic<uint32_t> atomicMaxUint32(numeric_limits<uint32_t>::min());
    parallelFor(
        0, numberOfElements,
        [&](size_t i) {
          // below a na�ve but thread-safe way of finding min/max in parallel
          CPUParallelismUtilityFunctions::atomicMin(
              atomicMinUint32, checkHostPtr[i]); // global atomic min
          CPUParallelismUtilityFunctions::atomicMax(
              atomicMaxUint32, checkHostPtr[i]); // global atomic max
        },
        threadPool);
    // run GPU atomicMinMaxUint32 kernel
    array<uint32_t, 2> hostAtomicMinMaxUint32 = {
        {numeric_limits<uint32_t>::max(), numeric_limits<uint32_t>::min()}};
    DeviceMemory<uint32_t> atomicMinMaxUint32Memory(2);
    atomicMinMaxUint32Memory.copyHostToDeviceAsync(
        hostAtomicMinMaxUint32.data(), cudaStreamsHandler[0]);
    {
      ProfileGPUTimer profileGPUTimer(
          "parallelForCUDAAtomicMinMaxUint32<<<>>> kernel time taken:", device,
          cudaStreamsHandler[0]);
      CUDAParallelFor::launchCUDAParallelForInStream(
          numberOfElements, 0, cudaStreamsHandler[0],
          [] __device__(size_t index, uint32_t *__restrict atomicMinMaxUint32,
                        const uint32_t *__restrict uint32Values) {
            atomicMin(&atomicMinMaxUint32[0],
                      uint32Values[index]); // global atomic min
            atomicMax(&atomicMinMaxUint32[1],
                      uint32Values[index]); // global atomic max
          },
          atomicMinMaxUint32Memory.device(), cudaMemoryHandler.device());
    }
    // note the non-async function usage here
    atomicMinMaxUint32Memory.copyDeviceToHost(hostAtomicMinMaxUint32.data(),
                                              cudaStreamsHandler[0]);
    cudaMemoryHandler.copyDeviceToHost(cudaStreamsHandler[0]);
    // test results
    const bool uint32MinResult1 = MathFunctions::equal(
        CUDAUtilityFunctions::float32Unflip(atomicMinUint32.load()),
        CUDAUtilityFunctions::float32Unflip(hostAtomicMinMaxUint32[0]));
    const bool uint32MaxResult1 = MathFunctions::equal(
        CUDAUtilityFunctions::float32Unflip(atomicMaxUint32.load()),
        CUDAUtilityFunctions::float32Unflip(hostAtomicMinMaxUint32[1]));
    const bool uint32MinResult2 =
        (atomicMinUint32.load() == hostAtomicMinMaxUint32[0]);
    const bool uint32MaxResult2 =
        (atomicMaxUint32.load() == hostAtomicMinMaxUint32[1]);
    EXPECT_TRUE(uint32MinResult1 && uint32MaxResult1 && uint32MinResult2 &&
                uint32MaxResult2);

    DebugConsole_consoleOutLine("Uint32 CPU-vs-GPU verification succeeded: ",
                                StringAuxiliaryFunctions::toString<bool>(
                                    uint32MinResult1 && uint32MaxResult1 &&
                                    uint32MinResult2 && uint32MaxResult2));
  }

#if __CUDA_ARCH__ >=                                                           \
    500 // Maxwell GPUs onwards for good atomicMin/Max 64bit support
  // float64 -> uint64 representation case
  {
    HostDeviceMemory<uint64_t> cudaMemoryHandler;
    auto future1 = cudaMemoryHandler.allocateAsync(numberOfElements);
    const auto checkHostPtr = make_unique<uint64_t[]>(numberOfElements);
    // wait until the reallyAsync() upload above is done
    future1.wait();
    const auto hostPtr = cudaMemoryHandler.host();
    parallelForThreadLocal(
        0, numberOfElements,
        [&](size_t i, size_t threadIdx) {
          checkHostPtr[i] = hostPtr[i] = CUDAUtilityFunctions::float64Flip(
              2.0 * threadLocalStorage[threadIdx]() -
              1.0); // -1.0f - +1.0f range
        },
        threadPool);
    cudaMemoryHandler.copyHostToDeviceAsync(cudaStreamsHandler[0]);
    // run CPU atomicMinMaxFloat64 kernel
    atomic<uint64_t> atomicMinUint64(numeric_limits<uint64_t>::max());
    atomic<uint64_t> atomicMaxUint64(numeric_limits<uint64_t>::min());
    parallelFor(
        0, numberOfElements,
        [&](size_t i) {
          // below a na�ve but thread-safe way of finding min/max in parallel
          CPUParallelismUtilityFunctions::atomicMin(
              atomicMinUint64, checkHostPtr[i]); // global atomic min
          CPUParallelismUtilityFunctions::atomicMax(
              atomicMaxUint64, checkHostPtr[i]); // global atomic max
        },
        threadPool);
    // run GPU atomicMinMaxUint64 kernel
    array<uint64_t, 2> hostAtomicMinMaxUint64 = {
        {numeric_limits<uint64_t>::max(), numeric_limits<uint64_t>::min()}};
    DeviceMemory<uint64_t> atomicMinMaxUint64Values(2);
    atomicMinMaxUint64Values.copyHostToDeviceAsync(
        hostAtomicMinMaxUint64.data(), cudaStreamsHandler[0]);
    {
      ProfileGPUTimer profileGPUTimer(
          "parallelForCUDAAtomicMinMaxUint64<<<>>> kernel time taken:", device,
          cudaStreamsHandler[0]);
      CUDAParallelFor::launchCUDAParallelForInStream(
          numberOfElements, 0, cudaStreamsHandler[0],
          [] __device__(size_t index,
                        uint64_t *__restrict atomicMinMaxUint64Values,
                        const uint64_t *__restrict uint64Values) {
            setAtomicMinMaxUint64(index, atomicMinMaxUint64Values,
                                  uint64Values);
          },
          atomicMinMaxUint64Values.device(), cudaMemoryHandler.device());
    }
    // note the non-async function usage here
    atomicMinMaxUint64Values.copyDeviceToHost(hostAtomicMinMaxUint64.data(),
                                              cudaStreamsHandler[0]);
    cudaMemoryHandler.copyDeviceToHost(cudaStreamsHandler[0]);
    // test results
    const bool uint64MinResult1 = MathFunctions::equal(
        CUDAUtilityFunctions::float64Unflip(atomicMinUint64.load()),
        CUDAUtilityFunctions::float64Unflip(hostAtomicMinMaxUint64[0]));
    const bool uint64MaxResult1 = MathFunctions::equal(
        CUDAUtilityFunctions::float64Unflip(atomicMaxUint64.load()),
        CUDAUtilityFunctions::float64Unflip(hostAtomicMinMaxUint64[1]));
    const bool uint64MinResult2 =
        (atomicMinUint64.load() == hostAtomicMinMaxUint64[0]);
    const bool uint64MaxResult2 =
        (atomicMaxUint64.load() == hostAtomicMinMaxUint64[1]);
    EXPECT_TRUE(uint64MinResult1 && uint64MaxResult1 && uint64MinResult2 &&
                uint64MaxResult2);

    DebugConsole_consoleOutLine("Uint64 CPU-vs-GPU verification succeeded: ",
                                StringAuxiliaryFunctions::toString<bool>(
                                    uint64MinResult1 && uint64MaxResult1 &&
                                    uint64MinResult2 && uint64MaxResult2));
  }
#endif // __CUDA_ARCH__
}

TEST(DeviceGoogleTest07__UTILS_CUDA_Classes, CUDADeviceUtilityFunctions) {
  DeviceGoogleTest07__UTILS_CUDA_Classes::executeTest();
}

void DeviceGoogleTest08__UTILS_CUDA_Classes::executeTest() {
  // create below the CUDA driver info for testing the GPU(s) with optional CUDA
  // profiling enabled
  const CUDADriverInfo cudaDriverInfo(cudaDeviceScheduleAuto, true);
  const CUDAStreamsHandler cudaStreamsHandler(cudaDriverInfo);
  cudaStreamsHandler.addCallback(0, testStreamCallback, nullptr);

  // Part 1 Test: BasicLaunch
  {
    HostDeviceMemory<size_t> memory(10);
    // we create a raw memory view on it
    RawDeviceMemory<size_t> rawMemory(memory.device());
    // we pass that to a kernel and change it (each thread writes its index)
    KernelLauncher::create().setBlock(10).runCUDAParallelFor(
        10,
        [] __device__(size_t index, const RawDeviceMemory<size_t> &memory) {
          memory[index] = index;
        },
        rawMemory);
    memory.copyDeviceToHost();
    checkArray<size_t, 10>(memory.host());
  }

  // Part 2 Test: MultipleBlocks
  {
    HostDeviceMemory<size_t> memory(10);
    Span<size_t> span(memory.device(), 10);
    // we pass that to a kernel and change it (each thread writes its index)
    KernelLauncher::create()
        .setGrid(10)
        .setBlock({1, 1})
        .setStream(nullptr)
        .synchronized()
        .runCUDAParallelFor(
            10,
            [] __device__(size_t index, const Span<size_t> &memory) {
              memory[index] = index;
            },
            span);
    memory.copyDeviceToHost();
    checkArray<size_t, 10>(memory.host());
  }

  // Part 3 Test: OtherStream
  {
    HostDeviceMemory<size_t> memory(10);
    Span<size_t, 10> span(memory.device());
    // we pass that to a kernel and change it (each thread writes its index)
    KernelLauncher::create()
        .setGrid(1)
        .setBlock(1)
        .setStream(cudaStreamsHandler[0])
        .synchronized()
        .runCUDAParallelFor(
            1,
            [] __device__(size_t, const Span<size_t, 10> &memory) {
              size_t idx = 0;
              for (auto &i : memory) {
                i = idx++;
              }
            },
            span);
    // note the non-async function usage below
    memory.copyDeviceToHost(cudaStreamsHandler[0]);
    checkArray<size_t, 10>(memory.host());
  }

  // Part 4 Test: MultipleArguments with given grid & block sizes
  {
    HostDeviceMemory<float> memory(10);
    HostDeviceMemory<float> a(10);
    HostDeviceMemory<size_t> b(10);
    fill(a.host(), a.host() + 10, 3.0f);
    fill(b.host(), b.host() + 10, 3);
    a.copyHostToDevice();
    b.copyHostToDevice();
    dim3 blocks;
    dim3 threads;
    tie(blocks, threads) =
        CUDAUtilityFunctions::calculateCUDA1DKernelDimensions(10);
    KernelLauncher::create()
        .setGrid(blocks)
        .setBlock(threads)
        .setStream(cudaStreamsHandler[0])
        .synchronized()
        .run(kernelTestRun, 10, memory.device(), a.device(), b.device());
    // note the non-async function usage below
    memory.copyDeviceToHost(cudaStreamsHandler[0]);
    checkArray<float, 10, true>(memory.host(), 6.0f);
  }

  // Part 5 Test: GlobalIndex
  {
    HostDeviceMemory<size_t> memory(64);
    // we pass that to a kernel and change it (each thread writes its index)
    KernelLauncher::create()
        .setGrid(8)
        .setBlock(8)
        .setStream(cudaStreamsHandler[0])
        .synchronized()
        .runCUDAParallelFor(
            64,
            [] __device__(size_t index, size_t *__restrict__ memory) {
              memory[index] = CUDAUtilityDeviceFunctions::globalLinearIndex();
            },
            memory.device());
    // note the non-async function usage below
    memory.copyDeviceToHost(cudaStreamsHandler[0]);
    parallelFor(0, 64, [&](size_t i) {
      checkArray<size_t, 1, true>(memory.host() + i, i);
    });
  }
}

TEST(DeviceGoogleTest08__UTILS_CUDA_Classes, CUDAKernelLauncher) {
  DeviceGoogleTest08__UTILS_CUDA_Classes::executeTest();
}

void DeviceGoogleTest09__UTILS_CUDA_Classes::executeTest() {
  // create below the CUDA driver info for testing the GPU(s) with optional CUDA
  // profiling enabled
  const CUDADriverInfo cudaDriverInfo(cudaDeviceScheduleAuto, true);

  // explicit CUDAMemoryRegistry test with host/device memory
  // register/allocation respectively
  constexpr size_t ARRAY_SIZE = 1000000;
  const int device = 0;

  // the CUDAMemoryPool test
  CUDAMemoryPool cudaMemoryPool(cudaDriverInfo,
                                false); // proper memory pool usage
  CUDAMemoryPool cudaMemoryPoolSeparateAllocations(
      cudaDriverInfo, true); // separate allocations, for testing
  testCUDAMemoryPool(cudaMemoryPool, ARRAY_SIZE, device);
  testCUDAMemoryPool(cudaMemoryPoolSeparateAllocations, ARRAY_SIZE, device);

  // the CUDAProcessMemoryPool test
  CUDAProcessMemoryPool cudaProcessMemoryPool(
      cudaDriverInfo, false); // proper process memory pool usage
  CUDAProcessMemoryPool cudaProcessMemoryPoolSeparateAllocations(
      cudaDriverInfo, true); // separate allocations, for testing
  testCUDAProcessMemoryPool(cudaDriverInfo, cudaProcessMemoryPool, ARRAY_SIZE,
                            device);
  testCUDAProcessMemoryPool(cudaDriverInfo,
                            cudaProcessMemoryPoolSeparateAllocations,
                            ARRAY_SIZE, device);
}

TEST(DeviceGoogleTest09__UTILS_CUDA_Classes, CUDAMemoryPool) {
  DeviceGoogleTest09__UTILS_CUDA_Classes::executeTest();
}

void DeviceGoogleTest10__UTILS_CUDA_Classes::executeTest() {
  // create below the CUDA driver info for testing the GPU(s) with optional CUDA
  // profiling enabled
  const CUDADriverInfo cudaDriverInfo(cudaDeviceScheduleAuto, true);
  CUDAEventTimer timer;

  EXPECT_TRUE(testFastMemset(0, 256, 256));
  EXPECT_TRUE(testFastMemset(15, 17, 256));
  EXPECT_TRUE(testFastMemset(1, 20, 256));
  EXPECT_TRUE(testFastMemset(10, 10, 100));

  const size_t TEST_SIZE = 8;
  array<size_t, TEST_SIZE> alignmentSizes = {
      {1, 7, 8, 9, 1107, 1024, 1024 * 1024, 10 * 1024 * 1024}};
  const size_t OFFSET_SIZE = 8;
  for (const auto alignmentSize : alignmentSizes) {
    DebugConsole_consoleOutLine("Alignment check for size: ", alignmentSize);
    for (size_t offset = 0; offset < OFFSET_SIZE; ++offset) {
      EXPECT_TRUE(verifyFastMemset(timer, offset, alignmentSize));
    }
  }
}

TEST(DeviceGoogleTest10__UTILS_CUDA_Classes, CUDAUtilityFunctionsMemset) {
  DeviceGoogleTest10__UTILS_CUDA_Classes::executeTest();
}

void DeviceGoogleTest11__UTILS_CUDA_Classes::executeTest() {
  // create below the CUDA driver info for testing the GPU(s) with optional CUDA
  // profiling enabled
  const CUDADriverInfo cudaDriverInfo(cudaDeviceScheduleAuto, true);

  // uint32_t-uint64_t tests
  {
    constexpr size_t TEST_EXPONENT_1 = 17;
    constexpr size_t TEST_EXPONENT_2 = 47;
    HostDeviceMemory<uint32_t> hostDeviceUint32Handler(1);
    HostDeviceMemory<uint64_t> hostDeviceUint64Handler(1);

    uint32_t groundTruthUint32 = uint32_t(2) << (TEST_EXPONENT_1 - 1);
    uint64_t groundTruthUint64 = uint64_t(2) << (TEST_EXPONENT_2 - 1);
    uint32_t resultUint32 =
        CUDAUtilityFunctions::pow<TEST_EXPONENT_1>(uint32_t(2));
    uint64_t resultUint64 =
        CUDAUtilityFunctions::pow<TEST_EXPONENT_2>(uint64_t(2));

    CUDAParallelFor::launchCUDAParallelFor(
        1,
        [] __device__(size_t, uint32_t *__restrict memoryUint32,
                      uint64_t *__restrict memoryUint64) {
          memoryUint32[0] =
              CUDAUtilityFunctions::pow<TEST_EXPONENT_1>(uint32_t(2));
          memoryUint64[0] =
              CUDAUtilityFunctions::pow<TEST_EXPONENT_2>(uint64_t(2));
        },
        hostDeviceUint32Handler.device(), hostDeviceUint64Handler.device());

    hostDeviceUint32Handler.copyDeviceToHost();
    hostDeviceUint64Handler.copyDeviceToHost();

    // check ground-truth vs. CPU
    EXPECT_TRUE(groundTruthUint32 == resultUint32);
    EXPECT_TRUE(groundTruthUint64 == resultUint64);

    // check CPU vs. GPU
    EXPECT_TRUE(resultUint32 == hostDeviceUint32Handler[0]);
    EXPECT_TRUE(resultUint64 == hostDeviceUint64Handler[0]);
  }

  // float-double tests
  {
    constexpr size_t TEST_EXPONENT_1 = 32;
    constexpr size_t TEST_EXPONENT_2 = 37;
    HostDeviceMemory<float> hostDeviceFloatHandler(2);
    HostDeviceMemory<double> hostDeviceDoubleHandler(2);

    float groundTruthFloat1 = powf(PI_FLT, TEST_EXPONENT_1);
    double groundTruthDouble1 = pow(PI_DBL, TEST_EXPONENT_1);
    float groundTruthFloat2 = powf(PI_FLT, TEST_EXPONENT_2);
    double groundTruthDouble2 = pow(PI_DBL, TEST_EXPONENT_2);
    float resultFloat1 = CUDAUtilityFunctions::pow<TEST_EXPONENT_1>(PI_FLT);
    double resultDouble1 = CUDAUtilityFunctions::pow<TEST_EXPONENT_1>(PI_DBL);
    float resultFloat2 = CUDAUtilityFunctions::pow<TEST_EXPONENT_2>(PI_FLT);
    double resultDouble2 = CUDAUtilityFunctions::pow<TEST_EXPONENT_2>(PI_DBL);

    CUDAParallelFor::launchCUDAParallelFor(
        1,
        [] __device__(size_t, float *__restrict memoryFloat,
                      double *__restrict memoryDouble) {
          memoryFloat[0] = CUDAUtilityFunctions::pow<TEST_EXPONENT_1>(PI_FLT);
          memoryDouble[0] = CUDAUtilityFunctions::pow<TEST_EXPONENT_1>(PI_DBL);
          memoryFloat[1] = CUDAUtilityFunctions::pow<TEST_EXPONENT_2>(PI_FLT);
          memoryDouble[1] = CUDAUtilityFunctions::pow<TEST_EXPONENT_2>(PI_DBL);
        },
        hostDeviceFloatHandler.device(), hostDeviceDoubleHandler.device());

    hostDeviceFloatHandler.copyDeviceToHost();
    hostDeviceDoubleHandler.copyDeviceToHost();

    // check ground-truth std:pow() vs. CPU
    EXPECT_TRUE(!UnitTestUtilityFunctions_flt::checkRelativeError(
        groundTruthFloat1, resultFloat1, float(1e-7)));
    EXPECT_TRUE(!UnitTestUtilityFunctions_dbl::checkRelativeError(
        groundTruthDouble1, resultDouble1, double(1e-15)));
    EXPECT_TRUE(!UnitTestUtilityFunctions_flt::checkRelativeError(
        groundTruthFloat2, resultFloat2, float(1e-6)));
    EXPECT_TRUE(!UnitTestUtilityFunctions_dbl::checkRelativeError(
        groundTruthDouble2, resultDouble2, double(1e-15)));

    // check CPU vs. GPU (totally equal to CPU due to GPU having zero ULP for
    // multiplications)
    EXPECT_TRUE(MathFunctions::equal(resultFloat1, hostDeviceFloatHandler[0]));
    EXPECT_TRUE(
        MathFunctions::equal(resultDouble1, hostDeviceDoubleHandler[0]));
    EXPECT_TRUE(MathFunctions::equal(resultFloat2, hostDeviceFloatHandler[1]));
    EXPECT_TRUE(
        MathFunctions::equal(resultDouble2, hostDeviceDoubleHandler[1]));
  }
}

TEST(DeviceGoogleTest11__UTILS_CUDA_Classes, CUDAUtilityFunctionsPow) {
  DeviceGoogleTest11__UTILS_CUDA_Classes::executeTest();
}

void DeviceGoogleTest12__UTILS_CUDA_Classes::executeTest() {
  // create below the CUDA driver info for testing the GPU(s) with optional CUDA
  // profiling enabled
  const CUDADriverInfo cudaDriverInfo(cudaDeviceScheduleAuto, true);

  // simple CUDAQueue memory copy test
  {
    constexpr size_t QUEUE_SIZE = 20;
    vector<float> points(QUEUE_SIZE);
    for (size_t i = 0; i < QUEUE_SIZE; ++i) {
      points[i] = float(i);
    }

    DeviceMemory<float> buffer(QUEUE_SIZE);
    buffer.copyHostToDevice(points.data());

    CUDAQueue<float> cudaQueue(QUEUE_SIZE);
    cudaQueue.push_back(buffer, QUEUE_SIZE);

    DeviceMemory<float> dstBuffer(QUEUE_SIZE);
    cudaQueue.front(dstBuffer, QUEUE_SIZE);

    vector<float> dstPoints(QUEUE_SIZE);
    dstBuffer.copyDeviceToHost(dstPoints.data());

    EXPECT_EQ(dstPoints, points);
  }

  // advanced CUDAQueue test
  {
    constexpr size_t BUFFER_SIZE = 100;
    vector<size_t> points(BUFFER_SIZE);
    for (size_t i = 0; i < BUFFER_SIZE; ++i) {
      points[i] = i;
    }

    DeviceMemory<size_t> buffer(BUFFER_SIZE);
    buffer.copyHostToDevice(points.data());

    size_t QUEUE_SIZE = 37;
    DeviceMemory<size_t> queueBuffer(QUEUE_SIZE);
    CUDAQueue<size_t> cudaQueue(queueBuffer, QUEUE_SIZE);

    cudaQueue.push_back(buffer, cudaQueue.reserved());
    size_t total = 0;
    for (size_t i = 0; i < QUEUE_SIZE; i += 7) {
      total += cudaQueue.pop_front(7);
    }

    EXPECT_EQ(total, cudaQueue.reserved());
    EXPECT_TRUE(cudaQueue.empty());

    for (size_t i = 0; i < 100; ++i) {
      cudaQueue.push_back(buffer, 12);
      cudaQueue.pop_front(10);
      cudaQueue.push_back(buffer, 8);
      cudaQueue.pop_front(10);
    }
    EXPECT_TRUE(cudaQueue.empty());

    constexpr size_t QUEUE_OFFSET = 10;
    cudaQueue.push_back(buffer, QUEUE_SIZE);
    DeviceMemory<size_t> dstBuffer(QUEUE_SIZE);
    cudaQueue.pop_front(QUEUE_OFFSET);
    cudaQueue.front(dstBuffer, QUEUE_SIZE);
    vector<size_t> b(QUEUE_SIZE);
    dstBuffer.copyDeviceToHost(b.data());
    b.resize(QUEUE_SIZE - QUEUE_OFFSET);
    vector<size_t> a(points.begin() + QUEUE_OFFSET,
                     points.begin() + QUEUE_SIZE);

    EXPECT_EQ(a, b);
  }
}

// void DeviceGoogleTest13__Color_Histogram_GPU::executeTest() {
//   // Create CUDA driver info for testing the GPU
//   const CUDADriverInfo cudaDriverInfo(cudaDeviceScheduleAuto);

//   // Load test image
//   vector<uint8_t> imageData;
//   unsigned width, height;
//   const string currentPath =
//       Utils::UtilityFunctions::StdReadWriteFileFunctions::getCurrentPath();
//   const string inputFile =
//       string(currentPath + "/" + "Assets" + "/" + "alps.png");

//   unsigned error =
//       lodepng::decode(imageData, width, height, inputFile, LCT_RGB);
//   EXPECT_EQ(error, 0u) << "Error loading image: " << lodepng_error_text(error);
//   if (error)
//     return;

//   // Create and run GPU implementation
//   ColorHistogramGPUTest gpuHist(cudaDriverInfo, 0,
//                                 true); // Use device 0 with unified memory
//   gpuHist.initializeFromImage(imageData.data(), width, height);
//   gpuHist.initializeGPUMemory();
//   gpuHist.performGPUComputing();
//   gpuHist.retrieveGPUResults();
//   EXPECT_TRUE(gpuHist.verifyComputingResults());
//   gpuHist.releaseGPUComputingResources();

//   DebugConsole_consoleOutLine(
//       "GPU Color histogram computation time: ", gpuHist.getTotalTime(), " ms");
// }

// TEST(HostGoogleTest11__Color_Histogram_GPU, ColorHistogramGPUTest) {
//   DeviceGoogleTest13__Color_Histogram_GPU::executeTest();
// }

// The main entry point of the DeviceUnitTests executable.
int main(int argc, char *argv[]) {
#ifdef GPU_FRAMEWORK_DEBUG
  DebugConsole::setUseLogFile(true);
  DebugConsole::setLogFileName("DeviceUnitTests.log");
#endif // GPU_FRAMEWORK_DEBUG

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}