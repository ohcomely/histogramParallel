#include "Tests/CUDALinearAlgebraGPUComputingTest.h"
#include "CUDAEventTimer.h"
#include "CUDAParallelFor.h"
#include "CUDAUtilityFunctions.h"
#include "CPUParallelism/CPUParallelismNCP.h"
#include "UtilityFunctions.h"
#include <algorithm>

using namespace std;
using namespace UtilsCUDA;
using namespace UtilsCUDA::CUDAParallelFor;
using namespace Utils::CPUParallelism;

namespace // anonymous namespace used instead of deprecated 'static' keyword used for cpp variable locality
{
  constexpr size_t NUMBER_OF_CUDA_STREAMS = 1;

  inline void startMemoryPool(const CUDADriverInfo& cudaDriverInfo, CUDAProcessMemoryPool& cudaProcessMemoryPool, int device, size_t arraySize, bool useUnifiedMemory)
  {
    const size_t hostBytesToAllocate = (3 * arraySize + 3 * cudaDriverInfo.getTextureAlignment(device)) * sizeof(int32_t); // Note: allocation is in bytes, plus some padding space
    if (useUnifiedMemory)
    {
      // use the Device Memory Pool for allocation of device memory (with Unified Memory enabled) for the given device
      array<size_t, CUDAProcessMemoryPool::MAX_DEVICES> deviceBytesToAllocatePerDevice = { { 0 } }; // default-initialize all devices to zero bytes usage
      deviceBytesToAllocatePerDevice[device] = hostBytesToAllocate; // allocate bytes in given device
      cudaProcessMemoryPool.allocateDeviceMemoryPool(deviceBytesToAllocatePerDevice, 1ull << device);
    }
    else
    {
      // use the Host/Device Memory Pool for allocation of host/device memory for the given device
      array<size_t, CUDAProcessMemoryPool::MAX_DEVICES> deviceBytesToAllocatePerDevice = { { 0 } }; // default-initialize all devices to zero bytes usage
      deviceBytesToAllocatePerDevice[device] = hostBytesToAllocate; // allocate bytes in given device
      cudaProcessMemoryPool.allocateHostDeviceMemoryPool(hostBytesToAllocate, deviceBytesToAllocatePerDevice);
    }
  }

  inline void stopMemoryPool(CUDAProcessMemoryPool& cudaProcessMemoryPool, bool useUnifiedMemory)
  {
    if (useUnifiedMemory)
    {
      // use the Device Memory Pool for de-allocation of host memory
      cudaProcessMemoryPool.freeDeviceMemoryPool();
    }
    else
    {
      // use the Host/Device Memory Pool for de-allocation of host/device memory
      cudaProcessMemoryPool.freeHostDeviceMemoryPool();
    }
  }
}

CUDALinearAlgebraGPUComputingTest::CUDALinearAlgebraGPUComputingTest(const CUDADriverInfo& cudaDriverInfo, int device, bool useUnifiedMemory, size_t arraySize) noexcept
  : CUDAGPUComputingAbstraction(cudaDriverInfo, device)
  , arraySize_(max<size_t>(1, arraySize * arraySize))
  , useUnifiedMemory_(useUnifiedMemory&& cudaDriverInfo.hasUnifiedMemory(device) && cudaDriverInfo.getConcurrentManagedAccess(device) && cudaDriverInfo.isAtLeastGPUType(CUDADriverInfo::GPUTypes::PASCAL, device)) // last option is for using UVA prefetching which is only possible at least on Pascal GPU hardware & onwards
  , cudaProcessMemoryPool_(cudaDriverInfo, false) // do NOT use default allocations for all devices (in this test each instance handles one device)
  , cudaStreamsHandler_(cudaDriverInfo, device, NUMBER_OF_CUDA_STREAMS)
  , gpuTimer_(device, cudaStreamsHandler_[0])
{
  // choose which GPU to run on for a multi-GPU system
  CUDAError_checkCUDAError(cudaSetDevice(device));
  DebugConsole_consoleOutLine("CUDALinearAlgebraGPUComputingTest::constructor() information below:\nUsing device: ", device_, " with array size: '", arraySize_, "'.");
}

void CUDALinearAlgebraGPUComputingTest::initializeGPUMemory()
{
  gpuTimer_.startTimer();

  startMemoryPool(cudaDriverInfo_, cudaProcessMemoryPool_, device_, arraySize_, useUnifiedMemory_);

  if (useUnifiedMemory_)
  {
    cudaProcessMemoryPool_.reserve(arrayA_, arraySize_, device_);
    cudaProcessMemoryPool_.reserve(arrayB_, arraySize_, device_);
    cudaProcessMemoryPool_.reserve(arrayC_, arraySize_, device_);
    cudaProcessMemoryPool_.reportHostDeviceMemoryPoolInformation("Linear Algebra Test");

    // populate with data
    parallelFor(0, arraySize_, [&](size_t i)
    {
      arrayA_[i] = -int32_t(i);
      arrayB_[i] =  int32_t(i * i);
    });

    // using UVA prefetching below is only possible on devices with Concurrent Managed Access
    if (cudaDriverInfo_.getConcurrentManagedAccess(device_))
    {
      // using UVA prefetching below with explicit memAdvise enabled for faster transfers
      arrayA_.memPrefetchWithAdviseAsync(device_, cudaStreamsHandler_[0]);
      arrayB_.memPrefetchWithAdviseAsync(device_, cudaStreamsHandler_[0]);
      arrayC_.memPrefetchWithAdviseAsync(device_, cudaStreamsHandler_[0]);
    }
  }
  else
  {
    cudaProcessMemoryPool_.reserve(hostDeviceArrayA_, arraySize_, device_);
    cudaProcessMemoryPool_.reserve(hostDeviceArrayB_, arraySize_, device_);
    cudaProcessMemoryPool_.reserve(hostDeviceArrayC_, arraySize_, device_);
    cudaProcessMemoryPool_.reportHostDeviceMemoryPoolInformation("Linear Algebra Test");

    // populate with data
    parallelFor(0, arraySize_, [&](size_t i)
    {
      hostDeviceArrayA_[i] = -int32_t(i);
      hostDeviceArrayB_[i] =  int32_t(i * i);
    });

    // copy the arrays 'hostDeviceArrayA' & 'hostDeviceArrayB' from the CPU to the GPU
    hostDeviceArrayA_.copyHostToDeviceAsync(cudaStreamsHandler_[0]);
    hostDeviceArrayB_.copyHostToDeviceAsync(cudaStreamsHandler_[0]);
  }

  totalTimeTakenInMs_ += gpuTimer_.getElapsedTimeInMilliSecs();
}

void CUDALinearAlgebraGPUComputingTest::performGPUComputing()
{
  gpuTimer_.startTimer();

  // choose which GPU to run on for a multi-GPU system
  CUDAError_checkCUDAError(cudaSetDevice(device_));

  if (useUnifiedMemory_)
  {
    launchCUDAParallelForInStream(arraySize_, 0, cudaStreamsHandler_[0], [] __device__ (size_t index, const int32_t* __restrict a, const int32_t* __restrict b, int32_t* __restrict c)
    {
      c[index] = a[index] + b[index];
    }, arrayA_.device(), arrayB_.device(), arrayC_.device());
  }
  else
  {
    launchCUDAParallelForInStream(arraySize_, 0, cudaStreamsHandler_[0], [] __device__ (size_t index, const int32_t* __restrict a, const int32_t* __restrict b, int32_t* __restrict c)
    {
      c[index] = a[index] + b[index];
    }, hostDeviceArrayA_.device(), hostDeviceArrayB_.device(), hostDeviceArrayC_.device());
  }

  totalTimeTakenInMs_ += gpuTimer_.getElapsedTimeInMilliSecs();
}

void CUDALinearAlgebraGPUComputingTest::retrieveGPUResults()
{
  gpuTimer_.startTimer();

  // choose which GPU to run on for a multi-GPU system
  CUDAError_checkCUDAError(cudaSetDevice(device_));

  if (useUnifiedMemory_)
  {
    // using UVA prefetching below is only possible on devices with Concurrent Managed Access
    if (cudaDriverInfo_.getConcurrentManagedAccess(device_))
    {
      // using UVA prefetching below with explicit memAdvise enabled for faster transfers
      arrayC_.memPrefetchWithAdviseAsync(cudaCpuDeviceId, cudaStreamsHandler_[0]);
    }
    // cudaDeviceSynchronize() waits for the kernel to finish, and returns any errors encountered during the launch
    CUDAError_checkCUDAError(cudaDeviceSynchronize());
  }
  else // synchronize needed with Unified Memory
  {
    // copy the array 'hostDeviceArrayC' back from the GPU to the CPU
    hostDeviceArrayC_.copyDeviceToHost(cudaStreamsHandler_[0]);
  }

  totalTimeTakenInMs_ += gpuTimer_.getElapsedTimeInMilliSecs();
}

bool CUDALinearAlgebraGPUComputingTest::verifyComputingResults()
{
  int32_t* resultsArrayC = useUnifiedMemory_ ? arrayC_.device() : hostDeviceArrayC_.host();
  for (size_t i = 0; i < arraySize_; ++i)
  {
    if (resultsArrayC[i] != -int32_t(i) + int32_t(i * i))
    {
      DebugConsole_consoleOutLine("CUDALinearAlgebraGPUComputingTest::verifyComputingResults() error found at index: ", i);
      return false;
    }
  }

  DebugConsole_consoleOutLine("CUDALinearAlgebraGPUComputingTest::verifyComputingResults() passed.");
  return true;
}

void CUDALinearAlgebraGPUComputingTest::releaseGPUComputingResources()
{
  stopMemoryPool(cudaProcessMemoryPool_, useUnifiedMemory_);

  DebugConsole_consoleOutLine("CUDALinearAlgebraGPUComputingTest total time taken: ", totalTimeTakenInMs_, " ms.\n");
}