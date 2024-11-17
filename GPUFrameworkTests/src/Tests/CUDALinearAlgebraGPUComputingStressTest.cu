#include "Tests/CUDALinearAlgebraGPUComputingStressTest.h"
#include "CUDAEventTimer.h"
#include "CUDAParallelFor.h"
#include "CUDASpinLock.h"
#include "CUDAUtilityFunctions.h"
#include "CPUParallelism/CPUParallelismNCP.h"
#include "UtilityFunctions.h"
#include <string>
#include <algorithm>
#include <limits>

using namespace std;
using namespace UtilsCUDA;
using namespace UtilsCUDA::CUDAParallelFor;
using namespace Utils::CPUParallelism;

namespace // anonymous namespace used instead of deprecated 'static' keyword used for cpp variable locality
{
  constexpr size_t NUMBER_OF_CUDA_STREAMS = 2;

  inline string reportRunType(const CUDALinearAlgebraGPUComputingStressTest::RunTypes& runType)
  {
    return (runType == CUDALinearAlgebraGPUComputingStressTest::RunTypes::RUN_CPU) ? "RUN_CPU" : ((runType == CUDALinearAlgebraGPUComputingStressTest::RunTypes::RUN_GPU) ? "RUN_GPU" : "RUN_BOTH");
  }

  __forceinline__ __device__ __host__
  int32_t randomizer(size_t index)
  {
    float2 randSeed = CUDAUtilityFunctions::rand2(make_float2(float(index), float(index * index)));
    uint32_t   seed = CUDAUtilityFunctions::seedGenerator(uint32_t(randSeed.x * numeric_limits<uint32_t>::max()), uint32_t(randSeed.y * numeric_limits<uint32_t>::max()));
    return int32_t(CUDAUtilityFunctions::rand1u(seed) >> 1); // truncate uint32_t -> int32_t
  }

  __forceinline__ __device__ __host__
  void linearAlgebraFunction(size_t index, const int32_t* __restrict a, const int32_t* __restrict b, int32_t* __restrict c, size_t randomizerIndex)
  {
    // Note: enforce the compiler to NOT optimize away the randomizer() call & use 'randomizerIndex' and NOT 'index' to not cache the randomizer() result in CPU-side
    c[index] = a[index] + b[index] + int32_t(randomizer(randomizerIndex) * numeric_limits<float>::epsilon());
  }

  template<bool USE_UVA>
  __forceinline__ __device__
  void __syncblocks(int32_t* __restrict kernelExecutionCounter, int32_t* __restrict globalSynchronization, int32_t* __restrict globalBarrier, int32_t* __restrict gpuStopFlag, int32_t* __restrict cpuStopFlag)
  {
    // sync the block threads (all block warps) before entering the main syncblocks() code block
    __syncthreads();
    // only the master thread ({0}) for each block performs the __syncblocks() equivalent call below
    if (threadIdx.x == 0)
    {
      // try to emulate __syncblocks() call, in-order atomic incrementing globalSynchronization variable for each block with spin locks
      CUDASpinLock<int32_t> cudaSpinLock(globalSynchronization, globalBarrier);
      cudaSpinLock.acquireGrid();
      if (blockIdx.x == 0)
      {
        // we don't need an atomicAdd() as only the master thread ({0, 0}) performs the increment
        ++kernelExecutionCounter[0];
        // use an atomicCAS() call to enforce that the cpuStopFlag is updated in global memory
        // only the master thread ({0, 0}) performs the check & conditionally sets the gpuStopFlag with atomicExch()
        if (atomicCAS(cpuStopFlag, 1, 1)) { atomicExch(gpuStopFlag, 1); }
        if (USE_UVA) __threadfence_system();
        else         __threadfence();
      }
      // try to emulate __syncblocks() call, in-order atomic decrementing globalSynchronization variable for each block with spin locks
      cudaSpinLock.releaseGrid();
    }
    // sync the block threads (all block warps) before continuing
    __syncthreads();
  }

  // Note: using a persistent kernel approach for better load balancing better different GPUs
  template<bool USE_UVA>
  __forceinline__ __device__
  void linearAlgebraPersistentKernelGPU(size_t index, const int32_t* __restrict a, const int32_t* __restrict b, int32_t* __restrict c, size_t arraySize,
                                        int32_t* __restrict kernelExecutionCounter, int32_t* __restrict globalSynchronization, int32_t* __restrict globalBarrier,  int32_t* __restrict gpuStopFlag, int32_t* __restrict cpuStopFlag,
                                        bool useCounter)
  {
    do // do one iteration at least to pass the CPU verification (rare scenario of GPU being slower than CPU)
    {
      // core algorithm execution
      size_t currentIndex = index;
      while (currentIndex < arraySize)
      {
        linearAlgebraFunction(currentIndex, a, b, c, currentIndex);
        currentIndex += CUDAUtilityDeviceFunctions::globalThreadCount();
      }
      if (useCounter)
      {
        // sync all active blocks in persistent kernel via the syncblocks call()
        __syncblocks<USE_UVA>(kernelExecutionCounter, globalSynchronization, globalBarrier, gpuStopFlag, cpuStopFlag);
      }
    }
    while (!atomicCAS(useCounter ? gpuStopFlag : cpuStopFlag, 1, 1));
  }

  inline void linearAlgebraKernelCPU(size_t index, const int32_t* __restrict a, const int32_t* __restrict b, int32_t* __restrict c, size_t iterations)
  {
    for (size_t iteration = 0; iteration < iterations; ++iteration)
    {
      linearAlgebraFunction(index, a, b, c, iteration);
    }
  }
}

CUDALinearAlgebraGPUComputingStressTest::CUDALinearAlgebraGPUComputingStressTest(const CUDADriverInfo& cudaDriverInfo, CUDAProcessMemoryPool& cudaProcessMemoryPool, int device, size_t arraySize, const RunTypes& runType,
                                                                                 size_t numberOfCPUTheads, size_t numberOfCPUKernelIterations, bool useUnifiedMemory, bool useCounter) noexcept
  : CUDAGPUComputingAbstraction(cudaDriverInfo, device)
  , arraySize_(max<size_t>(1, arraySize * arraySize))
  , runType_(runType)
  , numberOfCPUThreads_(max<size_t>(1, numberOfCPUTheads))
  , numberOfCPUKernelIterations_(max<size_t>(1, numberOfCPUKernelIterations))
  , useUnifiedMemory_(useUnifiedMemory && cudaDriverInfo.hasUnifiedMemory(device) && cudaDriverInfo.getConcurrentManagedAccess(device) && cudaDriverInfo.isAtLeastGPUType(CUDADriverInfo::GPUTypes::PASCAL, device)) // last option is for using UVA prefetching which is only possible at least on Pascal GPU hardware & onwards
  , useCounter_(useCounter)
  , cudaProcessMemoryPool_(cudaProcessMemoryPool)
  , cudaStreamsHandler_(cudaDriverInfo, device, NUMBER_OF_CUDA_STREAMS)
  , gpuTimer_(device, cudaStreamsHandler_[0])
  , cpuArrayC_((runType == RunTypes::RUN_GPU) ? nullptr : make_unique<int32_t[]>(arraySize_))
{
  // choose which GPU to run on for a multi-GPU system
  CUDAError_checkCUDAError(cudaSetDevice(device));
  DebugConsole_consoleOutLine("CUDALinearAlgebraGPUComputingStressTest::constructor() information below:\nUsing device '", device_, "' with array size '", arraySize_, "' and runType '", reportRunType(runType_), "'.");
}

void CUDALinearAlgebraGPUComputingStressTest::initializeGPUMemory()
{
  gpuTimer_.startTimer();

  if (useUnifiedMemory_)
  {
    // add handlers to HostDevice memory pool
    cudaProcessMemoryPool_.reserve(arrayA_,           arraySize_, device_);
    cudaProcessMemoryPool_.reserve(arrayB_,           arraySize_, device_);
    cudaProcessMemoryPool_.reserve(arrayC_,           arraySize_, device_);
    cudaProcessMemoryPool_.reserve(kernelExecutionCounterUVA_, 1, device_);
    cudaProcessMemoryPool_.reserve(globalSynchronizationUVA_,  1, device_);
    cudaProcessMemoryPool_.reserve(globalBarrierUVA_,          1, device_);
    cudaProcessMemoryPool_.reserve(gpuStopFlagUVA_,            1, device_);
    cudaProcessMemoryPool_.reserve(cpuStopFlagUVA_,            1, device_);
    cudaProcessMemoryPool_.reportHostDeviceMemoryPoolInformation("Stress Test");

    parallelFor(0, arraySize_, [&](size_t i)
    {
      arrayA_[i] = -int32_t(i);
      arrayB_[i] =  int32_t(i * i);
    }, max<size_t>(1, numberOfCPUThreads_), AFFINITY_MASK_NONE); // let stress test use the underlying OS scheduler for thread execution
    kernelExecutionCounterUVA_[0] = 0;
    globalSynchronizationUVA_[0]  = 0;
    globalBarrierUVA_[0]          = 0;
    gpuStopFlagUVA_[0]            = 0;
    cpuStopFlagUVA_[0]            = 0;

    // using UVA prefetching below is only possible on devices with Concurrent Managed Access
    if (cudaDriverInfo_.getConcurrentManagedAccess(device_))
    {
      // using UVA prefetching below with explicit memAdvise enabled for faster transfers
      arrayA_.memPrefetchWithAdviseAsync(device_, cudaStreamsHandler_[0]);
      arrayB_.memPrefetchWithAdviseAsync(device_, cudaStreamsHandler_[0]);
      kernelExecutionCounterUVA_.memPrefetchWithAdviseAsync(device_, cudaStreamsHandler_[0]);
      globalSynchronizationUVA_.memPrefetchWithAdviseAsync( device_, cudaStreamsHandler_[0]);
      globalBarrierUVA_.memPrefetchWithAdviseAsync(device_, cudaStreamsHandler_[0]);
      gpuStopFlagUVA_.memPrefetchWithAdviseAsync(  device_, cudaStreamsHandler_[0]);
      cpuStopFlagUVA_.memPrefetchWithAdviseAsync(  device_, cudaStreamsHandler_[0]);
    }
  }
  else
  {
    cudaProcessMemoryPool_.reserve(hostDeviceArrayA_, arraySize_, device_);
    cudaProcessMemoryPool_.reserve(hostDeviceArrayB_, arraySize_, device_);
    cudaProcessMemoryPool_.reserve(hostDeviceArrayC_, arraySize_, device_);
    cudaProcessMemoryPool_.reserve(kernelExecutionCounter_,    1, device_);
    cudaProcessMemoryPool_.reserve(globalSynchronization_,     1, device_);
    cudaProcessMemoryPool_.reserve(globalBarrier_,             1, device_);
    cudaProcessMemoryPool_.reserve(gpuStopFlag_,               1, device_);
    cudaProcessMemoryPool_.reserve(cpuStopFlag_,               1, device_);
    cudaProcessMemoryPool_.reportHostDeviceMemoryPoolInformation("Stress Test");

    parallelFor(0, arraySize_, [&](size_t i)
    {
      hostDeviceArrayA_[i] = -int32_t(i);
      hostDeviceArrayB_[i] =  int32_t(i * i);
    }, max<size_t>(1, numberOfCPUThreads_), AFFINITY_MASK_NONE); // let stress test use the underlying OS scheduler for thread execution
    kernelExecutionCounter_[0] = 0;
    globalSynchronization_[0]  = 0;
    globalBarrier_[0]          = 0;
    gpuStopFlag_[0]            = 0;
    cpuStopFlag_[0]            = 0;

    // copy the arrays 'hostDeviceArrayA', 'hostDeviceArrayB' & 'stopFlag' from the CPU to the GPU
    hostDeviceArrayA_.copyHostToDeviceAsync(cudaStreamsHandler_[0]);
    hostDeviceArrayB_.copyHostToDeviceAsync(cudaStreamsHandler_[0]);
    kernelExecutionCounter_.copyHostToDeviceAsync(cudaStreamsHandler_[0]);
    globalSynchronization_.copyHostToDeviceAsync(cudaStreamsHandler_[0]);
    globalBarrier_.copyHostToDeviceAsync(cudaStreamsHandler_[0]);
    gpuStopFlag_.copyHostToDeviceAsync(cudaStreamsHandler_[0]);
    cpuStopFlag_.copyHostToDeviceAsync(cudaStreamsHandler_[0]);
  }

  totalTimeTakenInMs_ += gpuTimer_.getElapsedTimeInMilliSecs();
}

void CUDALinearAlgebraGPUComputingStressTest::performGPUComputing()
{
  gpuTimer_.startTimer();

  // GPU kernel has to be executed first, as it is asynchronous
  if (runType_ == RunTypes::RUN_GPU || runType_ == RunTypes::RUN_BOTH)
  {
    const auto gridSize       = UtilsCUDA::CUDAUtilityFunctions::calculateCUDAPersistentKernel(cudaDriverInfo_, device_, cudaDriverInfo_.getMaxThreadsPerBlock(device_));
    const size_t totalThreads = get<0>(gridSize).x * get<1>(gridSize).x;

    // choose which GPU to run the GPU kernel on for a multi-GPU system
    CUDAError_checkCUDAError(cudaSetDevice(device_));

    if (useUnifiedMemory_)
    {
      // run kernel multiple times while the CPU is busy running the CPU kernel (GPU calls are asynchronous with CUDA stream usage
      launchCUDAParallelForInStream(totalThreads, 0, cudaStreamsHandler_[0], [] __device__(size_t index, const int32_t* __restrict a, const int32_t* __restrict b, int32_t* __restrict c, size_t arraySize,
                                                                                           int32_t* __restrict kernelExecutionCounter, int32_t* __restrict globalSynchronization, int32_t* __restrict globalBarrier, int32_t* __restrict gpuStopFlag, int32_t* __restrict cpuStopFlag,
                                                                                           bool executionCounter)
      {
        linearAlgebraPersistentKernelGPU<true>(index, a, b, c, arraySize, kernelExecutionCounter, globalSynchronization, globalBarrier, gpuStopFlag, cpuStopFlag, executionCounter);
      }, arrayA_.device(), arrayB_.device(), arrayC_.device(), arraySize_,
         kernelExecutionCounterUVA_.device(), globalSynchronizationUVA_.device(), globalBarrierUVA_.device(), gpuStopFlagUVA_.device(), cpuStopFlagUVA_.device(),
         useCounter_);
    }
    else
    {
      // run kernel multiple times while the CPU is busy running the CPU kernel (GPU calls are asynchronous with CUDA stream usage
      launchCUDAParallelForInStream(totalThreads, 0, cudaStreamsHandler_[0], [] __device__(size_t index, const int32_t* __restrict a, const int32_t* __restrict b, int32_t* __restrict c, size_t arraySize,
                                                                                           int32_t* __restrict kernelExecutionCounter, int32_t* __restrict globalSynchronization, int32_t* __restrict globalBarrier, int32_t* __restrict gpuStopFlag, int32_t* __restrict cpuStopFlag,
                                                                                           bool executionCounter)
      {
        linearAlgebraPersistentKernelGPU<false>(index, a, b, c, arraySize, kernelExecutionCounter, globalSynchronization, globalBarrier, gpuStopFlag, cpuStopFlag, executionCounter);
      }, hostDeviceArrayA_.device(), hostDeviceArrayB_.device(), hostDeviceArrayC_.device(), arraySize_,
         kernelExecutionCounter_.device(), globalSynchronization_.device(), globalBarrier_.device(), gpuStopFlag_.device(), cpuStopFlag_.device(),
         useCounter_);
    }
  }

  // CPU kernel has to be executed second, as it is synchronous (blocking)
  if (runType_ == RunTypes::RUN_CPU || runType_ == RunTypes::RUN_BOTH)
  {
    if (useUnifiedMemory_)
    {
      // run the CPU kernel
      parallelFor(0, arraySize_, [&](size_t index)
      {
        linearAlgebraKernelCPU(index, arrayA_.get(), arrayB_.get(), cpuArrayC_.get(), numberOfCPUKernelIterations_);
      }, max<size_t>(1, numberOfCPUThreads_), AFFINITY_MASK_NONE); // let stress test use the underlying OS scheduler for thread execution
    }
    else
    {
      // run the CPU kernel
      parallelFor(0, arraySize_, [&](size_t index)
      {
        linearAlgebraKernelCPU(index, hostDeviceArrayA_.host(), hostDeviceArrayB_.host(), cpuArrayC_.get(), numberOfCPUKernelIterations_);
      }, max<size_t>(1, numberOfCPUThreads_), AFFINITY_MASK_NONE); // let stress test use the underlying OS scheduler for thread execution
    }
  }
  else // if (runType_ == RunTypes::RUN_GPU)
  {
    // for GPU-only stress tests, CPU just waits
    for (size_t i = 0; i < numberOfCPUKernelIterations_; ++i)
    {
      threadSleep(arraySize_ >> 16);
    }
  }

  if (useUnifiedMemory_)
  {
    // stop the persistent kernel via another stream uploading the cpuStopFlag
    cpuStopFlagUVA_[0] = 1;
    CUDAError_checkCUDAError(cudaDeviceSynchronize()); // device synchronize to enforce the stopFlag upload
  }
  else
  {
    // stop the persistent kernel via another stream uploading the cpuStopFlag
    cpuStopFlag_[0] = 1;
    cpuStopFlag_.copyHostToDeviceAsync(cudaStreamsHandler_[1]);
    CUDAError_checkCUDAError(cudaStreamSynchronize(cudaStreamsHandler_[1])); // stream synchronize to enforce the cpuStopFlag upload
  }

  totalTimeTakenInMs_ += gpuTimer_.getElapsedTimeInMilliSecs();
}

void CUDALinearAlgebraGPUComputingStressTest::retrieveGPUResults()
{
  gpuTimer_.startTimer();

  if (useUnifiedMemory_)
  {
    // using UVA prefetching below is only possible on devices with Concurrent Managed Access
    if (cudaDriverInfo_.getConcurrentManagedAccess(device_))
    {
      // using UVA prefetching below with explicit memAdvise enabled for faster transfers
      arrayC_.memPrefetchWithAdviseAsync(cudaCpuDeviceId, cudaStreamsHandler_[0]);
      if (useCounter_)
      {
        kernelExecutionCounterUVA_.memPrefetchWithAdviseAsync(cudaCpuDeviceId, cudaStreamsHandler_[0]);
      }
    }
    // cudaDeviceSynchronize() waits for the kernel to finish & data to prefetch, and returns any errors encountered during the launch
    CUDAError_checkCUDAError(cudaDeviceSynchronize());
  }
  else
  {
    // copy the array 'hostDeviceArrayC' back from the GPU to the CPU
    hostDeviceArrayC_.copyDeviceToHost(cudaStreamsHandler_[0]);
    if (useCounter_)
    {
      kernelExecutionCounter_.copyDeviceToHost(cudaStreamsHandler_[0]);
    }
  }

  totalTimeTakenInMs_ += gpuTimer_.getElapsedTimeInMilliSecs();
}

bool CUDALinearAlgebraGPUComputingStressTest::verifyComputingResults()
{
  if (runType_ == RunTypes::RUN_BOTH)
  {
    int32_t* arrayC = useUnifiedMemory_ ? arrayC_.get() : hostDeviceArrayC_.host();
    for (size_t i = 0; i < arraySize_; ++i)
    {
      if (cpuArrayC_[i] != arrayC[i])
      {
        DebugConsole_consoleOutLine("CUDALinearAlgebraGPUComputingStressTest::verifyComputingResults() error found at index: ", i);
        return false;
      }
    }
  }

  DebugConsole_consoleOutLine("CUDALinearAlgebraGPUComputingStressTest::verifyComputingResults() passed.");
  if (useCounter_)
  {
    DebugConsole_consoleOutLine("CUDALinearAlgebraGPUComputingStressTest: GPU kernel executed ", useUnifiedMemory_ ? kernelExecutionCounterUVA_[0] : kernelExecutionCounter_[0], " times.");
  }

  return true;
}

void CUDALinearAlgebraGPUComputingStressTest::releaseGPUComputingResources()
{
  cpuArrayC_.reset(nullptr); // release & delete CPU memory so as to avoid temporary memory consumption

  DebugConsole_consoleOutLine("CUDALinearAlgebraGPUComputingStressTest total time taken: ", totalTimeTakenInMs_, " ms.\n");
}