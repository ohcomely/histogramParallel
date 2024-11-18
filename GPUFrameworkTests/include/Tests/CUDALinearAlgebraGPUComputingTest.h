

#pragma once

#ifndef __CUDALinearAlgebraGPUComputingTest_h
#define __CUDALinearAlgebraGPUComputingTest_h

#include "CUDAGPUComputingAbstraction.h"
#include "CUDAMemoryHandlers.h"
#include "CUDAProcessMemoryPool.h"
#include "CUDAStreamsHandler.h"
#include "CUDAEventTimer.h"
#include "UnitTests.h"
#include <cstdint>

/** @brief namespace UtilsCUDA for encapsulating all the CUDA related code compiled by the NVCC compiler.
*
* @author Thanos Theo, 2018
* @version 14.0.0.0
*/
namespace UtilsCUDA
{
  /** @brief This class contains a basic Linear Algebra GPU Computing test case in CUDA. Using the Curiously Recurring Template Pattern (CRTP).
  *
  *  CUDALinearAlgebraGPUComputingTest.h:
  *  ===================================
  *  This class contains a basic Linear Algebra GPU Computing test case in CUDA. Using the Curiously Recurring Template Pattern (CRTP).
  *
  * @author Thanos Theo, 2018
  * @version 14.0.0.0
  */
  class CUDALinearAlgebraGPUComputingTest final : private CUDAGPUComputingAbstraction<CUDALinearAlgebraGPUComputingTest>, private Utils::UnitTests::UnitTestUtilityFunctions_flt // private inheritance used for composition and prohibiting up-casting
  {
  public:

    /** @brief Initializes GPU memory.
    */
    void initializeGPUMemory();

    /** @brief Performs the GPU Computing calculations.
    */
    void performGPUComputing();

    /** @brief Retrieves the results from the GPU.
    */
    void retrieveGPUResults();

    /** @brief Verifies the computing results between the CPU and the GPU.
    */
    bool verifyComputingResults();

    /** @brief Releases the GPU Computing resources.
    */
    void releaseGPUComputingResources();

    CUDALinearAlgebraGPUComputingTest(const CUDADriverInfo& cudaDriverInfo, int device = 0, bool useUnifiedMemory = false, std::size_t arraySize = 8192) noexcept; // should be 16384 for large VRAM/GPU systems
    ~CUDALinearAlgebraGPUComputingTest() = default; // no virtual destructor for data-oriented design (no up-casting should ever be used)
    CUDALinearAlgebraGPUComputingTest(const CUDALinearAlgebraGPUComputingTest&) = delete; // copy-constructor delete
    CUDALinearAlgebraGPUComputingTest(CUDALinearAlgebraGPUComputingTest&&)      = delete; // move-constructor delete
    CUDALinearAlgebraGPUComputingTest& operator=(const CUDALinearAlgebraGPUComputingTest&) = delete; //      assignment operator delete
    CUDALinearAlgebraGPUComputingTest& operator=(CUDALinearAlgebraGPUComputingTest&&)      = delete; // move-assignment operator delete

  private:

    std::size_t arraySize_ = 8192 * 8192; // should be 16384 * 16384 for large VRAM/GPU systems
    bool useUnifiedMemory_ = false;

    DeviceMemory<std::int32_t>     arrayA_;
    DeviceMemory<std::int32_t>     arrayB_;
    DeviceMemory<std::int32_t>     arrayC_;
    HostDeviceMemory<std::int32_t> hostDeviceArrayA_;
    HostDeviceMemory<std::int32_t> hostDeviceArrayB_;
    HostDeviceMemory<std::int32_t> hostDeviceArrayC_;
    CUDAProcessMemoryPool cudaProcessMemoryPool_;
    const CUDAStreamsHandler cudaStreamsHandler_;
    CUDAEventTimer gpuTimer_;
  };
} // namespace UtilsCUDA

#endif // __CUDALinearAlgebraGPUComputingTest_h