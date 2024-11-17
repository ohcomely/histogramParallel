/*

Copyright (c) 2009-2018, Thanos Theo. All rights reserved.
Released Under a Simplified BSD (FreeBSD) License
for academic, personal & non-commercial use.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the author and should not be interpreted as representing official policies,
either expressed or implied, of the FreeBSD Project.

A Commercial License is also available for commercial use with
special restrictions and obligations at a one-off fee. See links at:
1. http://www.dotredconsultancy.com/openglrenderingenginetoolrelease.php
2. http://www.dotredconsultancy.com/openglrenderingenginetoolsourcecodelicence.php
Please contact Thanos Theo (thanos.theo@dotredconsultancy.com) for more information.

*/

#pragma once

#ifndef __CUDALinearAlgebraGPUComputingStressTest_h
#define __CUDALinearAlgebraGPUComputingStressTest_h

#include "CUDAGPUComputingAbstraction.h"
#include "CUDAMemoryHandlers.h"
#include "CUDAProcessMemoryPool.h"
#include "CUDAEventTimer.h"
#include "CUDAStreamsHandler.h"
#include "UnitTests.h"
#include <cstdint>
#include <memory>

/** @brief namespace UtilsCUDA for encapsulating all the CUDA related code compiled by the NVCC compiler.
*
* @author Thanos Theo, 2018
* @version 14.0.0.0
*/
namespace UtilsCUDA
{
  /** @brief This class contains a basic Linear Algebra GPU Computing stress test case in host & device. Using the Curiously Recurring Template Pattern (CRTP).
  *
  *  CUDALinearAlgebraGPUComputingStressTest.h:
  *  =========================================
  *  This class contains a basic Linear Algebra GPU Computing stress test case in host & device. Using the Curiously Recurring Template Pattern (CRTP).
  *
  * @author Thanos Theo, Amir Shahvaran, 2019
  * @version 14.0.0.0
  */
  class CUDALinearAlgebraGPUComputingStressTest final : private CUDAGPUComputingAbstraction<CUDALinearAlgebraGPUComputingStressTest>, private Utils::UnitTests::UnitTestUtilityFunctions_flt // private inheritance used for composition and prohibiting up-casting
  {
  public:

    enum class RunTypes : std::size_t { RUN_CPU = 0, RUN_GPU = 1, RUN_BOTH = 2 };

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

    CUDALinearAlgebraGPUComputingStressTest(const CUDADriverInfo& cudaDriverInfo, CUDAProcessMemoryPool& cudaProcessMemoryPool, int device = 0, std::size_t arraySize = 8192, const RunTypes& runType = RunTypes::RUN_BOTH,
                                            std::size_t numberOfCPUTheads = 1, std::size_t numberOfCPUKernelIterations = 160, bool useUnifiedMemory = false, bool useCounter = false) noexcept; // should be 16384 for large VRAM/GPU systems
    ~CUDALinearAlgebraGPUComputingStressTest() = default; // no virtual destructor for data-oriented design (no up-casting should ever be used)
    CUDALinearAlgebraGPUComputingStressTest(const CUDALinearAlgebraGPUComputingStressTest&) = delete; // copy-constructor delete
    CUDALinearAlgebraGPUComputingStressTest(CUDALinearAlgebraGPUComputingStressTest&&)      = delete; // move-constructor delete
    CUDALinearAlgebraGPUComputingStressTest& operator=(const CUDALinearAlgebraGPUComputingStressTest&) = delete; //      assignment operator delete
    CUDALinearAlgebraGPUComputingStressTest& operator=(CUDALinearAlgebraGPUComputingStressTest&&)      = delete; // move-assignment operator delete

  private:

    std::size_t arraySize_                   = 8192 * 8192; // should be 16384 * 16384 for large VRAM/GPU systems
    RunTypes runType_                        = RunTypes::RUN_BOTH;
    std::size_t numberOfCPUThreads_          = 0;
    std::size_t numberOfCPUKernelIterations_ = 0;
    bool useUnifiedMemory_                   = false;
    bool useCounter_                         = false;

    DeviceMemory<std::int32_t> arrayA_;
    DeviceMemory<std::int32_t> arrayB_;
    DeviceMemory<std::int32_t> arrayC_;
    DeviceMemory<std::int32_t> kernelExecutionCounterUVA_;
    DeviceMemory<std::int32_t> globalSynchronizationUVA_;
    DeviceMemory<std::int32_t> globalBarrierUVA_;
    DeviceMemory<std::int32_t> gpuStopFlagUVA_;
    DeviceMemory<std::int32_t> cpuStopFlagUVA_;
    HostDeviceMemory<std::int32_t> hostDeviceArrayA_;
    HostDeviceMemory<std::int32_t> hostDeviceArrayB_;
    HostDeviceMemory<std::int32_t> hostDeviceArrayC_;
    HostDeviceMemory<std::int32_t> kernelExecutionCounter_;
    HostDeviceMemory<std::int32_t> globalSynchronization_;
    HostDeviceMemory<std::int32_t> globalBarrier_;
    HostDeviceMemory<std::int32_t> gpuStopFlag_;
    HostDeviceMemory<std::int32_t> cpuStopFlag_;
    CUDAProcessMemoryPool& cudaProcessMemoryPool_;
    const CUDAStreamsHandler cudaStreamsHandler_;
    CUDAEventTimer gpuTimer_;
    std::unique_ptr<std::int32_t[]> cpuArrayC_ = nullptr;
  };
} // namespace UtilsCUDA

#endif // __CUDALinearAlgebraGPUComputingStressTest_h