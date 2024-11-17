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