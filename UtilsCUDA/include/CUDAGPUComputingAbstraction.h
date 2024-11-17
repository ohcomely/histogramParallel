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

#ifndef __CUDAGPUComputing_h
#define __CUDAGPUComputing_h

#include "ModuleDLL.h"
#include "CUDADriverInfo.h"

/** @brief namespace UtilsCUDA for encapsulating all the CUDA related code compiled by the NVCC compiler.
*
* @author Thanos Theo, 2018
* @version 14.0.0.0
*/
namespace UtilsCUDA
{
  /** @brief This class encapsulates a basic abstraction layer for CUDA GPU Computing. Using the Curiously Recurring Template Pattern (CRTP).
  *
  *  CUDAGPUComputingAbstraction.h:
  *  =============================
  *  This class encapsulates a basic abstraction layer for CUDA GPU Computing
  *  (abstract class CUDAGPUComputingAbstraction, ie no direct instantiation allowed).
  *  Note: No virtual destructor is needed for data-oriented design, ie no up-casting should ever be used.
  *  Using the Curiously Recurring Template Pattern (CRTP).
  *
  * @author Thanos Theo, 2018
  * @version 14.0.0.0
  */
  template <typename Derived>
  class CRTP_MODULE_API CUDAGPUComputingAbstraction
  {
  public:

    /** @brief Initializes GPU memory.
    */
    void initializeGPUMemory()          {        asDerived()->initializeGPUMemory(); }

    /** @brief Performs the GPU Computing calculations.
    */
    void performGPUComputing()          {        asDerived()->performGPUComputing(); }

    /** @brief Retrieves the results from the GPU.
    */
    void retrieveGPUResults()           {        asDerived()->retrieveGPUResults(); }

    /** @brief Verifies the computing results between the CPU and the GPU.
    */
    bool verifyComputingResults()       { return asDerived()->verifyComputingResults(); }

    /** @brief Releases the GPU Computing resources.
    */
    void releaseGPUComputingResources() {        asDerived()->releaseGPUComputingResources(); }

  protected:

    const CUDADriverInfo& cudaDriverInfo_;
    int device_                = 0;
    int deviceCount_           = 0;
    double totalTimeTakenInMs_ = 0.0;

    CUDAGPUComputingAbstraction(const CUDADriverInfo& cudaDriverInfo, int device) noexcept : cudaDriverInfo_(cudaDriverInfo), device_(device), deviceCount_(cudaDriverInfo.getDeviceCount()), totalTimeTakenInMs_(0.0) {}
    ~CUDAGPUComputingAbstraction() = default; // no virtual destructor for data-oriented design (no up-casting should ever be used)
    CUDAGPUComputingAbstraction(const CUDAGPUComputingAbstraction&) = delete; // copy-constructor delete
    CUDAGPUComputingAbstraction(CUDAGPUComputingAbstraction&&)      = delete; // move-constructor delete
    CUDAGPUComputingAbstraction& operator=(const CUDAGPUComputingAbstraction&) = delete; //      assignment operator delete
    CUDAGPUComputingAbstraction& operator=(CUDAGPUComputingAbstraction&&)      = delete; // move-assignment operator delete

  private:

          Derived* asDerived()       { return reinterpret_cast<      Derived*>(this); }
    const Derived* asDerived() const { return reinterpret_cast<const Derived*>(this); }
  };
} // namespace UtilsCUDA

#endif // __CUDAGPUComputing_h