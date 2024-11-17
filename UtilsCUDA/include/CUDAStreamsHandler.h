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

#ifndef __CUDAStreamsHandler_h
#define __CUDAStreamsHandler_h

#include "ModuleDLL.h"
#include "CUDADriverInfo.h"
#include <cuda_runtime_api.h>
#include <memory>

/** @brief namespace UtilsCUDA for encapsulating all the CUDA related code compiled by the NVCC compiler.
*
* @author Thanos Theo, 2018
* @version 14.0.0.0
*/
namespace UtilsCUDA
{
  /** @brief This class encapsulates usage of a collection of CUDA streams & the RAII C++ idiom.
  *
  * @author Thanos Theo, 2018
  * @version 14.0.0.0
  */
  class UTILS_CUDA_MODULE_API CUDAStreamsHandler final
  {
  public:

    CUDAStreamsHandler(const CUDADriverInfo& cudaDriverInfo, int device = 0, size_t numberOfStreams = 1, bool useStreamPriorities = true, int priorityType = cudaStreamNonBlocking) noexcept;
    ~CUDAStreamsHandler() noexcept; // no virtual destructor for data-oriented design (no up-casting should ever be used)

    void addCallback(std::size_t index, const cudaStreamCallback_t& callback, void* data) const noexcept;
    const cudaStream_t& operator[](std::size_t index) const noexcept { return cudaStreams_[index]; }

    CUDAStreamsHandler(const CUDAStreamsHandler&) = delete; // copy-constructor deleted
    CUDAStreamsHandler(CUDAStreamsHandler&&)      = delete; // move-constructor deleted
    CUDAStreamsHandler& operator=(const CUDAStreamsHandler&) = delete; //      assignment operator deleted
    CUDAStreamsHandler& operator=(CUDAStreamsHandler&&)      = delete; // move-assignment operator deleted

  private:

    void initialize(int device) noexcept;
    void uninitialize() const noexcept;

    std::size_t numberOfStreams_                 = 0;
    std::unique_ptr<cudaStream_t[]> cudaStreams_ = nullptr;

    bool useStreamPriorities_                    = false;
    int priorityType_                            = cudaStreamNonBlocking;
    int priorityHighest_                         = 0;
    int priorityLowest_                          = 0;
  };
} // namespace UtilsCUDA

#endif // __CUDAStreamsHandler_h