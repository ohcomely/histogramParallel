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

#ifndef __CPUParallelismNCP_h
#define __CPUParallelismNCP_h

#include "../ModuleDLL.h"
#include "FunctionView.h"
#include "ThreadOptions.h"
#include <cstdint>

/** @brief Namespace Utils contains utility classes with mainly static CPU related methods.
* @author Thanos Theo, 2009-2018
* @version 14.0.0.0
*/
namespace Utils
{
  /** @brief Namespace CPUParallelism encapsulates usage of the N-CP parallelism idea.
  *
  *  This namespace encapsulates usage of the N-CP parallelism idea.\n
  *  CPUParallelism libraries originally based on with further extensions: http://www.manning.com/williams/.\n
  *  The N-CP idea was based on: http://www.biolayout.org/wp-content/uploads/2013/01/Manuscript.pdf.\n
  *  Further inspiration was found here: http://jcip.net.s3-website-us-east-1.amazonaws.com/.\n
  *
  * @author Thanos Theo, 2009-2018
  * @version 14.0.0.0
  */
  namespace CPUParallelism
  {
    // forward declarations
    class ThreadPool;

    /// auxiliary parallelism functions
    UTILS_MODULE_API std::size_t numberOfHardwareThreads();
    UTILS_MODULE_API void threadSleep(std::size_t millisecs);

    /// parallelFor() versions with only the index provided
    UTILS_MODULE_API void parallelFor(                        std::size_t indexEnd, const FunctionView<void(std::size_t)>& kernelFunction, std::size_t numberOfThreads = numberOfHardwareThreads(), std::uint64_t affinityMask = AFFINITY_MASK_ALL, std::size_t priority = PRIORITY_NONE);
    UTILS_MODULE_API void parallelFor(std::size_t indexStart, std::size_t indexEnd, const FunctionView<void(std::size_t)>& kernelFunction, std::size_t numberOfThreads = numberOfHardwareThreads(), std::uint64_t affinityMask = AFFINITY_MASK_ALL, std::size_t priority = PRIORITY_NONE);
    UTILS_MODULE_API void parallelFor(                        std::size_t indexEnd, const FunctionView<void(std::size_t)>& kernelFunction, ThreadPool& threadPool);
    UTILS_MODULE_API void parallelFor(std::size_t indexStart, std::size_t indexEnd, const FunctionView<void(std::size_t)>& kernelFunction, ThreadPool& threadPool);

    /// parallelFor() versions with both the index & threadIdx provided
    UTILS_MODULE_API void parallelForThreadLocal(                        std::size_t indexEnd, const FunctionView<void(std::size_t, std::size_t)>& kernelFunction, std::size_t numberOfThreads = numberOfHardwareThreads(), std::uint64_t affinityMask = AFFINITY_MASK_ALL, std::size_t priority = PRIORITY_NONE);
    UTILS_MODULE_API void parallelForThreadLocal(std::size_t indexStart, std::size_t indexEnd, const FunctionView<void(std::size_t, std::size_t)>& kernelFunction, std::size_t numberOfThreads = numberOfHardwareThreads(), std::uint64_t affinityMask = AFFINITY_MASK_ALL, std::size_t priority = PRIORITY_NONE);
    UTILS_MODULE_API void parallelForThreadLocal(                        std::size_t indexEnd, const FunctionView<void(std::size_t, std::size_t)>& kernelFunction, ThreadPool& threadPool);
    UTILS_MODULE_API void parallelForThreadLocal(std::size_t indexStart, std::size_t indexEnd, const FunctionView<void(std::size_t, std::size_t)>& kernelFunction, ThreadPool& threadPool);
  }
}

#endif // __CPUParallelismNCP_h