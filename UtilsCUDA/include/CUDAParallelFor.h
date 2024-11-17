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

#ifndef __CUDAParallelFor_h
#define __CUDAParallelFor_h

#include "CUDAUtilityFunctions.h"
#include "CUDAUtilityDeviceFunctions.h"
#include <cuda_runtime_api.h>
#include <vector_types.h>
#include <cstdint>
#include <tuple>
#include <utility>

/** @brief namespace UtilsCUDA for encapsulating all the CUDA related code compiled by the NVCC compiler.
*
* @author Thanos Theo, 2018
* @version 14.0.0.0
*/
namespace UtilsCUDA
{
  /** @brief namespace CUDAParallelFor for encapsulating a CUDA related parallelFor() construct using a combination of C99 variadic macros & C++11 variadic templates.
  *
  * @author Thanos Theo, Amir Shahvarani, 2019
  * @version 14.0.0.0
  */
  namespace CUDAParallelFor
  {
    template<typename FunctionType, typename... Args>
    __global__
    void kernel(std::size_t arraySize, const FunctionType& lambdaFunction, Args... args)
    {
      std::size_t index = std::size_t(CUDAUtilityDeviceFunctions::globalLinearIndex());
      if (index < arraySize)
      {
        lambdaFunction(index, args...);
      }
    }

    template<typename FunctionType, typename... Args>
    static inline void launchCUDAParallelForWithDimensions(std::size_t arraySize, const dim3& blocks, const dim3& threads, const FunctionType& lambdaFunction, Args&&... args)
    {
      kernel<<<blocks, threads>>>(arraySize, lambdaFunction, std::forward<Args>(args)...);
      CUDAError_checkCUDAErrorDebug(cudaPeekAtLastError());
    }

    template<typename FunctionType, typename... Args>
    static inline void launchCUDAParallelFor(std::size_t arraySize, const FunctionType& lambdaFunction, Args&&... args)
    {
      dim3 blocks;
      dim3 threads;
      std::tie(blocks, threads) = CUDAUtilityFunctions::calculateCUDA1DKernelDimensions(arraySize);
      launchCUDAParallelForWithDimensions<FunctionType, Args...>(arraySize, blocks, threads, lambdaFunction, std::forward<Args>(args)...);
    }

    template<typename FunctionType, typename... Args>
    static inline void launchCUDAParallelForInStreamWithDimensions(std::size_t arraySize, const dim3& blocks, const dim3& threads, std::size_t sharedMemoryBytes, const cudaStream_t& stream, const FunctionType& lambdaFunction, Args&&... args)
    {
      kernel<<<blocks, threads, sharedMemoryBytes, stream>>>(arraySize, lambdaFunction, std::forward<Args>(args)...);
      CUDAError_checkCUDAErrorDebug(cudaPeekAtLastError());
    }

    template<typename FunctionType, typename... Args>
    static inline void launchCUDAParallelForInStream(std::size_t arraySize, std::size_t sharedMemoryBytes, const cudaStream_t& stream, const FunctionType& lambdaFunction, Args&&... args)
    {
      dim3 blocks;
      dim3 threads;
      std::tie(blocks, threads) = CUDAUtilityFunctions::calculateCUDA1DKernelDimensions(arraySize);
      launchCUDAParallelForInStreamWithDimensions<FunctionType, Args...>(arraySize, blocks, threads, sharedMemoryBytes, stream, lambdaFunction, std::forward<Args>(args)...);
    }
  }
} // namespace UtilsCUDA

#endif // __CUDAParallelFor_h