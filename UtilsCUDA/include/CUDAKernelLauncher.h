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

#ifndef __CUDAKernelLauncher_h
#define __CUDAKernelLauncher_h

#include "CUDAUtilityFunctions.h"
#include "CUDAParallelFor.h"
#include <cuda_runtime_api.h>
#include <vector_types.h>
#include <cstdint>
#include <utility>

/** @brief namespace UtilsCUDA for encapsulating all the CUDA related code compiled by the NVCC compiler.
*
* @author Thanos Theo, 2018
* @version 14.0.0.0
*/
namespace UtilsCUDA
{
  /**
  * @brief CUDA helper class to perform a more readable kernel launch in code with the fluent builder pattern.
  *
  * The main purpose of this class to make kernel launches more readable and explicit without knowing the exact launch syntax of CUDA.
  * It also improves readability in an IDE that does not properly parse CUDA <<<>>> syntax.
  *
  * Example Usage:
  *    KernelLauncher::create().setGrid({1, 100}).setBlock({1}).setStream(stream).synchronized().run(kernelFunction, arg1, arg2, arg3);
  *    This is equivalent to:
  *      kernelFunction<<<{1, 100}, {1}, 0, stream>>>(arg1, arg2, arg3);
  *      CUDAError_checkCUDAErrorDebug(cudaPeekAtLastError());
  *      CUDAError_checkCUDAError(cudaStreamSynchronize(stream));
  *
  * Or:
  *
  *    KernelLauncher::create().setGrid({1, 100}).setBlock({1}).setStream(stream).synchronized().runCUDAParallelFor(arraySize, kernelLambda, arg1, arg2, arg3);
  *    This is equivalent to:
  *      kernelFunction<<<{1, 100}, {1}, 0, stream>>>(arraySize, kernelLambda, arg1, arg2, arg3);
  *      CUDAError_checkCUDAErrorDebug(cudaPeekAtLastError());
  *      CUDAError_checkCUDAError(cudaStreamSynchronize(stream));
  *
  * default values are:
  *   gridSize     = {1}
  *   blockSize    = {1}
  *   stream       = 0 (default CUDA stream)
  *   sharedMemory = 0
  *
  * @author David Lenz, Amir Shahvarani, Thanos Theo, 2019
  * @version 14.0.0.0
  */
  class KernelLauncher final
  {
  public:

    /**
    * Create a KernelLauncher.
    */
    static KernelLauncher create()
    {
      return KernelLauncher{};
    }

    /**
    * Launches the CUDAParallelFor kernel with its given lambda and arguments.
    */
    template <typename FunctionType, typename... Args>
    void runCUDAParallelFor(std::size_t arraySize, const FunctionType& lambdaFunction, Args&&... args)
    {
      CUDAParallelFor::launchCUDAParallelForInStreamWithDimensions<FunctionType, Args...>(arraySize, gridSize_, blockSize_, sharedMemoryBytes_, stream_, lambdaFunction, std::forward<Args>(args)...);
      CUDAError_checkCUDAErrorDebug(cudaPeekAtLastError());
      if (synchronize_)
      {
        CUDAError_checkCUDAError(cudaStreamSynchronize(stream_));
      }
    }

    /**
    * Launches the generic CUDA kernel with its given arguments.
    */
    template <typename FunctionType, typename... Args>
    void run(const FunctionType& kernelFunction, Args&&... args)
    {
      kernelFunction<<<gridSize_, blockSize_, sharedMemoryBytes_, stream_>>>(std::forward<Args>(args)...);
      CUDAError_checkCUDAErrorDebug(cudaPeekAtLastError());
      if (synchronize_)
      {
        CUDAError_checkCUDAError(cudaStreamSynchronize(stream_));
      }
    }

    /**
    * Specifies the  grid size of the problem (i.e. the number of blocks).
    */
    KernelLauncher& setGrid(const dim3& gridSize)
    {
      gridSize_ = gridSize;
      return *this;
    }

    /**
    * Specifies the block size of the problem (i.e. the number of threads).
    */
    KernelLauncher& setBlock(const dim3& blockSize)
    {
      blockSize_ = blockSize;
      return *this;
    }

    /**
    * Specifies the grid and block size of the problem
    */
    KernelLauncher& setGridAndBlock(const std::tuple<dim3, dim3>& gridBlockSizes)
    {
      gridSize_  = std::get<0>(gridBlockSizes);
      blockSize_ = std::get<1>(gridBlockSizes);
      return *this;
    }

    /**
    * Preallocate shared memory of a specific size.
    */
    KernelLauncher& setSharedMemory(std::size_t sharedMemoryBytes)
    {
      sharedMemoryBytes_ = sharedMemoryBytes;
      return *this;
    }

    /**
    * Perform the kernel launch in a specific stream.
    */
    KernelLauncher& setStream(const cudaStream_t& stream)
    {
      stream_ = stream;
      return *this;
    }

    /**
    * Block kernel execution until the kernel finishes in the given thread.
    */
    KernelLauncher& synchronized()
    {
      synchronize_ = true;
      return *this;
    }

    /**
    * Does not block kernel execute until the kernel finishes in the given thread.
    */
    KernelLauncher& asynchronous()
    {
      synchronize_ = false;
      return *this;
    }

    // object can only be moved out (in create() function)
    ~KernelLauncher() = default;
    KernelLauncher(KernelLauncher&)  = default;
    KernelLauncher(KernelLauncher&&) = default;
    KernelLauncher& operator=(KernelLauncher&)  = default;
    KernelLauncher& operator=(KernelLauncher&&) = default;

private:

    // constructor is private and object can only be created with the static function KernelLauncher::run()
    KernelLauncher()  = default;

    dim3 gridSize_                 = 1;
    dim3 blockSize_                = 1;
    std::size_t sharedMemoryBytes_ = 0;
    cudaStream_t stream_           = nullptr; // 0 (default CUDA stream)
    bool synchronize_              = false;
  };
} // namespace UtilsCUDA

#endif // __CUDAKernelLauncher_h