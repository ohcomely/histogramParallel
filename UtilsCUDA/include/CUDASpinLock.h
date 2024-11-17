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

#ifndef __CUDASpinLock_h
#define __CUDASpinLock_h

#include <cuda_runtime_api.h>

/** @brief namespace UtilsCUDA for encapsulating all the CUDA related code compiled by the NVCC compiler.
*
* @author Thanos Theo, 2018
* @version 14.0.0.0
*/
namespace UtilsCUDA
{
  /** @brief This class is based on the book'The CUDA Handbook - A comprehensive Guide to GPU Programming'.
  *
  *  CUDASpinLock.h:
  *  ==============
  *  Note from the book:
  *    The CUDA execution model imposes restrictions on the use of global memory atomics for synchronization, like for this CUDASpinLock class.
  *    Unlike CPU threads, some CUDA threads within a kernel launch may not begin execution until other threads in the same kernel have exited.
  *    On CUDA hardware, each SM can context switch a limited number of thread blocks, so any kernel launch with more than
  *    MaxThreadBlocksPerSM * NumSMs requires the first thread blocks to exit before more thread blocks can begin execution.
  *    As a result, it is important that developers not assume all of the threads in a given kernel launch are active.
  *
  *    Note 1: the CUDASpinLock::acquireBlock() function below is prone to deadlock if used for INTRABLOCK synchronization.
  *    Expected usage is for one thread in each block to attempt to acquireBlock() the CUDASpinLock, otherwise the divergent code execution tends to deadlock.
  *    This is unsuitable in any case, since the hardware supports so many better ways for threads within the same block to communicate and synchronize with one another,
  *    for example shared memory and __syncthreads(), respectively.
  *
  *    Note 2: the CUDASpinLock::acquireGrid() function below is prone to deadlock if used for INTRAGRID synchronization.
  *    Expected usage is for one thread in each block to attempt to acquireGrid() the CUDASpinLock, otherwise the divergent code execution tends to deadlock.
  *    Additionally, it also NEEDS a persistent kernel to use this way for grid synchronization, ie all blocks to be executing for this to work. See the CUDA Stress Test for a workable use case.
  *    The modern sanctioned way is to use CUDA 9.0 Cooperative Groups, ie <cooperative_groups.h>.
  *
  *  Example code for INTRABLOCK synchronization usage:
  *    __forceinline__ __device__
  *    void sumDoubles(double* pSum, int* spinlock, const double* in, size_t N, int* acquireCount)
  *    {
  *      SharedMemory<double> shared;
  *      CUDASpinLock<int> globalSpinlock(spinlock);
  *      for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < N; i += blockDim.x*gridDim.x)
  *      {
  *        shared[threadIdx.x] = in[i];
  *        __syncthreads();
  *        double blockSum = reduceBlock<double, double>();
  *        __syncthreads();
  *
  *        if (threadIdx.x == 0)
  *        {
  *          globalSpinlock.acquireBlock();
  *          *pSum += blockSum;
  *          __threadfence(); // function stalls current thread until its writes to global memory are guaranteed to be visible by all other threads in the grid
  *          globalSpinlock.releaseBlock();
  *        }
  *      }
  *    }
  *
  * @author Thanos Theo, Amir Shahvaran, 2019
  * @version 14.0.0.0
  */
  template<typename T>
  class CUDASpinLock final
  {
  public:

    __forceinline__ __device__ void acquireBlock() { while (atomicCAS( sequencer_, T(0), T(1))); }
    __forceinline__ __device__ void releaseBlock() {        atomicExch(sequencer_, T(0));        }

    __forceinline__ __device__ void acquireGrid()
    {
      while (atomicCAS(barrier_,   T(gridDim.x),  T(gridDim.x))  != T(gridDim.x))  {}
      while (atomicCAS(sequencer_, T(blockIdx.x), T(blockIdx.x)) != T(blockIdx.x)) {}
             atomicAdd(sequencer_, T( 1));
      while (atomicCAS(sequencer_, T(gridDim.x),  T(gridDim.x))  != T(gridDim.x))  {}
             atomicAdd(barrier_,   T(-1));
    }
    __forceinline__ __device__ void releaseGrid()
    {
      while (atomicCAS(sequencer_, T(blockIdx.x + 1), T(blockIdx.x + 1)) != T(blockIdx.x + 1)) {}
             atomicAdd(sequencer_, T(-1));
      while (atomicCAS(sequencer_,              T(0),              T(0)) !=              T(0)) {}
             atomicAdd(barrier_,   T( 1));
    }

    __forceinline__ __device__ explicit CUDASpinLock(T* sequencer, T* barrier = nullptr) noexcept : sequencer_(sequencer), barrier_(barrier)
    {
      *barrier = gridDim.x;
    }
    CUDASpinLock()  = delete;
    ~CUDASpinLock() = default; // no virtual destructor for data-oriented design (no up-casting should ever be used)
    CUDASpinLock(const CUDASpinLock&) = delete; // copy-constructor delete
    CUDASpinLock(CUDASpinLock&&)      = delete; // move-constructor delete
    CUDASpinLock& operator=(const CUDASpinLock&) = delete; //      assignment operator delete
    CUDASpinLock& operator=(CUDASpinLock&&)      = delete; // move-assignment operator delete

  private:

    T* sequencer_ = nullptr;
    T* barrier_   = nullptr;
  };
} // namespace UtilsCUDA

#endif // __CUDASpinLock_h