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

#ifndef __CUDAUtilityDeviceFunctions_h
#define __CUDAUtilityDeviceFunctions_h

#include "ModuleDLL.h"
#include <cuda.h>
#include <cstdint>
#include <functional>

/** @brief namespace UtilsCUDA for encapsulating all the CUDA related code compiled by the NVCC compiler.
*
* @author Thanos Theo, 2018
* @version 14.0.0.0
*/
namespace UtilsCUDA
{
  /** @brief This class encapsulates all the CUDA related device only utility functions.
  *
  * NOTE: *** These do NOT work for shared-memory atomics at this time! ***
  * More information: https://github.com/treecode/Bonsai/blob/master/runtime/profiling/derived_atomic_functions.h
  *
  * @author Thanos Theo, 2019
  * @version 14.0.0.0
  */
  class UTILS_CUDA_MODULE_API CUDAUtilityDeviceFunctions final
  {
  public:

    /** @brief globalThreadCount() calculates the total number of threads in the running grid.
    *
    * @author Amir Shahvarani, 2019
    */
    static __forceinline__ __device__ std::uint32_t globalThreadCount()
    {
      return blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z;
    }

    /** @brief globalIdx() calculates the global index of a given thread.
    *
    * @author Amir Shahvarani, 2019
    */
    static __forceinline__ __device__ uint3 globalIndex()
    {
      return uint3{ threadIdx.x + blockIdx.x * blockDim.x,
                    threadIdx.y + blockIdx.y * blockDim.y,
                    threadIdx.z + blockIdx.z * blockDim.z };
    }

    /** @brief globalLinearIdx() calculates the linear order of a given thread based on given dimension.
    *
    * NOTE: index must be within the boundaries of dimension.
    *
    * @author Amir Shahvarani, 2019
    */
    static __forceinline__ __device__ std::uint32_t linearIndex(const uint3& index, const dim3& dimension)
    {
      return index.x +
              index.y * dimension.x +
              index.z * dimension.y * dimension.x;
    }

    /** @brief globalLinearIdx() calculates the global linear order of a given thread.
    *
    * @author Amir Shahvarani, 2019
    */
    static __forceinline__ __device__ std::uint32_t globalLinearIndex()
    {
      uint3 globalIndex = CUDAUtilityDeviceFunctions::globalIndex();
      return CUDAUtilityDeviceFunctions::linearIndex(globalIndex, uint3{ blockDim.x * gridDim.x, blockDim.y * gridDim.y, blockDim.z * gridDim.z });
    }

    /** @brief Device atomic min for floats.
    */
    static __forceinline__ __device__ float atomicMin(float* address, float value)
    {
      return atomicArithmeticOpFloat(address, value, std::less<float>());
    }

    /** @brief Device atomic max for floats.
    */
    static __forceinline__ __device__ float atomicMax(float* address, float value)
    {
      return atomicArithmeticOpFloat(address, value, std::greater<float>());
    }

    /** @brief Device atomic min for doubles.
    */
    static __forceinline__ __device__ double atomicMin(double* address, double value)
    {
      return atomicArithmeticOpDouble(address, value, std::less<double>());
    }

    /** @brief Device atomic max for doubles.
    */
    static __forceinline__ __device__ double atomicMax(double* address, double value)
    {
      return atomicArithmeticOpDouble(address, value, std::greater<double>());
    }

    CUDAUtilityDeviceFunctions()  = delete;
    ~CUDAUtilityDeviceFunctions() = delete;
    CUDAUtilityDeviceFunctions(const CUDAUtilityDeviceFunctions&) = delete;
    CUDAUtilityDeviceFunctions(CUDAUtilityDeviceFunctions&&)      = delete;
    CUDAUtilityDeviceFunctions& operator=(const CUDAUtilityDeviceFunctions&) = delete;
    CUDAUtilityDeviceFunctions& operator=(CUDAUtilityDeviceFunctions&&)      = delete;

  private:

    /** @brief atomicArithmeticOpFloat() for floats
    *
    * For all float & double atomics below:
    *       Must do the compare with integers, not floating point,
    *       since NaN is never equal to any other NaN
    *
    * NOTE: *** These do NOT work for shared-memory atomics at this time! ***
    * More information: https://github.com/treecode/Bonsai/blob/master/runtime/profiling/derived_atomic_functions.h
    *
    * @author Thanos Theo, 2019
    */
    template<typename ComparisonOp>
    static __forceinline__ __device__ float atomicArithmeticOpFloat(float* address, float value, ComparisonOp comparisonOp)
    {
      int ret = __float_as_int(*address);
      while (comparisonOp(value, __int_as_float(ret)))
      {
        int old = ret;
        if ((ret = atomicCAS(reinterpret_cast<int*>(address), old, __float_as_int(value))) == old)
        {
          break;
        }
      }

      return __int_as_float(ret);
    }

    /** @brief atomicArithmeticOpDouble() for doubles
    *
    * For all float & double atomics below:
    *       Must do the compare with integers, not floating point,
    *       since NaN is never equal to any other NaN
    *
    * NOTE: *** These do NOT work for shared-memory atomics at this time! ***
    * More information: https://github.com/treecode/Bonsai/blob/master/runtime/profiling/derived_atomic_functions.h
    *
    * @author Thanos Theo, 2019
    */
    template<typename ComparisonOp>
    static __forceinline__ __device__ double atomicArithmeticOpDouble(double* address, double value, ComparisonOp comparisonOp)
    {
      unsigned long long ret = __double_as_longlong(*address);
      while (comparisonOp(value, __longlong_as_double(ret)))
      {
        unsigned long long old = ret;
        if ((ret = atomicCAS(reinterpret_cast<unsigned long long*>(address), old, __double_as_longlong(value))) == old)
        {
          break;
        }
      }

      return __longlong_as_double(ret);
    }
  };
} // namespace UtilsCUDA

#endif // __CUDAUtilityDeviceFunctions_h