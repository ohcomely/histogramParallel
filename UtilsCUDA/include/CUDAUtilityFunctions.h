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

#ifndef __CUDAUtilityFunctions_h
#define __CUDAUtilityFunctions_h

#include "ModuleDLL.h"
#include "EnvironmentConfig.h"
#include "MathConstants.h"
#include "CUDADriverInfo.h"
#include <cuda_runtime_api.h>
#include <vector_functions.h>
#include <vector_types.h>
#include <cuda.h>
#include <curand.h>
#include <string>
#include <sstream>
#include <cstring> // for memcpy() call
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <type_traits>
#include <tuple>

// DebugConsole macro functions below

#define DebugConsole_printfCUDALine(...) \
          do { if (GPU_FRAMEWORK_CUDA_CONSOLE) UtilsCUDA::CUDAUtilityFunctions::printfCUDAImpl(__VA_ARGS__); } while (0)

#define DebugConsole_printfCUDALineDetailed(...)                                                                                                                                          \
          do { if (GPU_FRAMEWORK_CUDA_CONSOLE) { UtilsCUDA::CUDAUtilityFunctions::printfCUDAImpl("%s%s%s%s%s%d\n", "\nFile: ", __FILE__, "\nFunction: ", __func__, "\nLine: ", __LINE__); \
                                                 UtilsCUDA::CUDAUtilityFunctions::printfCUDAImpl(__VA_ARGS__); } } while (0)

// ReleaseConsole macro functions below

#define ReleaseConsole_printfCUDALine(...) \
          do { UtilsCUDA::CUDAUtilityFunctions::printfCUDAImpl(__VA_ARGS__); } while (0)

#define ReleaseConsole_printfCUDALineDetailed(...)                                                                                                        \
          do { { UtilsCUDA::CUDAUtilityFunctions::printfCUDAImpl("%s%s%s%s%s%d\n", "\nFile: ", __FILE__, "\nFunction: ", __func__, "\nLine: ", __LINE__); \
                 UtilsCUDA::CUDAUtilityFunctions::printfCUDAImpl(__VA_ARGS__); } } while (0)

// CUDAError & CUDADriverInfo macro functions below

#define CUDAError_checkCUDAError(x) \
          do { auto errnum = (x); if (GPU_FRAMEWORK_CUDA_ERROR) { UtilsCUDA::CUDAUtilityFunctions::checkCUDAErrorImpl(errnum, __FILE__, __func__, __LINE__); } } while (0)

#define CUDAError_checkCUDAErrorDebug(x) \
          do { if (GPU_FRAMEWORK_CUDA_ERROR_DEBUG) { auto errnum = (x); UtilsCUDA::CUDAUtilityFunctions::checkCUDAErrorImpl(errnum, __FILE__, __func__, __LINE__); } } while (0)

#define CUDADriverInfo_report(x) \
                                   do { if (GPU_FRAMEWORK_CUDA_DRIVER_INFO) (x); } while (0)

/** @brief namespace UtilsCUDA for encapsulating all the CUDA related code compiled by the NVCC compiler.
*
* @author Thanos Theo, 2018
* @version 14.0.0.0
*/
namespace UtilsCUDA
{
  /** @brief This struct encapsulates all the CUDA related utility functions.
  *
  * @author Thanos Theo, 2018
  * @version 14.0.0.0
  */
  struct UTILS_CUDA_MODULE_API CUDAUtilityFunctions final
  {
    template<typename... Args>
    static __forceinline__ __host__ __device__ void printfCUDAImpl(const char* format, Args... args)
    {
      printf(format, args...);
    }

    /**
    *  GLSL-style equal function.
    */
    template<typename T>
    static __forceinline__ __host__ __device__ bool equal(const T left, const T right, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr)
    {
      return std::abs(left - right) <= std::numeric_limits<T>::epsilon();
    }

    /*
    *  GLSL-style sign function.
    */
    template<typename T>
    static __forceinline__ __host__ __device__ T sign(const T x, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr)
    {
      return (x < T(0)) ? T(-1) : (x > T(0)) ? T(1) : T(0);
    }

    /**
    *  GLSL-style fract function (float version).
    */
    template<typename T>
    static __forceinline__ __host__ __device__ T fract(T x, std::enable_if_t<std::is_same<T, float>::value>* = nullptr)
    {
      return x - floorf(x);
    }

    /**
    *  GLSL-style fract function (double version).
    */
    template<typename T>
    static __forceinline__ __host__ __device__ T fract(T x, std::enable_if_t<std::is_same<T, double>::value>* = nullptr)
    {
      return x - floor(x);
    }

    /**
    *  Conversion function from degrees to radians.
    */
    template <typename T>
    static __forceinline__ __host__ __device__ T toRadians(const T degrees, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr)
    {
      return degrees * (T(Utils::PI<T>) / T(180));
    }

    /**
    *  Conversion function from radians to degrees.
    */
    template <typename T>
    static __forceinline__ __host__ __device__ T toDegrees(const T radians, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr)
    {
      return radians / (T(Utils::PI<T>) / T(180));
    }

    /**
    *  GLSL-style dot function (float version).
    */
    static __forceinline__ __host__ __device__ float dot(const float2& a, const float2& b)
    {
      return a.x * b.x + a.y * b.y;
    }

    /**
    *  GLSL-style dot function (double version).
    */
    static __forceinline__ __host__ __device__ double dot(const double2& a, const double2& b)
    {
      return a.x * b.x + a.y * b.y;
    }

    /** @brief This function returns uniformly distributed float values in the range [0, 1] (float version).
    *
    * @author Thanos Theo, 2018
    */
    static __forceinline__ __host__ __device__ float rand1(const float2& seed)
    {
      const float dotProduct = dot(seed, make_float2(12.9898f, 78.233f));
      return fract(sinf(dotProduct) * 43758.5453f);
    }

    /** @brief This function returns uniformly distributed double values in the range [0, 1] (double version).
    *
    * @author Thanos Theo, 2018
    */
    static __forceinline__ __host__ __device__ double rand1(const double2& seed)
    {
      const double dotProduct = dot(seed, make_double2(12.9898, 78.233));
      return fract(sin(dotProduct) * 43758.5453);
    }

    /** @brief This function returns uniformly distributed float2 values in the range [0, 1] (float version).
    *
    * @author Thanos Theo, 2018
    */
    static __forceinline__ __host__ __device__ float2 rand2(const float2& seed)
    {
      const float dotProduct = dot(seed, make_float2(12.9898f, 78.233f));
      return make_float2(fract(sinf(dotProduct) * 43758.5453f), fract(sinf(2.0f * dotProduct) * 43758.5453f));
    }

    /** @brief This function returns uniformly distributed double2 values in the range [0, 1] (double version).
    *
    * @author Thanos Theo, 2018
    */
    static __forceinline__ __host__ __device__ double2 rand2(const double2& seed)
    {
      const double dotProduct = dot(seed, make_double2(12.9898, 78.233));
      return make_double2(fract(sin(dotProduct) * 43758.5453), fract(sin(2.0 * dotProduct) * 43758.5453));
    }

    /** @brief This function returns uniformly distributed float3 values in the range [0, 1] (float version).
    *
    * @author Thanos Theo, 2018
    */
    static __forceinline__ __host__ __device__ float3 rand3(const float2& seed)
    {
      const float dotProduct = dot(seed, make_float2(12.9898f, 78.233f));
      return make_float3(fract(sinf(dotProduct) * 43758.5453f), fract(sinf(2.0f * dotProduct) * 43758.5453f), fract(sinf(3.0f * dotProduct) * 43758.5453f));
    }

    /** @brief This function returns uniformly distributed double3 values in the range [0, 1] (double version).
    *
    * @author Thanos Theo, 2018
    */
    static __forceinline__ __host__ __device__ double3 rand3(const double2& seed)
    {
      const double dotProduct = dot(seed, make_double2(12.9898, 78.233));
      return make_double3(fract(sin(dotProduct) * 43758.5453), fract(sin(2.0 * dotProduct) * 43758.5453), fract(sin(3.0 * dotProduct) * 43758.5453));
    }

    /** @brief This function returns uniformly distributed float4 values in the range [0, 1] (float version).
    *
    * @author Thanos Theo, 2018
    */
    static __forceinline__ __host__ __device__ float4 rand4(const float2& seed)
    {
      const float dotProduct = dot(seed, make_float2(12.9898f, 78.233f));
      return make_float4(fract(sinf(dotProduct) * 43758.5453f), fract(sinf(2.0f * dotProduct) * 43758.5453f), fract(sinf(3.0f * dotProduct) * 43758.5453f), fract(sinf(4.0f * dotProduct) * 43758.5453f));
    }

    /** @brief This function returns uniformly distributed double4 values in the range [0, 1] (double version).
    *
    * @author Thanos Theo, 2018
    */
    static __forceinline__ __host__ __device__ double4 rand4(const double2& seed)
    {
      const double dotProduct = dot(seed, make_double2(12.9898, 78.233));
      return make_double4(fract(sin(dotProduct) * 43758.5453), fract(sin(2.0 * dotProduct) * 43758.5453), fract(sin(3.0 * dotProduct) * 43758.5453), fract(sin(4.0 * dotProduct) * 43758.5453));
    }

    /** @brief Seed generator for the Linear Congruential Generator (LGC).
    *
    * Note: default loop (for unrolling) with value of 16.
    *
    * @author Thanos Theo, 2018
    */
    template<std::uint32_t N = 16>
    static __forceinline__ __host__ __device__ std::uint32_t seedGenerator(std::uint32_t value0, std::uint32_t value1)
    {
      std::uint32_t v0 = value0;
      std::uint32_t v1 = value1;
      std::uint32_t s0 = 0;

      for (std::uint32_t n = 0; n < N; ++n)
      {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
      }

      return v0;
    }

    /** @brief Generate random uint32_t values in the [0, 2^24) range with the Linear Congruential Generator (LGC).
    *
    * @author Thanos Theo, 2018
    */
    static __forceinline__ __host__ __device__ std::uint32_t rand1u(std::uint32_t& seed)
    {
      const std::uint32_t LCG_A = 1664525u;
      const std::uint32_t LCG_C = 1013904223u;
      seed = (LCG_A * seed + LCG_C);

      return seed & 0x00FFFFFF;
    }

    /** @brief Generate random float values in the [0, 1) range with the Linear Congruential Generator (LGC).
    *
    * @author Thanos Theo, 2018
    */
    static __forceinline__ __host__ __device__ float rand1f(std::uint32_t& seed)
    {
      return float(rand1u(seed)) / float(0x01000000);
    }

    /** @brief Generate random float2 values in the [0, 1) range with the Linear Congruential Generator (LGC).
    *
    * @author Thanos Theo, 2018
    */
    static __forceinline__ __host__ __device__ float2 rand2f(std::uint32_t& seed)
    {
      return make_float2(rand1f(seed), rand1f(seed));
    }

    /** @brief Generate random float3 values in the [0, 1) range with the Linear Congruential Generator (LGC).
    *
    * @author Thanos Theo, 2018
    */
    static __forceinline__ __host__ __device__ float3 rand3f(std::uint32_t& seed)
    {
      return make_float3(rand1f(seed), rand1f(seed), rand1f(seed));
    }

    /** @brief Generate random float4 values in the [0, 1) range with the Linear Congruential Generator (LGC).
    *
    * @author Thanos Theo, 2018
    */
    static __forceinline__ __host__ __device__ float4 rand4f(std::uint32_t& seed)
    {
      return make_float4(rand1f(seed), rand1f(seed), rand1f(seed), rand1f(seed));
    }

    /** @brief This function is the GPU version checkAbsoluteError to be used with CUDA.
    *
    * @author Thanos Theo, 2018
    */
    template <typename T>
    static __forceinline__ __host__ __device__ bool checkAbsoluteError(T a, T b, T error)
    {
      return (std::abs(a - b) > error);
    }

    /** @brief This function is the GPU version checkRelativeError to be used with CUDA.
    *
    * @author Thanos Theo, 2018
    */
    template <typename T>
    static inline bool checkRelativeError(T a, T b, T relativeError)
    {
      return (std::abs((a - b) / a) > relativeError);
    }

    /** @brief Get the float32 bit representation to a uint32.
    *
    * @author Thanos Theo, 2018
    */
    static __forceinline__ __host__ __device__ std::uint32_t asUint32(float value)
    {
      // the C language explicitly permits type-punning through a union whereas modern C++ has no such permission
      // union { std::uint32_t uint32Value; float float32Value; }; // anonymous union
      std::uint32_t uint32Value;
      std::memcpy(&uint32Value, &value, sizeof(std::uint32_t));

      return uint32Value;
    }

    /** @brief Get the uint32 bit representation to a float32.
    *
    * @author Thanos Theo, 2018
    */
    static __forceinline__ __host__ __device__ float asFloat32(std::uint32_t value)
    {
      // the C language explicitly permits type-punning through a union whereas modern C++ has no such permission
      // union { std::uint32_t uint32Value; float float32Value; }; // anonymous union
      float float32Value;
      std::memcpy(&float32Value, &value, sizeof(float));

      return float32Value;
    }

    /** @brief Flip a float32 for make it sortable: finds SIGN of fp number, so: if it's 1 (negative float32) it flips all bits, if it's 0 (positive float32) it flips the sign only. Needs IEEE 754 hardware compliance. Based on http://stereopsis.com/radix.html.
    *
    * @author Thanos Theo, 2018
    */
    static __forceinline__ __host__ __device__ std::uint32_t float32Flip(float unflippedFloatValue)
    {
      const std::uint32_t f    = asUint32(unflippedFloatValue);
      const std::uint32_t mask = -std::int32_t(f >> 31) | 0x80000000;

      return f ^ mask;
    }

    /** @brief Unflip a float32 back (invert float32Flip() above): signed was flipped from above, so: if sign is 1 (negative) it flips the sign bit back, if if sign is 0 (positive) it flips all bits back. Needs IEEE 754 hardware compliance. Based on http://stereopsis.com/radix.html.
    *
    * @author Thanos Theo, 2018
    */
    static __forceinline__ __host__ __device__ float float32Unflip(std::uint32_t flippedFloatValue)
    {
      const std::uint32_t f    = flippedFloatValue;
      const std::uint32_t mask = ((f >> 31) - 1) | 0x80000000;

      return asFloat32(f ^ mask);
    }

    /** @brief Get the float64 bit representation to a uint64.
    *
    * @author Thanos Theo, 2018
    */
    static __forceinline__ __host__ __device__ std::uint64_t asUint64(double value)
    {
      // the C language explicitly permits type-punning through a union whereas modern C++ has no such permission
      // union { std::uint64_t uint64Value; double float64Value; }; // anonymous union
      std::uint64_t uint64Value;
      std::memcpy(&uint64Value, &value, sizeof(std::uint64_t));

      return uint64Value;
    }

    /** @brief Get the uint64 bit representation to a float64.
    *
    * @author Thanos Theo, 2018
    */
    static __forceinline__ __host__ __device__ double asFloat64(std::uint64_t value)
    {
      // the C language explicitly permits type-punning through a union whereas modern C++ has no such permission
      // union { std::uint64_t uint64Value; double float64Value; }; // anonymous union
      double float64Value;
      std::memcpy(&float64Value, &value, sizeof(double));

      return float64Value;
    }

    /** @brief Flip a float64 for make it sortable: finds SIGN of fp number, so: if it's 1 (negative float64) it flips all bits, if it's 0 (positive float64) it flips the sign only. Needs IEEE 754 hardware compliance. Based on http://stereopsis.com/radix.html.
    *
    * @author Thanos Theo, 2018
    */
    static __forceinline__ __host__ __device__ std::uint64_t float64Flip(double unflippedFloatValue)
    {
      const std::uint64_t f    = asUint64(unflippedFloatValue);
      const std::uint64_t mask = -std::int64_t(f >> 63) | 0x8000000000000000;

      return f ^ mask;
    }

    /** @brief Unflip a float64 back (invert float64Flip() above): signed was flipped from above, so: if sign is 1 (negative) it flips the sign bit back, if if sign is 0 (positive) it flips all bits back. Needs IEEE 754 hardware compliance. Based on http://stereopsis.com/radix.html.
    *
    * @author Thanos Theo, 2018
    */
    static __forceinline__ __host__ __device__ double float64Unflip(std::uint64_t flippedFloatValue)
    {
      const std::uint64_t f    = flippedFloatValue;
      const std::uint64_t mask = ((f >> 63) - 1) | 0x8000000000000000;

      return asFloat64(f ^ mask);
    }

    /** @brief Templatized pow<EXPONENT>(T) function with 1 base case needed: pow<0> where T is an arithmetic primitive type.
    *
    * @author Ben Dart, Amir Shahvarani, Inaki Pujol, Thanos Theo, 2019
    */
    template<std::size_t EXPONENT, typename T>
    static __forceinline__ __host__ __device__ T pow(T value, std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr)
    {
      switch (EXPONENT)
      {
        case 0:
           return T(1);

        default:
          const T halfExp = pow<(EXPONENT >> 1)>(value);
          return halfExp * halfExp * ((EXPONENT & 0x1) ? value : T(1));
      }
    }

    /** Fast memset implementation with alignment considered. For use in both host & device code.
    *
    *   Note: This memset variant is meant to be used within device code and not to replace the cudaMemset() kernel call.
    *         For the CPU side, we assume that std::memset() call is already optimal and probably vectorized.
    *         We provide below the __host__ variant for verification purposes.
    *
    * @author Leonid Volnin, 2019
    * @version 14.0.0.0
    */
    static __forceinline__ __host__ __device__ void* memset(void* __restrict ptr, std::uint8_t value, std::size_t length)
    {
      std::uint8_t* __restrict beginPtr   = reinterpret_cast<std::uint8_t*>(ptr);
      // if the buffer is too short, just set alignedPtr to the end of the buffer, so it will be memset in first loop
      std::uint8_t* __restrict alignedPtr = (length > 8) ? reinterpret_cast<std::uint8_t*>((std::size_t(ptr) + 7) & ~0x7) : beginPtr + length;
      std::uint64_t value8                = (value) | (value << 8) | (value << 16) | (value << 24);
      value8 |= (value8 << 32);

      std::uint8_t* __restrict p = beginPtr;
      for (; p < alignedPtr; ++p)
      {
         *p = value;
      }

      for (; p <= beginPtr + length - 8; p += 8)
      {
        *reinterpret_cast<std::uint64_t*>(p) = value8;
      }

      for (; p < beginPtr + length; ++p)
      {
        *p = value;
      }

      return ptr;
    }

    static std::uint8_t getCUDAExitCodeOffset();

    static void checkCUDAErrorImpl(const cudaError_t&    errnum, const char* file, const char* function, int line, bool abort = true);

    static void checkCUDAErrorImpl(const CUresult&       errnum, const char* file, const char* function, int line, bool abort = true);

    static void checkCUDAErrorImpl(const curandStatus_t& errnum, const char* file, const char* function, int line, bool abort = true);

    static std::uint32_t getWarpSize();

    static dim3 getDefaultThreads1DDimensions();

    static dim3 getDefaultThreads2DDimensions();

    /** The calculateCUDA2DKernelDimensions() function efficiently calculates the dimensions for a CUDA 1D kernel.
    *
    * @author Amir Shahvarani, 2019
    * @version 14.0.0.0
    */
    static std::tuple<dim3, dim3> calculateCUDA1DKernelDimensions(std::size_t arraySize, const dim3& threads1D = getDefaultThreads1DDimensions());

    /** The calculateCUDA2DKernelDimensions() function efficiently calculates the (non power-of-two) square dimensions for a CUDA 2D kernel.
    *
    * @author Thanos Theo, 2019
    * @version 14.0.0.0
    */
    static std::tuple<dim3, dim3> calculateCUDA2DKernelDimensions(std::size_t arraySize, const dim3& threads2D = getDefaultThreads2DDimensions());

    /** The calculateCUDA2DKernelDimensionsXY() function efficiently calculates the XY dimensions for a CUDA 2D kernel.
    *
    * @author Amir Shahvarani, 2019
    * @version 14.0.0.0
    */
    static std::tuple<dim3, dim3> calculateCUDA2DKernelDimensionsXY(std::size_t arraySizeX, std::size_t arraySizeY, const dim3& threads2D = getDefaultThreads2DDimensions());

    /** The calculateCUDAPersistentKernel() function efficiently calculates the dimensions of persistent kernel to run on current device.
    *
    * @author Amir Shahvarani, 2019
    * @version 14.0.0.0
    */
    static std::tuple<dim3, dim3> calculateCUDAPersistentKernel(const CUDADriverInfo& cudaDriverInfo, int device, uint32_t threadsPerBlock, uint32_t sharedMemoryPerBlock = 0);

    /** The checkAndReportCUDAMemory() function checks & reports CUDA memory per given device.
    *
    * @author Thanos Theo, 2019
    * @version 14.0.0.0
    */
    static std::string checkAndReportCUDAMemory(int device, bool useUVA = false);

    CUDAUtilityFunctions()  = delete;
    ~CUDAUtilityFunctions() = delete;
    CUDAUtilityFunctions(const CUDAUtilityFunctions&) = delete;
    CUDAUtilityFunctions(CUDAUtilityFunctions&&)      = delete;
    CUDAUtilityFunctions& operator=(const CUDAUtilityFunctions&) = delete;
    CUDAUtilityFunctions& operator=(CUDAUtilityFunctions&&)      = delete;
  };
} // namespace UtilsCUDA

#endif // __CUDAUtilityFunctions_h