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

#ifndef __UtilityFunctions_h
#define __UtilityFunctions_h

#include "ModuleDLL.h"
#include "EnvironmentConfig.h"
#include "MathConstants.h"
#include "VectorTypes.h"
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring> // for memcpy() call
#include <type_traits>
#include <array>
#include <algorithm>
#include <tuple>
#include <string>
#include <sstream>
#include <ios>
#include <vector>
#include <list>
#include <set>
#include <fstream>
#include <iostream>
#include <memory>

// DebugConsole macro functions below

#define DebugConsole_printfConsoleOutLine(...) \
          do { if (GPU_FRAMEWORK_DEBUG_CONSOLE) Utils::UtilityFunctions::DebugConsole::printfConsoleOutLineImpl(__VA_ARGS__); } while (0)

#define DebugConsole_printfFileOutLine(...) \
          do { if (GPU_FRAMEWORK_DEBUG_CONSOLE) Utils::UtilityFunctions::DebugConsole::printfFileOutLineImpl(__VA_ARGS__); } while (0)

#define DebugConsole_printfConsoleOutLineDetailed(...)                                                                                                                                                     \
          do { if (GPU_FRAMEWORK_DEBUG_CONSOLE) { Utils::UtilityFunctions::DebugConsole::printfConsoleOutLineImpl("%s%s%s%s%s%d\n", "\nFile: ", __FILE__, "\nFunction: ", __func__, "\nLine: ", __LINE__); \
                                                  Utils::UtilityFunctions::DebugConsole::printfConsoleOutLineImpl(__VA_ARGS__); } } while (0)

#define DebugConsole_printfFileOutLineDetailed(...)                                                                                                                                                     \
          do { if (GPU_FRAMEWORK_DEBUG_CONSOLE) { Utils::UtilityFunctions::DebugConsole::printfFileOutLineImpl("%s%s%s%s%s%d\n", "\nFile: ", __FILE__, "\nFunction: ", __func__, "\nLine: ", __LINE__); \
                                                  Utils::UtilityFunctions::DebugConsole::printfFileOutLineImpl(__VA_ARGS__); } } while (0)

#define DebugConsole_consoleOutLine(...) \
          do { if (GPU_FRAMEWORK_DEBUG_CONSOLE) Utils::UtilityFunctions::DebugConsole::consoleOutLineImpl(__VA_ARGS__); } while (0)

#define DebugConsole_fileOutLine(...) \
          do { if (GPU_FRAMEWORK_DEBUG_CONSOLE) Utils::UtilityFunctions::DebugConsole::fileOutLineImpl(__VA_ARGS__); } while (0)

#define DebugConsole_consoleOutLineDetailed(...)                                                                                                                \
          do { if (GPU_FRAMEWORK_DEBUG_CONSOLE) { std::ostringstream ss;                                                                                        \
                                       ss << "\nFile: " << __FILE__ << std::endl << "Function: " << __func__ << std::endl << "Line: " << __LINE__ << std::endl; \
                                       std::cout << ss.str();                                                                                                   \
                                       Utils::UtilityFunctions::DebugConsole::consoleOutLineImpl(ss.str());                                                     \
                                       Utils::UtilityFunctions::DebugConsole::consoleOutLineImpl(__VA_ARGS__); } } while (0)

#define DebugConsole_fileOutLineDetailed(...)                                                                                                                   \
          do { if (GPU_FRAMEWORK_DEBUG_CONSOLE) { std::ostringstream ss;                                                                                        \
                                       ss << "\nFile: " << __FILE__ << std::endl << "Function: " << __func__ << std::endl << "Line: " << __LINE__ << std::endl; \
                                       std::cout << ss.str();                                                                                                   \
                                       Utils::UtilityFunctions::DebugConsole::fileOutLineImpl(ss.str());                                                        \
                                       Utils::UtilityFunctions::DebugConsole::fileOutLineImpl(__VA_ARGS__); } } while (0)

// ReleaseConsole macro functions below

#define ReleaseConsole_printfConsoleOutLine(...) \
          do { Utils::UtilityFunctions::DebugConsole::printfConsoleOutLineImpl(__VA_ARGS__); } while (0)

#define ReleaseConsole_printfFileOutLine(...) \
          do { Utils::UtilityFunctions::DebugConsole::printfFileOutLineImpl(__VA_ARGS__); } while (0)

#define ReleaseConsole_printfConsoleOutLineDetailed(...)                                                                                                                  \
          do { { Utils::UtilityFunctions::DebugConsole::printfConsoleOutLineImpl("%s%s%s%s%s%d\n", "\nFile: ", __FILE__, "\nFunction: ", __func__, "\nLine: ", __LINE__); \
                 Utils::UtilityFunctions::DebugConsole::printfConsoleOutLineImpl(__VA_ARGS__); } } while (0)

#define ReleaseConsole_printfFileOutLineDetailed(...)                                                                                                                  \
          do { { Utils::UtilityFunctions::DebugConsole::printfFileOutLineImpl("%s%s%s%s%s%d\n", "\nFile: ", __FILE__, "\nFunction: ", __func__, "\nLine: ", __LINE__); \
                 Utils::UtilityFunctions::DebugConsole::printfFileOutLineImpl(__VA_ARGS__); } } while (0)

#define ReleaseConsole_consoleOutLine(...) \
          do { Utils::UtilityFunctions::DebugConsole::consoleOutLineImpl(__VA_ARGS__); } while (0)

#define ReleaseConsole_fileOutLine(...) \
          do { Utils::UtilityFunctions::DebugConsole::fileOutLineImpl(__VA_ARGS__); } while (0)

#define ReleaseConsole_consoleOutLineDetailed(...)                                                                                        \
          do { { std::ostringstream ss;                                                                                                   \
                 ss << "\nFile: " << __FILE__ << std::endl << "Function: " << __func__ << std::endl << "Line: " << __LINE__ << std::endl; \
                 std::cout << ss.str();                                                                                                   \
                 Utils::UtilityFunctions::DebugConsole::consoleOutLineImpl(ss.str());                                                     \
                 Utils::UtilityFunctions::DebugConsole::consoleOutLineImpl(__VA_ARGS__); } } while (0)

#define ReleaseConsole_fileOutLineDetailed(...)                                                                                           \
          do { { std::ostringstream ss;                                                                                                   \
                 ss << "\nFile: " << __FILE__ << std::endl << "Function: " << __func__ << std::endl << "Line: " << __LINE__ << std::endl; \
                 std::cout << ss.str();                                                                                                   \
                 Utils::UtilityFunctions::DebugConsole::fileOutLineImpl(ss.str());                                                        \
                 Utils::UtilityFunctions::DebugConsole::fileOutLineImpl(__VA_ARGS__); } } while (0)

/** @brief Namespace Utils contains utility classes with mainly static CPU related methods.
* @author Thanos Theo, 2009-2018
* @version 14.0.0.0
*/
namespace Utils
{
  // Partially substituting the lack of proper Reflection support in C++
  // and the std::type_info::name() with a cross-compiler/platform name convention
  #define DECLARE_TYPE_INFO_TEMPLATE() \
  template<typename T> struct Class;

  #define CREATE_TYPE_INFO_NAME(T) \
  struct T;                        \
  template<>                       \
  struct Class<T>{ static std::string name() { return #T; } };

  // Anonymous lambda type deduction from container name with usage: ContainerType(container)& value;
  #define ContainerType(container) std::remove_reference<decltype(*std::begin(container))>::type

  /** @brief The ReverseIterationWrapper dummy struct provides additional generic functionality which std doesn't still provide. Note that ReverseIterationWrapper with its related functions have to reside in namespace scope.
  *
  *  Usage: for (const auto& value : Utils::reverse(container))
  *
  * @author Thanos Theo, 2009-2018
  * @version 14.0.0.0
  */
  template <typename Container>
  struct ReverseIterationWrapper { Container& iterable; };

  template <typename Container>
  auto begin(ReverseIterationWrapper<Container> wrapper) { return std::rbegin(wrapper.iterable); }

  template <typename Container>
  auto end(ReverseIterationWrapper<Container> wrapper) { return std::rend(wrapper.iterable); }

  template <typename Container>
  ReverseIterationWrapper<Container> reverse(Container&& iterable) { return{ iterable }; }

  /** @brief Namespace UtilityFunctions contains classes with only static CG GLSL-style & CPU related methods.
  *
  *  UtilityFunctions.h:
  *  ==================
  *  Namespace UtilityFunctions contains classes with only static CG GLSL-style & CPU related methods.
  *
  * @author Thanos Theo, 2009-2018
  * @version 14.0.0.0
  */
  namespace UtilityFunctions
  {
    /** @brief The StdAuxiliaryFunctions struct provides additional generic functionality which std doesn't (currently) still provide.
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    struct StdAuxiliaryFunctions final
    {
      /** @brief Returns the size of an array as a compile-time constant (the array parameter has no name, because we care only about the number of elements it contains).
      */
      template <typename T, std::size_t N>
      static inline constexpr std::size_t arraySize(T(&)[N]) noexcept
      {
        return N;
      }

      /** @brief Returns the size_t value from a given enumerator.
      */
      template <typename E>
      static inline constexpr auto toUnsignedType(E enumerator) noexcept
      {
        return static_cast<std::underlying_type_t<E>>(enumerator);
      }

      /** @brief Sort an array using insertion sort with a constant small size of N.
      *
      * While some divide-and-conquer algorithms such as quicksort and mergesort outperform insertion sort for larger arrays,
      * non-recursive sorting algorithms such as insertion sort or selection sort are generally faster for very small arrays
      * (the exact size varies by environment and implementation, but is typically between seven and fifty elements).
      * Therefore, a useful optimization in the implementation of those algorithms is a hybrid approach,
      * using the simpler algorithm when the array has been divided to a small size.
      */
      template<std::size_t N, typename T>
      static inline void insertionSort(T* __restrict arrayData)
      {
        for (std::size_t i = 1; i < N; ++i)
        {
          T value = arrayData[i];
          std::size_t index = i;

          // move elements of arrayData[0..i-1], that are greater than key, to one position ahead of their current position
          while (index > 0 && arrayData[index - 1] > value)
          {
            arrayData[index] = arrayData[index - 1];
            index = index - 1;
          }

          arrayData[index] = value;
        }
      }

      StdAuxiliaryFunctions()  = delete;
      ~StdAuxiliaryFunctions() = delete;
      StdAuxiliaryFunctions(const StdAuxiliaryFunctions&) = delete;
      StdAuxiliaryFunctions(StdAuxiliaryFunctions&&)      = delete;
      StdAuxiliaryFunctions& operator=(const StdAuxiliaryFunctions&) = delete;
      StdAuxiliaryFunctions& operator=(StdAuxiliaryFunctions&&)      = delete;
    };

    /** @brief The Base64CompressorScrambler struct provides encoding/decoding functionality to strings.
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    struct UTILS_MODULE_API Base64CompressorScrambler final
    {
      static std::string encodeBase64String(const std::string& str);
      static std::string decodeBase64String(const std::string& str);
      static std::string flipString(const std::string& line);
      static std::string xorSwapString(const std::string& line);

      Base64CompressorScrambler()  = delete;
      ~Base64CompressorScrambler() = delete;
      Base64CompressorScrambler(const Base64CompressorScrambler&) = delete;
      Base64CompressorScrambler(Base64CompressorScrambler&&)      = delete;
      Base64CompressorScrambler& operator=(const Base64CompressorScrambler&) = delete;
      Base64CompressorScrambler& operator=(Base64CompressorScrambler&&)      = delete;
    };

    /** @brief The BitManipulationFunctions struct provides bit manipulation functionality.
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    struct UTILS_MODULE_API BitManipulationFunctions final
    {
      /**
      *  Find if the given number is a power-of-two number.
      *  Extremely efficient implementation taken from http://graphics.stanford.edu/~seander/bithacks.html
      */
      template <typename T>
      static inline bool isPowerOfTwo(T value, std::enable_if_t<std::is_integral<T>::value>* = nullptr)
      {
        return (value && !(value & (value - 1)));
      }

      /**
      *  Find the lowest bit position of a given power-of-two integer number.
      *  Extremely efficient implementation taken from http://graphics.stanford.edu/~seander/bithacks.html
      */
      static int getLowestBitPositionOfPowerOfTwoNumber(int value);

      /**
      *  Count turned on bits of a given integer number.
      *  Extremely efficient implementation taken from http://graphics.stanford.edu/~seander/bithacks.html
      */
      static int countTurnedOnBitsOfNumber(int value);

      /**
      *  Gets the previous power-of-two of a given number.
      *  Extremely efficient implementation taken from http://graphics.stanford.edu/~seander/bithacks.html
      */
      static unsigned int getPrevPowerOfTwo(unsigned int value);

      /**
      *  Gets the next power-of-two of a given number.
      *  Extremely efficient implementation taken from http://graphics.stanford.edu/~seander/bithacks.html
      */
      static unsigned int getNextPowerOfTwo(unsigned int value);

      /**
      *  Checks if the enumType has the enumSelection (for C-style enums).
      *  Using the extremely efficient getLowestBitPositionOfPowerOfTwoNumber() implementation.
      */
      template <typename T, typename I>
      static inline bool hasCStyleEnumType(T enumType, I enumSelection)
      {
        return bool((enumType >> getLowestBitPositionOfPowerOfTwoNumber(enumSelection)) & 1);
      }

      /**
      *  Checks if the enumType has the enumSelection (for C++11 class enums).
      *  Using the extremely efficient getLowestBitPositionOfPowerOfTwoNumber() implementation.
      */
      template <typename T, typename I>
      static inline bool hasClassEnumType(T enumType, I enumSelection)
      {
        return bool((StdAuxiliaryFunctions::toUnsignedType(enumType) >> getLowestBitPositionOfPowerOfTwoNumber(StdAuxiliaryFunctions::toUnsignedType(enumSelection))) & 1);
      }

      BitManipulationFunctions()  = delete;
      ~BitManipulationFunctions() = delete;
      BitManipulationFunctions(const BitManipulationFunctions&) = delete;
      BitManipulationFunctions(BitManipulationFunctions&&)      = delete;
      BitManipulationFunctions& operator=(const BitManipulationFunctions&) = delete;
      BitManipulationFunctions& operator=(BitManipulationFunctions&&)      = delete;
    };

    /** @brief The ArrayIndicingFunctions struct provides array indexing functionality.
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    struct ArrayIndicingFunctions final
    {
      /**
      *  Flattens the 2D array coordinates to an 1D index.
      */
      static inline std::size_t flattenArray2DIndex(std::size_t x, std::size_t y, std::size_t dimensionY)
      {
        return x * dimensionY + y;
      }

      /**
      *  Unflattens the 1D array index to 2D array coordinates.
      */
      static inline std::tuple<std::size_t, std::size_t> unflattenArray2DIndex(std::size_t array2DIndex, std::size_t dimensionY)
      {
        return std::make_tuple(array2DIndex / dimensionY, array2DIndex % dimensionY);
      }

      /**
      *  Getter from a 2D array laid out linearly in memory.
      */
      template <typename T>
      static inline T getArray2D(const T* __restrict array2D, std::size_t x, std::size_t y, std::size_t dimensionY)
      {
        return array2D[x * dimensionY + y];
      }

      /**
      *  Setter for a 2D array laid out linearly in memory.
      */
      template <typename T>
      static inline void setArray2D(T* __restrict array2D, std::size_t x, std::size_t y, std::size_t dimensionY, const T& value)
      {
        array2D[x * dimensionY + y] = value;
      }

      /**
      *  Flattens the 3D array coordinates to an 1D index.
      */
      static inline std::size_t flattenArray3DIndex(std::size_t x, std::size_t y, std::size_t z, std::size_t dimensionY, std::size_t dimensionZ)
      {
        return x * dimensionY * dimensionZ + y * dimensionZ + z;
      }

      /**
      *  Unflattens the 1D array index to 3D array coordinates.
      */
      static inline std::tuple<std::size_t, std::size_t, std::size_t> unflattenArray3DIndex(std::size_t array3DIndex, std::size_t dimensionY, std::size_t dimensionZ)
      {
        return std::make_tuple(array3DIndex / (dimensionY * dimensionZ), (array3DIndex / dimensionZ) % dimensionY, array3DIndex % dimensionZ);
      }

      /**
      *  Getter from a 3D array laid out linearly in memory.
      */
      template <typename T>
      static inline T getArray3D(const T* __restrict array3D, std::size_t x, std::size_t y, std::size_t z, std::size_t dimensionY, std::size_t dimensionZ)
      {
        return array3D[x * dimensionY * dimensionZ + y * dimensionZ + z];
      }

      /**
      *  Setter for a 3D array laid out linearly in memory.
      */
      template <typename T>
      static inline void setArray3D(T* __restrict array3D, std::size_t x, std::size_t y, std::size_t z, std::size_t dimensionY, std::size_t dimensionZ, const T& value)
      {
        array3D[x * dimensionY * dimensionZ + y * dimensionZ + z] = value;
      }

      ArrayIndicingFunctions()  = delete;
      ~ArrayIndicingFunctions() = delete;
      ArrayIndicingFunctions(const ArrayIndicingFunctions&) = delete;
      ArrayIndicingFunctions(ArrayIndicingFunctions&&)      = delete;
      ArrayIndicingFunctions& operator=(const ArrayIndicingFunctions&) = delete;
      ArrayIndicingFunctions& operator=(ArrayIndicingFunctions&&)      = delete;
    };

    /** @brief The TrigonometricFunctions struct covers essential operations involving angles on the trigonometric circle.
    * @author Jacobo Cabaleiro, Teodor Cioaca, 2019
    * @version 1.0.0.0
    */
    struct TrigonometricFunctions final
    {
      /**
      *  Conversion function from degrees to radians.
      */
      template <typename T>
      static inline T toRadians(const T degrees, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr)
      {
        return degrees * (T(PI_DBL) / T(180));
      }

      /**
      *  Conversion function from radians to degrees.
      */
      template <typename T>
      static inline T toDegrees(const T radians, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr)
      {
        return radians / (T(PI_DBL) / T(180));
      }

      /** @brief Normalizes an angle argument into a range covering the full trigonometric circle.
      * @tparam T input argument data type
      * @tparam AngleRange structure type exposing the boundaries of the reference angular range
      * @param angle the input angular argument
      * @return the corresponding normalized angle
      */
      template <typename T, typename AngleRange>
      static inline T normalizeRadAngle(const T angle, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr)
      {
        constexpr const T TWO_PI = T(2.0 * PI_DBL);
        double r = std::fmod(angle, TWO_PI);
        if (r < AngleRange::LOWER())
        {
          r += TWO_PI;
        }
        if (r >= AngleRange::UPPER())
        {
          r -= TWO_PI;
        }
        return r;
      }

      /** @brief Normalizes an angle argument into the [-PI, PI) range.
      * @param angle in radians
      * @return the corresponding normalized angle
      */
      template <typename T>
      static inline T normalizePI(const T angle, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr)
      {
        struct Range
        {
          static constexpr T LOWER() { return -T(PI_DBL); }
          static constexpr T UPPER() { return  T(PI_DBL); }
        };
        return normalizeRadAngle<T, Range>(angle);
      }

      /** @brief Normalizes an angle argument into the [0, 2*PI) range.
      * @param angle in radians
      * @return the corresponding normalized angle
      */
      template <typename T>
      static inline T normalize2PI(const T angle, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr)
      {
        struct Range
        {
          static constexpr T LOWER() { return T(0.0);          }
          static constexpr T UPPER() { return T(2.0 * PI_DBL); }
        };
        return normalizeRadAngle<T, Range>(angle);
      }

      /** @brief Computes the shortest arc length in rads on the trigonometric circle for two given input angles expressed in the [0, 2*PI) range.
      */
      template <typename T>
      static inline double getAngularDifference(const T fromAngle, const T targetAngle, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr)
      {
        return normalizePI(targetAngle - fromAngle);
      }

      TrigonometricFunctions()  = delete;
      ~TrigonometricFunctions() = delete;
      TrigonometricFunctions(const TrigonometricFunctions&) = delete;
      TrigonometricFunctions(TrigonometricFunctions&&)      = delete;
      TrigonometricFunctions& operator=(const TrigonometricFunctions&) = delete;
      TrigonometricFunctions& operator=(TrigonometricFunctions&&)      = delete;
    };

    /** @brief The MathFunctions struct provides some needed mathematical functions functionality (note that some functions emulate GLSL-style CPU functionality).
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    struct MathFunctions final
    {
      /**
      *  Strictly equal, *using machine epsilon as tolerance*.
      */
      template<typename T>
      static inline bool equal(const T left, const T right, std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr)
      {
        return std::abs(left - right) <= std::numeric_limits<T>::epsilon();
      }

      /**
      *  Floating point error epsilon value scaled according to the range of the compared values.
      */
      template<typename T>
      static inline bool scaledEpsilon(const T left, const T right, std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr)
      {
        // fractional numbers can be compared using the standard numeric_limits epsilon value, as the machine
        // epsilon is A RELATIVE ERROR (normalized around 1)
        // for larger values, that error needs to be scaled to account for quantization artifacts
        T maxXYOne = std::max({ T(1), std::abs(left), std::abs(right) });
        return std::numeric_limits<T>::epsilon() * maxXYOne;
      }

      /**
      *  Relatively equal, *using scaled machine epsilon as tolerance*.
      */
      template<typename T>
      static inline bool relativelyEqual(const T left, const T right, std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr)
      {
        return std::abs(left - right) <= scaledEpsilon(left, right);
      }

      /**
        * Approximately equal comparison for floating point numbers with explicitly specified tolerance.
        */
      template<typename T>
      static inline bool approximatelyEqual(const T left, const T right, const T tolerance, std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr)
      {
        T absDifference = std::abs(left - right);
        return absDifference <= scaledEpsilon(left, right) + tolerance;
      }

      /**
        * Checks if left < right + margin
        */
      template<typename T>
      static inline bool marginallyLessThan(const T left, const T right, const T margin, std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr)
      {
        T difference = left - right;
        return difference <= margin;
      }

      /*
      *  GLSL-style sign function.
      */
      template<typename T>
      static inline T sign(const T x, std::enable_if_t<std::is_arithmetic<T>::value && std::is_signed<T>::value>* = nullptr)
      {
        return (x < T(0)) ? T(-1) : (x > T(0)) ? T(1) : T(0);
      }

      /**
      *  GLSL-style fract function (integral version).
      */
      template <typename T>
      static inline T fract(const T x, std::enable_if_t<std::is_integral<T>::value>* = nullptr)
      {
        return x - T(std::floor<T>(x));
      }

      /**
      *  GLSL-style fract function (float/double version).
      */
      template <typename T>
      static inline T fract(const T x, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr)
      {
        return x - std::floor(x);
      }

      /**
      *  GLSL-style clamp function.
      */
      template <typename T>
      static inline T clamp(const T& value, const T& minVal, const T& maxVal, std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr)
      {
        return (std::max)(minVal, (std::min)(maxVal, value)); // explicit (std::max&min) to avoid VS compilation problems
      }

      /**
      *  GLSL-style reinterval function.
      */
      template <typename T>
      static inline T reinterval(const T& inVal, const T& oldMin, const T& oldMax, const T& newMin, const T& newMax, std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr)
      {
        if (equal(oldMax, oldMin))
        {
          return newMin;
        }

        T value = inVal;
        value  -= oldMin;
        value  *= (newMax - newMin);
        value  /= (oldMax - oldMin);
        value  += newMin;

        return value;
      }

      /**
      *  GLSL-style reintervalClamped function.
      */
      template <typename T>
      static inline T reintervalClamped(const T& inVal, const T& oldMin, const T& oldMax, const T& newMin, const T& newMax, std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr)
      {
        if (inVal <= oldMin)
        {
          return newMin;
        }
        if (inVal >= oldMax)
        {
          return newMax;
        }

        return clamp<T>(reinterval(inVal, oldMin, oldMax, newMin, newMax), newMin, newMax);
      }

      /**
      *  GLSL-style mix function.
      */
      template <typename T, typename I>
      static inline T mix(const T& left, const T& right, const I& t, std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr)
      {
        I one_minus_t = I(1) - t;
        return left * one_minus_t + right * t;
      }

      /**
      *  GLSL-style smoothstep function.
      */
      template <typename T>
      static inline T smoothstep(const T& edge0, const T& edge1, T x, std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr) // pass x by value to avoid temporary variable creation inside the function
      {
        x = clamp<T>((x - edge0) / (edge1 - edge0), T(0), T(1));
        return x * x * (T(3) - T(2) * x);
      }

      /**
      *  Prof. Ken Perlin suggests an improved version of the smoothstep function which has zero 1st and 2nd order derivatives at t=0 and t=1.
      *  Scale, and clamp x to 0...1 (first line) range & evaluate polynomial (second line).
      *  Look at http://en.wikipedia.org/wiki/Smoothstep -> smootherstep.
      *  GLSL-style smootherstep function.
      */
      template <typename T>
      static inline T smootherstep(const T& edge0, const T& edge1, T x, std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr) // pass x by value to avoid temporary variable creation inside the function
      {
        x = clamp<T>((x - edge0) / (edge1 - edge0), T(0), T(1));
        return x * x * x * (x * (x * T(6) - T(15)) + T(10));
      }

      /**
      *  Matlab MOD function emulation (integral version).
      */
      template <typename T>
      static inline T matlabMOD(const T a, const T b, std::enable_if_t<std::is_integral<T>::value>* = nullptr)
      {
        T c  = a - b * T(std::floor<T>(a / b)) + b;
        return c - b * T(std::floor<T>(c / b));
      }

      /**
      *  Matlab MOD function emulation (float/double version).
      */
      template <typename T>
      static inline T matlabMOD(const T a, const T b, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr)
      {
        T c  = a - b * std::floor(a / b) + b;
        return c - b * std::floor(c / b);
      }

      /**
      *  GLSL-style dot function (float version).
      */
      static inline float dot(const VectorTypes::float2& a, const VectorTypes::float2& b)
      {
        return a.x * b.x + a.y * b.y;
      }

      /**
      *  GLSL-style dot function (double version).
      */
      static inline double dot(const VectorTypes::double2& a, const VectorTypes::double2& b)
      {
        return a.x * b.x + a.y * b.y;
      }

      /** @brief This function returns uniformly distributed float values in the range [0, 1] (float version).
      *
      * @author Thanos Theo, 2018
      */
      static inline float rand1(const VectorTypes::float2& seed)
      {
        const float dotProduct = dot(seed, VectorTypes::float2(12.9898f, 78.233f));
        return fract(sinf(dotProduct) * 43758.5453f);
      }

      /** @brief This function returns uniformly distributed double values in the range [0, 1] (double version).
      *
      * @author Thanos Theo, 2018
      */
      static inline double rand1(const VectorTypes::double2& seed)
      {
        const double dotProduct = dot(seed, VectorTypes::double2(12.9898, 78.233));
        return fract(sin(dotProduct) * 43758.5453);
      }

      /** @brief This function returns uniformly distributed float2 values in the range [0, 1] (float version).
      *
      * @author Thanos Theo, 2018
      */
      static inline VectorTypes::float2 rand2(const VectorTypes::float2& seed)
      {
        const float dotProduct = dot(seed, VectorTypes::float2(12.9898f, 78.233f));
        return VectorTypes::float2(fract(sinf(dotProduct) * 43758.5453f), fract(sinf(2.0f * dotProduct) * 43758.5453f));
      }

      /** @brief This function returns uniformly distributed double2 values in the range [0, 1] (double version).
      *
      * @author Thanos Theo, 2018
      */
      static inline VectorTypes::double2 rand2(const VectorTypes::double2& seed)
      {
        const double dotProduct = dot(seed, VectorTypes::double2(12.9898, 78.233));
        return VectorTypes::double2(fract(sin(dotProduct) * 43758.5453), fract(sin(2.0 * dotProduct) * 43758.5453));
      }

      /** @brief This function returns uniformly distributed float3 values in the range [0, 1] (float version).
      *
      * @author Thanos Theo, 2018
      */
      static inline VectorTypes::float3 rand3(const VectorTypes::float2& seed)
      {
        const float dotProduct = dot(seed, VectorTypes::float2(12.9898f, 78.233f));
        return VectorTypes::float3(fract(sinf(dotProduct) * 43758.5453f), fract(sinf(2.0f * dotProduct) * 43758.5453f), fract(sinf(3.0f * dotProduct) * 43758.5453f));
      }

      /** @brief This function returns uniformly distributed double3 values in the range [0, 1] (double version).
      *
      * @author Thanos Theo, 2018
      */
      static inline VectorTypes::double3 rand3(const VectorTypes::double2& seed)
      {
        const double dotProduct = dot(seed, VectorTypes::double2(12.9898, 78.233));
        return VectorTypes::double3(fract(sin(dotProduct) * 43758.5453), fract(sin(2.0 * dotProduct) * 43758.5453), fract(sin(3.0 * dotProduct) * 43758.5453));
      }

      /** @brief This function returns uniformly distributed float4 values in the range [0, 1] (float version).
      *
      * @author Thanos Theo, 2018
      */
      static inline VectorTypes::float4 rand4(const VectorTypes::float2& seed)
      {
        const float dotProduct = dot(seed, VectorTypes::float2(12.9898f, 78.233f));
        return VectorTypes::float4(fract(sinf(dotProduct) * 43758.5453f), fract(sinf(2.0f * dotProduct) * 43758.5453f), fract(sinf(3.0f * dotProduct) * 43758.5453f), fract(sinf(4.0f * dotProduct) * 43758.5453f));
      }

      /** @brief This function returns uniformly distributed double4 values in the range [0, 1] (double version).
      *
      * @author Thanos Theo, 2018
      */
      static inline VectorTypes::double4 rand4(const VectorTypes::double2& seed)
      {
        const double dotProduct = dot(seed, VectorTypes::double2(12.9898, 78.233));
        return VectorTypes::double4(fract(sin(dotProduct) * 43758.5453), fract(sin(2.0 * dotProduct) * 43758.5453), fract(sin(3.0 * dotProduct) * 43758.5453), fract(sin(4.0 * dotProduct) * 43758.5453));
      }

      /** @brief Seed generator for the Linear Congruential Generator (LGC).
      *
      * @author Thanos Theo, 2018
      */
      template<std::uint32_t N>
      static inline std::uint32_t seedGenerator(std::uint32_t value0, std::uint32_t value1)
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
      static inline std::uint32_t rand1u(std::uint32_t& seed)
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
      static inline float rand1f(std::uint32_t& seed)
      {
        return float(rand1u(seed)) / float(0x01000000);
      }

      /** @brief Generate random float2 values in the [0, 1) range with the Linear Congruential Generator (LGC).
      *
      * @author Thanos Theo, 2018
      */
      static inline VectorTypes::float2 rand2f(std::uint32_t& seed)
      {
        return VectorTypes::float2(rand1f(seed), rand1f(seed));
      }

      /** @brief Generate random float3 values in the [0, 1) range with the Linear Congruential Generator (LGC).
      *
      * @author Thanos Theo, 2018
      */
      static inline VectorTypes::float3 rand3f(std::uint32_t& seed)
      {
        return VectorTypes::float3(rand1f(seed), rand1f(seed), rand1f(seed));
      }

      /** @brief Generate random float4 values in the [0, 1) range with the Linear Congruential Generator (LGC).
      *
      * @author Thanos Theo, 2018
      */
      static inline VectorTypes::float4 rand4f(std::uint32_t& seed)
      {
        return VectorTypes::float4(rand1f(seed), rand1f(seed), rand1f(seed), rand1f(seed));
      }

      /** @brief Get the float32 bit representation to a uint32.
      *
      * @author Thanos Theo, 2018
      */
      static inline std::uint32_t asUint32(float value)
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
      static inline float asFloat32(std::uint32_t value)
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
      static inline std::uint32_t float32Flip(float unflippedFloatValue)
      {
        const std::uint32_t f    = asUint32(unflippedFloatValue);
        const std::uint32_t mask = -std::int32_t(f >> 31) | 0x80000000;

        return f ^ mask;
      }

      /** @brief Unflip a float32 back (invert float32Flip() above): signed was flipped from above, so: if sign is 1 (negative) it flips the sign bit back, if if sign is 0 (positive) it flips all bits back. Needs IEEE 754 hardware compliance. Based on http://stereopsis.com/radix.html.
      *
      * @author Thanos Theo, 2018
      */
      static inline float float32Unflip(std::uint32_t flippedFloatValue)
      {
        const std::uint32_t f    = flippedFloatValue;
        const std::uint32_t mask = ((f >> 31) - 1) | 0x80000000;

        return asFloat32(f ^ mask);
      }

      /** @brief Get the float64 bit representation to a uint64.
      *
      * @author Thanos Theo, 2018
      */
      static inline std::uint64_t asUint64(double value)
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
      static inline double asFloat64(std::uint64_t value)
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
      static inline std::uint64_t float64Flip(double unflippedFloatValue)
      {
        const std::uint64_t f    = asUint64(unflippedFloatValue);
        const std::uint64_t mask = -std::int64_t(f >> 63) | 0x8000000000000000;

        return f ^ mask;
      }

      /** @brief Unflip a float64 back (invert float64Flip() above): signed was flipped from above, so: if sign is 1 (negative) it flips the sign bit back, if if sign is 0 (positive) it flips all bits back. Needs IEEE 754 hardware compliance. Based on http://stereopsis.com/radix.html.
      *
      * @author Thanos Theo, 2018
      */
      static inline double float64Unflip(std::uint64_t flippedFloatValue)
      {
        const std::uint64_t f    = flippedFloatValue;
        const std::uint64_t mask = ((f >> 63) - 1) | 0x8000000000000000;

        return asFloat64(f ^ mask);
      }

      MathFunctions()  = delete;
      ~MathFunctions() = delete;
      MathFunctions(const MathFunctions&) = delete;
      MathFunctions(MathFunctions&&)      = delete;
      MathFunctions& operator=(const MathFunctions&) = delete;
      MathFunctions& operator=(MathFunctions&&)      = delete;
    };

    /** @brief The StringAuxiliaryFunctions class provides additional string functionality which std doesn't (currently) still provide.
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    class UTILS_MODULE_API StringAuxiliaryFunctions final
    {
    public:

      /**
      *  String manipulation auxiliary function (bool version).
      */
      template<typename T>
      static inline std::string toString(const bool value, std::enable_if_t<std::is_same<T, bool>::value>* = nullptr)
      {
        return (value ? "true" : "false");
      }

      /**
      *  String manipulation auxiliary function (integral signed version).
      */
      template<typename T>
      static inline std::string toString(const T value, std::enable_if_t<!std::is_same<T, bool>::value && std::is_integral<T>::value && std::is_signed<T>::value>* = nullptr)
      {
        return parseNumberCStyle("%i", value);
      }

      /**
      *  String manipulation auxiliary function (integral unsigned version).
      */
      template<typename T>
      static inline std::string toString(const T value, std::enable_if_t<!std::is_same<T, bool>::value && std::is_integral<T>::value && std::is_unsigned<T>::value>* = nullptr)
      {
        return parseNumberCStyle("%u", value);
      }

      /**
      *  String manipulation auxiliary function (float/double version).
      */
      template<typename T>
      static inline std::string toString(const T value, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr)
      {
        return parseNumberCStyle("%f", value);
      }

      /**
      *  String manipulation auxiliary function (T 'as a string' version).
      */
      template<typename T>
      static inline std::string toString(const T& value, std::enable_if_t<!std::is_arithmetic<T>::value && std::is_same<T, std::string>::value>* = nullptr)
      {
        return std::string(value);
      }

      /**
      *  String manipulation auxiliary function (T 'as a generic writable object' version).
      */
      template<typename T>
      static inline std::string toString(const T& value, std::enable_if_t<!std::is_arithmetic<T>::value && !std::is_same<T, std::string>::value>* = nullptr)
      {
        std::ostringstream os;
        os << value;
        return os.str();
      }

      /**
      *  String manipulation auxiliary function (Args... 'as a variadic template' encapsulation for printf version).
      */
      template<typename... Args>
      static inline std::string printfToString(const char* format, const Args... args)
      {
        std::array<char, STRING_BUFFER_SIZE> resultString{ { ' ' } }; // double braces because we initialize an array inside an std::array object
        std::snprintf(resultString.data(), STRING_BUFFER_SIZE, format, args...);

        return std::string(resultString.data());
      }

      /**
      *  String manipulation auxiliary function (bool version).
      */
      template<typename T>
      static inline T fromString(const std::string& str, std::enable_if_t<std::is_same<T, bool>::value>* = nullptr)
      {
        return T((toLowerCase(str) == "true") || (fromString<std::intmax_t>(str) > 0));
      }

      /**
      *  String manipulation auxiliary function (integral signed version).
      */
      template<typename T>
      static inline T fromString(const std::string& str, std::enable_if_t<!std::is_same<T, bool>::value && std::is_integral<T>::value && std::is_signed<T>::value>* = nullptr)
      {
        return T(strtoll(str.c_str(), nullptr, 0));
      }

      /**
      *  String manipulation auxiliary function (integral unsigned version).
      */
      template<typename T>
      static inline T fromString(const std::string& str, std::enable_if_t<!std::is_same<T, bool>::value && std::is_integral<T>::value && std::is_unsigned<T>::value>* = nullptr)
      {
        return T(strtoull(str.c_str(), nullptr, 0));
      }

      /**
      *  String manipulation auxiliary function (float/double version).
      */
      template<typename T>
      static inline T fromString(const std::string& str, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr)
      {
        return T(stod(str));
      }

      /**
      *  String manipulation auxiliary function (T 'as a string' version).
      */
      template<typename T>
      static inline T fromString(const std::string& str, std::enable_if_t<!std::is_arithmetic<T>::value && std::is_same<T, std::string>::value>* = nullptr)
      {
        return std::string(str);
      }

      /**
      *  String manipulation auxiliary function (T 'as a generic writable object' version).
      */
      template<typename T>
      static inline T fromString(const std::string& str, std::enable_if_t<!std::is_arithmetic<T>::value && !std::is_same<T, std::string>::value>* = nullptr)
      {
        std::istringstream input(str);
        T t(0);
        return (input >> t) ? t : T(0);
      }

      /**
      *  String manipulation auxiliary function.
      */
      static bool startsWith(const std::string& str, const std::string& starting);

      /**
      *  String manipulation auxiliary function.
      */
      static bool endsWith(const std::string& str, const std::string& ending);

      /**
      *  String manipulation auxiliary function.
      */
      static std::string trimLeft(const std::string& str);

      /**
      *  String manipulation auxiliary function.
      */
      static std::string trimRight(const std::string& str);

      /**
      *  String manipulation auxiliary function.
      */
      static std::string trim(const std::string& str);

      /**
      *  String manipulation auxiliary function.
      */
      static std::string toUpperCase(const std::string& str);

      /**
      *  String manipulation auxiliary function.
      */
      static std::string toLowerCase(const std::string& str);

      /**
      *  String manipulation auxiliary function.
      */
      static std::string formatNumberString(std::size_t number, std::size_t totalNumbers);

      /**
      *  String manipulation auxiliary function.
      *  Note: the Container has to support the 'insert()' function (std::vector, ordered & unordered std::map/std::set, std::dequeue, but NOT std::queue).
      */
      template <typename Container>
      static Container tokenize(const std::string& str, const std::string& delimiters = " ")
      {
        Container results;
        if (str.empty())
        {
          return results; // early return of empty filename string
        }

        // trim first the given string
        std::string trimmedStr = trim(str);
        // skip delimiters at beginning
        std::string::size_type lastPosition = trimmedStr.find_first_not_of(delimiters, 0);
        // find first "non-delimiter"
        std::string::size_type position     = trimmedStr.find_first_of(delimiters, lastPosition);

        while ((std::string::npos != position) || (std::string::npos != lastPosition))
        {
          // found a token, add it to the Container
          results.insert(results.end(), trimmedStr.substr(lastPosition, position - lastPosition));
          // skip delimiters, note the "not_of"
          lastPosition = trimmedStr.find_first_not_of(delimiters, position);
          // find next "non-delimiter"
          position     = trimmedStr.find_first_of(delimiters, lastPosition);
        }

        return results;
      }

      StringAuxiliaryFunctions()  = delete;
      ~StringAuxiliaryFunctions() = delete;
      StringAuxiliaryFunctions(const StringAuxiliaryFunctions&) = delete;
      StringAuxiliaryFunctions(StringAuxiliaryFunctions&&)      = delete;
      StringAuxiliaryFunctions& operator=(const StringAuxiliaryFunctions&) = delete;
      StringAuxiliaryFunctions& operator=(StringAuxiliaryFunctions&&)      = delete;

    private:

      static constexpr std::size_t    STRING_BUFFER_SIZE = 2048;
      static constexpr std::size_t CHARACTER_BUFFER_SIZE = 64;

      /**
      *  String manipulation auxiliary function.
      */
      template<typename T>
      static inline std::string parseNumberCStyle(const char* format, const T value)
      {
        std::array<char, CHARACTER_BUFFER_SIZE> resultString{ { ' ' } }; // double braces because we initialize an array inside an std::array object
        std::snprintf(resultString.data(), CHARACTER_BUFFER_SIZE, format, value);

        return std::string(resultString.data());
      }
    };

    /** @brief The StdReadWriteFileFunctions struct provides additional i/o functionality.
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    struct UTILS_MODULE_API StdReadWriteFileFunctions final
    {
      /**
      *  Checks if stream is open.
      */
      static bool assure(const std::ios& stream, const std::string& fullpathWithFileName);

      /**
      *  Checks if file is empty.
      */
      static bool assure(std::size_t numberOfElements, const std::string& fullpathWithFileName);

      /**
      *  Reads a text file into a list of line strings.
      */
      static std::list<std::string> readTextFile(const std::string& fullpathWithFileName, bool trimString = true);

      /**
      *  Writes a text file with a given text.
      */
      static void writeTextFile(const std::string& fullpathWithFileName, const std::string& textToWrite, std::ios_base::openmode mode = std::ios::out);

      /**
      *  Writes a text file with a given list of texts.
      */
      static void writeTextFile(const std::string& fullpathWithFileName, const std::list<std::string>& textToWrite, std::ios_base::openmode mode = std::ios::out);

      /**
      *  Checks if a given path exists using the C++17 <filesystem>.
      */
      static bool pathExists(const std::string& fullpath);

      /**
      *  Gets the file size of a given file using the C++17 <filesystem>.
      */
      static std::size_t getFileSize(const std::string& fullpathWithFileName);

      /**
      *  Gets the current path using the C++17 <filesystem>.
      */
      static std::string getCurrentPath();

      /**
      *  Removes the given file using the C++17 <filesystem>.
      */
      static bool removeFile(const std::string& fullpathWithFileName);

      /**
      *  Removes all files with given extension in given directory using the C++17 <filesystem>.
      */
      static bool removeAllFilesWithExtension(const std::string& fullpath, const std::string& fileExtension);

      /**
      *  Creates the given directory using the C++17 <filesystem>.
      */
      static bool createDirectory(const std::string& fullpath);

      /**
      *  Removes the given directory with anything in it recursively using the C++17 <filesystem>.
      */
      static std::uintmax_t removeDirectory(const std::string& fullpath);

      /**
      *  Cast T as bytes.
      */
      template<typename T>
      static inline char* asBytes(const T* obj)
      {
        return reinterpret_cast<char*>(obj); // treat memory as bytes
      }

      /**
      *  Cast void* as object T.
      */
      template<typename T>
      static inline T* asObject(void* data)
      {
        return reinterpret_cast<T*>(data);
      }

      /**
      *  Write a binary file from the given T* pointer array.
      */
      template<typename T>
      static bool writeBinaryFile(const std::string& fullpathWithFileName, const T* __restrict ptr, std::size_t arraySize)
      {
        std::ofstream out;
        out.open(fullpathWithFileName, std::ios::out | std::ios::binary);
        if (!assure(out, fullpathWithFileName))
        {
          return false;
        }

        out.write(asBytes<T>(ptr), arraySize * sizeof(T));
        out.close();

        return true;
      } // also runs the destructor of ofstream in case of early function exit (the RAII principle)

      /**
      *  Read the given binary file to an std::vector<T>.
      */
      template<typename T>
      static bool readBinaryFile(const std::string& fullpathWithFileName, std::vector<T>& vec)
      {
        std::ifstream in;
        in.open(fullpathWithFileName, std::ios::in | std::ios::binary);
        if (!assure(in, fullpathWithFileName))
        {
          return false;
        }

        std::size_t numberOfElements = getFileSize(fullpathWithFileName) / sizeof(T);
        if (!assure(numberOfElements, fullpathWithFileName))
        {
          return false;
        }

        vec.resize(numberOfElements);
        in.read(asBytes<T>(vec.data()), numberOfElements * sizeof(T));
        in.close();

        return true;
      } // also runs the destructor of ofstream in case of early function exit (the RAII principle)

      /**
      *  Read the given binary file to an std::unique_ptr<T[]>.
      */
      template<typename T>
      static std::tuple<bool, std::size_t> readBinaryFile(const std::string& fullpathWithFileName, std::unique_ptr<T[]>& ptr)
      {
        std::ifstream in;
        in.open(fullpathWithFileName, std::ios::in | std::ios::binary);
        if (!assure(in, fullpathWithFileName))
        {
          return std::make_tuple(false, 0);
        }

        std::size_t numberOfElements = getFileSize(fullpathWithFileName) / sizeof(T);
        if (!assure(numberOfElements, fullpathWithFileName))
        {
          return std::make_tuple(false, 0);
        }

        ptr = std::unique_ptr<T[]>(new T[numberOfElements]); // only reserve the memory and set all the values later from the read() call
        in.read(asBytes<T>(ptr.get()), numberOfElements * sizeof(T));
        in.close();

        return std::make_tuple(true, numberOfElements);
      } // also runs the destructor of ofstream in case of early function exit (the RAII principle)

      StdReadWriteFileFunctions()  = delete;
      ~StdReadWriteFileFunctions() = delete;
      StdReadWriteFileFunctions(const StdReadWriteFileFunctions&) = delete;
      StdReadWriteFileFunctions(StdReadWriteFileFunctions&&)      = delete;
      StdReadWriteFileFunctions& operator=(const StdReadWriteFileFunctions&) = delete;
      StdReadWriteFileFunctions& operator=(StdReadWriteFileFunctions&&)      = delete;
    };

    /** @brief The DebugConsole class provides debugging & logging functionality.
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    class UTILS_MODULE_API DebugConsole final
    {
    public:

      static void setLogFileName(const std::string& givenLogFileName);
      static void setUseLogFile(bool givenUseLogFile);

      template<typename... Args>
      static inline void printfConsoleOutLineImpl(const char* format, const Args... args)
      {
        std::printf(format, args...);
        printfFileOutLineImpl(format, args...);
      }

      template<typename... Args>
      static inline void printfFileOutLineImpl(const char* format, const Args... args)
      {
        if (getUseLogFile())
        {
          const std::string msg(StringAuxiliaryFunctions::printfToString(format, args...));
          writeLogFileImpl(msg);
        }
      }

      static inline void consoleOutLineImpl()
      {
        const std::string msg(1U, '\n');
        std::cout << msg;
        checkAndWriteLogFileImpl(msg);
      }

      template<typename... Args>
      static inline void consoleOutLineImpl(const Args... args)
      {
        std::ostringstream ss;
        const std::initializer_list<int> x = { ((ss << args), 0)... }; // abusing the comma operator when expanding the parameter pack
        if (!std::empty(x))                                            // check to avoid an error for empty initializer_list
        {
          ss << '\n';
          std::cout << ss.str();
          checkAndWriteLogFileImpl(ss.str());
        }
      }

      static inline void fileOutLineImpl()
      {
        if (getUseLogFile())
        {
          const std::string msg(1U, '\n');
          writeLogFileImpl(msg);
        }
      }

      template<typename... Args>
      static inline void fileOutLineImpl(const Args... args)
      {
        if (getUseLogFile())
        {
          std::ostringstream ss;
          const std::initializer_list<int> x = { ((ss << args), 0)... }; // abusing the comma operator when expanding the parameter pack
          if (!std::empty(x))                                            // check to avoid an error for empty initializer_list
          {
            ss << '\n';
            writeLogFileImpl(ss.str());
          }
        }
      }

      DebugConsole()  = delete;
      ~DebugConsole() = delete;
      DebugConsole(const DebugConsole&) = delete;
      DebugConsole(DebugConsole&&)      = delete;
      DebugConsole& operator=(const DebugConsole&) = delete;
      DebugConsole& operator=(DebugConsole&&)      = delete;

    private:

      static std::string getLogFileName();
      static bool getUseLogFile();
      static void checkAndWriteLogFileImpl(const std::string& msg);
      static void writeLogFileImpl(const std::string& msg);
    };
  }
} // namespace Utils

#endif // __UtilityFunctions_h