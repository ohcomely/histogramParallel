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

#ifndef __UnitTests_h
#define __UnitTests_h

// Note: Due to NVCC not being C++14 compliant yet, we cannot use the UtilityFunctions header directly within UnitTests.h.
//       Thus, it was decided to use a UnitTests.cpp compilation unit to hide the usage of the UtilityFunctions header.
//       As a result, an extra explicit instantiation definition of function template was needed, along with suppressing a Windows-side warning.
#if defined _WIN32
  #pragma warning (push)
  #pragma warning (disable : 4661) // for UnitTestUtilityFunctions::parseComplexDataFromText() cpp-side template definition
#endif

#include "ModuleDLL.h"
#include <type_traits>
#include <complex>
#include <cmath>
#include <tuple>
#include <sstream>
#include <string>
#include <list>
#include <vector>
#include <cstdint>

/** @brief Namespace Utils contains utility classes with mainly static CPU related methods.
* @author Thanos Theo, 2009-2018
* @version 14.0.0.0
*/
namespace Utils
{
  /** @brief Namespace UnitTests contains classes used for unit testing.
  * @author Thanos Theo, 2009-2018
  * @version 14.0.0.0
  */
  namespace UnitTests
  {
    /** @brief The UnitTestsInterface struct encapsulate a basic unit test interface using the Curiously Recurring Template Pattern (CRTP).
    * @author Thanos Theo, 2018
    * @version 14.0.0.0
    */
    template <typename Derived>
    class CRTP_MODULE_API UnitTestsInterface
    {
    public:

      void resetTests()        {        asDerived()->resetTests();        }
      bool conductTests()      { return asDerived()->conductTests();      }
      void reportTestResults() {        asDerived()->reportTestResults(); }

      UnitTestsInterface()  = default;
      ~UnitTestsInterface() = default; // no virtual destructor for data-oriented design (no up-casting should ever be used)
      UnitTestsInterface(const UnitTestsInterface&) = delete; // copy-constructor default
      UnitTestsInterface(UnitTestsInterface&&)      = delete; // move-constructor delete
      UnitTestsInterface& operator=(const UnitTestsInterface&) = delete; //      assignment operator default
      UnitTestsInterface& operator=(UnitTestsInterface&&)      = delete; // move-assignment operator delete

    private:

            Derived* asDerived()       { return reinterpret_cast<      Derived*>(this); }
      const Derived* asDerived() const { return reinterpret_cast<const Derived*>(this); }
    };

    /** @brief The UnitTestUtilityFunctions class adds unit testing utility function support through private inheritance.
    * @author Thanos Theo, 2018
    * @version 14.0.0.0
    */
    template <typename T>
    class UTILS_MODULE_API UnitTestUtilityFunctions
    {
    public:

      static inline T delta(T a, T b)
      {
        return std::abs(a - b);
      }

      static inline bool checkAbsoluteError(T a, T b, T error = getDefaultError())
      {
        return (std::abs(a - b) > error);
      }

      static inline bool checkRelativeError(T a, T b, T relativeError = getDefaultError())
      {
        return (std::abs((a - b) / a) > relativeError);
      }

      static inline bool checkComplexAbsoluteError(std::complex<T> a, std::complex<T> b, T error = getDefaultError())
      {
        return (std::abs(a.real() - b.real()) > error) && (std::abs(a.imag() - b.imag()) > error);
      }

      static inline bool checkComplexRelativeError(std::complex<T> a, std::complex<T> b, T relativeError = getDefaultError())
      {
        T magnitude = std::abs(a);
        return (std::abs((a.real() - b.real()) / magnitude) > relativeError) && (std::abs((a.imag() - b.imag()) / magnitude) > relativeError);
      }

      /// Note: Template types I & W should not be decimals but only complex numbers of types std::complex, fftw & cufft.
      ///       These template I & W types are not checked with template metaprogramming (besides if not decimal),
      ///       so as to avoid dependencies to non-std complex numbers structs (fftw & cufft) in the Utils component.
      template <typename I, typename W>
      static inline bool checkComplexRootMeanSquaredError(const I* __restrict arrayA, const W* __restrict arrayB, std::size_t arraySize,
                                                         T squaredError = getDefaultError(),
                                                         std::enable_if_t<!std::is_floating_point<I>::value && !std::is_floating_point<W>::value>* = nullptr)
      {
        const T* arrayAptr = nullptr;
        const T* arrayBptr = nullptr;
        T tempReal         = T(0);
        T tempImag         = T(0);
        T sumRealError     = T(0);
        T sumImagError     = T(0);
        for (std::size_t i = 0; i < arraySize; ++i)
        {
          // C-style cast for complex objects with aligned memory access patterns
          // verified to work for std::complex, fftw & cufft structs
          arrayAptr     = reinterpret_cast<const T*>(&arrayA[i]);
          arrayBptr     = reinterpret_cast<const T*>(&arrayB[i]);
          tempReal      = arrayAptr[0] - arrayBptr[0];
          tempImag      = arrayAptr[1] - arrayBptr[1];
          sumRealError += tempReal * tempReal;
          sumImagError += tempImag * tempImag;
        }
        sumRealError /= T(arraySize);
        sumImagError /= T(arraySize);
        sumRealError  = T(std::sqrt(sumRealError));
        sumImagError  = T(std::sqrt(sumImagError));

        return (sumRealError > squaredError) || (sumImagError > squaredError);
      }

      /// Note: Template types I & W should not be decimals but only complex numbers of types std::complex, fftw & cufft.
      ///       These template I & W types are not checked with template metaprogramming (besides if not decimal),
      ///       so as to avoid dependencies to non-std complex numbers structs (fftw & cufft) in the Utils component.
      template <typename I, typename W>
      static inline bool checkComplexTwoNormError(const I* __restrict arrayA, const W* __restrict arrayB, std::size_t arraySize,
                                                  T error = getDefaultError(),
                                                  std::enable_if_t<!std::is_floating_point<I>::value && !std::is_floating_point<W>::value>* = nullptr)
      {
        const T* arrayAptr = nullptr;
        const T* arrayBptr = nullptr;
        T absNumerator     = T(0);
        T absDenumerator   = T(0);
        T sumNumerator     = T(0);
        T sumDenumerator   = T(0);
        for (std::size_t i = 0; i < arraySize; ++i)
        {
          // C-style cast for complex objects with aligned memory access patterns
          // verified to work for std::complex, fftw & cufft structs
          arrayAptr       = reinterpret_cast<const T*>(&arrayA[i]);
          arrayBptr       = reinterpret_cast<const T*>(&arrayB[i]);
          absNumerator    = (arrayAptr[0] - arrayBptr[0]) * (arrayAptr[0] - arrayBptr[0]) + (arrayAptr[1] - arrayBptr[1]) * (arrayAptr[1] - arrayBptr[1]);
          absDenumerator  = arrayBptr[0] * arrayBptr[0] + arrayBptr[1] * arrayBptr[1];
          sumNumerator   += absNumerator;
          sumDenumerator += absDenumerator;
        }
        sumNumerator   = T(std::sqrt(sumNumerator));
        sumDenumerator = T(std::sqrt(sumDenumerator));

        return (sumNumerator / sumDenumerator > error);
      }

      static inline std::tuple<bool, std::string> checkSeriesError(const T* __restrict arrayA, const T* __restrict arrayB, std::size_t arraySize,
                                                                   bool frequencyData = false, T timeError = getDefaultTimeError(), T frequencyError = getDefaultFrequencyError())
      {
        T rootMeanSquare = T(0);
        T maxArrayValue  = arrayB[0];

        for (std::size_t i = 0; i < arraySize; ++i)
        {
          rootMeanSquare += arrayB[i] * arrayB[i];
          if (maxArrayValue < arrayB[i])
          {
            maxArrayValue = arrayB[i];
          }
        }
        rootMeanSquare = std::sqrt(rootMeanSquare / arraySize);

        if (frequencyData)
        {
          for (std::size_t i = 0; i < arraySize; ++i)
          {
            T relativeError = std::abs(arrayA[i] - arrayB[i]) / std::abs(arrayB[i]);
            T absoluteError = std::abs(arrayA[i] - arrayB[i]);
            if ((relativeError > frequencyError) && (absoluteError > 1.0e-6f  * maxArrayValue))
            {
              std::ostringstream failMsg;
              failMsg << "checkSeriesError() for frequency data failed at index : " << i << " size: " << arraySize << std::endl;
              failMsg << " with relative error found: " << relativeError << " allowed: " << frequencyError << " absolute error found: " << absoluteError << " allowed: " << 1.0e-6f * maxArrayValue << std::endl;
              return std::make_tuple(false, failMsg.str());
            }
          }
        }
        else
        {
          for (std::size_t i = 0; i < arraySize; ++i)
          {
            T absoluteError = std::abs(arrayA[i] - arrayB[i]);
            if ((absoluteError / rootMeanSquare) > timeError)
            {
              std::ostringstream failMsg;
              failMsg << "checkSeriesError() for time domain data failed at index : " << i << " size: " << arraySize << std::endl;
              failMsg << " with error relative to RMS found: " << (absoluteError / rootMeanSquare) << " allowed: " << timeError << std::endl;
              return std::make_tuple(false, failMsg.str());
            }
          }
        }

        return std::make_tuple(true, "");
      }

      /// Note: Template types I & W should not be decimals but only complex numbers of types std::complex, fftw & cufft.
      ///       These template I & W types are not checked with template metaprogramming (besides if not decimal),
      ///       so as to avoid dependencies to non-std complex numbers structs (fftw & cufft) in the Utils component.
      template <typename I, typename W>
      static inline std::tuple<bool, std::string> checkSeriesError(const I* __restrict arrayA, const W* __restrict arrayB, std::size_t arraySize,
                                                                   bool frequencyData = false, T timeError = getDefaultTimeError(), T frequencyError = getDefaultFrequencyError(),
                                                                   std::enable_if_t<!std::is_floating_point<I>::value && !std::is_floating_point<W>::value>* = nullptr)
      {
        T rootMeanSquare    = T(0);
        T tempAbsoluteValue = T(0);
        const T* arrayAptr  = nullptr;
        const T* arrayBptr  = nullptr;
        arrayBptr           = reinterpret_cast<const T*>(&arrayB[0]);
        T maxAbsoluteValue  = std::sqrt(arrayBptr[0] * arrayBptr[0] + arrayBptr[1] * arrayBptr[1]);

        for (std::size_t i = 0; i < arraySize; ++i)
        {
          // C-style cast for complex objects with aligned memory access patterns
          // verified to work for std::complex, fftw & cufft structs
          arrayBptr        = reinterpret_cast<const T*>(&arrayB[i]);
          rootMeanSquare   += (arrayBptr[0] * arrayBptr[0] + arrayBptr[1] * arrayBptr[1]);
          tempAbsoluteValue = std::sqrt(arrayBptr[0] * arrayBptr[0] + arrayBptr[1] * arrayBptr[1]);
          if (maxAbsoluteValue < tempAbsoluteValue)
          {
            maxAbsoluteValue = tempAbsoluteValue;
          }
        }
        rootMeanSquare = std::sqrt(rootMeanSquare / arraySize);

        if (frequencyData)
        {
          for (std::size_t i = 0; i < arraySize; ++i)
          {
            arrayAptr = reinterpret_cast<const T*>(&arrayA[i]);
            arrayBptr = reinterpret_cast<const T*>(&arrayB[i]);
            T absoluteError = std::sqrt((arrayAptr[0] - arrayBptr[0]) * (arrayAptr[0] - arrayBptr[0]) + (arrayAptr[1] - arrayBptr[1]) * (arrayAptr[1] - arrayBptr[1]));
            T relativeError = absoluteError / std::sqrt(arrayBptr[0] * arrayBptr[0] + arrayBptr[1] * arrayBptr[1]);

            if ((relativeError > frequencyError) && (absoluteError > 1.0e-6f  * maxAbsoluteValue))
            {
              std::ostringstream failMsg;
              failMsg << "checkSeriesError() for complex frequency data failed at index : " << i << " size: " << arraySize << std::endl;
              failMsg << " with relative error found: " << relativeError << " allowed: " << frequencyError << " absolute error found: " << absoluteError << " allowed: " << 1.0e-6f * maxAbsoluteValue << std::endl;
              return std::make_tuple(false, failMsg.str());
            }
          }
        }
        else
        {
          for (std::size_t i = 0; i < arraySize; ++i)
          {
            arrayAptr = reinterpret_cast<const T*>(&arrayA[i]);
            arrayBptr = reinterpret_cast<const T*>(&arrayB[i]);
            T absoluteError = std::sqrt((arrayAptr[0] - arrayBptr[0]) * (arrayAptr[0] - arrayBptr[0]) + (arrayAptr[1] - arrayBptr[1]) * (arrayAptr[1] - arrayBptr[1]));

            if ((absoluteError / rootMeanSquare) > timeError)
            {
              std::ostringstream failMsg;
              failMsg << "checkSeriesError() for complex time domain data failed at index : " << i << " size: " << arraySize << std::endl;
              failMsg << " with error relative to RMS found: " << (absoluteError / rootMeanSquare) << " allowed: " << timeError << std::endl;
              return std::make_tuple(false, failMsg.str());
            }
          }
        }

        return std::make_tuple(true, "");
      }

      /// Note: Template types I & W should not be decimals but only complex numbers of types std::complex, fftw & cufft.
      ///       These template I & W types are not checked with template metaprogramming (besides if not decimal),
      ///       so as to avoid dependencies to non-std complex numbers structs (fftw & cufft) in the Utils component.
      template <typename I, typename W>
      static inline std::tuple<bool, std::string> verifyComplexArraysAbsoluteError(const std::string& arrayAName, const I* __restrict arrayA, std::size_t arrayASize,
                                                                                   const std::string& arrayBName, const W* __restrict arrayB, std::size_t arrayBSize,
                                                                                   T error = getDefaultError(),
                                                                                   std::enable_if_t<!std::is_floating_point<I>::value && !std::is_floating_point<W>::value>* = nullptr)
      {
        if (arrayASize != arrayBSize)
        {
          std::ostringstream failMsg;
          failMsg << "verifyComplexArraysAbsoluteError() failed with array sizes : \n"
                  << " with " << arrayAName << " size: " << arrayASize << std::endl
                  << " with " << arrayBName << " size: " << arrayBSize << std::endl;
          return std::make_tuple(false, failMsg.str());
        }

        for (std::size_t i = 0; i < arrayASize; ++i)
        {
          if (checkComplexAbsoluteError(arrayA[i], arrayB[i], error))
          {
            std::ostringstream failMsg;
            failMsg << "verifyComplexArraysAbsoluteError() failed at index: " << i         << " with values:\n"
                    << " with " << arrayAName << " value: "                   << arrayA[i] << std::endl
                    << " with " << arrayBName << " value: "                   << arrayB[i] << std::endl;
            return std::make_tuple(false, failMsg.str());
          }
        }

        return std::make_tuple(true, "");
      }

      /// Note: Template types I & W should not be decimals but only complex numbers of types std::complex, fftw & cufft.
      ///       These template I & W types are not checked with template metaprogramming (besides if not decimal),
      ///       so as to avoid dependencies to non-std complex numbers structs (fftw & cufft) in the Utils component.
      template <typename I, typename W>
      static inline std::tuple<bool, std::string> verifyComplexArraysRelativeError(const std::string& arrayAName, const I* __restrict arrayA, std::size_t arrayASize,
                                                                                   const std::string& arrayBName, const W* __restrict arrayB, std::size_t arrayBSize,
                                                                                   T relativeError = getDefaultError(),
                                                                                   std::enable_if_t<!std::is_floating_point<I>::value && !std::is_floating_point<W>::value>* = nullptr)
      {
        if (arrayASize != arrayBSize)
        {
          std::ostringstream failMsg;
          failMsg << "verifyComplexArraysRelativeError() failed with array sizes : \n"
                  << " with " << arrayAName << " size: " << arrayASize << std::endl
                  << " with " << arrayBName << " size: " << arrayBSize << std::endl;
          return std::make_tuple(false, failMsg.str());
        }

        for (std::size_t i = 0; i < arrayASize; ++i)
        {
          if (checkComplexRelativeError(arrayA[i], arrayB[i], relativeError))
          {
            std::ostringstream failMsg;
            failMsg << "verifyComplexArraysRelativeError() failed at index: " << i         << " with values:\n"
                    << " with " << arrayAName << " value: "                   << arrayA[i] << std::endl
                    << " with " << arrayBName << " value: "                   << arrayB[i] << std::endl;
            return std::make_tuple(false, failMsg.str());
          }
        }

        return std::make_tuple(true, "");
      }

      static void parseComplexArrayFromTextRowMajor(   const std::list<std::string>& dataLines, std::complex<T>* __restrict complexArray, std::uint32_t dataSize);

      static void parseComplexArrayFromTextColumnMajor(const std::list<std::string>& dataLines, std::complex<T>* __restrict complexArray);

      UnitTestUtilityFunctions()  = default;
      ~UnitTestUtilityFunctions() = default; // no virtual destructor for data-oriented design (no up-casting should ever be used)
      UnitTestUtilityFunctions(const UnitTestUtilityFunctions&) = delete; // copy-constructor default
      UnitTestUtilityFunctions(UnitTestUtilityFunctions&&)      = delete; // move-constructor delete
      UnitTestUtilityFunctions& operator=(const UnitTestUtilityFunctions&) = delete; //      assignment operator default
      UnitTestUtilityFunctions& operator=(UnitTestUtilityFunctions&&)      = delete; // move-assignment operator delete

    protected:

      static inline T getDefaultError()          { return T(1e-5); } // 5th decimal default error
      static inline T getDefaultTimeError()      { return T(1e-2); } // 2th decimal default error
      static inline T getDefaultFrequencyError() { return T(1e-2); } // 2th decimal default error
    };

    template class UnitTestUtilityFunctions<float>;  // explicit instantiation definition of class template
    template class UnitTestUtilityFunctions<double>; // explicit instantiation definition of class template
    using UnitTestUtilityFunctions_flt = UnitTestUtilityFunctions<float>;
    using UnitTestUtilityFunctions_dbl = UnitTestUtilityFunctions<double>;
  } // namespace UnitTests
} // namespace Utils

#endif // __UnitTests_h