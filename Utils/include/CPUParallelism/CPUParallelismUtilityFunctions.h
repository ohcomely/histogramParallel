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

#ifndef __CPUParallelismUtilityFunctions_h
#define __CPUParallelismUtilityFunctions_h

#include "../ModuleDLL.h"
#include <type_traits>
#include <future>
#include <utility>
#include <atomic>

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
    /** @brief This class encapsulates all the CPUParalellism related utility functions.
    *
    * @author Thanos Theo, 2018
    * @version 14.0.0.0
    */
    class UTILS_MODULE_API CPUParallelismUtilityFunctions final
    {
    public:

      /** @brief According to Scott Meyers, enforce task parallelism execution with the std::launch::async parameter in std::async().
      */
      template<typename F, typename... Ts>
      static inline auto reallyAsync(F&& f, Ts&&... params)
      {
        return std::async(std::launch::async, std::forward<F>(f), std::forward<Ts>(params)...);
      }

      /** @brief Perform an atomic addition to the T (decimal type only allowed for T, as C++ has specialized versions for integral types in its atomic library).
      */
      template<typename T>
      static inline T atomicAdd(std::atomic<T>& value, T newValue, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr)
      {
        return atomicArithmeticOp<T>(value, newValue, std::plus<T>());
      }

      /** @brief Perform an atomic multiply to the T (decimal type only allowed for T, as C++ has specialized versions for integral types in its atomic library).
      */
      template<typename T>
      static inline T atomicMultiply(std::atomic<T>& value, T newValue, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr)
      {
        return atomicArithmeticOp<T>(value, newValue, std::multiplies<T>());
      }

      /** @brief Perform an atomic min to the T (arithmetic type only allowed for T).
      */
      template<typename T>
      static inline T atomicMin(std::atomic<T>& value, T newValue, std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr)
      {
        return atomicComparisonOp<T>(value, newValue, std::less<T>());
      }

      /** @brief Perform an atomic max to the T (arithmetic type only allowed for T).
      */
      template<typename T>
      static inline T atomicMax(std::atomic<T>& value, T newValue, std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr)
      {
        return atomicComparisonOp<T>(value, newValue, std::greater<T>());
      }

      CPUParallelismUtilityFunctions()  = delete;
      ~CPUParallelismUtilityFunctions() = delete;
      CPUParallelismUtilityFunctions(const CPUParallelismUtilityFunctions&) = delete;
      CPUParallelismUtilityFunctions(CPUParallelismUtilityFunctions&&)      = delete;
      CPUParallelismUtilityFunctions& operator=(const CPUParallelismUtilityFunctions&) = delete;
      CPUParallelismUtilityFunctions& operator=(CPUParallelismUtilityFunctions&&)      = delete;

    private:

      /** @brief Perform an atomic arithmetic operation to the T via spin-locking on compare_exchange_weak(), the Compare-and-Swap (CAS) algorithm.
      */
      template<typename T, typename ArithmeticOp>
      static inline T atomicArithmeticOp(std::atomic<T>& value, T newValue, ArithmeticOp arithmeticOp)
      {
        T oldValue     = value.load();
        T desiredValue = arithmeticOp(oldValue, newValue);
        while (!value.compare_exchange_weak(oldValue, desiredValue))
        {
          desiredValue = arithmeticOp(oldValue, newValue);
        }

        return oldValue;
      }

      /** @brief Perform an atomic comparison operation to the T via spin-locking on compare_exchange_weak(), the Compare-and-Swap (CAS) algorithm.
      */
      template<typename T, typename ComparisonOp>
      static inline T atomicComparisonOp(std::atomic<T>& value, T newValue, ComparisonOp comparisonOp)
      {
        T oldValue     = value.load();
        T desiredValue = comparisonOp(newValue, oldValue) ? newValue : oldValue;
        while (!value.compare_exchange_weak(oldValue, desiredValue))
        {
          desiredValue = comparisonOp(newValue, oldValue) ? newValue : oldValue;
        }

        return oldValue;
      }
    };
  }
}

#endif // __CPUParallelismUtilityFunctions_h