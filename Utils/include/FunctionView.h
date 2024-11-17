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

#ifndef __FunctionView_h
#define __FunctionView_h

#include <cstdint>
#include <type_traits>
#include <utility>

/** @brief Namespace Utils contains utility classes with mainly static CPU related methods.
* @author Thanos Theo, 2009-2018
* @version 14.0.0.0
*/
namespace Utils
{
  /** @brief This class encapsulates usage of a function view (lightweight replacement of std::function).
  *
  * FunctionView<R(T...)> is a lightweight non-owning generic callable
  * object view, similar to a std::function<R(T...)>, but with much less overhead.
  *
  * A FunctionView invocation should have the same cost as a function
  * pointer (which it basically is underneath). The function-like object that the
  * FunctionView refers to MUST have a lifetime that outlasts any use of the FunctionView.
  *
  * In contrast, a full std::function<> is an owning container for a
  * callable object. It's more robust, especially with respect to object
  * lifetimes, but the call overhead is quite high. So use a FunctionView when you can.
  *
  * This implementation comes from LLVM:
  * https://github.com/llvm-mirror/llvm/blob/master/include/llvm/ADT/STLExtras.h
  *
  * For more information & profiling tests:
  * https://vittorioromeo.info/index/blog/passing_functions_to_functions.html
  *
  * @author Thanos Theo, 2009-2018
  * @version 14.0.0.0
  */
  template<typename Fn>
  class FunctionView;

  template<typename Ret, typename ...Params>
  class FunctionView<Ret(Params...)>
  {
  public:

    FunctionView() = default;
    FunctionView(std::nullptr_t)
    {}

    template <typename Callable>
    FunctionView(Callable&& callable, std::enable_if_t<!std::is_same<std::remove_reference_t<Callable>, FunctionView>::value>* = nullptr)
      : callback_(callback_fn<std::remove_reference_t<Callable>>)
      , callable_(reinterpret_cast<std::intptr_t>(&callable))
    {}

    Ret operator()(Params ...params) const
    {
      return callback_(callable_, std::forward<Params>(params)...);
    }

    operator bool() const { return callback_; }

  private:

    Ret(*callback_)(intptr_t callable, Params... params) = nullptr;
    intptr_t callable_ = 0;

    template<typename Callable>
    static Ret callback_fn(intptr_t callable, Params... params)
    {
      return (*reinterpret_cast<Callable*>(callable))(std::forward<Params>(params)...);
    }
  };
} // namespace Utils

#endif // __FunctionView_h