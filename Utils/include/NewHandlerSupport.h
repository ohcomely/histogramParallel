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

#ifndef __NewHandlerSupport_h
#define __NewHandlerSupport_h

#include <new>

/** @brief Namespace Utils contains utility classes with mainly static CPU related methods.
* @author Thanos Theo, 2009-2018
* @version 14.0.0.0
*/
namespace Utils
{
  /** @brief "Mixin-style" base class for class-specific std::set_new_handler support.
  * @author Thanos Theo, 2009-2018
  * @version 14.0.0.0
  */
  template<typename T> class NewHandlerSupport
  {
  public:

    static std::new_handler set_new_handler(std::new_handler newHandler) noexcept
    {
      const std::new_handler oldHandler = currentHandler_;
      currentHandler_ = newHandler;

      return oldHandler;
    }

    // normal new/delete
    static void* operator new(std::size_t size) noexcept(false)
    {
      try
      {
        NewHandlerHolder newHandlerHolder(std::set_new_handler(currentHandler_)); // using RAII for set_new_handler
      }
      catch (...)
      {
        std::bad_alloc badAllocException;
        throw badAllocException;
      }

      return ::operator new(size);
    }
    static void operator delete(void* pMemory) noexcept
    {
      NewHandlerHolder newHandlerHolder(std::set_new_handler(currentHandler_)); // using RAII for set_new_handler

      ::operator delete(pMemory);
    }

    // placement new/delete
    static void* operator new(std::size_t size, void *ptr) noexcept
    {
      NewHandlerHolder newHandlerHolder(std::set_new_handler(currentHandler_)); // using RAII for set_new_handler

      return ::operator new(size, ptr);
    }
    static void operator delete(void* pMemory, void* ptr) noexcept
    {
      NewHandlerHolder newHandlerHolder(std::set_new_handler(currentHandler_)); // using RAII for set_new_handler

      ::operator delete(pMemory, ptr);
    }

    // nothrow new/delete
    static void* operator new(std::size_t size, const std::nothrow_t& nt) noexcept
    {
      NewHandlerHolder newHandlerHolder(std::set_new_handler(currentHandler_)); // using RAII for set_new_handler

      return ::operator new(size, nt);
    }
    static void operator delete(void* pMemory, const std::nothrow_t& nt) noexcept
    {
      NewHandlerHolder newHandlerHolder(std::set_new_handler(currentHandler_)); // using RAII for set_new_handler

      ::operator delete(pMemory, nt);
    }

    NewHandlerSupport()          = default;
    virtual ~NewHandlerSupport() = default;
    NewHandlerSupport(const NewHandlerSupport&) = delete;
    NewHandlerSupport(NewHandlerSupport&&)      = delete;
    NewHandlerSupport& operator=(const NewHandlerSupport&) = delete;
    NewHandlerSupport& operator=(NewHandlerSupport&&)      = delete;

  private:

    class NewHandlerHolder // used as a RAII object for set_new_handler
    {
    public:

      explicit NewHandlerHolder(std::new_handler newHandler) : handler_(newHandler) {}
      ~NewHandlerHolder() { std::set_new_handler(handler_); }
      NewHandlerHolder(const NewHandlerHolder&) = delete;
      NewHandlerHolder(NewHandlerHolder&&)      = delete;
      NewHandlerHolder& operator=(const NewHandlerHolder&) = delete;
      NewHandlerHolder& operator=(NewHandlerHolder&&)      = delete;

    private:

      std::new_handler handler_;
    };

    static std::new_handler currentHandler_;
  };

  template<typename T> std::new_handler NewHandlerSupport<T>::currentHandler_ = nullptr;
} // namespace Utils

#endif // __NewHandlerSupport_h