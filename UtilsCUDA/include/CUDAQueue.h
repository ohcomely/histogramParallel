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

#ifndef __CUDAQueue_h
#define __CUDAQueue_h

#include "CUDAMemoryHandlers.h"
#include "CUDAUtilityFunctions.h"
#include <type_traits>
#include <algorithm>

/** @brief namespace UtilsCUDA for encapsulating all the CUDA related code compiled by the NVCC compiler.
*
* @author Thanos Theo, 2018
* @version 14.0.0.0
*/
namespace UtilsCUDA
{
  /** @brief CUDAQueueView to linearly access elements in the CUDAQueue class.
  *
  * @tparam T see @class CUDAQueue
  *
  * @author Leonid Volnin, 2019
  * @version 14.0.0.0
  */
  template<typename T>
  class CUDAQueueView final
  {
  public:

    CUDAQueueView(const T* ptr, std::size_t offset, std::size_t length)
        : devicePtr_(ptr)
        , offset_(offset)
        , length_(length)
    {
    }

    __forceinline__ __host__ __device__       T& operator[](std::size_t index)       { return devicePtr_[(offset_ + index) % length_]; }
    __forceinline__ __host__ __device__ const T& operator[](std::size_t index) const { return devicePtr_[(offset_ + index) % length_]; }

  private:

    const T* devicePtr_ = nullptr;
    std::size_t offset_ = 0;
    std::size_t length_ = 0;
  };

  /** @brief CUDAQueue class for GPU memory. It uses a pre-allocated memory buffer to avoid expensive memory allocations and de-allocations. Elements are stored as a ring buffer (also known as a circular buffer).
  *
  * @tparam T - should be a Plain Old Data (POD) type
  *
  * @author Leonid Volnin, 2019
  * @version 14.0.0.0
  */
  template<typename T>
  class CUDAQueue final
  {
  public:

    static_assert(std::is_trivially_copyable<T>::value, "A CUDAQueue<T> element should be trivially copyable.");

    /** @brief First constructor for the CUDAQueue: with a given buffer.
    *
    * Note: Due to two step initialization of device memory allocation via the CUDA memory pool (see dotred_utils_cuda::CUDAMemoryPool),
    *       it is necessary to pass the buffer size because memory may not be allocated yet. This should be updated once the memory pool allocation strategy changes.
    *
    * @param buffer     storage for CUDAQueue elements. Stored as a reference, so it should have longer lifetime than @class CUDAQueue object.
    * @param bufferSize size of the storage buffer. It is the user's responsibility to allocate enough memory to store all data.
    */
    explicit CUDAQueue(DeviceMemory<T>& buffer, std::size_t bufferSize)
      : buffer_(buffer)
      , bufferSize_(bufferSize)
    {
    }

    /** @brief Second constructor for the CUDAQueue: with an owned storage buffer allocated directly, optionally on a given device & with Unified Memory enabled.
    *
    * @param bufferSize size of the storage buffer. It is the user's responsibility to allocate enough memory to store all data.
    * @param device           optional parameter to use a given device.
    * @param useUnifiedMemory optional parameter to enable Unified Memory.
    */
    explicit CUDAQueue(std::size_t bufferSize, int device = 0, bool useUnifiedMemory = false)
      : ownedStorage_(bufferSize, device, useUnifiedMemory)
      , buffer_(ownedStorage_)
      , bufferSize_(bufferSize)
    {
    }

    /** @brief Copies @param elements to the end of the CUDAQueue.
    *
    * @param elements
    * @param length number of elements to copy
    */
    void push_back(const DeviceMemory<T>& elements, std::size_t length)
    {
      assert(actualSize_ + length <= bufferSize_);
      if (offset_ + actualSize_ + length > bufferSize_)
      {
        if (offset_ + actualSize_ < bufferSize_)
        {
          std::size_t leftPart = bufferSize_ - actualSize_ - offset_;
          CUDAError_checkCUDAError(cudaMemcpy(buffer_.device() + offset_ + actualSize_, elements.device(),            sizeof(T) * leftPart,            cudaMemcpyDeviceToDevice));
          CUDAError_checkCUDAError(cudaMemcpy(buffer_.device(),                         elements.device() + leftPart, sizeof(T) * (length - leftPart), cudaMemcpyDeviceToDevice));
        }
        else
        {
          std::size_t offset = (offset_ + actualSize_) % bufferSize_;
          CUDAError_checkCUDAError(cudaMemcpy(buffer_.device() + offset, elements.device(), sizeof(T) * length, cudaMemcpyDeviceToDevice));
        }
      }
      else
      {
        std::size_t offset = offset_ + actualSize_;
        CUDAError_checkCUDAError(cudaMemcpy(buffer_.device() + offset, elements.device(), sizeof(T) * length, cudaMemcpyDeviceToDevice));
      }
      actualSize_ += length;
    }

    /** @brief Copies elements from begin of the CUDAQueue to @param dst buffer. If the CUDAQueue has less than @param length elements, copies only what the CUDAQueue has.
    *
    * @param dst destination buffer
    * @param length number of elements to copy
    * @return number of elements that were copied
    */
    std::size_t front(DeviceMemory<T>& dst, std::size_t length) const
    {
      assert(dst.getNumberOfElements() >= length);
      length = std::min(length, actualSize_);
      if (offset_ + length > bufferSize_)
      {
        std::size_t leftPart = bufferSize_ - offset_;
        CUDAError_checkCUDAError(cudaMemcpy(dst.device(), buffer_.device() + offset_,  sizeof(T) * leftPart,            cudaMemcpyDeviceToDevice));
        CUDAError_checkCUDAError(cudaMemcpy(dst.device() + leftPart, buffer_.device(), sizeof(T) * (length - leftPart), cudaMemcpyDeviceToDevice));
      }
      else
      {
        CUDAError_checkCUDAError(cudaMemcpy(dst.device(), buffer_.device() + offset_,  sizeof(T) * length,              cudaMemcpyDeviceToDevice));
      }
      return length;
    }

    /** @brief Removes no more than @param length elements from begin of CUDAQueue.
    *
    * @param length
    * @return number of actual elements removed
    */
    std::size_t pop_front(std::size_t length)
    {
      length       = std::min(length, actualSize_);
      actualSize_ -= length;
      offset_      = (offset_ + length) % bufferSize_;
      return length;
    }

    /** @brief Number of elements in the CUDAQueue.
    */
    std::size_t size()     const { return actualSize_;      }

    /** @brief Check if the CUDAQueue is empty.
    */
    bool empty()           const { return actualSize_ == 0; }

    /** @brief Maximum size of the CUDAQueue.
    */
    std::size_t reserved() const { return bufferSize_;      }

    /** @brief A CUDAQueueView to access CUDAQueue elements in sequential order.
    */
    CUDAQueueView<T> view() const
    {
      // make sure the CUDAQueueView can be passed to device function by value
      static_assert(std::is_trivially_copyable<CUDAQueueView<T>>::value, "A CUDAQueueView<T> element should be trivially copyable.");
      return CUDAQueueView<T>{buffer_.device(), offset_, bufferSize_};
    }

  private:

    DeviceMemory<T>  ownedStorage_;
    DeviceMemory<T>& buffer_;
    std::size_t bufferSize_ = 0;
    std::size_t actualSize_ = 0;
    std::size_t offset_     = 0;
  };
} // namespace UtilsCUDA

#endif // __CUDAQueue_h