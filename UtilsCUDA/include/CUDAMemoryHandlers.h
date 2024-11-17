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

#ifndef __CUDAMemoryHandlers_h
#define __CUDAMemoryHandlers_h

#include "ModuleDLL.h"
#include "CUDAUtilityFunctions.h"
#include "CUDAMemoryWrappers.h"
#include "CPUParallelism/CPUParallelismUtilityFunctions.h"
#include <cuda_runtime_api.h>
#include <cstring> // for memset() host call
#include <cassert> // for host and device side asserts()
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <type_traits>
#include <future>
#include <memory>

/** @brief namespace UtilsCUDA for encapsulating all the CUDA related code compiled by the NVCC compiler.
*
* @author Thanos Theo, 2018
* @version 14.0.0.0
*/
namespace UtilsCUDA
{
  /** @brief Custom deleter class for (pinned) host memory.
  *
  * @author David Lenz, Thanos Theo, 2018
  * @version 14.0.0.0
  */
  template <typename T>
  struct PinnedDeleter
  {
    PinnedDeleter(bool useDeleter = true) noexcept : useDeleter_(useDeleter) {};

    void operator()(T* ptr) noexcept
    {
      if (useDeleter_)
      {
        assert(ptr != nullptr);
        assert(isValidHostDevicePointer(ptr));
        CUDAError_checkCUDAError(cudaFreeHost(ptr));
      }
    }

    bool useDeleter_ = false;
  };

  /** @brief Custom deleter class for device memory.
  *
  * @author David Lenz, Thanos Theo, 2018
  * @version 14.0.0.0
  */
  template <typename T>
  struct CUDADeleter
  {
    CUDADeleter(bool useDeleter = true) noexcept : useDeleter_(useDeleter) {};

    void operator()(T* ptr) noexcept
    {
      if (useDeleter_)
      {
        assert(ptr != nullptr);
        assert(isValidHostDevicePointer(ptr));
        CUDAError_checkCUDAError(cudaFree(ptr));
      }
    }

    bool useDeleter_ = false;
  };

  template <typename T>
  using PinnedUniquePtr = std::unique_ptr<T, PinnedDeleter<T>>;
  template <typename T>
  using DeviceUniquePtr = std::unique_ptr<T, CUDADeleter<T>>;

  template <typename T>
  PinnedUniquePtr<T> make_unique_pinned(std::size_t numberOfElements, unsigned int flags = cudaHostRegisterDefault) noexcept
  {
    void* ptr = nullptr;
    std::size_t bytesToAllocate = numberOfElements * sizeof(T);
    CUDAError_checkCUDAError(cudaHostAlloc(&ptr, bytesToAllocate, flags));
    return PinnedUniquePtr<T>(reinterpret_cast<T*>(ptr), PinnedDeleter<T>{true});
  };

  template <typename T>
  DeviceUniquePtr<T> make_unique_device(std::size_t numberOfElements, int device = 0, bool useUnifiedMemory = false) noexcept
  {
    void* ptr = nullptr;
    std::size_t bytesToAllocate = numberOfElements * sizeof(T);
    CUDAError_checkCUDAError(cudaSetDevice(device));
    CUDAError_checkCUDAError(useUnifiedMemory ? cudaMallocManaged(&ptr, bytesToAllocate)
                                              : cudaMalloc(       &ptr, bytesToAllocate));
    return DeviceUniquePtr<T>(reinterpret_cast<T*>(ptr), CUDADeleter<T>{true});
  };

  /** @brief The MemoryHandlersAbstraction class encapsulates a basic abstraction layer for the memory handlers using the Curiously Recurring Template Pattern (CRTP).
  * @author Thanos Theo, 2019
  * @version 14.0.0.0
  */
  template <typename T, typename Derived>
  class CRTP_MODULE_API MemoryHandlersAbstraction
  {
  public:

    /// Allocate functions (with future<void> variant)
    void              allocate(     std::size_t numberOfElements, int device = 0) noexcept {        asDerived()->allocate(     numberOfElements, device); }
    std::future<void> allocateAsync(std::size_t numberOfElements, int device = 0) noexcept { return asDerived()->allocateAsync(numberOfElements, device); }
    /// Reset functions (with future<void> variant)
    void              reset()      noexcept {        asDerived()->reset();      }
    std::future<void> resetAsync() noexcept { return asDerived()->resetAsync(); }
    /// Swap function
    void swap(Derived& other) noexcept { return asDerived()->swap(other);   }
    /// Memset function
    void memset(int value)    noexcept { return asDerived()->memset(value); }

    /// Convenience functions
          T* get()       { return asDerived()->get(); }
    const T* get() const { return asDerived()->get(); }
          T& operator[](std::size_t index)       { return asDerived()->operator[](index); }
    const T& operator[](std::size_t index) const { return asDerived()->operator[](index); }
    explicit operator bool()      const noexcept { return asDerived()->operator bool();   }
    std::size_t getNumberOfElements()      const { return asDerived()->getNumberOfElements(); }
    bool           isMemoryPoolMode()      const { return asDerived()->isMemoryPoolMode();    }

  protected:

    std::size_t numberOfElements_ = 0;
    bool          memoryPoolMode_ = false;

    MemoryHandlersAbstraction()  = default;
    ~MemoryHandlersAbstraction() = default; // no virtual destructor for data-oriented design (no up-casting should ever be used)
    MemoryHandlersAbstraction(const MemoryHandlersAbstraction&) = delete; // copy-constructor default
    MemoryHandlersAbstraction(MemoryHandlersAbstraction&&)      = delete; // move-constructor delete
    MemoryHandlersAbstraction& operator=(const MemoryHandlersAbstraction&) = delete; //      assignment operator default
    MemoryHandlersAbstraction& operator=(MemoryHandlersAbstraction&&)      = delete; // move-assignment operator delete

  private:

          Derived* asDerived()       { return reinterpret_cast<      Derived*>(this); }
    const Derived* asDerived() const { return reinterpret_cast<const Derived*>(this); }
  };

  /** @brief This class encapsulates usage of a collection of host handling techniques (host side only) & the RAII C++ idiom. Using the Curiously Recurring Template Pattern (CRTP).
  *
  * @author Thanos Theo, 2019
  * @version 14.0.0.0
  */
  template <typename T>
  class HostMemory final : private MemoryHandlersAbstraction<T, HostMemory<T>> // private inheritance used for prohibiting up-casting
  {
  public:

    using MemoryHandlersAbstraction<T, HostMemory<T>>::numberOfElements_;
    using MemoryHandlersAbstraction<T, HostMemory<T>>::memoryPoolMode_;

    void allocate(std::size_t numberOfElements, unsigned int flags = cudaHostRegisterDefault) noexcept
    {
      // calls to allocate() should not be made in memory pool mode
      assert(!memoryPoolMode_);
      if (memoryPoolMode_) return;

      hostPtr_          = make_unique_pinned<T>(numberOfElements, flags);
      numberOfElements_ = numberOfElements;
      assert(hostPtr_  != nullptr);
      assert(isValidHostDevicePointer(hostPtr_.get()));
    }

    std::future<void> allocateAsync(std::size_t numberOfElements, unsigned int flags = cudaHostRegisterDefault) noexcept
    {
      // copy-by-value below to avoid template inlining issues with temporary lifetime variables
      return Utils::CPUParallelism::CPUParallelismUtilityFunctions::reallyAsync([this, numberOfElements, flags] { allocate(numberOfElements, flags); });
    }

    void reset() noexcept
    {
      // calls to reset() should not be made in memory pool mode
      assert(!memoryPoolMode_);
      if (memoryPoolMode_) return;

      // release & delete the owned ptr memory
      hostPtr_.reset(nullptr);
      numberOfElements_ = 0;
      assert(hostPtr_ == nullptr);
      assert(!isValidHostDevicePointer(hostPtr_.get()));
    }

    std::future<void> resetAsync() noexcept
    {
      return Utils::CPUParallelism::CPUParallelismUtilityFunctions::reallyAsync([this] { reset(); });
    }

    void swap(HostMemory<T>& other) noexcept
    {
      hostPtr_.swap(other.hostPtr_);
      std::swap(memoryPoolHostIndex_, other.memoryPoolHostIndex_);
      std::swap(numberOfElements_,    other.numberOfElements_);
      std::swap(memoryPoolMode_,      other.memoryPoolMode_);
    }

    void memset(int value) noexcept
    {
      assert(hostPtr_ != nullptr);
      assert(isValidHostDevicePointer(hostPtr_.get()));
      std::memset(hostPtr_.get(), value, sizeof(T) * numberOfElements_);
    }

          T* get()          { assert(hostPtr_ != nullptr); return hostPtr_.get(); }
    const T* get()    const { assert(hostPtr_ != nullptr); return hostPtr_.get(); }
          T* host()         { assert(hostPtr_ != nullptr); return hostPtr_.get(); }
    const T* host()   const { assert(hostPtr_ != nullptr); return hostPtr_.get(); }
          T& operator[](std::size_t index)       { return host()[index]; }
    const T& operator[](std::size_t index) const { return host()[index]; }
    explicit operator bool()      const noexcept { return hostPtr_ && isValidHostDevicePointer(hostPtr_.get()); }
    std::size_t getNumberOfElements()      const { return numberOfElements_; }
    bool           isMemoryPoolMode()      const { return memoryPoolMode_;   }

    /// Explicit conversions to spans (for host)
    Span<      T> hostSpan()       { return Span<      T>(host(), numberOfElements_); }
    Span<const T> hostSpan() const { return Span<const T>(host(), numberOfElements_); }
    Span<      T> hostSpan(std::size_t count)       { assert(count <= numberOfElements_); return Span<      T>(host(), count); }
    Span<const T> hostSpan(std::size_t count) const { assert(count <= numberOfElements_); return Span<const T>(host(), count); }
    template<std::size_t SIZE>
    Span<      T, SIZE> hostSpan()       { assert(SIZE <= numberOfElements_); return Span<      T, SIZE>(host()); }
    template<std::size_t SIZE>
    Span<const T, SIZE> hostSpan() const { assert(SIZE <= numberOfElements_); return Span<const T, SIZE>(host()); }
    //@}

    HostMemory()  = default;
    explicit HostMemory(std::size_t numberOfElements, unsigned int flags = cudaHostRegisterDefault) { allocate(numberOfElements, flags); }
    ~HostMemory() = default; // no virtual destructor for data-oriented design (no up-casting should ever be used)
    HostMemory(const HostMemory&) = delete; // copy-constructor deleted
    HostMemory(HostMemory&&)      = delete; // move-constructor deleted
    HostMemory& operator=(const HostMemory&) = delete; //      assignment operator deleted
    HostMemory& operator=(HostMemory&&)      = delete; // move-assignment operator deleted

  private:

    PinnedUniquePtr<T> hostPtr_      = nullptr;
    std::size_t memoryPoolHostIndex_ = 0;

    friend class CUDAMemoryPool;
    friend class CUDAProcessMemoryPool;

    /// Note that the host ptr will NOT be automatically deleted (deleter set to false)
    void setHostPtr(std::uint8_t* hostPtr) noexcept
    {
      hostPtr_        = PinnedUniquePtr<T>(reinterpret_cast<T*>(hostPtr), PinnedDeleter<T>{false});
      memoryPoolMode_ = true;
    }
  };

  /** @brief This class encapsulates usage of a collection of CUDA memory handling techniques (device side only) & the RAII C++ idiom. Using the Curiously Recurring Template Pattern (CRTP).
  *
  * @author David Lenz, Thanos Theo, 2019
  * @version 14.0.0.0
  */
  template <typename T>
  class DeviceMemory final : private MemoryHandlersAbstraction<T, DeviceMemory<T>> // private inheritance used for composition and prohibiting up-casting
  {
  public:

    using MemoryHandlersAbstraction<T, DeviceMemory<T>>::numberOfElements_;
    using MemoryHandlersAbstraction<T, DeviceMemory<T>>::memoryPoolMode_;

    void allocate(std::size_t numberOfElements, int device = 0, bool useUnifiedMemory = false) noexcept
    {
      // calls to allocate() should not be made in memory pool mode
      assert(!memoryPoolMode_);
      if (memoryPoolMode_) return;

      devicePtr_        = make_unique_device<T>(numberOfElements, device, useUnifiedMemory);
      numberOfElements_ = numberOfElements;
      useUnifiedMemory_ = useUnifiedMemory;
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
    }

    std::future<void> allocateAsync(std::size_t numberOfElements, int device = 0, bool useUnifiedMemory = false) noexcept
    {
      // copy-by-value below to avoid template inlining issues with temporary lifetime variables
      return Utils::CPUParallelism::CPUParallelismUtilityFunctions::reallyAsync([this, numberOfElements, device, useUnifiedMemory] { allocate(numberOfElements, device, useUnifiedMemory); });
    }

    void reset() noexcept
    {
      // calls to reset() should not be made in memory pool mode
      assert(!memoryPoolMode_);
      if (memoryPoolMode_) return;

      // release & delete the owned ptr memory
      devicePtr_.reset(nullptr);
      numberOfElements_ = 0;
      useUnifiedMemory_ = false;
      assert(devicePtr_ == nullptr);
      assert(!isValidHostDevicePointer(devicePtr_.get()));
    }

    std::future<void> resetAsync() noexcept
    {
      return Utils::CPUParallelism::CPUParallelismUtilityFunctions::reallyAsync([this] { reset(); });
    }

    void swap(DeviceMemory<T>& other) noexcept
    {
      devicePtr_.swap(other.devicePtr_);
      std::swap(memoryPoolDeviceIndex_, other.memoryPoolDeviceIndex_);
      std::swap(useUnifiedMemory_,      other.useUnifiedMemory_);
      std::swap(numberOfElements_,      other.numberOfElements_);
      std::swap(memoryPoolMode_,        other.memoryPoolMode_);
    }

    //@{
    /// Function overloads use the class member numberOfElements and copy all
    void memset(int value) noexcept
    {
      memset(value, numberOfElements_);
    }

    void memsetAsync(int value, const cudaStream_t& stream) noexcept
    {
      memsetAsync(value, numberOfElements_, stream);
    }

    void copyHostToDevice(const void* hostPtr) noexcept
    {
      copyHostToDevice(hostPtr, numberOfElements_);
    }

    void copyHostToDeviceAsync(const void* hostPtr, const cudaStream_t& stream) noexcept
    {
      copyHostToDeviceAsync(hostPtr, numberOfElements_, stream);
    }

    void copyDeviceToDevice(const void* devicePtr) noexcept
    {
      copyDeviceToDevice(devicePtr, numberOfElements_);
    }

    void copyDeviceToDeviceAsync(const void* devicePtr, const cudaStream_t& stream) noexcept
    {
      copyDeviceToDeviceAsync(devicePtr, numberOfElements_, stream);
    }

    void copyDeviceFromDevice(void* devicePtr) const noexcept
    {
      copyDeviceFromDevice(devicePtr, numberOfElements_);
    }

    void copyDeviceFromDeviceAsync(void* devicePtr, const cudaStream_t& stream) const noexcept
    {
      copyDeviceFromDeviceAsync(devicePtr, numberOfElements_, stream);
    }

    void copyDeviceToHost(void* hostPtr) const noexcept
    {
      copyDeviceToHost(hostPtr, numberOfElements_);
    }

    /** @brief Note: asynchronous memcpy on the given stream, enforce further synchronization with device -> host copy for that stream.
    */
    void copyDeviceToHost(void* hostPtr, const cudaStream_t& stream) const noexcept
    {
      copyDeviceToHost(hostPtr, numberOfElements_, stream);
    }

    void copyDeviceToHostAsync(void* hostPtr, const cudaStream_t& stream) const noexcept
    {
      copyDeviceToHostAsync(hostPtr, numberOfElements_, stream);
    }

    void memPrefetch(int dstDevice) const noexcept
    {
      memPrefetch(numberOfElements_, dstDevice);
    }

    void memPrefetchAsync(int dstDevice, const cudaStream_t& stream) const noexcept
    {
      memPrefetchAsync(numberOfElements_, dstDevice, stream);
    }

    void memPrefetchWithAdvise(int dstDevice) const noexcept
    {
      memPrefetchWithAdvise(numberOfElements_, dstDevice);
    }

    void memPrefetchWithAdviseAsync(int dstDevice, const cudaStream_t& stream) const noexcept
    {
      memPrefetchWithAdviseAsync(numberOfElements_, dstDevice, stream);
    }
    //@}

    //@{
    /// Function overloads with a given numberOfElements as function parameter
    void memset(int value, std::size_t numberOfElements) noexcept
    {
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
      assert(numberOfElements <= numberOfElements_);
      // cudaMemset() is a kernel launch, the current device has to be set on the same one as the allocated memory
      assert(isValidDevicePointerWithCurrentDevice(devicePtr_.get()));
      CUDAError_checkCUDAError(cudaMemset(devicePtr_.get(), value, sizeof(T) * numberOfElements));
    }

    void memsetAsync(int value, std::size_t numberOfElements, const cudaStream_t& stream) noexcept
    {
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
      assert(numberOfElements <= numberOfElements_);
      // cudaMemset() is a kernel launch, the current device has to be set on the same one as the allocated memory
      assert(isValidDevicePointerWithCurrentDevice(devicePtr_.get()));
      CUDAError_checkCUDAError(cudaMemsetAsync(devicePtr_.get(), value, sizeof(T) * numberOfElements, stream));
    }

    void copyHostToDevice(const void* hostPtr, std::size_t numberOfElements) noexcept
    {
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
      assert(numberOfElements <= numberOfElements_);
      CUDAError_checkCUDAError(cudaMemcpy(devicePtr_.get(), hostPtr, sizeof(T) * numberOfElements, cudaMemcpyHostToDevice));
    }

    void copyHostToDeviceAsync(const void* hostPtr, std::size_t numberOfElements, const cudaStream_t& stream) noexcept
    {
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
      assert(numberOfElements <= numberOfElements_);
      CUDAError_checkCUDAError(cudaMemcpyAsync(devicePtr_.get(), hostPtr, sizeof(T) * numberOfElements, cudaMemcpyHostToDevice, stream));
    }

    void copyDeviceToDevice(const void* devicePtr, std::size_t numberOfElements) noexcept
    {
      assert(devicePtr  != nullptr);
      assert(isValidHostDevicePointer(devicePtr));
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
      assert(numberOfElements <= numberOfElements_);
      CUDAError_checkCUDAError(cudaMemcpy(devicePtr_.get(), devicePtr, sizeof(T) * numberOfElements, cudaMemcpyDeviceToDevice));
    }

    void copyDeviceToDeviceAsync(const void* devicePtr, std::size_t numberOfElements, const cudaStream_t& stream) noexcept
    {
      assert(devicePtr  != nullptr);
      assert(isValidHostDevicePointer(devicePtr));
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
      assert(numberOfElements <= numberOfElements_);
      CUDAError_checkCUDAError(cudaMemcpyAsync(devicePtr_.get(), devicePtr, sizeof(T) * numberOfElements, cudaMemcpyDeviceToDevice, stream));
    }

    void copyDeviceFromDevice(void* devicePtr, std::size_t numberOfElements) const noexcept
    {
      assert(devicePtr  != nullptr);
      assert(isValidHostDevicePointer(devicePtr));
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
      assert(numberOfElements <= numberOfElements_);
      CUDAError_checkCUDAError(cudaMemcpy(devicePtr, devicePtr_.get(), sizeof(T) * numberOfElements, cudaMemcpyDeviceToDevice));
    }

    void copyDeviceFromDeviceAsync(void* devicePtr, std::size_t numberOfElements, const cudaStream_t& stream) const noexcept
    {
      assert(devicePtr  != nullptr);
      assert(isValidHostDevicePointer(devicePtr));
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
      assert(numberOfElements <= numberOfElements_);
      CUDAError_checkCUDAError(cudaMemcpyAsync(devicePtr, devicePtr_.get(), sizeof(T) * numberOfElements, cudaMemcpyDeviceToDevice, stream));
    }

    void copyDeviceToHost(void* hostPtr, std::size_t numberOfElements) const noexcept
    {
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
      assert(numberOfElements <= numberOfElements_);
      CUDAError_checkCUDAError(cudaMemcpy(hostPtr, devicePtr_.get(), sizeof(T) * numberOfElements, cudaMemcpyDeviceToHost));
    }

    /** @brief Note: asynchronous memcpy on the given stream, enforce further synchronization with device -> host copy for that stream.
    */
    void copyDeviceToHost(void* hostPtr, std::size_t numberOfElements, const cudaStream_t& stream) const noexcept
    {
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
      assert(numberOfElements <= numberOfElements_);
      CUDAError_checkCUDAError(cudaMemcpyAsync(hostPtr, devicePtr_.get(), sizeof(T) * numberOfElements, cudaMemcpyDeviceToHost, stream));
      CUDAError_checkCUDAError(cudaStreamSynchronize(stream));
    }

    void copyDeviceToHostAsync(void* hostPtr, std::size_t numberOfElements, const cudaStream_t& stream) const noexcept
    {
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
      assert(numberOfElements <= numberOfElements_);
      CUDAError_checkCUDAError(cudaMemcpyAsync(hostPtr, devicePtr_.get(), sizeof(T) * numberOfElements, cudaMemcpyDeviceToHost, stream));
    }

    void memPrefetch(std::size_t numberOfElements, int dstDevice) const noexcept
    {
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
      assert(numberOfElements <= numberOfElements_);
      assert(useUnifiedMemory_);
      CUDAError_checkCUDAError(cudaMemPrefetchAsync(devicePtr_.get(), sizeof(T) * numberOfElements, dstDevice));
    }

    void memPrefetchAsync(std::size_t numberOfElements, int dstDevice, const cudaStream_t& stream) const noexcept
    {
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
      assert(numberOfElements <= numberOfElements_);
      assert(useUnifiedMemory_);
      CUDAError_checkCUDAError(cudaMemPrefetchAsync(devicePtr_.get(), sizeof(T) * numberOfElements, dstDevice, stream));
    }

    void memPrefetchWithAdvise(std::size_t numberOfElements, int dstDevice) const noexcept
    {
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
      assert(numberOfElements <= numberOfElements_);
      assert(useUnifiedMemory_);
      std::size_t numberOfBytes = sizeof(T) * numberOfElements;
      // optimize with pinning regions to CPU memory and establish a direct mapping from the GPU by using
      // a combination of 'cudaMemAdviseSetPreferredLocation' and 'cudaMemAdviseSetAccessedBy' usage hints
      CUDAError_checkCUDAError(cudaMemAdvise(devicePtr_.get(), numberOfBytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
      CUDAError_checkCUDAError(cudaMemAdvise(devicePtr_.get(), numberOfBytes, cudaMemAdviseSetAccessedBy, dstDevice));
      CUDAError_checkCUDAError(cudaMemPrefetchAsync(devicePtr_.get(), numberOfBytes, dstDevice));
    }

    void memPrefetchWithAdviseAsync(std::size_t numberOfElements, int dstDevice, const cudaStream_t& stream) const noexcept
    {
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
      assert(numberOfElements <= numberOfElements_);
      assert(useUnifiedMemory_);
      std::size_t numberOfBytes = sizeof(T) * numberOfElements;
      // optimize with pinning regions to CPU memory and establish a direct mapping from the GPU by using
      // a combination of 'cudaMemAdviseSetPreferredLocation' and 'cudaMemAdviseSetAccessedBy' usage hints
      CUDAError_checkCUDAError(cudaMemAdvise(devicePtr_.get(), numberOfBytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
      CUDAError_checkCUDAError(cudaMemAdvise(devicePtr_.get(), numberOfBytes, cudaMemAdviseSetAccessedBy, dstDevice));
      CUDAError_checkCUDAError(cudaMemPrefetchAsync(devicePtr_.get(), numberOfBytes, dstDevice, stream));
    }
    //@}

    //@{
    /// below are function overloads with a given number of bytes as function parameter
    void memsetAsBytes(int value, std::size_t numberOfBytes) noexcept
    {
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
      assert(numberOfBytes <= sizeof(T) * numberOfElements_);
      // cudaMemset() is a kernel launch, the current device has to be set on the same one as the allocated memory
      assert(isValidDevicePointerWithCurrentDevice(devicePtr_.get()));
      CUDAError_checkCUDAError(cudaMemset(devicePtr_.get(), value, numberOfBytes));
    }

    void memsetAsBytesAsync(int value, std::size_t numberOfBytes, const cudaStream_t& stream) noexcept
    {
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
      assert(numberOfBytes <= sizeof(T) * numberOfElements_);
      // cudaMemset() is a kernel launch, the current device has to be set on the same one as the allocated memory
      assert(isValidDevicePointerWithCurrentDevice(devicePtr_.get()));
      CUDAError_checkCUDAError(cudaMemsetAsync(devicePtr_.get(), value, numberOfBytes, stream));
    }

    void copyHostToDeviceAsBytes(const void* hostPtr, std::size_t numberOfBytes) noexcept
    {
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
      assert(numberOfBytes <= sizeof(T) * numberOfElements_);
      CUDAError_checkCUDAError(cudaMemcpy(devicePtr_.get(), hostPtr, numberOfBytes, cudaMemcpyHostToDevice));
    }

    void copyHostToDeviceAsBytesAsync(const void* hostPtr, std::size_t numberOfBytes, const cudaStream_t& stream) noexcept
    {
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
      assert(numberOfBytes <= sizeof(T) * numberOfElements_);
      CUDAError_checkCUDAError(cudaMemcpyAsync(devicePtr_.get(), hostPtr, numberOfBytes, cudaMemcpyHostToDevice, stream));
    }

    void copyDeviceToDeviceAsBytes(const void* devicePtr, std::size_t numberOfBytes) noexcept
    {
      assert(devicePtr  != nullptr);
      assert(isValidHostDevicePointer(devicePtr));
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
      assert(numberOfBytes <= sizeof(T) * numberOfElements_);
      CUDAError_checkCUDAError(cudaMemcpy(devicePtr_.get(), devicePtr, numberOfBytes, cudaMemcpyDeviceToDevice));
    }

    void copyDeviceToDeviceAsBytesAsync(const void* devicePtr, std::size_t numberOfBytes, const cudaStream_t& stream) noexcept
    {
      assert(devicePtr  != nullptr);
      assert(isValidHostDevicePointer(devicePtr));
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
      assert(numberOfBytes <= sizeof(T) * numberOfElements_);
      CUDAError_checkCUDAError(cudaMemcpyAsync(devicePtr_.get(), devicePtr, numberOfBytes, cudaMemcpyDeviceToDevice, stream));
    }

    void copyDeviceFromDeviceAsBytes(void* devicePtr, std::size_t numberOfBytes) const noexcept
    {
      assert(devicePtr  != nullptr);
      assert(isValidHostDevicePointer(devicePtr));
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
      assert(numberOfBytes <= sizeof(T) * numberOfElements_);
      CUDAError_checkCUDAError(cudaMemcpy(devicePtr, devicePtr_.get(), numberOfBytes, cudaMemcpyDeviceToDevice));
    }

    void copyDeviceFromDeviceAsBytesAsync(void* devicePtr, std::size_t numberOfBytes, const cudaStream_t& stream) const noexcept
    {
      assert(devicePtr  != nullptr);
      assert(isValidHostDevicePointer(devicePtr));
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
      assert(numberOfBytes <= sizeof(T) * numberOfElements_);
      CUDAError_checkCUDAError(cudaMemcpyAsync(devicePtr, devicePtr_.get(), numberOfBytes, cudaMemcpyDeviceToDevice, stream));
    }

    void copyDeviceToHostAsBytes(void* hostPtr, std::size_t numberOfBytes) const noexcept
    {
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
      assert(numberOfBytes <= sizeof(T) * numberOfElements_);
      CUDAError_checkCUDAError(cudaMemcpy(hostPtr, devicePtr_.get(), numberOfBytes, cudaMemcpyDeviceToHost));
    }

    /** @brief Note: asynchronous memcpy on the given stream, enforce further synchronization with device -> host copy for that stream.
    */
    void copyDeviceToHostAsBytes(void* hostPtr, std::size_t numberOfBytes, const cudaStream_t& stream) const noexcept
    {
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
      assert(numberOfBytes <= sizeof(T) * numberOfElements_);
      CUDAError_checkCUDAError(cudaMemcpyAsync(hostPtr, devicePtr_.get(), numberOfBytes, cudaMemcpyDeviceToHost, stream));
      CUDAError_checkCUDAError(cudaStreamSynchronize(stream));
    }

    void copyDeviceToHostAsBytesAsync(void* hostPtr, std::size_t numberOfBytes, const cudaStream_t& stream) const noexcept
    {
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
      assert(numberOfBytes <= sizeof(T) * numberOfElements_);
      CUDAError_checkCUDAError(cudaMemcpyAsync(hostPtr, devicePtr_.get(), numberOfBytes, cudaMemcpyDeviceToHost, stream));
    }

    void memPrefetchAsBytes(std::size_t numberOfBytes, int dstDevice) const noexcept
    {
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
      assert(numberOfBytes <= sizeof(T) * numberOfElements_);
      assert(useUnifiedMemory_);
      CUDAError_checkCUDAError(cudaMemPrefetchAsync(devicePtr_.get(), numberOfBytes, dstDevice));
    }

    void memPrefetchAsBytesAsync(std::size_t numberOfBytes, int dstDevice, const cudaStream_t& stream) const noexcept
    {
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
      assert(numberOfBytes <= sizeof(T) * numberOfElements_);
      assert(useUnifiedMemory_);
      CUDAError_checkCUDAError(cudaMemPrefetchAsync(devicePtr_.get(), numberOfBytes, dstDevice, stream));
    }

    void memPrefetchWithAdviseAsBytes(std::size_t numberOfBytes, int dstDevice) const noexcept
    {
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
      assert(numberOfBytes <= sizeof(T) * numberOfElements_);
      assert(useUnifiedMemory_);
      // optimize with pinning regions to CPU memory and establish a direct mapping from the GPU by using
      // a combination of 'cudaMemAdviseSetPreferredLocation' and 'cudaMemAdviseSetAccessedBy' usage hints
      CUDAError_checkCUDAError(cudaMemAdvise(devicePtr_.get(), numberOfBytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
      CUDAError_checkCUDAError(cudaMemAdvise(devicePtr_.get(), numberOfBytes, cudaMemAdviseSetAccessedBy, dstDevice));
      CUDAError_checkCUDAError(cudaMemPrefetchAsync(devicePtr_.get(), numberOfBytes, dstDevice));
    }

    void memPrefetchWithAdviseAsBytesAsync(std::size_t numberOfBytes, int dstDevice, const cudaStream_t& stream) const noexcept
    {
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
      assert(numberOfBytes <= sizeof(T) * numberOfElements_);
      assert(useUnifiedMemory_);
      // optimize with pinning regions to CPU memory and establish a direct mapping from the GPU by using
      // a combination of 'cudaMemAdviseSetPreferredLocation' and 'cudaMemAdviseSetAccessedBy' usage hints
      CUDAError_checkCUDAError(cudaMemAdvise(devicePtr_.get(), numberOfBytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
      CUDAError_checkCUDAError(cudaMemAdvise(devicePtr_.get(), numberOfBytes, cudaMemAdviseSetAccessedBy, dstDevice));
      CUDAError_checkCUDAError(cudaMemPrefetchAsync(devicePtr_.get(), numberOfBytes, dstDevice, stream));
    }
    //@}

          T* get()          { assert(devicePtr_ != nullptr); assert(useUnifiedMemory_); return devicePtr_.get(); }
    const T* get()    const { assert(devicePtr_ != nullptr); assert(useUnifiedMemory_); return devicePtr_.get(); }
          T* device()       { assert(devicePtr_ != nullptr); return devicePtr_.get(); }
    const T* device() const { assert(devicePtr_ != nullptr); return devicePtr_.get(); }
          T& operator[](std::size_t index)       { assert(useUnifiedMemory_); return device()[index]; } // for Unified Memory usage only (where a device ptr can be accessed on host as a normal ptr)
    const T& operator[](std::size_t index) const { assert(useUnifiedMemory_); return device()[index]; } // for Unified Memory usage only (where a device ptr can be accessed on host as a normal ptr)
    explicit operator bool()      const noexcept { return devicePtr_ && isValidHostDevicePointer(devicePtr_.get()); }
    std::size_t getNumberOfElements()      const { return numberOfElements_; }
    bool           isMemoryPoolMode()      const { return memoryPoolMode_;   }

    //@{
    /// Convenience functions to pass pointers to kernels
    operator RawDeviceMemory<      T>()       { return RawDeviceMemory<      T>(device()); }
    operator RawDeviceMemory<const T>() const { return RawDeviceMemory<const T>(device()); }
    //@}

    //@{
    /// Explicit conversions to spans
    Span<      T> deviceSpan()       { return Span<      T>(device(), numberOfElements_); }
    Span<const T> deviceSpan() const { return Span<const T>(device(), numberOfElements_); }
    Span<      T> deviceSpan(std::size_t count)       { assert(count <= numberOfElements_); return Span<      T>(device(), count); }
    Span<const T> deviceSpan(std::size_t count) const { assert(count <= numberOfElements_); return Span<const T>(device(), count); }
    template<std::size_t SIZE>
    Span<      T, SIZE> deviceSpan()       { assert(SIZE <= numberOfElements_); return Span<      T, SIZE>(device()); }
    template<std::size_t SIZE>
    Span<const T, SIZE> deviceSpan() const { assert(SIZE <= numberOfElements_); return Span<const T, SIZE>(device()); }
    //@}

    DeviceMemory()  = default;
    explicit DeviceMemory(std::size_t numberOfElements, int device = 0, bool useUnifiedMemory = false) { allocate(numberOfElements, device, useUnifiedMemory); }
    ~DeviceMemory() = default; // no virtual destructor for data-oriented design (no up-casting should ever be used)
    DeviceMemory(const DeviceMemory&) = delete; // copy-constructor deleted
    DeviceMemory(DeviceMemory&&)      = delete; // move-constructor deleted
    DeviceMemory& operator=(const DeviceMemory&) = delete; //      assignment operator deleted
    DeviceMemory& operator=(DeviceMemory&&)      = delete; // move-assignment operator deleted

private:

    DeviceUniquePtr<T> devicePtr_      = nullptr;
    std::size_t memoryPoolDeviceIndex_ = 0;
    bool useUnifiedMemory_             = false;

    friend class CUDAMemoryPool;
    friend class CUDAProcessMemoryPool;

    /// Note that the device ptr will NOT be automatically deleted (deleter set to false)
    void setDevicePtr(std::uint8_t* devicePtr, bool useUnifiedMemory = false) noexcept
    {
      devicePtr_        = DeviceUniquePtr<T>(reinterpret_cast<T*>(devicePtr), CUDADeleter<T>{false});
      useUnifiedMemory_ = useUnifiedMemory;
      memoryPoolMode_   = true;
    }
  };

  /** @brief This class encapsulates usage of a collection of host & CUDA memory handling techniques (host & device side) & the RAII C++ idiom. Using the Curiously Recurring Template Pattern (CRTP).
  *
  * @author David Lenz, Thanos Theo, 2019
  * @version 14.0.0.0
  */
  template <typename T>
  class HostDeviceMemory final : private MemoryHandlersAbstraction<T, HostDeviceMemory<T>> // private inheritance used for prohibiting up-casting
  {
  public:

    using MemoryHandlersAbstraction<T, HostDeviceMemory<T>>::numberOfElements_;
    using MemoryHandlersAbstraction<T, HostDeviceMemory<T>>::memoryPoolMode_;

    void allocate(std::size_t numberOfElements, int device = 0, unsigned int flags = cudaHostRegisterDefault) noexcept
    {
      // calls to allocate() should not be made in memory pool mode
      assert(!memoryPoolMode_);
      if (memoryPoolMode_) return;

      hostPtr_          = make_unique_pinned<T>(numberOfElements, flags );
      devicePtr_        = make_unique_device<T>(numberOfElements, device);
      numberOfElements_ = numberOfElements;
      assert(hostPtr_   != nullptr);
      assert(isValidHostDevicePointer(hostPtr_.get()));
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
    }

    std::future<void> allocateAsync(std::size_t numberOfElements, int device = 0, unsigned int flags = cudaHostRegisterDefault) noexcept
    {
      // copy-by-value below to avoid template inlining issues with temporary lifetime variables
      return Utils::CPUParallelism::CPUParallelismUtilityFunctions::reallyAsync([this, numberOfElements, device, flags] { allocate(numberOfElements, device, flags); });
    }

    void reset() noexcept
    {
      // calls to reset() should not be made in memory pool mode
      assert(!memoryPoolMode_);
      if (memoryPoolMode_) return;

      // release & delete the owned ptr memory
      hostPtr_.reset(nullptr);
      devicePtr_.reset(nullptr);
      numberOfElements_ = 0;
      assert(hostPtr_   == nullptr);
      assert(!isValidHostDevicePointer(hostPtr_.get()));
      assert(devicePtr_ == nullptr);
      assert(!isValidHostDevicePointer(devicePtr_.get()));
    }

    std::future<void> resetAsync() noexcept
    {
      return Utils::CPUParallelism::CPUParallelismUtilityFunctions::reallyAsync([this] { reset(); });
    }

    void swap(HostDeviceMemory<T>& other) noexcept
    {
      hostPtr_.swap(  other.hostPtr_);
      devicePtr_.swap(other.devicePtr_);
      std::swap(memoryPoolHostIndex_,   other.memoryPoolHostIndex_);
      std::swap(memoryPoolDeviceIndex_, other.memoryPoolDeviceIndex_);
      std::swap(numberOfElements_,      other.numberOfElements_);
      std::swap(memoryPoolMode_,        other.memoryPoolMode_);
    }

    void memset(int value, bool memsetHost = true) noexcept
    {
      assert(hostPtr_   != nullptr);
      assert(isValidHostDevicePointer(hostPtr_.get()));
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
      if (memsetHost)         std::memset(  hostPtr_.get(), value, sizeof(T) * numberOfElements_);
      // cudaMemset() is a kernel launch, the current device has to be set on the same one as the allocated memory
      assert(isValidDevicePointerWithCurrentDevice(devicePtr_.get()));
      CUDAError_checkCUDAError(cudaMemset(devicePtr_.get(), value, sizeof(T) * numberOfElements_));
    }

    void memsetAsync(int value, const cudaStream_t& stream, bool memsetHost = true) noexcept
    {
      assert(hostPtr_   != nullptr);
      assert(isValidHostDevicePointer(hostPtr_.get()));
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
      if (memsetHost)         std::memset(       hostPtr_.get(), value, sizeof(T) * numberOfElements_);
      // cudaMemset() is a kernel launch, the current device has to be set on the same one as the allocated memory
      assert(isValidDevicePointerWithCurrentDevice(devicePtr_.get()));
      CUDAError_checkCUDAError(cudaMemsetAsync(devicePtr_.get(), value, sizeof(T) * numberOfElements_, stream));
    }

    void copyHostToDevice() noexcept
    {
      assert(hostPtr_   != nullptr);
      assert(isValidHostDevicePointer(hostPtr_.get()));
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
      CUDAError_checkCUDAError(cudaMemcpy(devicePtr_.get(), hostPtr_.get(), sizeof(T) * numberOfElements_, cudaMemcpyHostToDevice));
    }

    void copyHostToDeviceAsync(const cudaStream_t& stream) noexcept
    {
      assert(hostPtr_   != nullptr);
      assert(isValidHostDevicePointer(hostPtr_.get()));
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
      CUDAError_checkCUDAError(cudaMemcpyAsync(devicePtr_.get(), hostPtr_.get(), sizeof(T) * numberOfElements_, cudaMemcpyHostToDevice, stream));
    }

    void copyDeviceToHost() const noexcept
    {
      assert(hostPtr_   != nullptr);
      assert(isValidHostDevicePointer(hostPtr_.get()));
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
      CUDAError_checkCUDAError(cudaMemcpy(hostPtr_.get(), devicePtr_.get(), sizeof(T) * numberOfElements_, cudaMemcpyDeviceToHost));
    }

    /** @brief Note: asynchronous memcpy on the given stream, enforce further synchronization with device -> host copy for that stream.
    */
    void copyDeviceToHost(const cudaStream_t& stream) const noexcept
    {
      assert(hostPtr_   != nullptr);
      assert(isValidHostDevicePointer(hostPtr_.get()));
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
      CUDAError_checkCUDAError(cudaMemcpyAsync(hostPtr_.get(), devicePtr_.get(), sizeof(T) * numberOfElements_, cudaMemcpyDeviceToHost, stream));
      CUDAError_checkCUDAError(cudaStreamSynchronize(stream));
    }

    void copyDeviceToHostAsync(const cudaStream_t& stream) const noexcept
    {
      assert(hostPtr_   != nullptr);
      assert(isValidHostDevicePointer(hostPtr_.get()));
      assert(devicePtr_ != nullptr);
      assert(isValidHostDevicePointer(devicePtr_.get()));
      CUDAError_checkCUDAError(cudaMemcpyAsync(hostPtr_.get(), devicePtr_.get(), sizeof(T) * numberOfElements_, cudaMemcpyDeviceToHost, stream));
    }

          T* get()          { assert(hostPtr_   != nullptr); return hostPtr_.get();   }
    const T* get()    const { assert(hostPtr_   != nullptr); return hostPtr_.get();   }
          T* host()         { assert(hostPtr_   != nullptr); return hostPtr_.get();   }
    const T* host()   const { assert(hostPtr_   != nullptr); return hostPtr_.get();   }
          T* device()       { assert(devicePtr_ != nullptr); return devicePtr_.get(); }
    const T* device() const { assert(devicePtr_ != nullptr); return devicePtr_.get(); }
          T& operator[](std::size_t index)       { return host()[index]; }
    const T& operator[](std::size_t index) const { return host()[index]; }
    explicit operator bool()      const noexcept { return hostPtr_ && isValidHostDevicePointer(hostPtr_.get()) && devicePtr_ && isValidHostDevicePointer(devicePtr_.get()); }
    std::size_t getNumberOfElements()      const { return numberOfElements_; }
    bool           isMemoryPoolMode()      const { return memoryPoolMode_;   }

    //@{
    /// Convenience functions to pass pointers to kernels
    operator RawDeviceMemory<      T>()       { return RawDeviceMemory<      T>(device()); }
    operator RawDeviceMemory<const T>() const { return RawDeviceMemory<const T>(device()); }
    //@}

    //@{
    /// Explicit conversions to spans (for host)
    Span<      T> hostSpan()       { return Span<      T>(host(), numberOfElements_); }
    Span<const T> hostSpan() const { return Span<const T>(host(), numberOfElements_); }
    Span<      T> hostSpan(std::size_t count)       { assert(count <= numberOfElements_); return Span<      T>(host(), count); }
    Span<const T> hostSpan(std::size_t count) const { assert(count <= numberOfElements_); return Span<const T>(host(), count); }
    template<std::size_t SIZE>
    Span<      T, SIZE> hostSpan()       { assert(SIZE <= numberOfElements_); return Span<      T, SIZE>(host()); }
    template<std::size_t SIZE>
    Span<const T, SIZE> hostSpan() const { assert(SIZE <= numberOfElements_); return Span<const T, SIZE>(host()); }

    /// Explicit conversions to spans (for device)
    Span<      T> deviceSpan()       { return Span<      T>(device(), numberOfElements_); }
    Span<const T> deviceSpan() const { return Span<const T>(device(), numberOfElements_); }
    Span<      T> deviceSpan(std::size_t count)       { assert(count <= numberOfElements_); return Span<      T>(device(), count); }
    Span<const T> deviceSpan(std::size_t count) const { assert(count <= numberOfElements_); return Span<const T>(device(), count); }
    template<std::size_t SIZE>
    Span<      T, SIZE> deviceSpan()       { assert(SIZE <= numberOfElements_); return Span<      T, SIZE>(device()); }
    template<std::size_t SIZE>
    Span<const T, SIZE> deviceSpan() const { assert(SIZE <= numberOfElements_); return Span<const T, SIZE>(device()); }
    //@}

    HostDeviceMemory()  = default;
    explicit HostDeviceMemory(std::size_t numberOfElements, int device = 0, unsigned int flags = cudaHostRegisterDefault) { allocate(numberOfElements, device, flags); }
    ~HostDeviceMemory() = default; // no virtual destructor for data-oriented design (no up-casting should ever be used)
    HostDeviceMemory(const HostDeviceMemory&) = delete; // copy-constructor deleted
    HostDeviceMemory(HostDeviceMemory&&)      = delete; // move-constructor deleted
    HostDeviceMemory& operator=(const HostDeviceMemory&) = delete; //      assignment operator deleted
    HostDeviceMemory& operator=(HostDeviceMemory&&)      = delete; // move-assignment operator deleted

private:

    PinnedUniquePtr<T> hostPtr_        = nullptr;
    DeviceUniquePtr<T> devicePtr_      = nullptr;
    std::size_t memoryPoolHostIndex_   = 0;
    std::size_t memoryPoolDeviceIndex_ = 0;

    friend class CUDAMemoryPool;
    friend class CUDAProcessMemoryPool;

    /// Note that the host ptr will NOT be automatically deleted (deleter set to false)
    void setHostPtr(std::uint8_t* hostPtr) noexcept
    {
      hostPtr_        = PinnedUniquePtr<T>(reinterpret_cast<T*>(hostPtr), PinnedDeleter<T>{false});
      memoryPoolMode_ = true;
    }

    /// Note that the device ptr will NOT be automatically deleted (deleter set to false)
    void setDevicePtr(std::uint8_t* devicePtr, bool = false) noexcept
    {
      devicePtr_      = DeviceUniquePtr<T>(reinterpret_cast<T*>(devicePtr), CUDADeleter<T>{false});
      memoryPoolMode_ = true;
    }
  };
} // namespace UtilsCUDA

#endif // __CUDAMemoryHandlers_h