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

#ifndef __CUDAMemoryWrappers_h
#define __CUDAMemoryWrappers_h

#include "CUDAUtilityFunctions.h"
#include <cuda_runtime_api.h>
#include <type_traits>
#include <cassert> // for host and device side asserts()
#include <cstddef>
#include <limits>
#include <array>

/** @brief namespace UtilsCUDA for encapsulating all the CUDA related code compiled by the NVCC compiler.
*
* @author Thanos Theo, 2018
* @version 14.0.0.0
*/
namespace UtilsCUDA
{
  /** @brief Check the validity of the given host/device pointer.
  *
  * @author Thanos Theo, 2019
  * @version 14.0.0.0
  */
  template <typename T>
  inline bool isValidHostDevicePointer(const T* ptr)
  {
    cudaPointerAttributes attributes{};
    return (ptr != nullptr && cudaPointerGetAttributes(&attributes, ptr) == cudaSuccess);
  }

  /** @brief Check the validity of the given device pointer which the memory was allocated or registered versus the current device for the calling host thread.
  *
  * @author Thanos Theo, 2019
  * @version 14.0.0.0
  */
  template <typename T>
  inline bool isValidDevicePointerWithCurrentDevice(const T* ptr)
  {
    int device = 0;
    CUDAError_checkCUDAError(cudaGetDevice(&device));
    cudaPointerAttributes attributes{};
    CUDAError_checkCUDAError(cudaPointerGetAttributes(&attributes, ptr));
    return (device == attributes.device);
  }

  /**
  * @brief Helper class for the Span class below.
  *
  * Helper class to ensure that a pointer is referencing a valid device memory area.
  * Its main purpose is obtaining type-safety at compile time when mixing host and device memory:
  *      - it prevents users from accidentally accessing the memory on the host side
  *      - allows function arguments (or struct members etc) to state the intent whether it expects device memory or host memory
  *
  * @tparam T underlying type of the array
  *
  * @author David Lenz, 2019
  * @version 14.0.0.0
  */
  template <typename T>
  class RawDeviceMemory final
  {
  public:

    using NonConstType = typename std::remove_const<T>::type;
    using    ConstType = const NonConstType;

    __forceinline__ __host__ __device__ operator T*() const { return ptr_; }
    __forceinline__ __host__ __device__ T* operator->()                  const { assert(ptr_ != nullptr); return ptr_; }
    __forceinline__ __host__ __device__ T& operator[](std::size_t index) const { assert(ptr_ != nullptr); return ptr_[index]; }

    /** @brief Get the raw pointer from the wrapper without any checks.
      */
    __forceinline__ __host__ __device__ T* data() const { return ptr_; }

    // main constructor
    __forceinline__ __host__ __device__ explicit RawDeviceMemory(T* ptr) : ptr_(ptr)
    {
    #ifndef __CUDA_ARCH__
      // host code path only
      assert(isValidHostDevicePointer(ptr));
    #endif // __CUDA_ARCH__
    }

    // constructor that allows instantiating a const version from a non-const version
    __forceinline__ __host__ __device__ RawDeviceMemory(const RawDeviceMemory<NonConstType>& other) noexcept : ptr_(other.data()) {}
    // if T is already const, this is our copy constructor, if T is non-const then this will still not compile
    __forceinline__ __host__ __device__ RawDeviceMemory(const RawDeviceMemory<   ConstType>& other) noexcept : ptr_(other.data()) {}

    // empty default constructor
    RawDeviceMemory()  = default;
    ~RawDeviceMemory() = default;  // no virtual destructor for data-oriented design (no up-casting should ever be used)
    RawDeviceMemory(RawDeviceMemory&&) = default;  // move-constructor defaulted

    RawDeviceMemory& operator=(const RawDeviceMemory&) = default;  //      assignment operator defaulted
    RawDeviceMemory& operator=(RawDeviceMemory&&) = default;       // move-assignment operator defaulted

  private:

    T* ptr_ = nullptr;
  };

  // special keyword for dynamic sizes
  constexpr std::size_t DYNAMIC_SIZE = std::numeric_limits<std::size_t>::max();

  /**
  * @brief Helper class to differentiate whether something is fixed size at compile time or of dynamic size.
  * Used for zero-cost abstraction of Span.
  * In case of a fixed size, the size does not need to be stored
  *
  * @tparam SIZE indicates whether the size of the underlying array could be dynamic (not known at compile time) or static
  *
  * @author David Lenz, 2019
  * @version 14.0.0.0
  */
  template <std::size_t SIZE = DYNAMIC_SIZE>
  class DynamicOrCompileTimeSize
  {
  public:

    DynamicOrCompileTimeSize() = default;
    __forceinline__ __host__ __device__ DynamicOrCompileTimeSize(std::size_t size)
    {
      (void)size;  // to ensure that we get no warning even if the assert is compiled out
      assert(size == SIZE);
    }

    // fixed size cannot be changed
    void resize(std::size_t newSize)
    {
      (void)newSize; // to ensure that we get no warning even if the assert is compiled out
      static_assert(SIZE != DYNAMIC_SIZE, "Resizing is not possible for a fixed-size array/span");
    }
    __forceinline__ __host__ __device__ std::size_t size() const { return SIZE; }
  };

  /**@brief Template specialization for dynamically sized objects. It needs to store the size in a size_t.
  *
  * @author David Lenz, 2019
  * @version 14.0.0.0
  */
  template <>
  class DynamicOrCompileTimeSize<DYNAMIC_SIZE>
  {
  public:

    __forceinline__ __host__ __device__ DynamicOrCompileTimeSize(std::size_t size) : size_(size) {}
    __forceinline__ __host__ __device__ std::size_t size() const { return size_; }

    /**
    * Allows to change the stored size
    */
    void resize(std::size_t newSize) { size_ = newSize; }

  private:

    std::size_t size_ = 0;
  };

  /**
  * @brief Helper class to store a pointer and an associated size with it.
  *
  * In case of compile-time sizes, the size of this helper class is only the size of the pointer.
  * storing of the size does not need any data @see DynamicOrCompileTimeSize
  *
  * @tparam T datatype of the pointer thats supposed to be stored
  * @tparam NUM_ELEMENTS compile time size or DYNAMIC_SIZE for runtime sized arrays
  *
  * @author David Lenz, 2019
  * @version 14.0.0.0
  */
  template <typename T, std::size_t NUM_ELEMENTS = DYNAMIC_SIZE>
  class SpanStorage final : public DynamicOrCompileTimeSize<NUM_ELEMENTS>
  {
  public:

    __forceinline__ __host__ __device__ SpanStorage(T* data, std::size_t size) : DynamicOrCompileTimeSize<NUM_ELEMENTS>(size), rawPtr_(data) {}
    __forceinline__ __host__ __device__ T* data() const { return rawPtr_; }

  private:

    T* rawPtr_ = nullptr;
  };

  /**
  * @brief Helper class implementing the "span" paradigm from the CppCoreGuidelines applied to device arrays.
  *
  * This class provides a view on an array hosted in device memory.
  * It serves the following purposes:
  *      - simplifies repeated function calls with void func(T*, size_t count) to void func(Span span)
  *      - allows automatic error and bounds checking for debug or other instrumented builds
  *
  * @note As it is to be used for device memory, all the stl iterator "Container" requirements are not fulfilled for now
  *
  * @tparam T underlying type of the array
  * @tparam SIZE the size of the underlying memory (in number of elements) if known at compile time or DYNAMIC_SIZE to
  *              allow sizes at runtime
  *
  * @author David Lenz, 2019
  * @version 14.0.0.0
  */
  template <typename T, std::size_t SIZE = DYNAMIC_SIZE>
  class Span final
  {
  public:

    using NonConstType = typename std::remove_const<T>::type;
    using ConstType = const NonConstType;

    __forceinline__ __host__ __device__ bool empty()       const noexcept { return size() == 0; }
    __forceinline__ __host__ __device__ std::size_t size() const noexcept { return storage_.size(); }
    __forceinline__ __host__ __device__ T* data()          const noexcept { return storage_.data(); }
    __forceinline__ __host__ __device__ operator T*()      const          { return data(); }

    //@{
    /// Enable ranged base for loops
    __forceinline__ __host__ __device__ T* begin() const { return data(); }
    __forceinline__ __host__ __device__ T* end()   const { return data() + size(); }
    //@}

    //@{
    /// Accessor functions for single elements
    __forceinline__ __host__ __device__ T& operator[](std::size_t index) const
    {
      assert(data() != nullptr);
      assert(index < this->size());

      return data()[index];
    }
    __forceinline__ __host__ __device__ T& at(std::size_t index)         const { return operator[](index); }
    __forceinline__ __host__ __device__ T& operator()(std::size_t index) const { return operator[](index); }
    //@}

    /**
    * Create a subspan, i.e. a view on a subset of the memory pointed by the current span
    * The returned span will point to the [ptr + startIndex -> ptr + startIndex + size]
    * The function assumes, that this memory region is overlapping the valid memory region for this span
    */
    __forceinline__ __host__ __device__ Span<T> subSpan(std::size_t startIndex, std::size_t size) const
    {
      assert(data() != nullptr);
      assert(startIndex + size <= this->size());

      return Span<T>(storage_.data() + startIndex, size);
    }

    /**
    * Same functionality @see subSpan but returning a fixed size span
    * @tparam SUBSPAN_SIZE
    * @param startIndex
    * @return
    */
    template <std::size_t SUBSPAN_SIZE>
    __forceinline__ __host__ __device__ Span<T, SUBSPAN_SIZE> subSpan(std::size_t startIndex) const
    {
      return Span<T, SUBSPAN_SIZE>(storage_.data() + startIndex);
    }

    /**
    * Resize the span
    * @note this will not alter the underlying memory, but only alter the view on that memory
    * @note only do this, if you know that the underlying memory has at least this size.
    */
    void resize(std::size_t newSize) { storage_.resize(newSize); }

    // device code can also generate the Span<T> type implicitly from a raw pointer
    __forceinline__ __host__ __device__ Span(T* ptr, std::size_t count) : storage_(ptr, count) {}
    __forceinline__ __host__ __device__ Span(T* ptr) : storage_(ptr, SIZE)
    {
      static_assert(SIZE != DYNAMIC_SIZE, "Calling the constructor without specifying the number of elements only possible for fixed size spans");
    }

    ~Span() = default;  // no virtual destructor for data-oriented design (no up-casting should ever be used)

        // constructor that allows instantiating a const version from a non-const version
    __forceinline__ __host__ __device__ Span(const Span<NonConstType, SIZE>& other) noexcept : storage_(other.data(), other.size()) {}

    // constructor that allows instantiating a const version from a const version
    __forceinline__ __host__ __device__ Span(const Span<ConstType, SIZE>& other) noexcept : storage_(other.data(), other.size()) {}

    // constructor that allows instantiating a dynamic sized span from a fixed size span or vice versa
    template <std::size_t OTHER_SIZE>
    __forceinline__ __host__ __device__ Span(const Span<NonConstType, OTHER_SIZE>& other) noexcept : storage_(other.data(), other.size()) {}
    // constructor that allows instantiating a dynamic sized span from a fixed size span or vice versa
    template <std::size_t OTHER_SIZE>
    __forceinline__ __host__ __device__ Span(const Span<ConstType, OTHER_SIZE>& other) noexcept : storage_(other.data(), other.size()) {}

    // allow implicit creation of std::arrays to spans
    template <std::size_t OTHER_SIZE>
    __host__ Span(const std::array<NonConstType, OTHER_SIZE>& other) noexcept : storage_(other.data(), other.size()) {}
    template <std::size_t OTHER_SIZE>
    __host__ Span(const std::array<ConstType,    OTHER_SIZE>& other) noexcept : storage_(other.data(), other.size()) {}

    Span(Span&&) = default; // move-constructor defaulted
    Span& operator=(const Span&) = default;  //      assignment operator defaulted
    Span& operator=(Span&&) = default;       // move-assignment operator defaulted

    __forceinline__ __host__ __device__ Span() : storage_(nullptr, SIZE == DYNAMIC_SIZE ? 0 : SIZE) {}

  private:

    SpanStorage<T, SIZE> storage_{};
  };
} // namespace UtilsCUDA

#endif // __CUDAMemoryWrappers_h