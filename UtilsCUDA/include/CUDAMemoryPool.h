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

#ifndef __CUDAMemoryPool_h
#define __CUDAMemoryPool_h

#include "ModuleDLL.h"
#include "CUDADriverInfo.h"
#include "CUDAMemoryHandlers.h"
#include "CUDAMemoryWrappers.h"
#include "EnvironmentConfig.h"
#include <cstdint>
#include <functional>
#include <bitset>
#include <vector>
#include <memory>
#include <string>

/** @brief namespace UtilsCUDA for encapsulating all the CUDA related code compiled by the NVCC compiler.
*
* @author Thanos Theo, 2018
* @version 14.0.0.0
*/
namespace UtilsCUDA
{
  /** @brief This class encapsulates CUDA Memory Pool functionality for both host & device with reporting.
  *
  *  CUDAMemoryPool.h:
  *  ================
  *  This class encapsulates CUDA Memory Pool functionality for both host & device with reporting.
  *
  *  The work pattern to use the CUDA Memory Pool is as follows:
  *    1. Use the addToHostMemoryPool(), addToDeviceMemoryPool() & addToHostDeviceMemoryPool() to add host/device T* data respectively to the memory pool (or together).
  *    2. Use the allocateHostMemoryPool(), allocateDeviceMemoryPool() & allocateHostDeviceMemoryPool() for host/device batch allocations.
  *    3. Use the freeHostMemoryPool(), freeDeviceMemoryPool() & freeHostDeviceMemoryPool() to delete all host/device data explicitly from the memory pool.
  *
  *  Note: 1. The destructor will also freeHostMemoryPool()/freeDeviceMemoryPool()/freeHostDeviceMemoryPool() in RAII fashion.
  *        2. The allocateDeviceMemoryPool() supports Unified Memory allocation per device (default is off).
  *        3. After using the allocateHostMemoryPool()/allocateDeviceMemoryPool()/allocateHostDeviceMemoryPool(),
  *           the addToHostMemoryPool()/addToDeviceMemoryPool()/addToHostDeviceMemoryPool() call is invalid (nothing added, false returned).
  *
  * @author Thanos Theo, 2019
  * @version 14.0.0.0
  */
  class UTILS_CUDA_MODULE_API CUDAMemoryPool final
  {
  public:

    static constexpr std::size_t MAX_DEVICES = 64;

    /// enum for Memory Pool Types
    enum class MemoryPoolTypes : std::size_t { HOST_MEMORY = 0, DEVICE_MEMORY = 1 };

    /// struct for Memory Pool Data
    struct MemoryPoolData final
    {
      std::uint8_t* ptr_            = nullptr;
      std::size_t numberOfElements_ = 0;
      std::size_t sizeOfElement_    = 0;
      std::size_t offset_           = 0;
      int device_                   = 0;
      const std::function<void(std::uint8_t* ptr, bool)> memoryHandlerSetFunction{nullptr};

      MemoryPoolData(std::uint8_t* ptr, std::size_t numberOfElements, std::size_t sizeOfElement, std::size_t offset, int device,
                     const std::function<void(std::uint8_t* ptr, bool)>& memoryHandlerSetFunction) noexcept
        : ptr_(ptr), numberOfElements_(numberOfElements), sizeOfElement_(sizeOfElement), offset_(offset), device_(device)
        , memoryHandlerSetFunction(memoryHandlerSetFunction) {}
      ~MemoryPoolData() = default;
      MemoryPoolData(const MemoryPoolData&) = delete;
      MemoryPoolData(MemoryPoolData&&)      = default;
      MemoryPoolData& operator=(const MemoryPoolData&) = delete;
      MemoryPoolData& operator=(MemoryPoolData&&)      = delete;
    };

    /// Adds to the Host Memory Pool (wrapping a non-template function) via a HostMemory handle
    template<typename T>
    bool addToHostMemoryPool(HostMemory<T>& hostHandler, std::size_t numberOfElements)
    {
      // the hostHandler should not have been pre-allocated before reserving from the pool
      assert(!hostHandler);
      if (hostHandler) return false;

      const auto hostHandlerPtr = &hostHandler;
      const bool flag = addMemoryPoolData(numberOfElements, sizeof(T), 0, MemoryPoolTypes::HOST_MEMORY, [hostHandlerPtr](std::uint8_t* ptr, bool = false)
                        {
                          assert(ptr != nullptr);
                          assert(isValidHostDevicePointer(ptr));
                          hostHandlerPtr->setHostPtr(ptr);
                        });
      if (flag)
      {
        // only do the updates with a successful addition to the Host Memory Pool
        hostHandler.numberOfElements_    = numberOfElements;
        hostHandler.memoryPoolHostIndex_ = hostMemoryPool_.size() - 1;
      }
      return flag;
    }
    /// Adds to the Device Memory Pool (wrapping a non-template function) via a memory handle
    template<typename T>
    bool addToDeviceMemoryPool(DeviceMemory<T>& deviceHandler, std::size_t numberOfElements, int device = 0)
    {
      // the deviceHandler should not have been pre-allocated before reserving from the pool
      assert(!deviceHandler);
      if (deviceHandler) return false;

      const auto deviceHandlerPtr = &deviceHandler;
      const bool flag = addMemoryPoolData(numberOfElements, sizeof(T), device, MemoryPoolTypes::DEVICE_MEMORY, [deviceHandlerPtr](std::uint8_t* ptr, bool unifiedMemoryFlag)
                        {
                          assert(ptr != nullptr);
                          assert(isValidHostDevicePointer(ptr));
                          deviceHandlerPtr->setDevicePtr(ptr, unifiedMemoryFlag);
                        });
      if (flag)
      {
        // only do the updates with a successful addition to the Device Memory Pool
        deviceHandler.numberOfElements_      = numberOfElements;
        deviceHandler.memoryPoolDeviceIndex_ = deviceMemoryPool_.size() - 1;
      }
      return flag;
    }
    /// Adds to the Host & Device Memory Pools (wrapping a non-template function) via a memory handle
    template<typename T>
    bool addToHostDeviceMemoryPool(HostDeviceMemory<T>& hostDeviceHandler, std::size_t numberOfElements, int device = 0)
    {
      // the hostDeviceHandler should not have been pre-allocated before reserving memory from the pool
      assert(!hostDeviceHandler);
      if (hostDeviceHandler) return false;

      const auto hostDeviceHandlerPtr = &hostDeviceHandler;
      const bool flag = addMemoryPoolData(numberOfElements, sizeof(T), 0,      MemoryPoolTypes::HOST_MEMORY,   [hostDeviceHandlerPtr](std::uint8_t* ptr, bool = false)
                        {
                          assert(ptr != nullptr);
                          assert(isValidHostDevicePointer(ptr));
                          hostDeviceHandlerPtr->setHostPtr(ptr);
                        }) &&
                        addMemoryPoolData(numberOfElements, sizeof(T), device, MemoryPoolTypes::DEVICE_MEMORY, [hostDeviceHandlerPtr](std::uint8_t* ptr, bool = false)
                        {
                          assert(ptr != nullptr);
                          assert(isValidHostDevicePointer(ptr));
                          hostDeviceHandlerPtr->setDevicePtr(ptr, false);
                        });
      if (flag)
      {
        // only do the updates with a successful addition to the Host & Device Memory Pools
        hostDeviceHandler.numberOfElements_      = numberOfElements;
        hostDeviceHandler.memoryPoolHostIndex_   = hostMemoryPool_.size()   - 1;
        hostDeviceHandler.memoryPoolDeviceIndex_ = deviceMemoryPool_.size() - 1;
      }
      return flag;
    }
    /// Registers host memory in the Host Memory Pool
    void allocateHostMemoryPool(const std::string& name = std::string(), unsigned int flags = cudaHostRegisterDefault);
    /// Allocates GPU-side memory in the Device Memory Pool
    void allocateDeviceMemoryPool(const std::string& name = std::string(), const std::bitset<MAX_DEVICES>& unifiedMemoryFlags = std::bitset<MAX_DEVICES>());
    /// Allocates CPU-side & GPU-side memory in the Host/Device Memory Pool
    void allocateHostDeviceMemoryPool(const std::string& name = std::string(), const std::bitset<MAX_DEVICES>& unifiedMemoryFlags = std::bitset<MAX_DEVICES>(), unsigned int flags = cudaHostRegisterDefault);
    ///  Frees (de-allocates) CPU-side memory & clears all the Host Memory Pool internal data structures
    void freeHostMemoryPool();
    /// Frees (de-allocates) GPU-side memory & clears all the Device Memory Pool internal data structures
    void freeDeviceMemoryPool();
    /// Frees (de-allocates) CPU-side & GPU-side memory & clears all the Host/Device Memory Pool internal data structures
    void freeHostDeviceMemoryPool();
    /// Gets the Host Memory Pool size
    std::size_t getHostMemoryPoolSize() const;
    /// Gets the Device Memory Pool size
    std::size_t getDeviceMemoryPoolSize(int device = 0) const;
    /// Gets the Host Memory Pool total bytes
    std::size_t getHostMemoryPoolTotalBytes() const;
    /// Gets the Device Memory Pool total bytes
    std::size_t getDeviceMemoryPoolTotalBytes(int device = 0) const;

    CUDAMemoryPool(const CUDADriverInfo& cudaDriverInfo, bool useSeparateAllocations = bool(GPU_FRAMEWORK_CUDA_MEMORY_POOL_USE_SEPARATE_ALLOCATIONS)) noexcept;
    ~CUDAMemoryPool() noexcept; // no virtual destructor for data-oriented design (no up-casting should ever be used)
    CUDAMemoryPool(const CUDAMemoryPool&) = delete;
    CUDAMemoryPool(CUDAMemoryPool&&)      = delete;
    CUDAMemoryPool& operator=(const CUDAMemoryPool&) = delete;
    CUDAMemoryPool& operator=(CUDAMemoryPool&&)      = delete;

  private:

    /// Use separate memory allocations (Note: to be enabled for debugging purposes only)
    bool useSeparateAllocations_ = false;
    /// The Host Memory Pool host allocated check
    bool isHostAllocated_ = false;
    /// The Host Memory Pool pointer
    std::uint8_t* hostMemoryPoolPtr_ = nullptr;
    /// The total Host Memory Pool bytes consumed
    std::size_t hostBytesToAllocate_ = 0;
    /// The Host Memory Pool is stored in a vector
    std::vector<MemoryPoolData> hostMemoryPool_;
    /// The Device Memory Pool device allocated check
    bool isDeviceAllocated_ = false;
    /// The number of available devices (default is 1)
    std::size_t deviceCount_ = 1;
    /// The texture alignment per device (1 per available GPU device)
    std::unique_ptr<std::size_t[]> textureAlignmentPerDevice_ = nullptr;
    /// The Device Memory Pool pointers (1 per available GPU device)
    std::unique_ptr<std::uint8_t*[]> deviceMemoryPoolPtrPerDevice_ = nullptr;
    /// The total Device Memory Pool bytes consumed (1 per available GPU device)
    std::unique_ptr<std::size_t[]> deviceBytesToAllocatePerDevice_ = nullptr;
    /// The Device Memory Pool is stored in a vector
    std::vector<MemoryPoolData> deviceMemoryPool_;

    /// Adds state to the CUDA Memory Pool
    bool addMemoryPoolData(std::size_t numberOfElements, std::size_t sizeOfElement, int device, const MemoryPoolTypes& type,
                           const std::function<void(std::uint8_t* ptr, bool)>& memoryHandlerSetFunction);
  };
} // namespace UtilsCUDA

#endif // __CUDAMemoryPool_h