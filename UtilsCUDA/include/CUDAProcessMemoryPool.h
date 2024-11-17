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

#ifndef __CUDAProcessMemoryPool_h
#define __CUDAProcessMemoryPool_h

#include "ModuleDLL.h"
#include "CUDADriverInfo.h"
#include "CUDAMemoryHandlers.h"
#include "CUDAMemoryWrappers.h"
#include "EnvironmentConfig.h"
#include <driver_types.h>
#include <cstdint>
#include <array>
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
  /** @brief This class encapsulates CUDA Process Memory Pool functionality for both host & device with reporting.
  *
  *  CUDAProcessMemoryPool.h:
  *  =======================
  *  This class encapsulates CUDA Process Memory Pool functionality for both host & device with reporting.
  *
  *  The work pattern to use the CUDA Process Memory Pool is as follows:
  *    1. Use the allocateHostMemoryPool(), allocateDeviceMemoryPool() & allocateHostDeviceMemoryPool() for host/device batch allocations.
  *    2. Use the reserve() calls to reserve host/device T* data from the memory pool.
  *    3. Use the freeHostMemoryPool(), freeDeviceMemoryPool() & freeHostDeviceMemoryPool() to delete all host/device data explicitly from the memory pool.
  *
  *  Note: 1. The destructor will also freeHostMemoryPool()/freeDeviceMemoryPool()/freeHostDeviceMemoryPool() in RAII fashion.
  *        2. The allocateDeviceMemoryPool() supports Unified Memory allocation per device (default is off).
  *        3. Before using the allocateHostMemoryPool()/allocateDeviceMemoryPool()/allocateHostDeviceMemoryPool(), all the reserve() calls are invalid (nothing added, false returned).
  *
  * @author Thanos Theo, 2019
  * @version 14.0.0.0
  */
  class UTILS_CUDA_MODULE_API CUDAProcessMemoryPool final
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

      MemoryPoolData(std::uint8_t* ptr, std::size_t numberOfElements, std::size_t sizeOfElement, std::size_t offset, int device) noexcept
        : ptr_(ptr), numberOfElements_(numberOfElements), sizeOfElement_(sizeOfElement), offset_(offset), device_(device) {}
      ~MemoryPoolData() = default;
      MemoryPoolData(const MemoryPoolData&) = delete;
      MemoryPoolData(MemoryPoolData&&)      = default;
      MemoryPoolData& operator=(const MemoryPoolData&) = delete;
      MemoryPoolData& operator=(MemoryPoolData&&)      = delete;
    };

    /// Registers host memory in the Host Memory Pool
    void allocateHostMemoryPool(std::size_t hostBytesToAllocate = 0, unsigned int flags = cudaHostRegisterDefault);
    /// Allocates GPU-side memory in the Device Memory Pool
    void allocateDeviceMemoryPool(const std::array<std::size_t, MAX_DEVICES>& deviceBytesToAllocatePerDevice = std::array<std::size_t, MAX_DEVICES>(),
                                  const std::bitset<MAX_DEVICES>& unifiedMemoryFlags = std::bitset<MAX_DEVICES>());
    /// Allocates CPU-side & GPU-side memory in the Host/Device Memory Pool
    void allocateHostDeviceMemoryPool(std::size_t hostBytesToAllocate = 0, const std::array<std::size_t, MAX_DEVICES>& deviceBytesToAllocatePerDevice = std::array<std::size_t, MAX_DEVICES>(),
                                      const std::bitset<MAX_DEVICES>& unifiedMemoryFlags = std::bitset<MAX_DEVICES>(), unsigned int flags = cudaHostRegisterDefault);
    /// Reserves in the Host Memory Pool (wrapping a non-template function) via a HostMemory handle
    template<typename T>
    bool reserve(HostMemory<T>& hostHandler, std::size_t numberOfElements)
    {
      // the hostHandler should not have been pre-allocated before reserving from the pool
      assert(!hostHandler);
      if (hostHandler) return false;

      const bool flag = reserveMemoryPoolData(numberOfElements, sizeof(T), 0, MemoryPoolTypes::HOST_MEMORY);
      if (flag)
      {
        // only do the updates with a successful addition to the Host Memory Pool
        hostHandler.numberOfElements_    = numberOfElements;
        hostHandler.memoryPoolHostIndex_ = hostMemoryPool_.size() - 1;
        uint8_t* hostPtr = hostMemoryPool_[hostHandler.memoryPoolHostIndex_].ptr_;
        assert(hostPtr != nullptr);
        assert(isValidHostDevicePointer(hostPtr));
        hostHandler.setHostPtr(hostPtr);
      }
      return flag;
    }
    /// Reserves in the Device Memory Pool (wrapping a non-template function) via a DeviceMemory handle
    template<typename T>
    bool reserve(DeviceMemory<T>& deviceHandler, std::size_t numberOfElements, int device = 0)
    {
      // the deviceHandler should not have been pre-allocated before reserving from the pool
      assert(!deviceHandler);
      if (deviceHandler) return false;

      const bool flag = reserveMemoryPoolData(numberOfElements, sizeof(T), device, MemoryPoolTypes::DEVICE_MEMORY);
      if (flag)
      {
        // only do the updates with a successful addition to the Device Memory Pool
        deviceHandler.numberOfElements_      = numberOfElements;
        deviceHandler.memoryPoolDeviceIndex_ = deviceMemoryPool_.size() - 1;
        uint8_t* devicePtr = deviceMemoryPool_[deviceHandler.memoryPoolDeviceIndex_].ptr_;
        assert(devicePtr != nullptr);
        assert(isValidHostDevicePointer(devicePtr));
        deviceHandler.setDevicePtr(devicePtr, unifiedMemoryFlags_[device]);
      }
      return flag;
    }
    /// Reserves in to the Host & Device Memory Pool (wrapping a non-template function) via a HostDeviceMemory handle
    template<typename T>
    bool reserve(HostDeviceMemory<T>& hostDeviceHandler, std::size_t numberOfElements, int device = 0)
    {
      // the hostDeviceHandler should not have been pre-allocated before reserving memory from the pool
      assert(!hostDeviceHandler);
      if (hostDeviceHandler) return false;

      const bool flag = reserveMemoryPoolData(numberOfElements, sizeof(T), 0,      MemoryPoolTypes::HOST_MEMORY) &&
                        reserveMemoryPoolData(numberOfElements, sizeof(T), device, MemoryPoolTypes::DEVICE_MEMORY);
      if (flag)
      {
        // only do the updates with a successful addition to the Host & Device Memory Pools
        hostDeviceHandler.numberOfElements_      = numberOfElements;
        hostDeviceHandler.memoryPoolHostIndex_   = hostMemoryPool_.size()   - 1;
        hostDeviceHandler.memoryPoolDeviceIndex_ = deviceMemoryPool_.size() - 1;
        uint8_t* hostPtr = hostMemoryPool_[hostDeviceHandler.memoryPoolHostIndex_].ptr_;
        assert(hostPtr != nullptr);
        assert(isValidHostDevicePointer(hostPtr));
        hostDeviceHandler.setHostPtr(hostPtr);
        uint8_t* devicePtr = deviceMemoryPool_[hostDeviceHandler.memoryPoolDeviceIndex_].ptr_;
        assert(devicePtr != nullptr);
        assert(isValidHostDevicePointer(devicePtr));
        hostDeviceHandler.setDevicePtr(devicePtr);
      }
      return flag;
    }
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
    /// Reports information from the Host Memory Pool
    void reportHostMemoryPoolInformation(const std::string& name = std::string()) const;
    /// Reports information from the Device Memory Pool
    void reportDeviceMemoryPoolInformation(const std::string& name = std::string()) const;
    /// Reports information from the Host/Device Memory Pool
    void reportHostDeviceMemoryPoolInformation(const std::string& name = std::string()) const;

    CUDAProcessMemoryPool(const CUDADriverInfo& cudaDriverInfo, bool useDefaultAllocations = true, bool useSeparateAllocations = bool(GPU_FRAMEWORK_CUDA_PROCESS_MEMORY_POOL_USE_SEPARATE_ALLOCATIONS)) noexcept;
    ~CUDAProcessMemoryPool() noexcept; // no virtual destructor for data-oriented design (no up-casting should ever be used)
    CUDAProcessMemoryPool(const CUDAProcessMemoryPool&) = delete;
    CUDAProcessMemoryPool(CUDAProcessMemoryPool&&)      = delete;
    CUDAProcessMemoryPool& operator=(const CUDAProcessMemoryPool&) = delete;
    CUDAProcessMemoryPool& operator=(CUDAProcessMemoryPool&&)      = delete;

  private:

    /// Use default memory allocations for all devices
    bool useDefaultAllocations_ = false;
    /// Use separate memory allocations (Note: to be enabled for debugging purposes only)
    bool useSeparateAllocations_ = false;
    /// The CUDADriverInfo reference
    const CUDADriverInfo& cudaDriverInfo_;
    /// The Host Memory Pool host allocated check
    bool isHostAllocated_ = false;
    /// The Host Memory Pool pointer
    std::uint8_t* hostMemoryPoolPtr_ = nullptr;
    /// The Host Memory Pool offset
    std::size_t hostMemoryPoolOffset_ = 0;
    /// The total Host Memory Pool bytes consumed
    std::size_t hostBytesToAllocate_ = 0;
    /// The Host Memory Pool is stored in a vector
    std::vector<MemoryPoolData> hostMemoryPool_;
    /// The Host Memory Pool flags
    unsigned int flags_ = cudaHostRegisterDefault;
    /// The Device Memory Pool device allocated check
    bool isDeviceAllocated_ = false;
    /// The number of available devices (default is 1)
    std::size_t deviceCount_ = 1;
    /// The texture alignment per device (1 per available GPU device)
    std::unique_ptr<std::size_t[]> textureAlignmentPerDevice_ = nullptr;
    /// The Device Memory Pool pointers (1 per available GPU device)
    std::unique_ptr<std::uint8_t*[]> deviceMemoryPoolPtrPerDevice_ = nullptr;
    /// The Device Memory Pool offsets (1 per available GPU device)
    std::unique_ptr<std::size_t[]> deviceMemoryPoolOffsetPerDevice_ = nullptr;
    /// The total Device Memory Pool bytes consumed (1 per available GPU device)
    std::unique_ptr<std::size_t[]> deviceBytesToAllocatePerDevice_ = nullptr;
    /// The Device Memory Pool is stored in a vector
    std::vector<MemoryPoolData> deviceMemoryPool_;
    /// The Device Memory Pool Unified Memory (UVA) flags
    std::bitset<MAX_DEVICES> unifiedMemoryFlags_;

    /// Reserves state to the CUDA Memory Pool
    bool reserveMemoryPoolData(std::size_t numberOfElements, std::size_t sizeOfElement, int device, const MemoryPoolTypes& type);
  };
} // namespace UtilsCUDA

#endif // __CUDAProcessMemoryPool_h