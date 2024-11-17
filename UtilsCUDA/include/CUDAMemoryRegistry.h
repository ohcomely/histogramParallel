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

#ifndef __CUDAMemoryRegistry_h
#define __CUDAMemoryRegistry_h

#include "ModuleDLL.h"
#include <driver_types.h>
#include <string>
#include <cstdint>
#include <vector>
#include <unordered_map>

/** @brief namespace UtilsCUDA for encapsulating all the CUDA related code compiled by the NVCC compiler.
*
* @author Thanos Theo, 2018
* @version 14.0.0.0
*/
namespace UtilsCUDA
{
  /** @brief This class encapsulates CUDA Memory Registry functionality for pre-allocated host memory with reporting.
  *
  *  CUDAMemoryRegistry.h:
  *  ====================
  *  This class encapsulates CUDA Memory Registry functionality for pre-allocated host memory with reporting.
  *
  *  The work pattern to use the CUDA Memory Registry is as follows:
  *    1. Use the addToMemoryRegistry() to register host T* data.
  *    2. Use the registerMemoryRegistry() for host batch register (per 'name' functions also available).
  *    3. Use the getPtrFromMemoryRegistry() to get T* the data.
  *    4. Use the unregisterMemoryRegistry() to delete all host data (per 'name' functions also available).
  *
  *  Note: 1. The destructor will also unregisterMemoryRegistry() in RAII fashion.
  *        2. Before using the registerMemoryRegistry(), using getPtrFromMemoryRegistry() call is invalid (nullptr returned).
  *        3. After using the registerMemoryRegistry(), the addToMemoryRegistry() call is invalid (nothing added, false returned).
  *        4. The returned getPtrFromMemoryRegistry() is invalid after a unregisterMemoryRegistry() call.
  *
  * @author Thanos Theo, 2019
  * @version 14.0.0.0
  */
  class UTILS_CUDA_MODULE_API CUDAMemoryRegistry final
  {
  public:

    /// struct for Memory Registry Data
    struct MemoryRegistryData final
    {
      std::uint8_t* ptr_            = nullptr;
      std::size_t numberOfElements_ = 0;
      std::size_t sizeOfElement_    = 0;

      MemoryRegistryData(std::uint8_t* ptr, std::size_t numberOfElements, std::size_t sizeOfElement) noexcept
          : ptr_(ptr), numberOfElements_(numberOfElements), sizeOfElement_(sizeOfElement) {}
      ~MemoryRegistryData() = default;
      MemoryRegistryData(const MemoryRegistryData&) = default;
      MemoryRegistryData(MemoryRegistryData&&)      = default;
      MemoryRegistryData& operator=(const MemoryRegistryData&) = default;
      MemoryRegistryData& operator=(MemoryRegistryData&&)      = default;
    };

    /// Registers host memory in the Memory Registry
    void registerMemoryRegistry(const std::string& name = std::string(), unsigned int flags = cudaHostRegisterDefault);
    /// Unregisters host memory from the Memory Registry
    void unregisterMemoryRegistry();
    /// Adds to the Memory Registry (wrapping a non-template function) a MemoryRegistryData
    template<typename T>
    bool addToMemoryRegistry(const std::string& name, T* ptr, std::size_t numberOfElements)
    {
        return addToMemoryRegistryPtr(name, reinterpret_cast<uint8_t*>(ptr), numberOfElements, sizeof(T));
    }
    /// Gets from the Memory Registry (wrapping a non-template function) a MemoryRegistryData
    template<typename T>
    MemoryRegistryData getPtrTupleFromMemoryRegistry(const std::string& name) const
    {
      const auto&   memoryRegistryData = getFromMemoryRegistryPtr(name);
      return (memoryRegistryData.ptr_ != nullptr) ? MemoryRegistryData{reinterpret_cast<T*>(memoryRegistryData.ptr_), memoryRegistryData.numberOfElements_, memoryRegistryData.sizeOfElement_}
                                                  : MemoryRegistryData{nullptr, 0, 0};
    }
    /// Gets from the Memory Registry (wrapping a non-template function) a MemoryRegistryData
    template<typename T>
    T* getPtrFromMemoryRegistry(const std::string& name) const
    {
      const auto& memoryRegistryData = getFromMemoryRegistryPtr(name);
      return (memoryRegistryData.ptr_ != nullptr) ? reinterpret_cast<T*>(memoryRegistryData.ptr_) : nullptr;
    }
    /// Gets the Memory Registry size
    std::size_t getMemoryRegistrySize() const { return memoryRegistry_.size(); };

    CUDAMemoryRegistry() = default;
    ~CUDAMemoryRegistry() noexcept; // no virtual destructor for data-oriented design (no up-casting should ever be used)
    CUDAMemoryRegistry(const CUDAMemoryRegistry&) = delete;
    CUDAMemoryRegistry(CUDAMemoryRegistry&&)      = delete;
    CUDAMemoryRegistry& operator=(const CUDAMemoryRegistry&) = delete;
    CUDAMemoryRegistry& operator=(CUDAMemoryRegistry&&)      = delete;

  private:

    /// The Memory Registry registered check
    bool isRegistered_ = false;
    /// The Memory Registry names is stored in a vector
    std::vector<std::string> memoryRegistryNames_;
    /// The Memory Registry is stored in an unordered map
    std::unordered_map<std::string, MemoryRegistryData> memoryRegistry_;
    /// Adds to the Memory Registry a MemoryRegistryData
    bool addToMemoryRegistryPtr(const std::string& name, std::uint8_t* ptr, std::size_t numberOfElements, std::size_t sizeOfElement);
    /// Gets from the Memory Registry a MemoryRegistryData
    MemoryRegistryData getFromMemoryRegistryPtr(const std::string& name) const;

    /// Reports information from the Memory Registry
    void reportMemoryRegistryInformation(const std::string& name = std::string()) const;
  };
} // namespace UtilsCUDA

#endif // __CUDAMemoryRegistry_h