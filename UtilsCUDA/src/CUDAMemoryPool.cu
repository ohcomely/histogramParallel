#include "CUDAMemoryPool.h"
#include "CUDAUtilityFunctions.h"
#include "CPUParallelism/CPUParallelismNCP.h"
#include "UtilityFunctions.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace UtilsCUDA;
using namespace Utils::CPUParallelism;
using namespace Utils::UtilityFunctions;

namespace // anonymous namespace used instead of deprecated 'static' keyword used for cpp variable locality
{
  inline size_t calculateMemoryBlockOffset(size_t& bytesToAllocate, size_t numberOfElements, size_t sizeOfElement, size_t textureAlignment)
  {
    // move the offset to a multiply of the alignment
    const size_t offset = bytesToAllocate + ((textureAlignment - (bytesToAllocate % textureAlignment)) % textureAlignment);
    bytesToAllocate     = offset + numberOfElements * sizeOfElement;
    return offset;
  }

  inline size_t getMemoryPoolSize(const vector<CUDAMemoryPool::MemoryPoolData>& memoryPool, const CUDAMemoryPool::MemoryPoolTypes& type, int device = 0)
  {
    if (type == CUDAMemoryPool::MemoryPoolTypes::HOST_MEMORY)
    {
      return memoryPool.size();
    }
    else // if (type == CUDAMemoryPool::MemoryPoolTypes::DEVICE_MEMORY)
    {
      size_t size = 0;
      for (const auto& memoryPoolData : memoryPool)
      {
        if (device == memoryPoolData.device_)
        {
          ++size;
        }
      }

      return size;
    }
  }

    inline size_t getMemoryPoolTotalBytes(const vector<CUDAMemoryPool::MemoryPoolData>& memoryPool, const CUDAMemoryPool::MemoryPoolTypes& type, int device = 0)
    {
      size_t totalBytes = 0;
      if (type == CUDAMemoryPool::MemoryPoolTypes::HOST_MEMORY)
      {
        for (const auto& memoryPoolData : memoryPool)
        {
          totalBytes += memoryPoolData.numberOfElements_ * memoryPoolData.sizeOfElement_;
        }
      }
      else // if (type == CUDAMemoryPool::MemoryPoolTypes::DEVICE_MEMORY)
      {
        for (const auto& memoryPoolData : memoryPool)
        {
          if (device == memoryPoolData.device_)
          {
            totalBytes += memoryPoolData.numberOfElements_ * memoryPoolData.sizeOfElement_;
          }
        }
      }

      return totalBytes;
    }

    inline void reportMemoryPoolInformationInternal(const vector<CUDAMemoryPool::MemoryPoolData>& memoryPool, const CUDAMemoryPool::MemoryPoolTypes& type, bool useSeparateAllocations,
                                                    const string& name = string(), const unique_ptr<size_t[]>& deviceBytesToAllocatePerDevice = nullptr, size_t deviceCount = 0,
                                                    const bitset<CUDAMemoryPool::MAX_DEVICES>& unifiedMemoryFlags = bitset<CUDAMemoryPool::MAX_DEVICES>())
    {
      const string memoryPoolName = name.empty() ? "" : name + " ";
      const string memoryPoolType = (type == CUDAMemoryPool::MemoryPoolTypes::HOST_MEMORY) ? "Host" : "Device";
      ostringstream ss;
      ss << '\n';
      ss << "   --- " << memoryPoolName << memoryPoolType << " Memory Pool Information ---" << '\n';
      if (!memoryPool.empty())
      {
        size_t index                    = 0;
        size_t totalHostBytesConsumed   = 0;
        size_t totalDeviceBytesConsumed = 0;
        ss << left << setw( 6) << "####"
                   << setw(18) << "Ptr"
                   << setw(13) << "Size"
                   << setw(10) << "SizeOf"
                   << setw(13) << "Offset"
                   << setw(13) << "Bytes";
        if (type == CUDAMemoryPool::MemoryPoolTypes::HOST_MEMORY)
        {
          ss << '\n';
        }
        else // if (type == CUDAMemoryPool::MemoryPoolTypes::DEVICE_MEMORY)
        {
          ss << setw( 3) << "Device" << '\n';
        }
        for (const auto& memoryPoolData : memoryPool)
        {
          ++index;
          const size_t totalBytesConsumed = memoryPoolData.numberOfElements_ * memoryPoolData.sizeOfElement_;
          ss << left << setw( 6) << StringAuxiliaryFunctions::formatNumberString(index, memoryPool.size())
                     << setw(18) << reinterpret_cast<void*>(memoryPoolData.ptr_) // Note: uint8_t (ie char) will confuse C++ and crash the log
                     << setw(13) << memoryPoolData.numberOfElements_
                     << setw(10) << memoryPoolData.sizeOfElement_
                     << setw(13) << memoryPoolData.offset_
                     << setw(13) << totalBytesConsumed;
          if (type == CUDAMemoryPool::MemoryPoolTypes::HOST_MEMORY)
          {
            ss << '\n';
            totalHostBytesConsumed += totalBytesConsumed;
          }
          else // if (type == CUDAMemoryPool::MemoryPoolTypes::DEVICE_MEMORY)
          {
            ss << setw( 3) << memoryPoolData.device_ << '\n';
            totalDeviceBytesConsumed += totalBytesConsumed;
          }
        }
        ss << '\n';
        if (type == CUDAMemoryPool::MemoryPoolTypes::HOST_MEMORY)
        {
          ss << "Total Host bytes consumed: "   << totalHostBytesConsumed   << '\n';
        }
        else // if (type == CUDAMemoryPool::MemoryPoolTypes::DEVICE_MEMORY)
        {
          ss << "Total Device bytes consumed: " << totalDeviceBytesConsumed << '\n';
        }
        if (useSeparateAllocations)
        {
          ss << "Note: Separate allocations functionality is being used." << '\n';
        }
        ss << '\n';
      }
      else
      {
        ss << "Note: " << memoryPoolType << " Memory Pool is empty." << '\n';
      }

      if (type == CUDAMemoryPool::MemoryPoolTypes::DEVICE_MEMORY)
      {
        for (size_t device = 0; device < deviceCount; ++device)
        {
          if (deviceBytesToAllocatePerDevice[device] > 0)
          {
            ss << CUDAUtilityFunctions::checkAndReportCUDAMemory(int(device), unifiedMemoryFlags[device]);
          }
        }
      }
      DebugConsole_consoleOutLine(ss.str());
    }
}

CUDAMemoryPool::CUDAMemoryPool(const CUDADriverInfo& cudaDriverInfo, bool useSeparateAllocations) noexcept
  : useSeparateAllocations_(useSeparateAllocations)
{
  deviceCount_                    = max<size_t>(1, size_t(cudaDriverInfo.getDeviceCount()));
  textureAlignmentPerDevice_      = make_unique<size_t[]>(deviceCount_);
  for (size_t device = 0; device < deviceCount_; ++device)
  {
    textureAlignmentPerDevice_[device] = cudaDriverInfo.getTextureAlignment(int(device));
  }
  deviceMemoryPoolPtrPerDevice_   = make_unique<uint8_t*[]>(deviceCount_);
  deviceBytesToAllocatePerDevice_ = make_unique<size_t[]>(deviceCount_);
}

CUDAMemoryPool::~CUDAMemoryPool() noexcept
{
  if (!hostMemoryPool_.empty())
  {
    // use the Host memory pool for the de-allocation of host memory (if necessary)
    freeHostMemoryPool();
  }

  if (!deviceMemoryPool_.empty())
  {
    // use the Device memory pool for the de-allocation of device memory (if necessary)
    freeDeviceMemoryPool();
  }
}

bool CUDAMemoryPool::addMemoryPoolData(size_t numberOfElements, size_t sizeOfElement, int device, const MemoryPoolTypes& type,
                                       const function<void(uint8_t* ptr, bool)>& memoryHandlerSetFunction)
{
  if (type == MemoryPoolTypes::HOST_MEMORY)
  {
    if (isHostAllocated_)
    {
      DebugConsole_consoleOutLine("Host Memory Pool error:\n Cannot add memory when the Host is already allocated.");
      return false;
    }

    const size_t offset = useSeparateAllocations_ ? 0 : calculateMemoryBlockOffset(hostBytesToAllocate_, numberOfElements, sizeOfElement, textureAlignmentPerDevice_[device]);
    hostMemoryPool_.emplace_back(MemoryPoolData{nullptr, numberOfElements, sizeOfElement, offset, device, memoryHandlerSetFunction});
  }
  else // if ((type == MemoryPoolTypes::DEVICE_MEMORY))
  {
    if (isDeviceAllocated_)
    {
      DebugConsole_consoleOutLine("Device Memory Pool error:\n Cannot add memory when the Device is already allocated.");
      return false;
    }

    const size_t offset = useSeparateAllocations_ ? 0 : calculateMemoryBlockOffset(deviceBytesToAllocatePerDevice_[device], numberOfElements, sizeOfElement, textureAlignmentPerDevice_[device]);
    deviceMemoryPool_.emplace_back(MemoryPoolData{nullptr, numberOfElements, sizeOfElement, offset, device, memoryHandlerSetFunction});
  }

  return true;
}

void CUDAMemoryPool::allocateHostMemoryPool(const string& name, unsigned int flags)
{
  if (!hostMemoryPool_.empty())
  {
    // allocate host memory below
    if (useSeparateAllocations_)
    {
      for (auto& memoryPoolHostData : hostMemoryPool_)
      {
        CUDAError_checkCUDAError(cudaHostAlloc(&memoryPoolHostData.ptr_, memoryPoolHostData.numberOfElements_ * memoryPoolHostData.sizeOfElement_, flags));
        memoryPoolHostData.memoryHandlerSetFunction(memoryPoolHostData.ptr_, false);
      }
    }
    else
    {
      CUDAError_checkCUDAError(cudaHostAlloc(&hostMemoryPoolPtr_, hostBytesToAllocate_, flags));
      for (auto& memoryPoolHostData : hostMemoryPool_)
      {
        memoryPoolHostData.ptr_ = hostMemoryPoolPtr_ + memoryPoolHostData.offset_; // ptr address + offset
        memoryPoolHostData.memoryHandlerSetFunction(memoryPoolHostData.ptr_, false);
      }
    }
    isHostAllocated_ = true;
  }
  else
  {
    DebugConsole_consoleOutLine("Host Memory Pool allocation warning:\n  Host Memory Pool is empty.");
  }

  // Note: Report the current host pointer(s) from the Host Memory Pool
  CUDADriverInfo_report(reportMemoryPoolInformationInternal(hostMemoryPool_, MemoryPoolTypes::HOST_MEMORY, useSeparateAllocations_, name));
}

void CUDAMemoryPool::allocateDeviceMemoryPool(const string& name, const bitset<MAX_DEVICES>& unifiedMemoryFlags)
{
  if (!deviceMemoryPool_.empty())
  {
    // allocate device memory below
    if (useSeparateAllocations_)
    {
      for (auto& memoryPoolDeviceData : deviceMemoryPool_)
      {
        CUDAError_checkCUDAError(cudaSetDevice(memoryPoolDeviceData.device_));
        CUDAError_checkCUDAError(unifiedMemoryFlags[memoryPoolDeviceData.device_] ? cudaMallocManaged(&memoryPoolDeviceData.ptr_, memoryPoolDeviceData.numberOfElements_ * memoryPoolDeviceData.sizeOfElement_)
                                                                                  : cudaMalloc(       &memoryPoolDeviceData.ptr_, memoryPoolDeviceData.numberOfElements_ * memoryPoolDeviceData.sizeOfElement_));
        memoryPoolDeviceData.memoryHandlerSetFunction(memoryPoolDeviceData.ptr_, unifiedMemoryFlags[memoryPoolDeviceData.device_]);
      }
    }
    else
    {
      parallelFor(0, deviceCount_, [&](size_t device)
      {
        if (deviceBytesToAllocatePerDevice_[device] > 0)
        {
          CUDAError_checkCUDAError(cudaSetDevice(int(device)));
          CUDAError_checkCUDAError(unifiedMemoryFlags[device] ? cudaMallocManaged(&deviceMemoryPoolPtrPerDevice_[device], deviceBytesToAllocatePerDevice_[device])
                                                              : cudaMalloc(       &deviceMemoryPoolPtrPerDevice_[device], deviceBytesToAllocatePerDevice_[device]));
          for (auto& memoryPoolDeviceData : deviceMemoryPool_)
          {
            if (size_t(memoryPoolDeviceData.device_) == device)
            {
              memoryPoolDeviceData.ptr_ = deviceMemoryPoolPtrPerDevice_[device] + memoryPoolDeviceData.offset_; // ptr address + offset
              memoryPoolDeviceData.memoryHandlerSetFunction(memoryPoolDeviceData.ptr_, unifiedMemoryFlags[device]);
            }
          }
        }
      });
    }
    isDeviceAllocated_  = true;
  }
  else
  {
    DebugConsole_consoleOutLine("Device Memory Pool allocation warning:\n  Device Memory Pool is empty.");
  }

  // Note: Report the current device pointer(s) from the Device Memory Pool
  CUDADriverInfo_report(reportMemoryPoolInformationInternal(deviceMemoryPool_, MemoryPoolTypes::DEVICE_MEMORY, useSeparateAllocations_, name, deviceBytesToAllocatePerDevice_, deviceCount_, unifiedMemoryFlags));
}

void CUDAMemoryPool::allocateHostDeviceMemoryPool(const string& name, const bitset<MAX_DEVICES>& unifiedMemoryFlags, unsigned int flags)
{
  allocateHostMemoryPool(  name, flags);
  allocateDeviceMemoryPool(name, unifiedMemoryFlags);
}

void CUDAMemoryPool::freeHostMemoryPool()
{
  if (!hostMemoryPool_.empty())
  {
    // free host memory below
    if (useSeparateAllocations_)
    {
      for (auto& memoryPoolHostData : hostMemoryPool_)
      {
        assert(memoryPoolHostData.ptr_ != nullptr);
        assert(isValidHostDevicePointer(memoryPoolHostData.ptr_));
        CUDAError_checkCUDAError(cudaFreeHost(memoryPoolHostData.ptr_));
        memoryPoolHostData.ptr_ = nullptr;
      }
    }
    else
    {
      assert(hostMemoryPoolPtr_ != nullptr);
      assert(isValidHostDevicePointer(hostMemoryPoolPtr_));
      CUDAError_checkCUDAError(cudaFreeHost(hostMemoryPoolPtr_));
      hostMemoryPoolPtr_   = nullptr;
    }
    hostBytesToAllocate_ = 0;
    isHostAllocated_     = false;
    hostMemoryPool_.clear();
  }
  else
  {
    DebugConsole_consoleOutLine("Host Memory Pool freeing warning:\n  Host Memory Pool is already empty.");
  }

  // Note: Report the current Host pointer(s) from the Host Memory Pool
  CUDADriverInfo_report(reportMemoryPoolInformationInternal(hostMemoryPool_, MemoryPoolTypes::HOST_MEMORY, useSeparateAllocations_));
}

void CUDAMemoryPool::freeDeviceMemoryPool()
{
  if (!deviceMemoryPool_.empty())
  {
    // de-allocate device memory below
    if (useSeparateAllocations_)
    {
      for (auto& memoryPoolDeviceData : deviceMemoryPool_)
      {
        assert(memoryPoolDeviceData.ptr_ != nullptr);
        assert(isValidHostDevicePointer(memoryPoolDeviceData.ptr_));
        CUDAError_checkCUDAError(cudaFree(memoryPoolDeviceData.ptr_));
        memoryPoolDeviceData.ptr_ = nullptr;
      }
    }
    else
    {
      parallelFor(0, deviceCount_, [&](size_t device)
      {
        if (deviceMemoryPoolPtrPerDevice_[device] != nullptr)
        {
          assert(isValidHostDevicePointer(deviceMemoryPoolPtrPerDevice_[device]));
          CUDAError_checkCUDAError(cudaFree(deviceMemoryPoolPtrPerDevice_[device]));
          deviceMemoryPoolPtrPerDevice_[device]   = nullptr;
          deviceBytesToAllocatePerDevice_[device] = 0;
        }
      });
    }
    isDeviceAllocated_ = false;
    deviceMemoryPool_.clear();
  }
  else
  {
    DebugConsole_consoleOutLine("Device Memory Pool freeing warning:\n  Device Memory Pool is already empty.");
  }

  // Note: Report the current device pointer(s) from the Device Memory Pool
  CUDADriverInfo_report(reportMemoryPoolInformationInternal(deviceMemoryPool_, MemoryPoolTypes::DEVICE_MEMORY, useSeparateAllocations_));
}

void CUDAMemoryPool::freeHostDeviceMemoryPool()
{
  freeHostMemoryPool();
  freeDeviceMemoryPool();
}

size_t CUDAMemoryPool::getHostMemoryPoolSize() const
{
  return getMemoryPoolSize(  hostMemoryPool_, MemoryPoolTypes::HOST_MEMORY);
}

size_t CUDAMemoryPool::getDeviceMemoryPoolSize(int device) const
{
  return getMemoryPoolSize(deviceMemoryPool_, MemoryPoolTypes::DEVICE_MEMORY, device);
}

size_t CUDAMemoryPool::getHostMemoryPoolTotalBytes() const
{
  return getMemoryPoolTotalBytes(  hostMemoryPool_, MemoryPoolTypes::HOST_MEMORY);
}

size_t CUDAMemoryPool::getDeviceMemoryPoolTotalBytes(int device) const
{
  return getMemoryPoolTotalBytes(deviceMemoryPool_, MemoryPoolTypes::DEVICE_MEMORY, device);
}