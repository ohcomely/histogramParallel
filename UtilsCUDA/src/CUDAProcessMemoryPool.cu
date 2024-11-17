#include "CUDAProcessMemoryPool.h"
#include "CUDAUtilityFunctions.h"
#include "CPUParallelism/CPUParallelismNCP.h"
#include "UtilityFunctions.h"
#include <cuda_runtime_api.h>
#include <mutex>
#include <algorithm>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace UtilsCUDA;
using namespace Utils::CPUParallelism;
using namespace Utils::UtilityFunctions;

namespace // anonymous namespace used instead of deprecated 'static' keyword used for cpp variable locality
{
  constexpr double GPU_MEMORY_PERCENTAGE = 0.500; // 50.0% of GPU global memory to be allocated by default
  constexpr double CPU_MEMORY_PERCENTAGE = 0.125; // 12.5% of GPU-to-CPU global memory to be allocated by default per GPU

  // used as static variable in local compilation unit
  mutex memoryPoolDataMutex;

  inline size_t calculateMemoryBlockOffset(size_t& bytesToAllocate, size_t numberOfElements, size_t sizeOfElement, size_t textureAlignment)
  {
    // move the offset to a multiply of the alignment
    const size_t offset = bytesToAllocate + ((textureAlignment - (bytesToAllocate % textureAlignment)) % textureAlignment);
    bytesToAllocate     = offset + numberOfElements * sizeOfElement;
    return offset;
  }

  inline size_t getMemoryPoolSize(const vector<CUDAProcessMemoryPool::MemoryPoolData>& memoryPool, const CUDAProcessMemoryPool::MemoryPoolTypes& type, int device = 0)
  {
    if (type == CUDAProcessMemoryPool::MemoryPoolTypes::HOST_MEMORY)
    {
      return memoryPool.size();
    }
    else // if (type == CUDAProcessMemoryPool::MemoryPoolTypes::DEVICE_MEMORY)
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

  inline size_t getMemoryPoolTotalBytes(const vector<CUDAProcessMemoryPool::MemoryPoolData>& memoryPool, const CUDAProcessMemoryPool::MemoryPoolTypes& type, int device = 0)
  {
    size_t totalBytes = 0;
    if (type == CUDAProcessMemoryPool::MemoryPoolTypes::HOST_MEMORY)
    {
      for (const auto& memoryPoolData : memoryPool)
      {
        totalBytes += memoryPoolData.numberOfElements_ * memoryPoolData.sizeOfElement_;
      }
    }
    else // if (type == CUDAProcessMemoryPool::MemoryPoolTypes::DEVICE_MEMORY)
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

  inline void reportMemoryPoolInformationInternal(const vector<CUDAProcessMemoryPool::MemoryPoolData>& memoryPool, const CUDAProcessMemoryPool::MemoryPoolTypes& type, bool useSeparateAllocations,
                                                  const string& name = string(), const unique_ptr<size_t[]>& deviceBytesToAllocatePerDevice = nullptr, size_t deviceCount = 0,
                                                  const bitset<CUDAProcessMemoryPool::MAX_DEVICES>& unifiedMemoryFlags = bitset<CUDAProcessMemoryPool::MAX_DEVICES>())
  {
    const string memoryPoolName = name.empty() ? "" : name + " ";
    const string memoryPoolType = (type == CUDAProcessMemoryPool::MemoryPoolTypes::HOST_MEMORY) ? "Host" : "Device";
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
      if (type == CUDAProcessMemoryPool::MemoryPoolTypes::HOST_MEMORY)
      {
        ss << '\n';
      }
      else // if (type == CUDAProcessMemoryPool::MemoryPoolTypes::DEVICE_MEMORY)
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
        if (type == CUDAProcessMemoryPool::MemoryPoolTypes::HOST_MEMORY)
        {
          ss << '\n';
          totalHostBytesConsumed += totalBytesConsumed;
        }
        else // if (type == CUDAProcessMemoryPool::MemoryPoolTypes::DEVICE_MEMORY)
        {
          ss << setw( 3) << memoryPoolData.device_ << '\n';
          totalDeviceBytesConsumed += totalBytesConsumed;
        }
      }
      ss << '\n';
      if (type == CUDAProcessMemoryPool::MemoryPoolTypes::HOST_MEMORY)
      {
        ss << "Total Host bytes consumed: "   << totalHostBytesConsumed   << '\n';
      }
      else // if (type == CUDAProcessMemoryPool::MemoryPoolTypes::DEVICE_MEMORY)
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

    if (type == CUDAProcessMemoryPool::MemoryPoolTypes::DEVICE_MEMORY)
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

CUDAProcessMemoryPool::CUDAProcessMemoryPool(const CUDADriverInfo& cudaDriverInfo, bool useDefaultAllocations, bool useSeparateAllocations) noexcept
  : useDefaultAllocations_(useDefaultAllocations)
  , useSeparateAllocations_(useSeparateAllocations)
  , cudaDriverInfo_(cudaDriverInfo)
{
  deviceCount_                     = max<size_t>(1, size_t(cudaDriverInfo.getDeviceCount()));
  textureAlignmentPerDevice_       = make_unique<size_t[]>(deviceCount_);
  deviceMemoryPoolPtrPerDevice_    = make_unique<uint8_t*[]>(deviceCount_);
  deviceMemoryPoolOffsetPerDevice_ = make_unique<size_t[]>(deviceCount_);
  deviceBytesToAllocatePerDevice_  = make_unique<size_t[]>(deviceCount_);
  // default sizes to allocate
  for (size_t device = 0; device < deviceCount_; ++device)
  {
    textureAlignmentPerDevice_[device] = cudaDriverInfo.getTextureAlignment(int(device));
    if (useDefaultAllocations_)
    {
      hostBytesToAllocate_                   += size_t(CPU_MEMORY_PERCENTAGE * cudaDriverInfo.getTotalGlobalMemory(int(device)));
      deviceBytesToAllocatePerDevice_[device] = size_t(GPU_MEMORY_PERCENTAGE * cudaDriverInfo.getTotalGlobalMemory(int(device)));
    }
  }
}

CUDAProcessMemoryPool::~CUDAProcessMemoryPool() noexcept
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

void CUDAProcessMemoryPool::allocateHostMemoryPool(size_t hostBytesToAllocate, unsigned int flags)
{
  if (isHostAllocated_)
  {
    DebugConsole_consoleOutLine("Host Memory Pool allocation error:\n  Cannot allocate host memory when the Host is already allocated.");
    return;
  }

  if (!useSeparateAllocations_)
  {
    if (hostBytesToAllocate > 0)
    {
      hostBytesToAllocate_ = hostBytesToAllocate;
    }
    // allocate host memory below
    CUDAError_checkCUDAError(cudaHostAlloc(&hostMemoryPoolPtr_, hostBytesToAllocate_, flags));
    DebugConsole_consoleOutLine("Host Memory Pool allocation of '", hostBytesToAllocate_, "' bytes for the host.");
  }
  isHostAllocated_ = true;
  flags_           = flags;
}

void CUDAProcessMemoryPool::allocateDeviceMemoryPool(const array<size_t, MAX_DEVICES>& deviceBytesToAllocatePerDevice,
                                                     const bitset<MAX_DEVICES>& unifiedMemoryFlags)
{
  if (isDeviceAllocated_)
  {
    DebugConsole_consoleOutLine("Device Memory Pool allocation error:\n  Cannot allocate device memory when the Device is already allocated.");
    return;
  }

  if (!useSeparateAllocations_)
  {
    parallelFor(0, deviceCount_, [&](size_t device)
    {
      if (deviceBytesToAllocatePerDevice[device] > 0)
      {
        deviceBytesToAllocatePerDevice_[device] = deviceBytesToAllocatePerDevice[device];
      }
      // allocate device memory below
      CUDAError_checkCUDAError(cudaSetDevice(int(device)));
      CUDAError_checkCUDAError(unifiedMemoryFlags[device] ? cudaMallocManaged(&deviceMemoryPoolPtrPerDevice_[device], deviceBytesToAllocatePerDevice_[device])
                                                          : cudaMalloc(       &deviceMemoryPoolPtrPerDevice_[device], deviceBytesToAllocatePerDevice_[device]));
      DebugConsole_consoleOutLine("Device Memory Pool allocation of '", deviceBytesToAllocatePerDevice_[device], "' bytes for device '", device, "'.");
    });
  }
  isDeviceAllocated_  = true;
  unifiedMemoryFlags_ = unifiedMemoryFlags;
}

void CUDAProcessMemoryPool::allocateHostDeviceMemoryPool(size_t hostBytesToAllocate, const array<size_t, MAX_DEVICES>& deviceBytesToAllocatePerDevice,
                                                         const bitset<MAX_DEVICES>& unifiedMemoryFlags, unsigned int flags)
{
  allocateHostMemoryPool(    hostBytesToAllocate,                       flags);
  allocateDeviceMemoryPool(deviceBytesToAllocatePerDevice, unifiedMemoryFlags);
}

bool CUDAProcessMemoryPool::reserveMemoryPoolData(size_t numberOfElements, size_t sizeOfElement, int device, const MemoryPoolTypes& type)
{
  if (type == MemoryPoolTypes::HOST_MEMORY)
  {
    if (!isHostAllocated_)
    {
      DebugConsole_consoleOutLine("Host Memory Pool reserve error:\n Cannot reserve memory when the Host is not already allocated.");
      return false;
    }

    unique_lock<mutex> lockDataMutex(memoryPoolDataMutex); // protect memory pool data accesses below
    if (useSeparateAllocations_)
    {
      uint8_t* ptr = nullptr;
      CUDAError_checkCUDAError(cudaHostAlloc(&ptr, numberOfElements * sizeOfElement, flags_));
      hostMemoryPool_.emplace_back(MemoryPoolData{ptr, numberOfElements, sizeOfElement, 0, device});
    }
    else
    {
      const size_t offset = calculateMemoryBlockOffset(hostMemoryPoolOffset_, numberOfElements, sizeOfElement, textureAlignmentPerDevice_[device]);
      if (hostMemoryPoolOffset_ <= hostBytesToAllocate_)
      {
        hostMemoryPool_.emplace_back(MemoryPoolData{hostMemoryPoolPtr_ + offset, numberOfElements, sizeOfElement, offset, device});
      }
      else
      {
        DebugConsole_consoleOutLine("Host Memory Pool reserve error:\n The Host has run out of pre-allocated memory:\n  host bytes allocated: ", hostBytesToAllocate_, " & host memory pool offset: ", hostMemoryPoolOffset_);
        CUDAError_checkCUDAError(cudaErrorMemoryAllocation);
        return false;
      }
    }
  }
  else // if ((type == MemoryPoolTypes::DEVICE_MEMORY))
  {
    if (!isDeviceAllocated_)
    {
      DebugConsole_consoleOutLine("Device Memory Pool reserve error:\n Cannot reserve memory when the Device is not already allocated.");
      return false;
    }

    unique_lock<mutex> lockDataMutex(memoryPoolDataMutex); // protect memory pool data accesses below
    if (useSeparateAllocations_)
    {
      uint8_t* ptr = nullptr;
      CUDAError_checkCUDAError(cudaSetDevice(device));
      CUDAError_checkCUDAError(unifiedMemoryFlags_[device] ? cudaMallocManaged(&ptr, numberOfElements * sizeOfElement)
                                                           : cudaMalloc(       &ptr, numberOfElements * sizeOfElement));
      deviceMemoryPool_.emplace_back(MemoryPoolData{ptr, numberOfElements, sizeOfElement, 0, device});
    }
    else
    {
      const size_t offset = calculateMemoryBlockOffset(deviceMemoryPoolOffsetPerDevice_[device], numberOfElements, sizeOfElement, textureAlignmentPerDevice_[device]);
      if (deviceMemoryPoolOffsetPerDevice_[device] <= deviceBytesToAllocatePerDevice_[device])
      {
        deviceMemoryPool_.emplace_back(MemoryPoolData{deviceMemoryPoolPtrPerDevice_[device] + offset, numberOfElements, sizeOfElement, offset, device});
      }
      else
      {
        DebugConsole_consoleOutLine("Device Memory Pool reserve error:\n The Device '", device, "' has run out of pre-allocated memory:\n  device bytes allocated: ", deviceBytesToAllocatePerDevice_[device], " & device memory pool offset: ", deviceMemoryPoolOffsetPerDevice_[device]);
        CUDAError_checkCUDAError(cudaErrorMemoryAllocation);
        return false;
      }
    }
  }

  return true;
}

void CUDAProcessMemoryPool::freeHostMemoryPool()
{
  if (! hostMemoryPool_.empty())
  {
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
      // free host memory below
      assert(hostMemoryPoolPtr_ != nullptr);
      assert(isValidHostDevicePointer(hostMemoryPoolPtr_));
      CUDAError_checkCUDAError(cudaFreeHost(hostMemoryPoolPtr_));
      hostMemoryPoolPtr_    = nullptr;
      hostMemoryPoolOffset_ = 0;
      hostBytesToAllocate_  = 0;
      if (useDefaultAllocations_)
      {
        for (size_t device = 0; device < deviceCount_; ++ device)
        {
          hostBytesToAllocate_ += size_t(CPU_MEMORY_PERCENTAGE * cudaDriverInfo_.getTotalGlobalMemory(int(device)));
        }
      }
    }
    isHostAllocated_ = false;
    hostMemoryPool_.clear();
    flags_           = cudaHostRegisterDefault;
  }
  else
  {
    DebugConsole_consoleOutLine("Host Memory Pool freeing warning:\n  Host Memory Pool is already empty.");
  }

  // Note: Report the current Host pointer(s) from the Host Memory Pool
  reportHostMemoryPoolInformation();
}

void CUDAProcessMemoryPool::freeDeviceMemoryPool()
{
  if (! deviceMemoryPool_.empty())
  {
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
      // de-allocate device memory below
      parallelFor(0, deviceCount_, [&](size_t device)
      {
        if (deviceMemoryPoolPtrPerDevice_[device] != nullptr)
        {
          assert(isValidHostDevicePointer(deviceMemoryPoolPtrPerDevice_[device]));
          CUDAError_checkCUDAError(cudaFree(deviceMemoryPoolPtrPerDevice_[device]));
          deviceMemoryPoolPtrPerDevice_[device]    = nullptr;
          deviceMemoryPoolOffsetPerDevice_[device] = 0;
          deviceBytesToAllocatePerDevice_[device]  = useDefaultAllocations_ ? size_t(GPU_MEMORY_PERCENTAGE * cudaDriverInfo_.getTotalGlobalMemory(int(device))) : 0;
        }
      });
    }
    isDeviceAllocated_  = false;
    deviceMemoryPool_.clear();
    unifiedMemoryFlags_ = bitset<MAX_DEVICES>();
  }
  else
  {
    DebugConsole_consoleOutLine("Device Memory Pool freeing warning:\n  Device Memory Pool is already empty.");
  }

  // Note: Report the current device pointer(s) from the Device Memory Pool
  reportDeviceMemoryPoolInformation();
}

void CUDAProcessMemoryPool::freeHostDeviceMemoryPool()
{
  freeHostMemoryPool();
  freeDeviceMemoryPool();
}

size_t CUDAProcessMemoryPool::getHostMemoryPoolSize() const
{
  return getMemoryPoolSize(  hostMemoryPool_, MemoryPoolTypes::HOST_MEMORY);
}

size_t CUDAProcessMemoryPool::getDeviceMemoryPoolSize(int device) const
{
  return getMemoryPoolSize(deviceMemoryPool_, MemoryPoolTypes::DEVICE_MEMORY, device);
}

size_t CUDAProcessMemoryPool::getHostMemoryPoolTotalBytes() const
{
  return getMemoryPoolTotalBytes(  hostMemoryPool_, MemoryPoolTypes::HOST_MEMORY);
}

size_t CUDAProcessMemoryPool::getDeviceMemoryPoolTotalBytes(int device) const
{
  return getMemoryPoolTotalBytes(deviceMemoryPool_, MemoryPoolTypes::DEVICE_MEMORY, device);
}

void CUDAProcessMemoryPool::reportHostMemoryPoolInformation(const string& name) const
{
  reportMemoryPoolInformationInternal(hostMemoryPool_,   MemoryPoolTypes::HOST_MEMORY,   useSeparateAllocations_, name);
}

void CUDAProcessMemoryPool::reportDeviceMemoryPoolInformation(const string& name) const
{
  reportMemoryPoolInformationInternal(deviceMemoryPool_, MemoryPoolTypes::DEVICE_MEMORY, useSeparateAllocations_, name, deviceBytesToAllocatePerDevice_, deviceCount_, unifiedMemoryFlags_);
}

void CUDAProcessMemoryPool::reportHostDeviceMemoryPoolInformation(const string& name) const
{
  reportMemoryPoolInformationInternal(hostMemoryPool_,   MemoryPoolTypes::HOST_MEMORY,   useSeparateAllocations_, name);
  reportMemoryPoolInformationInternal(deviceMemoryPool_, MemoryPoolTypes::DEVICE_MEMORY, useSeparateAllocations_, name, deviceBytesToAllocatePerDevice_, deviceCount_, unifiedMemoryFlags_);
}