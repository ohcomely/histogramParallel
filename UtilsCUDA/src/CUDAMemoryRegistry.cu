#include "CUDAMemoryRegistry.h"
#include "CUDAUtilityFunctions.h"
#include "UtilityFunctions.h"
#include <cuda_runtime_api.h>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace UtilsCUDA;
using namespace Utils::UtilityFunctions;

namespace // anonymous namespace used instead of deprecated 'static' keyword used for cpp variable locality
{
  inline void reportMemoryRegistryInformationInternal(const unordered_map<string, CUDAMemoryRegistry::MemoryRegistryData>& memoryRegistry, const vector<string>& memoryRegistryNames, const string& name)
  {
    stringstream ss;
    ss << '\n';
    ss << "   --- " << name << " Memory Registry Information ---" << '\n';
    if (!memoryRegistry.empty())
    {
      size_t index                      = 0;
      size_t totalRegistryBytesConsumed = 0;
      ss << left << setw( 6) << "####"
                 << setw(16) << "Ptr"
                 << setw(14) << "Size"
                 << setw(10) << "SizeOf"
                 << setw(14) << "Bytes" << '\n';
      for (const string& registryName : memoryRegistryNames)
      {
        ++index;
        const auto& memoryRegistryData  = memoryRegistry.at(registryName);
        const size_t totalBytesConsumed = memoryRegistryData.numberOfElements_ * memoryRegistryData.sizeOfElement_;
        ss << left << setw( 6) << StringAuxiliaryFunctions::formatNumberString(index, memoryRegistry.size())
                   << setw(16) << reinterpret_cast<void*>(memoryRegistryData.ptr_) // Note: uint8_t (ie char) will confuse C++ and crash the log
                   << setw(14) << memoryRegistryData.numberOfElements_
                   << setw(10) << memoryRegistryData.sizeOfElement_
                   << setw(14) << totalBytesConsumed << '\n';
        totalRegistryBytesConsumed += totalBytesConsumed;
      }
      ss << '\n';
      ss << "Total Memory Registry bytes consumed: " << totalRegistryBytesConsumed << '\n';
    }
    else
    {
      ss << "Note: Memory Registry is empty." << '\n';
    }
    ss << '\n';
    DebugConsole_consoleOutLine(ss.str());
  }
}

CUDAMemoryRegistry::~CUDAMemoryRegistry() noexcept
{
  if (!memoryRegistryNames_.empty())
  {
    // use the memory registry for unregistering (unpinning) the host memory with CUDA (if necessary)
    unregisterMemoryRegistry();
  }
}

void CUDAMemoryRegistry::registerMemoryRegistry(const string& name, unsigned int flags)
{
  if (!memoryRegistryNames_.empty())
  {
    for (const string& registryName : memoryRegistryNames_)
    {
      const auto& memoryRegistryData = memoryRegistry_.at(registryName);
      if (memoryRegistryData.ptr_ != nullptr) // ptr to register must be != null
      {
        // register host memory below
        size_t bytesToRegister = memoryRegistryData.numberOfElements_ * memoryRegistryData.sizeOfElement_;
        CUDAError_checkCUDAError(cudaHostRegister(memoryRegistryData.ptr_, bytesToRegister, flags));
      }
    }
    isRegistered_ = true;
  }
  else
  {
    DebugConsole_consoleOutLine("CUDAMemoryRegistry::registerMemoryRegistry warning:\n  Memory Registry is empty.");
  }

  // Note: Report the current Host pointer(s) from the Memory Registry
  CUDADriverInfo_report(reportMemoryRegistryInformation(name));
}

void CUDAMemoryRegistry::unregisterMemoryRegistry()
{
  if (!memoryRegistryNames_.empty())
  {
    for (const string& registryName : memoryRegistryNames_)
    {
      auto& memoryRegistryData = memoryRegistry_.at(registryName);
      if (memoryRegistryData.ptr_ != nullptr)
      {
        // unregister host memory below
        CUDAError_checkCUDAError(cudaHostUnregister(memoryRegistryData.ptr_));
        memoryRegistryData.ptr_ = nullptr;
      }
    }
    memoryRegistryNames_.clear();
    memoryRegistry_.clear();
    isRegistered_ = false;
  }
  else
  {
    DebugConsole_consoleOutLine("CUDAMemoryRegistry::unregisterAndClearMemoryRegistry warning:\n  Memory Registry is already empty.");
  }

  // Note: Report the current Host pointer(s) from the Memory Registry
  CUDADriverInfo_report(reportMemoryRegistryInformation());
}

bool CUDAMemoryRegistry::addToMemoryRegistryPtr(const string& name, uint8_t* ptr, size_t numberOfElements, size_t sizeOfElement)
{
  if (isRegistered_)
  {
    DebugConsole_consoleOutLine("Memory Registry error: cannot add a ptr when the Registry is already registered.");
    return false;
  }

  if (get<1>(memoryRegistry_.emplace(move(name), MemoryRegistryData{ptr, numberOfElements, sizeOfElement})))
  {
    memoryRegistryNames_.emplace_back(move(name));
    return true;
  }

  DebugConsole_consoleOutLine("Memory Registry error: name ", name, " already exists in the Memory Registry.");
  return false;
}

CUDAMemoryRegistry::MemoryRegistryData CUDAMemoryRegistry::getFromMemoryRegistryPtr(const string& name) const
{
  if (!isRegistered_)
  {
    DebugConsole_consoleOutLine("Memory Registry error: cannot get a ptr when the Registry is not registered.");
    return MemoryRegistryData{nullptr, 0, 0};
  }

  const auto& position = memoryRegistry_.find(name);
  if (position != memoryRegistry_.end())
  {
    return memoryRegistry_.at(name);
  }

  return MemoryRegistryData{nullptr, 0, 0};
}

void CUDAMemoryRegistry::reportMemoryRegistryInformation(const string& name) const
{
  reportMemoryRegistryInformationInternal(memoryRegistry_, memoryRegistryNames_, name);
}