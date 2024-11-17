#include "CUDADriverInfo.h"
#include "CUDAUtilityFunctions.h"
#include "UtilityFunctions.h"
#include <cuda_profiler_api.h>
#include <algorithm>
#include <sstream>
#include <cassert>
#include <iomanip>

using namespace std;
using namespace UtilsCUDA;
using namespace Utils::UtilityFunctions;

namespace // anonymous namespace used instead of deprecated 'static' keyword used for cpp variable locality
{
#ifndef NDEBUG
  inline bool checkCUDADeviceIsValid(int device, int cudaDeviceCount)
  {
    return ((cudaDeviceCount > 0) && (device < cudaDeviceCount));
  }
#endif // NDEBUG
}

CUDADriverInfo::CUDADriverInfo(unsigned int deviceFlags, bool enableProfiling) noexcept
  : enableProfiling_(enableProfiling)
{
  CUDAError_checkCUDAError(cudaGetDeviceCount(&cudaDeviceCount_));
  CUDAError_checkCUDAError(cudaDriverGetVersion(&cudaDriverVersion_));
  CUDAError_checkCUDAError(cudaRuntimeGetVersion(&cudaRuntimeVersion_));
  CUDAError_checkCUDAError(cudaGetDeviceCount(&cudaDeviceCount_));

  // make sure to enforce the default constructor of the cudaDeviceProp struct (C++03 array initialization syntax) via the make_unique() call
  allCudaDevicesProperties_ = make_unique<cudaDeviceProp[]>(max<size_t>(1, cudaDeviceCount_));

  // report all relevant CUDA information
  CUDADriverInfo_report(reportCUDAPlatformVersions());
  for (int device = 0; device < cudaDeviceCount_; ++device)
  {
    CUDAError_checkCUDAError(cudaGetDeviceProperties(&allCudaDevicesProperties_[device], device));
    CUDADriverInfo_report(reportCUDADeviceCapabilities(device));
  }

  // conditionally use custom CUDA device flags
  if (deviceFlags != cudaDeviceScheduleAuto)
  {
    int originalDevice = 0;
    if (cudaDeviceCount_ > 1)
    {
      CUDAError_checkCUDAError(cudaGetDevice(&originalDevice));
    }

    for (int device = 0; device < cudaDeviceCount_; ++device)
    {
      CUDAError_checkCUDAError(cudaSetDevice(device));
      CUDAError_checkCUDAError(cudaSetDeviceFlags(deviceFlags));
    }

    // original device, if overwritten, will be restored
    if (cudaDeviceCount_ > 1)
    {
      CUDAError_checkCUDAError(cudaSetDevice(originalDevice));
    }
  }

  // conditionally enable CUDA profiling
  if (enableProfiling_)
  {
    CUDAError_checkCUDAErrorDebug(cudaProfilerStart());
  }

  // Note: the canonical way to force runtime API context establishment is to call 'cudaFree(nullptr)'
  CUDAError_checkCUDAError(cudaFree(nullptr));
}

CUDADriverInfo::~CUDADriverInfo() noexcept
{
  if (enableProfiling_)
  {
    CUDAError_checkCUDAErrorDebug(cudaProfilerStop());

    // Note: cudaDeviceReset() must be called before exiting the CUDA context in order for profiling
    // and tracing tools such as Nsight and the NVidia Visual Profiler to show complete traces
    CUDAError_checkCUDAError(cudaDeviceReset());
  }
}

string CUDADriverInfo::getGPUArchitecture(int device) const noexcept
{
  const int major = getMajorVersion(device);
  const int minor = getMinorVersion(device);

  switch (major)
  {
  case 2:
    return "Fermi";

  case 3:
    return "Kepler";

  case 5:
    return "Maxwell";

  case 6:
    return "Pascal";

  case 7:
    return (minor < 5) ? "Volta" : "Turing";
  case 8:
    return "Ampere";

  default:
    CUDAError_checkCUDAError(cudaErrorUnknown);
    return "Unsupported GPU Architecture";
  }
}

void CUDADriverInfo::reportCUDAPlatformVersions() const noexcept
{
  ostringstream ss;
  ss << '\n';
  ss << "   --- CUDA Platform Information ---" << '\n';
  ss << "Driver Version:  " << getDriverVersion() << '\n';
  ss << "Runtime Version: " << getRuntimeVersion() << '\n';
  DebugConsole_consoleOutLine(ss.str());
}

void CUDADriverInfo::reportCUDADeviceCapabilities(int device) const noexcept
{
  ostringstream ss;
  ss << "   --- GPU Architecture Information for CUDA device " << device << " ---" << '\n';
  ss << "GPU Architecture:   " << getGPUArchitecture(device)   << '\n';
  ss << "Has Unified Memory: " << StringAuxiliaryFunctions::toString<bool>(hasUnifiedMemory(device)) << '\n';

  ss << '\n';
  ss << "   --- General Information for CUDA device " << device << " ---"                                                          << '\n';
  ss << "Name:                     " <<  allCudaDevicesProperties_[device].name                                                    << '\n';
  ss << "Compute Capability:       " <<  allCudaDevicesProperties_[device].major << "." << allCudaDevicesProperties_[device].minor << '\n';
  ss << "Clock Rate:                   " <<  allCudaDevicesProperties_[device].clockRate / 1000                              << " Mhz" << '\n';
  if (enableProfiling_)
  {
    ss << "Device Copy Overlap:          " <<  StringAuxiliaryFunctions::toString<bool>(allCudaDevicesProperties_[device].deviceOverlap             > 0) << '\n';
    ss << "Kernel Execution Timeout:     " <<  StringAuxiliaryFunctions::toString<bool>(allCudaDevicesProperties_[device].kernelExecTimeoutEnabled  > 0) << '\n';
    ss << "Concurrently Kernels:         " <<  StringAuxiliaryFunctions::toString<bool>(allCudaDevicesProperties_[device].concurrentKernels         > 0) << '\n';
    ss << "Concurrent Managed Access:    " <<  StringAuxiliaryFunctions::toString<bool>(allCudaDevicesProperties_[device].concurrentManagedAccess   > 0) << '\n';
    ss << "Host Native Atomic Supported: " <<  StringAuxiliaryFunctions::toString<bool>(allCudaDevicesProperties_[device].hostNativeAtomicSupported > 0) << '\n';
    ss << "Device Is Integrated:         " <<  StringAuxiliaryFunctions::toString<bool>(allCudaDevicesProperties_[device].integrated                > 0) << '\n';
    ss << "ECC Support Enabled:          " <<  StringAuxiliaryFunctions::toString<bool>(allCudaDevicesProperties_[device].ECCEnabled                > 0) << '\n';
    ss << "Memory Clock Rate:            " <<  left << setw(6) <<  allCudaDevicesProperties_[device].memoryClockRate / 1000 << "Mhz"  << '\n';
    ss << "Memory Bus Width:             " <<  left << setw(6) <<  allCudaDevicesProperties_[device].memoryBusWidth         << "bits" << '\n';
    ss << "L2 Cache Size:                " <<  left << setw(6) << (allCudaDevicesProperties_[device].l2CacheSize >> 10)     << "Kb"   << '\n';
    ss << "Asynchronous Engines:         " <<  allCudaDevicesProperties_[device].asyncEngineCount                   << '\n';
    ss << "Unified Addressing:           " <<  StringAuxiliaryFunctions::toString<bool>(allCudaDevicesProperties_[device].unifiedAddressing         > 0) << '\n';
    ss << "Global L1 cache:              " <<  StringAuxiliaryFunctions::toString<bool>(allCudaDevicesProperties_[device].globalL1CacheSupported    > 0) << '\n';
    ss << "Local  L1 cache:              " <<  StringAuxiliaryFunctions::toString<bool>(allCudaDevicesProperties_[device].localL1CacheSupported     > 0) << '\n';
    ss << "Multi GPU board:              " <<  StringAuxiliaryFunctions::toString<bool>(allCudaDevicesProperties_[device].isMultiGpuBoard           > 0) << '\n';
  }

  ss << '\n';
  ss << "   --- Memory Information for CUDA device " << device << " ---"                                 << '\n';
  ss << "Total Global Memory:   " << left << setw(6) << (allCudaDevicesProperties_[device].totalGlobalMem >> 20) << "Mb" << '\n';
  ss << "Total Constant Memory: " << left << setw(6) << (allCudaDevicesProperties_[device].totalConstMem  >> 10) << "Kb" << '\n';
  ss << "Max Memory Pitch:      " << left << setw(6) << (allCudaDevicesProperties_[device].memPitch       >> 20) << "Mb" << '\n';
  if (enableProfiling_)
  {
    ss << "Texture Alignment:     " <<  allCudaDevicesProperties_[device].textureAlignment                 << '\n';
    ss << "Managed Memory:        " <<  StringAuxiliaryFunctions::toString<bool>(allCudaDevicesProperties_[device].managedMemory > 0) << '\n';
  }

  ss << '\n';
  ss << "   --- MultiProcessor (MP) Information for CUDA device " << device << " ---"                                  << '\n';
  ss << "MultiProcessor Count:    " <<  allCudaDevicesProperties_[device].multiProcessorCount                          << '\n';
  ss << "Shared Memory Per Block: " << (allCudaDevicesProperties_[device].sharedMemPerBlock >> 10) << " Kb"            << '\n';
  ss << "Registers Per Block:     " <<  allCudaDevicesProperties_[device].regsPerBlock                                 << '\n';
  ss << "Threads In Warp:         " <<  allCudaDevicesProperties_[device].warpSize                                     << '\n';
  ss << "Max Threads Per Block:   " <<  allCudaDevicesProperties_[device].maxThreadsPerBlock                           << '\n';
  ss << "Max Threads Per MP:      " <<  allCudaDevicesProperties_[device].maxThreadsPerMultiProcessor                  << '\n';
  if (enableProfiling_)
  {
    ss << "Max Block Dimensions:   (" <<  allCudaDevicesProperties_[device].maxThreadsDim[0]                     << ", " <<
                                                              allCudaDevicesProperties_[device].maxThreadsDim[1] << ", " <<
                                                              allCudaDevicesProperties_[device].maxThreadsDim[2] << ")"  << '\n';
    ss << "Max Grid Dimensions:    (" <<  allCudaDevicesProperties_[device].maxGridSize[0]                       << ", " <<
                                                              allCudaDevicesProperties_[device].maxGridSize[1]   << ", " <<
                                                              allCudaDevicesProperties_[device].maxGridSize[2]   << ")"  << '\n';
  }
  ss << '\n';
  DebugConsole_consoleOutLine(ss.str());
}

void CUDADriverInfo::reportCUDADriverInfo() const noexcept
{
  CUDADriverInfo_report(reportCUDAPlatformVersions());
  for (int i = 0; i < cudaDeviceCount_; ++i)
  {
    CUDADriverInfo_report(reportCUDADeviceCapabilities(i));
  }
}

bool CUDADriverInfo::isGPUType(GPUTypes type, int device) const noexcept
{
  const int major = getMajorVersion(device);
  const int minor = getMinorVersion(device);

  switch (type)
  {
  case GPUTypes::FERMI:
    return major == 2;

  case GPUTypes::KEPLER:
    return major == 3;

  case GPUTypes::MAXWELL:
    return major == 5;

  case GPUTypes::PASCAL:
    return major == 6;

  case GPUTypes::VOLTA:
    return major == 7 && minor < 5;

  case GPUTypes::TURING:
    return major == 7 && minor >= 5;

  case GPUTypes::AMPERE:
    return major == 8;

  default:
    return false;
  }
}

bool CUDADriverInfo::isAtLeastGPUType(GPUTypes type, int device) const noexcept
{
  const int major = getMajorVersion(device);
  const int minor = getMinorVersion(device);

  switch (type)
  {
  case GPUTypes::FERMI:
    return major >= 2;

  case GPUTypes::KEPLER:
    return major >= 3;

  case GPUTypes::MAXWELL:
    return major >= 5;

  case GPUTypes::PASCAL:
    return major >= 6;

  case GPUTypes::VOLTA:
    return major >= 7;

  case GPUTypes::TURING:
    return major >= 7 && minor >= 5;

  case GPUTypes::AMPERE:
    return major >= 8;

  default:
    return false;
  }
}

int CUDADriverInfo::getDriverVersion() const noexcept
{
  return cudaDriverVersion_;
}

int CUDADriverInfo::getRuntimeVersion() const noexcept
{
  return cudaRuntimeVersion_;
}

int CUDADriverInfo::getDeviceCount() const noexcept
{
  return cudaDeviceCount_;
}

bool CUDADriverInfo::hasDynamicParallelism(int device) const noexcept
{
  // minimum compute capability expected of at least a Kepler Titan GPU of Compute Capability 3.5 for Dynamic Parallelism support or greater
  return isAtLeastGPUType(GPUTypes::MAXWELL, device) || (isAtLeastGPUType(GPUTypes::KEPLER, device) && getMinorVersion(device) >= 5);

}

bool CUDADriverInfo::hasUnifiedMemory(int device) const noexcept
{
  return getUnifiedAddressing(device) && getManagedMemory(device);
}

string CUDADriverInfo::getName(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return string(allCudaDevicesProperties_[device].name);
}

size_t CUDADriverInfo::getTotalGlobalMemory(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].totalGlobalMem;
}

size_t CUDADriverInfo::getSharedMemoryPerBlock(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].sharedMemPerBlock;
}

int CUDADriverInfo::getRegistersPerBlock(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].regsPerBlock;
}

int CUDADriverInfo::getWarpSize(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].warpSize;
}

size_t CUDADriverInfo::getMemoryPitch(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].memPitch;
}

int CUDADriverInfo::getMaxThreadsPerBlock(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].maxThreadsPerBlock;
}

const int* CUDADriverInfo::getMaxThreadsDimension(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].maxThreadsDim;
}

const int* CUDADriverInfo::getMaxGridSize(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].maxGridSize;
}

int CUDADriverInfo::getClockRate(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].clockRate;
}

size_t CUDADriverInfo::getTotalConstMemory(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].totalConstMem;
}

int CUDADriverInfo::getMajorVersion(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].major;
}

int CUDADriverInfo::getMinorVersion(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].minor;
}

size_t CUDADriverInfo::getTextureAlignment(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].textureAlignment;
}

size_t CUDADriverInfo::getTexturePitchAlignment(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].texturePitchAlignment;
}

int CUDADriverInfo::getDeviceOverlap(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].deviceOverlap;
}

int CUDADriverInfo::getMultiProcessorCount(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].multiProcessorCount;
}

int CUDADriverInfo::getKernelExecTimeoutEnabled(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].kernelExecTimeoutEnabled;
}

int CUDADriverInfo::getIntegrated(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].integrated;
}

int CUDADriverInfo::getCanMapHostMemory(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].canMapHostMemory;
}

int CUDADriverInfo::getComputeMode(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].computeMode;
}

int  CUDADriverInfo::getMaxTexture1D(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].maxTexture1D;
}

int  CUDADriverInfo::getMaxTexture1DMipmap(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].maxTexture1DMipmap;
}

int  CUDADriverInfo::getMaxTexture1DLinear(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].maxTexture1DLinear;
}

const int* CUDADriverInfo::getMaxTexture2D(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].maxTexture2D;
}

const int* CUDADriverInfo::getMaxTexture2DMipmap(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].maxTexture2DMipmap;
}

const int* CUDADriverInfo::getMaxTexture2DLinear(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].maxTexture2DLinear;
}

const int* CUDADriverInfo::getMaxTexture2DGather(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].maxTexture2DGather;
}

const int* CUDADriverInfo::getMaxTexture3D(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].maxTexture3D;
}

const int* CUDADriverInfo::getMaxTexture3DAlt(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].maxTexture3DAlt;
}

int  CUDADriverInfo::getMaxTextureCubemap(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].maxTextureCubemap;
}

const int* CUDADriverInfo::getMaxTexture1DLayered(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].maxTexture1DLayered;
}

const int* CUDADriverInfo::getMaxTexture2DLayered(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].maxTexture2DLayered;
}

const int* CUDADriverInfo::getMaxTextureCubemapLayered(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].maxTextureCubemapLayered;
}

int  CUDADriverInfo::getMaxSurface1D(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].maxSurface1D;
}

const int* CUDADriverInfo::getMaxSurface2D(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].maxSurface2D;
}

const int* CUDADriverInfo::getMaxSurface3D(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].maxSurface3D;
}

const int* CUDADriverInfo::getMaxSurface1DLayered(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].maxSurface1DLayered;
}

const int* CUDADriverInfo::getMaxSurface2DLayered(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].maxSurface2DLayered;
}

int  CUDADriverInfo::getMaxSurfaceCubemap(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].maxSurfaceCubemap;
}

const int* CUDADriverInfo::getMaxSurfaceCubemapLayered(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].maxSurfaceCubemapLayered;
}

size_t CUDADriverInfo::getSurfaceAlignment(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].surfaceAlignment;
}

int CUDADriverInfo::getConcurrentKernels(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].concurrentKernels;
}

int CUDADriverInfo::getConcurrentManagedAccess(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].concurrentManagedAccess;
}

int CUDADriverInfo::getHostNativeAtomicSupported(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].hostNativeAtomicSupported;
}

int CUDADriverInfo::getECCEnabled(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].ECCEnabled;
}

int CUDADriverInfo::getPciBusID(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].pciBusID;
}

int CUDADriverInfo::getPciDeviceID(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].pciDeviceID;
}

int CUDADriverInfo::getPciDomainID(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].pciDomainID;
}

int CUDADriverInfo::getTccDriver(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].tccDriver;
}

int CUDADriverInfo::getAsyncEngineCount(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].asyncEngineCount;
}

int CUDADriverInfo::getUnifiedAddressing(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].unifiedAddressing;
}

int CUDADriverInfo::getMemoryClockRate(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].memoryClockRate;
}

int CUDADriverInfo::getMemoryBusWidth(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].memoryBusWidth;
}

int CUDADriverInfo::getL2CacheSize(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].l2CacheSize;
}

int CUDADriverInfo::getMaxThreadsPerMultiProcessor(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].maxThreadsPerMultiProcessor;
}

int CUDADriverInfo::getStreamPrioritiesSupported(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].streamPrioritiesSupported;
}

int CUDADriverInfo::getGlobalL1CacheSupported(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].globalL1CacheSupported;
}

int CUDADriverInfo::getLocalL1CacheSupported(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].localL1CacheSupported;
}

size_t CUDADriverInfo::getSharedMemoryPerMultiprocessor(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].sharedMemPerMultiprocessor;
}

int CUDADriverInfo::getRegistersPerMultiprocessor(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].regsPerMultiprocessor;
}

int CUDADriverInfo::getManagedMemory(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].managedMemory;
}

int CUDADriverInfo::getIsMultiGpuBoard(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].isMultiGpuBoard;
}

int CUDADriverInfo::getMultiGpuBoardGroupID(int device) const noexcept
{
  assert(checkCUDADeviceIsValid(device, cudaDeviceCount_));
  return allCudaDevicesProperties_[device].multiGpuBoardGroupID;
}