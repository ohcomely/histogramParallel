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

#ifndef __CUDADriverInfo_h
#define __CUDADriverInfo_h

#include "ModuleDLL.h"
#include <cuda_runtime_api.h>
#include <string>
#include <cstdint>
#include <memory>

/** @brief namespace UtilsCUDA for encapsulating all the CUDA related code compiled by the NVCC compiler.
*
* @author Thanos Theo, 2018
* @version 14.0.0.0
*/
namespace UtilsCUDA
{
  /** @brief This class encapsulates CUDA driver info for detection & reporting.
  *
  *  CUDADriverInfo.h:
  *  ================
  *  This class encapsulates CUDA driver info for detection & reporting.
  *
  * @author Thanos Theo, 2018
  * @version 14.0.0.0
  */
  class UTILS_CUDA_MODULE_API CUDADriverInfo final
  {
  public:
    /// enum for Memory Pool Types
    enum class GPUTypes : std::size_t
    {
      FERMI   = 0,
      KEPLER  = 1,
      MAXWELL = 2,
      PASCAL  = 3,
      VOLTA   = 4,
      TURING  = 5,
      AMPERE  = 6
    };
    /// Device is the given GPU type
    bool isGPUType(GPUTypes type, int device) const noexcept;
    /// Device is at least the given GPU type
    bool isAtLeastGPUType(GPUTypes type, int device) const noexcept;
    /// CUDA driver version
    int getDriverVersion() const noexcept;
    /// CUDA runtime version
    int getRuntimeVersion() const noexcept;
    /// CUDA device count
    int getDeviceCount() const noexcept;
    /// Device support for Dynamic Parallelism
    bool hasDynamicParallelism(int device) const noexcept;
    /// Device support for Unified Memory
    bool hasUnifiedMemory(int device) const noexcept;
    /// ASCII string identifying device
    std::string getName(int device) const noexcept;
    /// Global memory available on device in bytes
    std::size_t getTotalGlobalMemory(int device) const noexcept;
    /// Shared memory available per block in bytes
    std::size_t getSharedMemoryPerBlock(int device) const noexcept;
    /// 32-bit registers available per block
    int getRegistersPerBlock(int device) const noexcept;
    /// Warp size in threads
    int getWarpSize(int device) const noexcept;
    /// Maximum pitch in bytes allowed by memory copies
    std::size_t getMemoryPitch(int device) const noexcept;
    /// Maximum number of threads per block
    int getMaxThreadsPerBlock(int device) const noexcept;
    /// Maximum size of each dimension of a block
    const int* getMaxThreadsDimension(int device) const noexcept;
    /// Maximum size of each dimension of a grid
    const int* getMaxGridSize(int device) const noexcept;
    /// Clock frequency in kilohertz
    int getClockRate(int device) const noexcept;
    /// Constant memory available on device in bytes
    std::size_t getTotalConstMemory(int device) const noexcept;
    /// Major compute capability
    int getMajorVersion(int device) const noexcept;
    /// Minor compute capability
    int getMinorVersion(int device) const noexcept;
    /// Alignment requirement for textures
    std::size_t getTextureAlignment(int device) const noexcept;
    /// Pitch alignment requirement for texture references bound to pitched memory
    std::size_t getTexturePitchAlignment(int device) const noexcept;
    /// Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount.
    int getDeviceOverlap(int device) const noexcept;
    /// Number of multiprocessors on device
    int getMultiProcessorCount(int device) const noexcept;
    /// Specified whether there is a run time limit on kernels
    int getKernelExecTimeoutEnabled(int device) const noexcept;
    /// Device is integrated as opposed to discrete
    int getIntegrated(int device) const noexcept;
    /// Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer
    int getCanMapHostMemory(int device) const noexcept;
    /// Compute mode (See ::cudaComputeMode)
    int getComputeMode(int device) const noexcept;
    /// Maximum 1D texture size
    int  getMaxTexture1D(int device) const noexcept;
    /// Maximum 1D mipmapped texture size
    int  getMaxTexture1DMipmap(int device) const noexcept;
    /// Maximum size for 1D textures bound to linear memory
    int  getMaxTexture1DLinear(int device) const noexcept;
    /// Maximum 2D texture dimensions
    const int* getMaxTexture2D(int device) const noexcept;
    /// Maximum 2D mipmapped texture dimensions
    const int* getMaxTexture2DMipmap(int device) const noexcept;
    /// Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory
    const int* getMaxTexture2DLinear(int device) const noexcept;
    /// Maximum 2D texture dimensions if texture gather operations have to be performed
    const int* getMaxTexture2DGather(int device) const noexcept;
    /// Maximum 3D texture dimensions
    const int* getMaxTexture3D(int device) const noexcept;
    /// Maximum alternate 3D texture dimensions
    const int* getMaxTexture3DAlt(int device) const noexcept;
    /// Maximum Cubemap texture dimensions
    int  getMaxTextureCubemap(int device) const noexcept;
    /// Maximum 1D layered texture dimensions
    const int* getMaxTexture1DLayered(int device) const noexcept;
    /// Maximum 2D layered texture dimensions
    const int* getMaxTexture2DLayered(int device) const noexcept;
    /// Maximum Cubemap layered texture dimensions
    const int* getMaxTextureCubemapLayered(int device) const noexcept;
    /// Maximum 1D surface size
    int  getMaxSurface1D(int device) const noexcept;
    /// Maximum 2D surface dimensions
    const int* getMaxSurface2D(int device) const noexcept;
    /// Maximum 3D surface dimensions
    const int* getMaxSurface3D(int device) const noexcept;
    /// Maximum 1D layered surface dimensions
    const int* getMaxSurface1DLayered(int device) const noexcept;
    /// Maximum 2D layered surface dimensions
    const int* getMaxSurface2DLayered(int device) const noexcept;
    /// Maximum Cubemap surface dimensions
    int  getMaxSurfaceCubemap(int device) const noexcept;
    /// Maximum Cubemap layered surface dimensions
    const int* getMaxSurfaceCubemapLayered(int device) const noexcept;
    /// Alignment requirements for surfaces
    std::size_t getSurfaceAlignment(int device) const noexcept;
    /// Device can possibly execute multiple kernels concurrently
    int getConcurrentKernels(int device) const noexcept;
    /// Device can coherently access managed memory concurrently with the CPU
    int getConcurrentManagedAccess(int device) const noexcept;
    /// Link between the device and the host supports native atomic operations
    int getHostNativeAtomicSupported(int device) const noexcept;
    /// Device has ECC support enabled
    int getECCEnabled(int device) const noexcept;
    /// PCI bus ID of the device
    int getPciBusID(int device) const noexcept;
    /// PCI device ID of the device
    int getPciDeviceID(int device) const noexcept;
    /// PCI domain ID of the device
    int getPciDomainID(int device) const noexcept;
    /// 1 if device is a Tesla device using TCC driver, 0 otherwise
    int getTccDriver(int device) const noexcept;
    /// Number of asynchronous engines
    int getAsyncEngineCount(int device) const noexcept;
    /// Device shares a unified address space with the host
    int getUnifiedAddressing(int device) const noexcept;
    /// Peak memory clock frequency in kilohertz
    int getMemoryClockRate(int device) const noexcept;
    /// Global memory bus width in bits
    int getMemoryBusWidth(int device) const noexcept;
    /// Size of L2 cache in bytes
    int getL2CacheSize(int device) const noexcept;
    /// Maximum resident threads per multiprocessor
    int getMaxThreadsPerMultiProcessor(int device) const noexcept;
    /// Device supports stream priorities
    int getStreamPrioritiesSupported(int device) const noexcept;
    /// Device supports caching globals in L1
    int getGlobalL1CacheSupported(int device) const noexcept;
    /// Device supports caching locals in L1
    int getLocalL1CacheSupported(int device) const noexcept;
    /// Shared memory available per multiprocessor in bytes
    std::size_t getSharedMemoryPerMultiprocessor(int device) const noexcept;
    /// 32-bit registers available per multiprocessor
    int getRegistersPerMultiprocessor(int device) const noexcept;
    /// Device supports allocating managed memory on this system
    int getManagedMemory(int device) const noexcept;
    /// Device is on a multi-GPU board
    int getIsMultiGpuBoard(int device) const noexcept;
    /// Unique identifier for a group of devices on the same multi-GPU board
    int getMultiGpuBoardGroupID(int device) const noexcept;

    /// Reports the full CUDA Driver Info
    void reportCUDADriverInfo() const noexcept;

    explicit CUDADriverInfo(unsigned int deviceFlags = cudaDeviceScheduleAuto, bool enableProfiling = false) noexcept;
    ~CUDADriverInfo() noexcept; // no virtual destructor for data-oriented design (no up-casting should ever be used)
    CUDADriverInfo(const CUDADriverInfo&) = delete;
    CUDADriverInfo(CUDADriverInfo&&)      = delete;
    CUDADriverInfo& operator=(const CUDADriverInfo&) = delete;
    CUDADriverInfo& operator=(CUDADriverInfo&&)      = delete;

  private:

    bool enableProfiling_   = false;
    int cudaDriverVersion_  = 0;
    int cudaRuntimeVersion_ = 0;
    int cudaDeviceCount_    = 0;
    std::unique_ptr<cudaDeviceProp[]> allCudaDevicesProperties_ = nullptr;

    std::string getGPUArchitecture(int device) const noexcept;
    void reportCUDAPlatformVersions() const noexcept;
    void reportCUDADeviceCapabilities(int device) const noexcept;
  };
} // namespace UtilsCUDA

#endif // __CUDADriverInfo_h