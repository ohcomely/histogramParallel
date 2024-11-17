#include "CUDAUtilityFunctions.h"
#include "EnvironmentConfig.h"
#include "UtilityFunctions.h"
#include <string>
#include <cassert>
#ifdef GPU_FRAMEWORK_USE_EXCEPTION_FOR_CUDA_ERROR
  #include <system_error>
#else
  #include <cstdlib>
#endif // GPU_FRAMEWORK_USE_EXCEPTION_FOR_CUDA_ERROR

#ifdef GPU_FRAMEWORK_USE_EXCEPTION_FOR_CUDA_ERROR
// std template specialization to make make_error_code() below work auto-magically
namespace std
{
  template<> struct is_error_code_enum<cudaError_t> : true_type{};
}

// make_error_code() has to be in the same namespace as cudaError_t, ie the global namespace
error_code make_error_code(cudaError_t e)
{
  return {int(e), cudaErrorCategory};
}
#endif // GPU_FRAMEWORK_USE_EXCEPTION_FOR_CUDA_ERROR

using namespace std;
using namespace UtilsCUDA;
using namespace Utils::UtilityFunctions;

namespace // anonymous namespace used instead of deprecated 'static' keyword used for cpp variable locality
{
  const uint8_t  CUDA_EXIT_CODE_OFFSET = 128;
  const uint32_t WARP_SIZE     = 32;
  const uint32_t SIZE_X_KERNEL = WARP_SIZE;
  const uint32_t SIZE_Y_KERNEL = WARP_SIZE / 4;

#ifdef GPU_FRAMEWORK_USE_EXCEPTION_FOR_CUDA_ERROR
  struct CUDAErrorCategory : public error_category
  {
    const char* name() const noexcept override { return "CUDAError"; }
    string message(int ev) const override;
  };

  string CUDAErrorCategory::message(int ev) const
  {
    return cudaGetErrorString(cudaError_t(ev));
  }

  const CUDAErrorCategory cudaErrorCategory{};
#endif // GPU_FRAMEWORK_USE_EXCEPTION_FOR_CUDA_ERROR

  inline const char* curandGetErrorString(const curandStatus_t& error)
  {
    switch (error)
    {
      case CURAND_STATUS_SUCCESS:
        return "CURAND_STATUS_SUCCESS";

      case CURAND_STATUS_VERSION_MISMATCH:
        return "CURAND_STATUS_VERSION_MISMATCH";

      case CURAND_STATUS_NOT_INITIALIZED:
        return "CURAND_STATUS_NOT_INITIALIZED";

      case CURAND_STATUS_ALLOCATION_FAILED:
        return "CURAND_STATUS_ALLOCATION_FAILED";

      case CURAND_STATUS_TYPE_ERROR:
        return "CURAND_STATUS_TYPE_ERROR";

      case CURAND_STATUS_OUT_OF_RANGE:
        return "CURAND_STATUS_OUT_OF_RANGE";

      case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
        return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";

      case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
        return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";

      case CURAND_STATUS_LAUNCH_FAILURE:
        return "CURAND_STATUS_LAUNCH_FAILURE";

      case CURAND_STATUS_PREEXISTING_FAILURE:
        return "CURAND_STATUS_PREEXISTING_FAILURE";

      case CURAND_STATUS_INITIALIZATION_FAILED:
        return "CURAND_STATUS_INITIALIZATION_FAILED";

      case CURAND_STATUS_ARCH_MISMATCH:
        return "CURAND_STATUS_ARCH_MISMATCH";

      case CURAND_STATUS_INTERNAL_ERROR:
        return "CURAND_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
  }

  inline void reportError(const string& error, int errnum, const char* file, const char* function, int line, bool abort)
  {
    ostringstream ss;
    ss << "\nFile: "   << file     << '\n';
    ss << "Function: " << function << '\n';
    ss << "Line: "     << line     << '\n';
    ss << "CUDA error reported: "  << error << " (" << errnum << ")" << '\n';
    DebugConsole_consoleOutLine(ss.str());
    if (abort)
    {
  #ifdef GPU_FRAMEWORK_USE_EXCEPTION_FOR_CUDA_ERROR
      throw system_error(errnum, ss.str()); // auto-magically use the make_error_code() from above
  #else
      exit(CUDA_EXIT_CODE_OFFSET + errnum);
  #endif // GPU_FRAMEWORK_USE_EXCEPTION_FOR_CUDA_ERROR
    }
  }
}

uint8_t CUDAUtilityFunctions::getCUDAExitCodeOffset()
{
  return CUDA_EXIT_CODE_OFFSET;
}

void CUDAUtilityFunctions::checkCUDAErrorImpl(const cudaError_t& errnum, const char* file, const char* function, int line, bool abort)
{
  if (errnum)
  {
    return reportError(cudaGetErrorString(errnum), errnum, file, function, line, abort);
  }
}

void CUDAUtilityFunctions::checkCUDAErrorImpl(const CUresult& errnum, const char* file, const char* function, int line, bool abort)
{
  if (errnum)
  {
    const char* errorStr = "<unknown>";
    // cuGetErrorString(errnum, &errorStr); // need to enable proper CUDA driver linkage
    return reportError(errorStr, errnum, file, function, line, abort);
  }
}

void CUDAUtilityFunctions::checkCUDAErrorImpl(const curandStatus_t& errnum, const char* file, const char* function, int line, bool abort)
{
  if (errnum)
  {
    return reportError(curandGetErrorString(errnum), errnum, file, function, line, abort);
  }
}

uint32_t CUDAUtilityFunctions::getWarpSize()
{
  return WARP_SIZE;
}

dim3 CUDAUtilityFunctions::getDefaultThreads1DDimensions()
{
  return dim3{ SIZE_X_KERNEL * SIZE_Y_KERNEL };
}

dim3 CUDAUtilityFunctions::getDefaultThreads2DDimensions()
{
  return dim3{ SIZE_X_KERNEL, SIZE_Y_KERNEL };
}

tuple<dim3, dim3> CUDAUtilityFunctions::calculateCUDA1DKernelDimensions(size_t arraySize, const dim3& threads1D)
{
  return make_tuple(dim3(uint32_t((arraySize + threads1D.x - 1) / threads1D.x)),
                    threads1D);
}

tuple<dim3, dim3> CUDAUtilityFunctions::calculateCUDA2DKernelDimensions(size_t arraySize, const dim3& threads2D)
{
  uint32_t sqrtSize = uint32_t(ceil(sqrt(arraySize)));
  return calculateCUDA2DKernelDimensionsXY(sqrtSize, sqrtSize, threads2D);
}

tuple<dim3, dim3> CUDAUtilityFunctions::calculateCUDA2DKernelDimensionsXY(size_t arraySizeX, size_t arraySizeY, const dim3& threads2D)
{
  return make_tuple(dim3{uint32_t((arraySizeX + threads2D.x - 1) / threads2D.x),
                         uint32_t((arraySizeY + threads2D.y - 1) / threads2D.y)},
                    threads2D);
}

tuple<dim3, dim3> CUDAUtilityFunctions::calculateCUDAPersistentKernel(const CUDADriverInfo& cudaDriverInfo, int device, uint32_t threadsPerBlock, uint32_t sharedMemoryPerBlock)
{
    assert(threadsPerBlock <= uint32_t(cudaDriverInfo.getMaxThreadsPerBlock(device)));
    const uint32_t maxThreadsPerMultiProcessor      = uint32_t(cudaDriverInfo.getMaxThreadsPerMultiProcessor(device));
    const uint32_t multiProcessorCount              = uint32_t(cudaDriverInfo.getMultiProcessorCount(device));
    const uint32_t maxSharedMemoryPerMultiProcessor = uint32_t(cudaDriverInfo.getSharedMemoryPerMultiprocessor(device));

    return make_tuple(dim3{(sharedMemoryPerBlock == 0) ?     multiProcessorCount * (maxThreadsPerMultiProcessor      / threadsPerBlock)
                                                       : min(multiProcessorCount * (maxThreadsPerMultiProcessor      / threadsPerBlock),
                                                             multiProcessorCount * (maxSharedMemoryPerMultiProcessor / sharedMemoryPerBlock))},
                      dim3{threadsPerBlock});
}

string CUDAUtilityFunctions::checkAndReportCUDAMemory(int device, bool useUVA)
{
    size_t freeBytes  = 0;
    size_t totalBytes = 0;
    CUDAError_checkCUDAError(cudaSetDevice(device));
    CUDAError_checkCUDAError(cudaMemGetInfo(&freeBytes, &totalBytes));

    // create return string
    ostringstream ss;
    ss << "   --- GPU Memory Usage ---" << '\n';
    ss << "Device: " << device << ": Used = " << ((totalBytes - freeBytes) >> 10) << " Kb, Free = " << (freeBytes >> 10) << " Kb, Total = " << (totalBytes >> 10) << " Kb" << ", UVA = " << StringAuxiliaryFunctions::toString<bool>(useUVA) << '\n';
    return ss.str();
}