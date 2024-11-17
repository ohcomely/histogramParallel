#include "CUDAStreamsHandler.h"
#include "CUDAUtilityFunctions.h"
#include <algorithm>

using namespace std;
using namespace UtilsCUDA;

CUDAStreamsHandler::CUDAStreamsHandler(const CUDADriverInfo& cudaDriverInfo, int device, size_t numberOfStreams, bool useStreamPriorities, int priorityType) noexcept
  : numberOfStreams_{max<size_t>(1, numberOfStreams)}
  , cudaStreams_{make_unique<cudaStream_t[]>(numberOfStreams_)}
  , useStreamPriorities_{useStreamPriorities && cudaDriverInfo.isAtLeastGPUType(CUDADriverInfo::GPUTypes::MAXWELL, device)}
  , priorityType_{priorityType}
{
  initialize(device);
}

CUDAStreamsHandler::~CUDAStreamsHandler() noexcept // no virtual destructor for data-oriented design (no up-casting should ever be used)
{
  uninitialize();
}

void CUDAStreamsHandler::addCallback(size_t index, const cudaStreamCallback_t& callback, void* data) const noexcept
{
  CUDAError_checkCUDAError(cudaStreamAddCallback(cudaStreams_[index], callback, data, 0));
}

void CUDAStreamsHandler::initialize(int device) noexcept
{
  if (useStreamPriorities_)
  {
    CUDAError_checkCUDAError(cudaDeviceGetStreamPriorityRange(&priorityLowest_, &priorityHighest_));
  }

  for (size_t i = 0; i < numberOfStreams_; ++i)
  {
    if (!cudaStreams_[i])
    {
      CUDAError_checkCUDAError(cudaSetDevice(int(device)));
      if (useStreamPriorities_)
      {
        // lowest number is highest priority (from Nvidia presentation 'GPU Technology Conference: CUDA Streams Best Practices and Common Pitfalls')
        CUDAError_checkCUDAError(cudaStreamCreateWithPriority(&cudaStreams_[i], priorityType_, priorityLowest_));
      }
      else
      {
        CUDAError_checkCUDAError(cudaStreamCreateWithFlags(&cudaStreams_[i], priorityType_));
      }
    }
  }
}

void CUDAStreamsHandler::uninitialize() const noexcept
{
  for (size_t i = 0; i < numberOfStreams_; ++i)
  {
    if (cudaStreams_[i])
    {
      CUDAError_checkCUDAError(cudaStreamDestroy(cudaStreams_[i]));
    }
  }
}