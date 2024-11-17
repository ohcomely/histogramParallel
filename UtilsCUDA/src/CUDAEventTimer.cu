#include "CUDAEventTimer.h"
#include "CUDAUtilityFunctions.h"
#include "UtilityFunctions.h"

using namespace UtilsCUDA;
using namespace Utils::UtilityFunctions;

CUDAEventTimer::CUDAEventTimer(int device, const cudaStream_t& cudaStream) noexcept
  : cudaStream_(cudaStream)
{
  CUDAError_checkCUDAError(cudaSetDevice(device));
  CUDAError_checkCUDAError(cudaEventCreate(&start_));
  CUDAError_checkCUDAError(cudaEventCreate(&stop_));
}

CUDAEventTimer::~CUDAEventTimer() noexcept
{
  CUDAError_checkCUDAError(cudaEventDestroy(start_));
  CUDAError_checkCUDAError(cudaEventDestroy(stop_));
}

void CUDAEventTimer::startTimer()
{
  stopped_ = false; // reset stop flag
  CUDAError_checkCUDAError(cudaEventRecord(start_, cudaStream_));
}

void CUDAEventTimer::stopTimer()
{
  if (stopped_)
  {
    return;
  }

  stopped_ = true; // set timer stopped flag
  CUDAError_checkCUDAError(cudaEventRecord(stop_, cudaStream_));
}

float CUDAEventTimer::getElapsedTime()
{
  if (!stopped_)
  {
    stopTimer();
  }

  float elapsedTime = 0.0f;
  CUDAError_checkCUDAError(cudaEventSynchronize(stop_));
  CUDAError_checkCUDAError(cudaEventElapsedTime(&elapsedTime, start_, stop_));
  return elapsedTime;
}

double CUDAEventTimer::getElapsedTimeInNanoSecs()
{
  return double(getElapsedTime() * 1000000.0f);
}

double CUDAEventTimer::getElapsedTimeInMicroSecs()
{
  return double(getElapsedTime() * 1000.0f);
}

double CUDAEventTimer::getElapsedTimeInMilliSecs()
{
  return double(getElapsedTime());
}

double CUDAEventTimer::getElapsedTimeInSecs()
{
  return double(getElapsedTime() / 1000.0f);
}

double CUDAEventTimer::getMeanTimeInNanoSecs()
{
  const auto index = StdAuxiliaryFunctions::toUnsignedType(TimerTypes::NANOSECS);
  return calculateMeanTime(getElapsedTimeInNanoSecs(), timersBookKeeping_[index], timersBookKeepingIndex_[index], firstTimersBookKeepingIterationCompleted_[index]);
}

double CUDAEventTimer::getMeanTimeInMicroSecs()
{
  const auto index = StdAuxiliaryFunctions::toUnsignedType(TimerTypes::MICROSECS);
  return calculateMeanTime(getElapsedTimeInMicroSecs(), timersBookKeeping_[index], timersBookKeepingIndex_[index], firstTimersBookKeepingIterationCompleted_[index]);
}

double CUDAEventTimer::getMeanTimeInMilliSecs()
{
  const auto index = StdAuxiliaryFunctions::toUnsignedType(TimerTypes::MILLISECS);
  return calculateMeanTime(getElapsedTimeInMilliSecs(), timersBookKeeping_[index], timersBookKeepingIndex_[index], firstTimersBookKeepingIterationCompleted_[index]);
}

double CUDAEventTimer::getMeanTimeInSecs()
{
  const auto index = StdAuxiliaryFunctions::toUnsignedType(TimerTypes::SECS);
  return calculateMeanTime(getElapsedTimeInSecs(), timersBookKeeping_[index], timersBookKeepingIndex_[index], firstTimersBookKeepingIterationCompleted_[index]);
}

double CUDAEventTimer::getDecimalElapsedTimeInMicroSecs()
{
  return getElapsedTimeInNanoSecs() / NANO_TO_MICROSECS_CONVERSION;
}

double CUDAEventTimer::getDecimalElapsedTimeInMilliSecs()
{
  return getElapsedTimeInNanoSecs() / NANO_TO_MILLISECS_CONVERSION;
}

double CUDAEventTimer::getDecimalElapsedTimeInSecs()
{
  return getElapsedTimeInNanoSecs() / NANO_TO_SECS_CONVERSION;
}

double CUDAEventTimer::getDecimalMeanTimeInMicroSecs()
{
  return getMeanTimeInNanoSecs() / NANO_TO_MICROSECS_CONVERSION;
}

double CUDAEventTimer::getDecimalMeanTimeInMilliSecs()
{
  return getMeanTimeInNanoSecs() / NANO_TO_MILLISECS_CONVERSION;
}

double CUDAEventTimer::getDecimalMeanTimeInSecs()
{
  return getMeanTimeInNanoSecs() / NANO_TO_SECS_CONVERSION;
}