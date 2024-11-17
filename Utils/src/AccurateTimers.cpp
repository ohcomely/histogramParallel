#include "AccurateTimers.h"
#include "UtilityFunctions.h"
#include <algorithm>
#include <array>
#include <limits>

using namespace std;
using namespace std::chrono;
using namespace Utils::AccurateTimers;
using namespace Utils::UtilityFunctions;

double AccurateTimerLog::calculateMeanTime(double currentTime, double* __restrict timersBookKeeping, int64_t& timersBookKeepingIndex, bool& firstTimersBookKeepingIterationCompleted)
{
  // keep a history of the time step for the last 11 frames timersBookKeepingIndex
  ++timersBookKeepingIndex;
  if (timersBookKeepingIndex == TIMERS_BOOK_KEEPING_SIZE)
  {
    firstTimersBookKeepingIterationCompleted = true;
  }
  timersBookKeepingIndex                   %= TIMERS_BOOK_KEEPING_SIZE;
  timersBookKeeping[timersBookKeepingIndex] = currentTime;
  if (firstTimersBookKeepingIterationCompleted)
  {
    // throw away the outliers, the two highest and the two lowest values
    array<double, TIMERS_BOOK_KEEPING_SIZE> timersBookKeepingLocalCopy = { { 0.0 } }; // double braces because we initialize an array inside an std::array object
    copy(timersBookKeeping, timersBookKeeping + TIMERS_BOOK_KEEPING_SIZE, timersBookKeepingLocalCopy.begin());
    StdAuxiliaryFunctions::insertionSort<TIMERS_BOOK_KEEPING_SIZE>(timersBookKeepingLocalCopy.data()); // sort an array using insertion sort with a constant small size of N
    // calculate the mean of the remaining 7 values
    double meanTime = 0.0;
    for (size_t i = 2; i <= TIMERS_BOOK_KEEPING_SIZE - 3; ++i)
    {
      meanTime += timersBookKeepingLocalCopy[i];
    }
    meanTime /= (TIMERS_BOOK_KEEPING_SIZE - 4);
    const double range = abs(timersBookKeepingLocalCopy[TIMERS_BOOK_KEEPING_SIZE - 3] - timersBookKeepingLocalCopy[2]);
    if (range >= numeric_limits<double>::epsilon())
    {
      meanTime += range * MathFunctions::smootherstep(timersBookKeepingLocalCopy[2], timersBookKeepingLocalCopy[TIMERS_BOOK_KEEPING_SIZE - 3], meanTime); // (adding more smoothness with smootherstep)
    }
    const int64_t previousIndex = !timersBookKeepingIndex ? (TIMERS_BOOK_KEEPING_SIZE - 1) : (timersBookKeepingIndex - 1);
    // lerp from the time step for the last frame to the calculated mean (adding more smoothness)
    return MathFunctions::mix(timersBookKeeping[previousIndex], meanTime, 0.5);
  }

  return currentTime;
}


// AccurateTimerLog class implementation following below

AccurateTimerLog::AccurateTimerLog() noexcept
{
  timersBookKeepingIndex_.fill(-1); // -1 for first iteration increment
  firstTimersBookKeepingIterationCompleted_.fill(false);
}


// AccurateCPUTimer class implementation following below

void AccurateCPUTimer::startTimer()
{
  stopped_ = false; // reset stop flag
  start_   = high_resolution_clock::now();
}

void AccurateCPUTimer::stopTimer()
{
  if (stopped_)
  {
    return;
  }

  stopped_ = true; // set timer stopped flag
  stop_    = high_resolution_clock::now();
}

template<typename ChronoType> double AccurateCPUTimer::getElapsedTime()
{
  if (!stopped_)
  {
    stop_ = high_resolution_clock::now();
  }

  return ChronoType(stop_ - start_).count();
}

double AccurateCPUTimer::getElapsedTimeInNanoSecs()
{
  return getElapsedTime<duration<double, nano>>();
}

double AccurateCPUTimer::getElapsedTimeInMicroSecs()
{
  return getElapsedTime<duration<double, micro>>();
}

double AccurateCPUTimer::getElapsedTimeInMilliSecs()
{
  return getElapsedTime<duration<double, milli>>();
}

double AccurateCPUTimer::getElapsedTimeInSecs()
{
  return getElapsedTime<duration<double, ratio<1>>>();
}

double AccurateCPUTimer::getMeanTimeInNanoSecs()
{
  const auto index = StdAuxiliaryFunctions::toUnsignedType(TimerTypes::NANOSECS);
  return calculateMeanTime(getElapsedTimeInNanoSecs(), timersBookKeeping_[index], timersBookKeepingIndex_[index], firstTimersBookKeepingIterationCompleted_[index]);
}

double AccurateCPUTimer::getMeanTimeInMicroSecs()
{
  const auto index = StdAuxiliaryFunctions::toUnsignedType(TimerTypes::MICROSECS);
  return calculateMeanTime(getElapsedTimeInMicroSecs(), timersBookKeeping_[index], timersBookKeepingIndex_[index], firstTimersBookKeepingIterationCompleted_[index]);
}

double AccurateCPUTimer::getMeanTimeInMilliSecs()
{
  const auto index = StdAuxiliaryFunctions::toUnsignedType(TimerTypes::MILLISECS);
  return calculateMeanTime(getElapsedTimeInMilliSecs(), timersBookKeeping_[index], timersBookKeepingIndex_[index], firstTimersBookKeepingIterationCompleted_[index]);
}

double AccurateCPUTimer::getMeanTimeInSecs()
{
  const auto index = StdAuxiliaryFunctions::toUnsignedType(TimerTypes::SECS);
  return calculateMeanTime(getElapsedTimeInSecs(), timersBookKeeping_[index], timersBookKeepingIndex_[index], firstTimersBookKeepingIterationCompleted_[index]);
}

double AccurateCPUTimer::getDecimalElapsedTimeInMicroSecs()
{
  return getElapsedTimeInNanoSecs() / NANO_TO_MICROSECS_CONVERSION;
}

double AccurateCPUTimer::getDecimalElapsedTimeInMilliSecs()
{
  return getElapsedTimeInNanoSecs() / NANO_TO_MILLISECS_CONVERSION;
}

double AccurateCPUTimer::getDecimalElapsedTimeInSecs()
{
  return getElapsedTimeInNanoSecs() / NANO_TO_SECS_CONVERSION;
}

double AccurateCPUTimer::getDecimalMeanTimeInMicroSecs()
{
  return getMeanTimeInNanoSecs() / NANO_TO_MICROSECS_CONVERSION;
}

double AccurateCPUTimer::getDecimalMeanTimeInMilliSecs()
{
  return getMeanTimeInNanoSecs() / NANO_TO_MILLISECS_CONVERSION;
}

double AccurateCPUTimer::getDecimalMeanTimeInSecs()
{
  return getMeanTimeInNanoSecs() / NANO_TO_SECS_CONVERSION;
}

uint64_t AccurateCPUTimer::getNanosecondsTimeSinceEpoch()
{
    return duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count();
}

uint64_t AccurateCPUTimer::getMicrosecondsTimeSinceEpoch()
{
    return duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count();
}

uint64_t AccurateCPUTimer::getMillisecondsTimeSinceEpoch()
{
  return duration_cast<milliseconds>(high_resolution_clock::now().time_since_epoch()).count();
}

uint64_t AccurateCPUTimer::getSecondsTimeSinceEpoch()
{
  return duration_cast<seconds>(high_resolution_clock::now().time_since_epoch()).count();
}