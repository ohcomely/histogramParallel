#include "CPUParallelism/ThreadOptions.h"
#include "UtilityFunctions.h"
#ifndef _WIN32
  #include <pthread.h>
#else
  #include <Windows.h>
#endif // _WIN32
#include <system_error>
#include <cerrno>

using namespace std;
using namespace Utils::CPUParallelism;
using namespace Utils::UtilityFunctions;

namespace // anonymous namespace used instead of deprecated 'static' keyword used for cpp variable locality
{
#ifndef _WIN32
  inline size_t pthreadSetNativePriority(size_t priority, int& policy)
  {
    if (priority == PRIORITY_NONE || priority < PRIORITY_MIN)
    {
      policy = SCHED_OTHER;
      return PRIORITY_NONE;
    }

    if (priority > PRIORITY_MAX)
    {
      priority = PRIORITY_MAX;
    }

    policy = SCHED_RR;
    const size_t getMinPriority = size_t(sched_get_priority_min(policy));
    const size_t getMaxPriority = size_t(sched_get_priority_max(policy));

    return size_t((getMaxPriority + (priority - PRIORITY_MIN) * (getMaxPriority - getMinPriority) / double(PRIORITY_MAX - PRIORITY_MIN)) + 0.5);
  }

  inline size_t pthreadGetNativePriority(size_t priority, int policy)
  {
    if (policy == SCHED_OTHER)
    {
      return PRIORITY_NONE;
    }

    const size_t minPriority = size_t(sched_get_priority_min(policy));
    const size_t maxPriority = size_t(sched_get_priority_max(policy));

    return size_t((PRIORITY_MIN + (priority - minPriority) * (PRIORITY_MAX - PRIORITY_MIN) / double(maxPriority - minPriority)) + 0.5);
  }
#else
  inline DWORD_PTR GetThreadAffinityMask(HANDLE thread)
  {
    DWORD_PTR mask = 1;
    DWORD_PTR old  = 0;

    // try every CPU one by one until one works or none are left
    while (mask)
    {
      old = ::SetThreadAffinityMask(thread, mask);
      if (old)
      {
        // this one worked
        ::SetThreadAffinityMask(thread, old); // restore original
        return old;
      }
      else
      {
        if (::GetLastError() != ERROR_INVALID_PARAMETER)
        {
          return THREAD_PRIORITY_ERROR_RETURN;
        }
      }
      mask <<= 1;
    }

    return THREAD_PRIORITY_ERROR_RETURN;
  }

  inline int winthreadSetNativePriority(size_t priority)
  {
    if (priority >= PRIORITY_MAX)
    {
      return THREAD_PRIORITY_HIGHEST;
    }
    else if (priority >= PRIORITY_MIN + (PRIORITY_MAX - PRIORITY_MIN) * 3 / 4)
    {
      return THREAD_PRIORITY_ABOVE_NORMAL;
    }
    else if (priority >= PRIORITY_MIN + (PRIORITY_MAX - PRIORITY_MIN) / 2)
    {
      return THREAD_PRIORITY_NORMAL;
    }
    else if (priority >= PRIORITY_MIN + (PRIORITY_MAX - PRIORITY_MIN) / 4)
    {
      return THREAD_PRIORITY_BELOW_NORMAL;
    }
    else if (priority >= PRIORITY_MIN)
    {
      return THREAD_PRIORITY_IDLE;
    }
    else
    {
      return THREAD_PRIORITY_NORMAL;
    }
  }

  inline size_t winthreadGetNativePriority(size_t priority)
  {
    switch (priority)
    {
      case THREAD_PRIORITY_IDLE:          return PRIORITY_MIN;
      case THREAD_PRIORITY_BELOW_NORMAL:  return PRIORITY_MIN + (PRIORITY_MAX - PRIORITY_MIN) / 4;
      case THREAD_PRIORITY_NORMAL:        return PRIORITY_MIN + (PRIORITY_MAX - PRIORITY_MIN) / 2;
      case THREAD_PRIORITY_ABOVE_NORMAL:  return PRIORITY_MIN + (PRIORITY_MAX - PRIORITY_MIN) * 3 / 4;
      case THREAD_PRIORITY_HIGHEST:
      case THREAD_PRIORITY_TIME_CRITICAL: return PRIORITY_MAX;
      default:                            return PRIORITY_MIN + (PRIORITY_MAX - PRIORITY_MIN) / 2;
    }
  }
#endif // _WIN32
}

#ifndef _WIN32
void Utils::CPUParallelism::setAffinity(thread& threadHandle, size_t threadIdx, uint64_t affinityMask)
{
  if ((affinityMask >> uint64_t(threadIdx)) & 1) // check if affinity mask bit is turned on
  {
    // create a cpu_set_t object representing a set of CPUs, clear it and mark only CPU i as set
    cpu_set_t cpuset{};
    CPU_ZERO(&cpuset);
    CPU_SET(threadIdx, &cpuset);
    const int errorCode = pthread_setaffinity_np(threadHandle.native_handle(), sizeof(cpu_set_t), &cpuset);
    if (errorCode != 0)
    {
      throw system_error(errorCode, system_category(), "Error calling pthread_setaffinity_np() for threadIdx: " + StringAuxiliaryFunctions::toString(threadIdx) + ".");
    }
  }
}

bool Utils::CPUParallelism::getAffinity(thread& threadHandle, size_t threadIdx)
{
  cpu_set_t cpuset{};
  CPU_ZERO(&cpuset);
  const int errorCode = pthread_getaffinity_np(threadHandle.native_handle(), sizeof(cpu_set_t), &cpuset);
  if (errorCode != 0)
  {
    throw system_error(errorCode, system_category(), "Error calling pthread_getaffinity_np() for threadIdx: " + StringAuxiliaryFunctions::toString(threadIdx) + ".");
  }

  return ((cpuset.__bits[0] >> uint64_t(threadIdx)) & 1);
}

void Utils::CPUParallelism::setPriority(thread& threadHandle, size_t threadIdx, size_t priority)
{
  pthread_attr_t attributes{};
  pthread_attr_init(&attributes);
  int policy = 0;
  pthread_attr_getschedpolicy(&attributes, &policy);
  pthread_attr_destroy(&attributes);
  const int nativePriority = int(pthreadSetNativePriority(priority, policy));
  const int errorCode = pthread_setschedprio(threadHandle.native_handle(), nativePriority);
  if (errorCode != 0)
  {
    throw system_error(errorCode, system_category(), "Error calling pthread_setschedprio() for threadIdx: " + StringAuxiliaryFunctions::toString(threadIdx) + ".");
  }
}

size_t Utils::CPUParallelism::getPriority(thread& threadHandle, size_t threadIdx)
{
  int policy = 0;
  sched_param schedParam{};
  const int errorCode = pthread_getschedparam(threadHandle.native_handle(), &policy, &schedParam);
  if (errorCode != 0)
  {
    throw system_error(errorCode, system_category(), "Error calling pthread_getschedparam() for threadIdx: " + StringAuxiliaryFunctions::toString(threadIdx) + ".");
  }

  return pthreadGetNativePriority(schedParam.sched_priority, policy);
}
#else
void Utils::CPUParallelism::setAffinity(thread& threadHandle, size_t threadIdx, uint64_t affinityMask)
{
  if ((affinityMask >> uint64_t(threadIdx)) & 1) // check if affinity mask bit is turned on
  {
    if (::SetThreadAffinityMask(threadHandle.native_handle(), DWORD_PTR(1) << threadIdx) == FALSE)
    {
      throw system_error(ENOTSUP, system_category(), "Error calling SetThreadAffinityMask() for threadIdx: " + StringAuxiliaryFunctions::toString(threadIdx) + ".");
    }
  }
}

bool Utils::CPUParallelism::getAffinity(thread& threadHandle, size_t threadIdx)
{
  const DWORD_PTR nativeAffinity = ::GetThreadAffinityMask(threadHandle.native_handle());
  if (size_t(nativeAffinity) == THREAD_PRIORITY_ERROR_RETURN)
  {
    throw system_error(ENOTSUP, system_category(), "Error calling GetThreadAffinityMask() for threadIdx: " + StringAuxiliaryFunctions::toString(threadIdx) + ".");
  }

  return ((nativeAffinity >> uint64_t(threadIdx)) & 1);
}

void Utils::CPUParallelism::setPriority(thread& threadHandle, size_t threadIdx, size_t priority)
{
  const int nativePriority = winthreadSetNativePriority(priority);
  if (::SetThreadPriority(threadHandle.native_handle(), nativePriority) == FALSE)
  {
    throw system_error(ENOTSUP, system_category(), "Error calling SetThreadPriority() for threadIdx: " + StringAuxiliaryFunctions::toString(threadIdx) + ".");
  }
}

size_t Utils::CPUParallelism::getPriority(thread& threadHandle, size_t threadIdx)
{
  const int nativePriority = ::GetThreadPriority(threadHandle.native_handle());
  if (nativePriority == THREAD_PRIORITY_ERROR_RETURN)
  {
    throw system_error(ENOTSUP, system_category(), "Error calling GetThreadPriority() for threadIdx: " + StringAuxiliaryFunctions::toString(threadIdx) + ".");
  }

  return winthreadGetNativePriority(nativePriority);
}
#endif // _WIN32