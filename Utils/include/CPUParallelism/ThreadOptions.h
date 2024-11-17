#pragma once

#ifndef THREAD_OPTIONS_H
#define THREAD_OPTIONS_H

#include "../ModuleDLL.h"
#include <cstdint>
#include <thread>

/** @brief Namespace Utils contains utility classes with mainly static CPU related methods.
* @author Thanos Theo, 2009-2018
* @version 14.0.0.0
*/
namespace Utils
{
  /** @brief Namespace CPUParallelism encapsulates usage of the N-CP parallelism idea.
  *
  *  This namespace encapsulates usage of the N-CP parallelism idea.\n
  *  CPUParallelism libraries originally based on with further extensions: http://www.manning.com/williams/.\n
  *  The N-CP idea was based on: http://www.biolayout.org/wp-content/uploads/2013/01/Manuscript.pdf.\n
  *  Further inspiration was found here: http://jcip.net.s3-website-us-east-1.amazonaws.com/.\n
  *
  * @author Thanos Theo, 2009-2018
  * @version 14.0.0.0
  */
  namespace CPUParallelism
  {
    /**  @brief Thread affinity mask constants.
    *
    *  Note: On non-Windows platforms, the pthread_setaffinity_np() call is being used for CPU affinity.\n
    *
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    enum ThreadAffinityMask : std::uint64_t
    {
      AFFINITY_MASK_NONE         = 0,                 // no affinity used
      AFFINITY_MASK_1_CPU_CORE   = 0x1,               // lower 1 bit  turned on
      AFFINITY_MASK_2_CPU_CORES  = 0x3,               // lower 2 bits turned on
      AFFINITY_MASK_4_CPU_CORES  = 0xf,               // lower 4 bits turned on
      AFFINITY_MASK_8_CPU_CORES  = 0xff,              // lower 8 bits turned on
      AFFINITY_MASK_16_CPU_CORES = 0xffff,            // lower 16 bits turned on
      AFFINITY_MASK_32_CPU_CORES = 0xffffffff,        // lower 32 bits turned on
      AFFINITY_MASK_ALL          = 0xffffffffffffffff // 0xffffffffffffffff (== ~0), all bits turned on
    };

    /**  @brief Thread priority constants.
    *
    *  Note: On non-Windows platforms, the pthread_setschedprio() call is being used for thread priority.\n
    *
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    enum ThreadPriorities : std::size_t
    {
      PRIORITY_NONE    =  0, // no priority control
      PRIORITY_MIN     =  1, // minimum priority
      PRIORITY_NORMAL  = 50, // normal  priority
      PRIORITY_MAX     = 99  // maximum priority
    };

    UTILS_MODULE_API        void setAffinity(std::thread& threadHandle, std::size_t threadIdx, std::uint64_t affinityMask);
    UTILS_MODULE_API        bool getAffinity(std::thread& threadHandle, std::size_t threadIdx);
    UTILS_MODULE_API        void setPriority(std::thread& threadHandle, std::size_t threadIdx, std::size_t priority);
    UTILS_MODULE_API std::size_t getPriority(std::thread& threadHandle, std::size_t threadIdx);
  }
}

#endif // THREAD_OPTIONS_H