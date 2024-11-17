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

#ifndef __Randomizers_h
#define __Randomizers_h

#include "ModuleDLL.h"
#include <random>
#include <limits>
#include <cstdint>
#include <array>

/** @brief Namespace Utils contains utility classes with mainly static CPU related methods.
* @author Thanos Theo, 2009-2018
* @version 14.0.0.0
*/
namespace Utils
{
  /** @brief Namespace Randomizers contains random number generator classes.
  * @author Thanos Theo, 2009-2018
  * @version 14.0.0.0
  */
  namespace Randomizers
  {
    /** @brief The UniformRandom class provides a uniform random number generator.
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    class UTILS_MODULE_API UniformRandom
    {
    public:

      std::uint64_t getUniformInteger()         { return uniformIntegerDistribution_(rng_); }
      double getUniformFloat()                  { return uniformRealDistribution_(rng_); }
      double operator()()                       { return uniformRealDistribution_(rng_); }
      void setSeed(std::uint64_t value = 5489U) { rng_.seed(value); }

      UniformRandom() noexcept;
      ~UniformRandom() = default; // no virtual destructor for data-oriented design (no up-casting should ever be used)
      UniformRandom(const UniformRandom&) = delete; // copy-constructor deleted
      UniformRandom(UniformRandom&&)      = delete; // move-constructor deleted
      UniformRandom& operator=(const UniformRandom&) = delete; //      assignment operator deleted
      UniformRandom& operator=(UniformRandom&&)      = delete; // move-assignment operator deleted

    protected:

      std::mt19937_64 rng_;

    private:

      std::uniform_int_distribution<std::uint64_t> uniformIntegerDistribution_;
      std::uniform_real_distribution<double>       uniformRealDistribution_;
    };

    /** @brief The NormalRandom class provides a normal random number generator.
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    class NormalRandom final : public UniformRandom
    {
    public:

      double getNormalFloat() { return normalDistribution_(rng_); }
      double operator()()     { return normalDistribution_(rng_); }

      NormalRandom()  = default;
      ~NormalRandom() = default; // no virtual destructor for data-oriented design (no up-casting should ever be used)
      NormalRandom(const NormalRandom&) = delete; // copy-constructor deleted
      NormalRandom(NormalRandom&&)      = delete; // move-constructor deleted
      NormalRandom& operator=(const NormalRandom&) = delete; //      assignment operator deleted
      NormalRandom& operator=(NormalRandom&&)      = delete; // move-assignment operator deleted

    private:

      std::normal_distribution<double> normalDistribution_;
    };

    /** @brief The ExponentialRandom class provides a exponential random number generator.
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    class ExponentialRandom final : public UniformRandom
    {
    public:

      double getExponentialFloat() { return exponentialDistribution_(rng_); }
      double operator()()          { return exponentialDistribution_(rng_); }

      ExponentialRandom()  = default;
      ~ExponentialRandom() = default; // no virtual destructor for data-oriented design (no up-casting should ever be used)
      ExponentialRandom(const ExponentialRandom&) = delete; // copy-constructor deleted
      ExponentialRandom(ExponentialRandom&&)      = delete; // move-constructor deleted
      ExponentialRandom& operator=(const ExponentialRandom&) = delete; //      assignment operator deleted
      ExponentialRandom& operator=(ExponentialRandom&&)      = delete; // move-assignment operator deleted

    private:

      std::exponential_distribution<double> exponentialDistribution_;
    };

    /** @brief The RandomRNGWELL512 class provides the very fast RNG WELL512 algorithm random number generator initialized with a random integer.
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    class UTILS_MODULE_API RandomRNGWELL512 final
    {
    public:

      static std::uint64_t getRandomMax() { return std::numeric_limits<std::uint64_t>::max(); }
      std::uint64_t getRandomInteger();   // in the interval [0, std::uint64_t max]
      double getRandomFloat()             { return getRandomInteger() / double(getRandomMax()); } // in the interval [0,1]
      double operator()()                 { return getRandomInteger() / double(getRandomMax()); } // in the interval [0,1]

      RandomRNGWELL512() noexcept;
      ~RandomRNGWELL512() = default; // no virtual destructor for data-oriented design (no up-casting should ever be used)
      RandomRNGWELL512(const RandomRNGWELL512&) = delete; // copy-constructor deleted
      RandomRNGWELL512(RandomRNGWELL512&&)      = delete; // move-constructor deleted
      RandomRNGWELL512& operator=(const RandomRNGWELL512&) = delete; //      assignment operator deleted
      RandomRNGWELL512& operator=(RandomRNGWELL512&&)      = delete; // move-assignment operator deleted

    private:

      std::uint64_t index_                    = 0; // init should also reset this to 0
      static constexpr std::size_t STATE_SIZE = 16;
      std::array<std::uint64_t, STATE_SIZE> state_{ { 0 } }; // double braces because we initialize an array inside an std::array object
    };
  } // namespace Randomizers
} // namespace Utils

#endif // __Randomizers_h