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

#ifndef __CPUParallelismTest_h
#define __CPUParallelismTest_h

#include "UnitTests.h"
#include "CPUParallelism/CPUParallelismNCP.h"
#include <tuple>

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
    /** @brief This class encapsulates unit testing of CPUParallelism libraries.
    *
    *  CPUParallelismTest.h:
    *  ====================
    *  This class encapsulates unit testing of CPUParallelism libraries.\n
    *  CPUParallelism libraries originally based on with further extensions: http://www.manning.com/williams/.\n
    *  The N-CP idea was based on: http://www.biolayout.org/wp-content/uploads/2013/01/Manuscript.pdf.\n
    *  Further inspiration was found here: http://jcip.net.s3-website-us-east-1.amazonaws.com/.\n
    *
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    class CPUParallelismTest final : private UnitTests::UnitTestsInterface<CPUParallelismTest>, private UnitTests::UnitTestUtilityFunctions_flt // private inheritance used for composition and prohibiting up-casting
    {
    public:

      // IUnitTests -> for Google Tests
      void resetTests();
      bool conductTests();
      void reportTestResults();

      CPUParallelismTest(std::size_t dimensions = 512, std::size_t numberOfThreads = numberOfHardwareThreads(), bool useRandomness = false) noexcept;
      CPUParallelismTest(std::tuple<std::size_t, std::size_t, std::size_t> dimensionsXYZ, std::size_t numberOfThreads = numberOfHardwareThreads(), bool useRandomness = false) noexcept;
      ~CPUParallelismTest() = default; // no virtual destructor for data-oriented design (no up-casting should ever be used)
      CPUParallelismTest(const CPUParallelismTest&) = delete; // copy-constructor deleted
      CPUParallelismTest(CPUParallelismTest&&)      = delete; // move-constructor deleted
      CPUParallelismTest& operator=(const CPUParallelismTest&) = delete; //      assignment operator deleted
      CPUParallelismTest& operator=(CPUParallelismTest&&)      = delete; // move-assignment operator deleted

    private:

      std::size_t dimensionX_      = 512;
      std::size_t dimensionY_      = 512;
      std::size_t dimensionZ_      = 512;
      std::size_t numberOfThreads_ = numberOfHardwareThreads();
      bool   useRandomness_        = false;

      int    testIterations_            = 0;
      double meanTimeCounterRandomizer_ = 0.0;
      double meanTimeCounterSingleCore_ = 0.0;
      double meanTimeCounterNCP_        = 0.0;
    };
  } // namespace CPUParallelism
} // namespace Utils

#endif // __CPUParallelismTest_h