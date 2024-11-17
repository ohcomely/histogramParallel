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
2.
http://www.dotredconsultancy.com/openglrenderingenginetoolsourcecodelicence.php
Please contact Thanos Theo (thanos.theo@dotredconsultancy.com) for more
information.

*/

#pragma once

#ifndef __HostUnitTests_h
#define __HostUnitTests_h

/** @brief Namespace Tests for all relevant unit testing host & device (CPU &
 * GPU) code.
 * @author Thanos Theo, 2018
 * @version 14.0.0.0
 */
namespace Tests {
/** @brief Host Google Test 01 for the Utils::AccurateTimers::AccurateCPUTimer
 * class functionality.
 * @author Thanos Theo, 2018
 * @version 14.0.0.0
 */
namespace HostGoogleTest01__UTILS_Classes {
void executeTest();
}

/** @brief Host Google Test 02 for the Utils::Randomizers::RandomRNGWELL512
 * class functionality.
 * @author Thanos Theo, 2018
 * @version 14.0.0.0
 */
namespace HostGoogleTest02__UTILS_Classes {
void executeTest();
}

/** @brief Host Google Test 03 for the Utils::SIMDVectorizations classes
 * functionality.
 * @author Thanos Theo, 2018
 * @version 14.0.0.0
 */
namespace HostGoogleTest03__UTILS_Classes {
void executeTest();
}

/** @brief Host Google Test 04 for the
 * Utils::UtilityFunctions::BitManipulationFunctions class functionality.
 * @author Thanos Theo, 2018
 * @version 14.0.0.0
 */
namespace HostGoogleTest04__UTILS_Classes {
void executeTest();
}

/** @brief Host Google Test 05 for the Utils::CPUParallelism parallelFor()
 * functionality.
 * @author Thanos Theo, 2018
 * @version 14.0.0.0
 */
namespace HostGoogleTest05__UTILS_CPUParallelism_Classes {
void executeTest();
}

/** @brief Host Google Test 06 for the Utils::CPUParallelism::CPUParallelismTest
 * class for the parallelFor() functionality.
 * @author Thanos Theo, 2018
 * @version 14.0.0.0
 */
namespace HostGoogleTest06__UTILS_CPUParallelism_Classes {
void executeTest();
}

/** @brief Host Google Test 07 for the lodepng class for png encoding/decoding
 * functionality.
 * @author Thanos Theo, 2018
 * @version 14.0.0.0
 */
namespace HostGoogleTest07__Lodepng_Classes {
void executeTest();
}

/** @brief Host Google Test 08 for the Utils::UtilityFunctions::MathFunctions
 * class functionality.
 * @author Thanos Theo, 2018
 * @version 14.0.0.0
 */
namespace HostGoogleTest08__UTILS_Classes {
void executeTest();
}

/** @brief Host Google Test 09 for the for the parallelFor() thread local &
 * ThreadPool functionality.
 * @author Thanos Theo, 2018
 * @version 14.0.0.0
 */
namespace HostGoogleTest09__UTILS_CPUParallelism_Classes {
void executeTest();
}

//Ritvik's test
namespace HostGoogleTest10__ColorHistogram {
void executeTest();
}
} // namespace Tests

#endif // __HostUnitTests_h