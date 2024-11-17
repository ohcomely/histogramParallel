#include "HostStressTests.h"
#include "Tests/LinearAlgebraCPUComputingStressTest.h"
#include "UtilityFunctions.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdio>

using namespace std;
using namespace Utils;
using namespace Utils::UtilityFunctions;
using namespace Tests;

namespace // anonymous namespace used instead of deprecated 'static' keyword used for cpp variable locality
{
  size_t cpu_iterations = 160;
  bool max_memory       = false;

  /*
  *  Displays the program usage information.
  */
  inline void giveUsage(const char* cmd, int exitCode = EXIT_FAILURE)
  {
    cout << "\nUsage: " << cmd << " [-h][-help][-cpu_iterations][-max_memory]\n";
    cout << "-------------------------------------------------------------------------------\n";
    cout << "   -h or -help          this help text\n";
    cout << "   -cpu_iterations #n   uses a given number of CPU iterations (default is 160) \n";
    cout << "   -max_memory          run stress test with max memory iterations             \n\n";

    exit(exitCode);
  }
}

void HostStressGoogleTest01::executeTest()
{
  bool testResult = true;
  size_t index    = 0;
  const size_t minDimensionSize = 64;
  const size_t maxDimensionSize = max_memory ? 16384 : 2048;
  for (size_t dimensionSize = minDimensionSize; testResult && (dimensionSize <= maxDimensionSize); dimensionSize <<= 1) // implies 'i *= 2'
  {
    ++index;
    DebugConsole_consoleOutLine("Running iteration '", index, "':");
    LinearAlgebraCPUComputingStressTest linearAlgebraCPUComputingStressTest(dimensionSize, cpu_iterations);
    linearAlgebraCPUComputingStressTest.resetTests();
    testResult = linearAlgebraCPUComputingStressTest.conductTests();
    linearAlgebraCPUComputingStressTest.reportTestResults();
  }
  EXPECT_TRUE(testResult);
}

TEST(HostStressGoogleTest01, LinearAlgebraCPUComputingStressTest)
{
  HostStressGoogleTest01::executeTest();
}

// The main entry point of the DeviceStressTests executable.
int main(int argc, char* argv[])
{
#ifdef GPU_FRAMEWORK_DEBUG
  DebugConsole::setUseLogFile(true);
  DebugConsole::setLogFileName("HostStressTests.log");
#endif // GPU_FRAMEWORK_DEBUG

  for (int i = 1; i < argc; ++i)
  {
    const string currentOption = StringAuxiliaryFunctions::toLowerCase(string(argv[i]));
    if ((currentOption == "-h") || (currentOption == "-help"))
    {
      giveUsage(argv[0], EXIT_SUCCESS);
    }
    else if (currentOption == "-cpu_iterations")
    {
      if ((i + 1) >= argc) giveUsage(argv[0]);
      cpu_iterations = StringAuxiliaryFunctions::fromString<size_t>(argv[i + 1]);
      if (cpu_iterations < 1)
      {
        DebugConsole_consoleOutLine("Command-line argument cpu_iterations should be > 0.");
        giveUsage(argv[0]);
      }
      ++i; // increment counter to skip the i + 1 option
      DebugConsole_consoleOutLine("Command-line argument cpu_iterations detected: ", cpu_iterations);
    }
    else if (currentOption == "-max_memory")
    {
      max_memory = true;
      DebugConsole_consoleOutLine("Command-line argument max_memory detected: ", StringAuxiliaryFunctions::toString<bool>(max_memory));
    }
    else
    {
      giveUsage(argv[0]);
    }
  }

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}