# GPU Framework version 14.0.0

set(UTILS_SRC_CPP
  src/lodepng.cpp
  src/AccurateTimers.cpp
  src/Randomizers.cpp
  src/UnitTests.cpp
  src/UtilityFunctions.cpp

  #This line is to make CMakeList.txt appear on the VS project tree.
  src/CMakeLists.txt
)
source_group("src" FILES ${UTILS_SRC_CPP})