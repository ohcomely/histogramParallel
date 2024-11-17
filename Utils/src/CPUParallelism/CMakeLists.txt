# GPU Framework version 14.0.0

set(UTILS_SRC_CPUParallel_CPP
  src/CPUParallelism/CPUParallelismNCP.cpp
  src/CPUParallelism/ThreadBarrier.cpp
  src/CPUParallelism/ThreadGuard.cpp
  src/CPUParallelism/ThreadJoiner.cpp
  src/CPUParallelism/ThreadOptions.cpp
  src/CPUParallelism/ThreadPool.cpp

  #This line is to make CMakeList.txt appear on the VS project tree.
  src/CPUParallelism/CMakeLists.txt
)
source_group("src\\CPUParallelism" FILES ${UTILS_SRC_CPUParallel_CPP})
