# GPU Framework version 14.0.0

set(UTILS_CPUParallel_HEADERS
  include/CPUParallelism/ConcurrentBlockingQueue.h
  include/CPUParallelism/CPUParallelismNCP.h
  include/CPUParallelism/CPUParallelismUtilityFunctions.h
  include/CPUParallelism/ThreadBarrier.h
  include/CPUParallelism/ThreadGuard.h
  include/CPUParallelism/ThreadJoiner.h
  include/CPUParallelism/ThreadOptions.h
  include/CPUParallelism/ThreadPool.h

  #This line is to make CMakeList.txt appear on the VS project tree.
  include/CPUParallelism/CMakeLists.txt
)
source_group("include\\CPUParallelism" FILES ${UTILS_CPUParallel_HEADERS})

install(FILES ${UTILS_CPUParallel_HEADERS} DESTINATION include/${PROJECT_NAME}/CPUParallelism)