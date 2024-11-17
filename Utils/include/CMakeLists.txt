# GPU Framework version 14.0.0

set(UTILS_HEADERS
  include/ModuleDLL.h
  include/lodepng.h
  include/EnvironmentConfig.h
  include/FunctionView.h
  include/MathConstants.h
  include/NewHandlerSupport.h
  include/AccurateTimers.h
  include/Randomizers.h
  include/SIMDVectorizations.h
  include/UnitTests.h
  include/UtilityFunctions.h
  include/VectorTypes.h

  #This line is to make CMakeList.txt appear on the VS project tree.
  include/CMakeLists.txt
)
source_group("include" FILES ${UTILS_HEADERS})

install(FILES ${UTILS_HEADERS} DESTINATION include/${PROJECT_NAME})