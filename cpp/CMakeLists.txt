cmake_minimum_required(VERSION 3.18)
project(cattrack_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Core tracking library (no Python, no ONNX dependency).
add_library(cattrack STATIC
  src/kalman.cpp
  src/hungarian.cpp
  src/geom.cpp
  src/track.cpp
  src/multi_tracker.cpp
  src/detection.cpp
  src/config.cpp
  src/preprocess.cpp
  src/camera.cpp
  src/draw.cpp
)
target_include_directories(cattrack PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_property(TARGET cattrack PROPERTY POSITION_INDEPENDENT_CODE ON)

# Python extension module (parity testing / transitional integration).
option(CATTRACK_BUILD_PYTHON "Build the pybind11 module" ON)
if(CATTRACK_BUILD_PYTHON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  find_package(pybind11 CONFIG REQUIRED)

  pybind11_add_module(cattrack_cpp bindings/module.cpp)
  target_link_libraries(cattrack_cpp PRIVATE cattrack)
endif()

# Standalone binary. Needs ONNX Runtime C++ (headers + libonnxruntime).
# Point ONNXRUNTIME_ROOT at an extracted onnxruntime-linux-<arch>-<ver> release:
#   cmake -S cpp -B cpp/build -DCATTRACK_WITH_ORT=ON -DONNXRUNTIME_ROOT=$HOME/onnxruntime
option(CATTRACK_WITH_ORT "Build the standalone cattrack binary (requires ONNX Runtime)" OFF)
if(CATTRACK_WITH_ORT)
  if(NOT ONNXRUNTIME_ROOT)
    message(FATAL_ERROR "CATTRACK_WITH_ORT=ON requires -DONNXRUNTIME_ROOT=<path>")
  endif()

  add_executable(cattrack_bin src/main.cpp src/inference.cpp)
  set_target_properties(cattrack_bin PROPERTIES OUTPUT_NAME cattrack)
  target_include_directories(cattrack_bin PRIVATE ${ONNXRUNTIME_ROOT}/include)
  target_link_libraries(cattrack_bin PRIVATE
    cattrack
    ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
  )
  set_target_properties(cattrack_bin PROPERTIES
    BUILD_RPATH ${ONNXRUNTIME_ROOT}/lib
    INSTALL_RPATH ${ONNXRUNTIME_ROOT}/lib
  )
endif()
