cmake_minimum_required(VERSION 3.0.2)
project(tanksim)

## ==== Catkin ====
find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
  message_generation
)

add_service_files(
  FILES
  link.srv
)

add_message_files(
  FILES
  actuator.msg
  sensor.msg
  info.msg
)

generate_messages(
  DEPENDENCIES
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp roslib message_runtime
)

## ==== Raylib ====
set(RAYLIB_DIR "$ENV{HOME}/raylib")
find_path(RAYLIB_INCLUDE_DIR raylib.h PATHS ${RAYLIB_DIR}/src)
find_library(RAYLIB_LIBRARY NAMES raylib PATHS ${RAYLIB_DIR}/src ${RAYLIB_DIR}/build/raylib)

## ==== OpenMP ====
find_package(OpenMP REQUIRED)

## ==== Local llama ====
# Path to your bundled .a files
set(LLAMA_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/lib")

## ==== libtorch ====
set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

# List the static libraries manually in proper order
set(LLAMA_DEPENDENCIES
  ${LLAMA_LIB_DIR}/libcommon.a
  ${LLAMA_LIB_DIR}/libllama.a
  ${LLAMA_LIB_DIR}/libggml-cpu.a
  ${LLAMA_LIB_DIR}/libggml-base.a
  ${LLAMA_LIB_DIR}/libggml.a
)

## ==== Include Paths ====
include_directories(
  include/llama
  ${catkin_INCLUDE_DIRS}
  ${RAYLIB_INCLUDE_DIR}
  include/libtorch/include
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

## ==== tanksim_simulator ====
add_executable(tanksim_simulator src/tanksim_simulator.cpp)
add_dependencies(tanksim_simulator ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(tanksim_simulator
  ${catkin_LIBRARIES}
  ${RAYLIB_LIBRARY}
  ode
  m
  dl
  pthread
)

## ==== tanksim_planner ====
add_executable(tanksim_planner src/tanksim_planner.cpp)

target_link_libraries(tanksim_planner
  ${catkin_LIBRARIES}
  OpenMP::OpenMP_CXX
  -Wl,--start-group
  ${LLAMA_DEPENDENCIES}
  -Wl,--end-group
  m
  dl
  pthread
  atomic
)

## ==== tanksim_controller ====

add_executable(tanksim_controller src/tanksim_controller.cpp)

target_link_libraries(tanksim_controller
  ${catkin_LIBRARIES}
  OpenMP::OpenMP_CXX
  -Wl,--start-group
  ${LLAMA_DEPENDENCIES}
  -Wl,--end-group
  "${TORCH_LIBRARIES}"
  m
  dl
  pthread
  atomic
)
