cmake_minimum_required(VERSION 3.0.2)
project(tanksim)

## Find catkin and any required components
find_package(catkin REQUIRED COMPONENTS
  roscpp
  # add any other dependencies here
)

# ==== find raylib ====
set(RAYLIB_DIR "$ENV{HOME}/raylib")
find_path(RAYLIB_INCLUDE_DIR raylib.h PATHS ${RAYLIB_DIR}/src)
find_library(RAYLIB_LIBRARY NAMES raylib PATHS ${RAYLIB_DIR}/src ${RAYLIB_DIR}/build/raylib)

# ==== find llama ====
set(LLAMA_DIR "$ENV{HOME}/llama.cpp")
find_path(LLAMA_INCLUDE_DIR raylib.h PATHS ${LLAMA_DIR}/src)
find_library(LLAMA_LIBRARY NAMES raylib PATHS ${LLAMA_DIR}/src ${LLAMA_DIR}/build/raylib)

# ==== Declare catkin package ====
catkin_package(
  # Include dirs and libs for catkin-dependent packages (if needed)
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp
)

# ==== Include directories ====
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${RAYLIB_INCLUDE_DIR}
)

# ==== Link your executable ====
add_executable(tanksim_simulator src/tanksim_simulator.cpp)

target_link_libraries(tanksim_simulator
  ${catkin_LIBRARIES}
  ${RAYLIB_LIBRARY}
  m
  dl
  pthread
)
