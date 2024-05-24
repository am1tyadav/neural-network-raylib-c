cmake_minimum_required(VERSION 3.27)

project(mnist_example C)

set(CMAKE_C_STANDARD 11)

set(CMAKE_BUILD_TYPE Debug)

set(BUILD_SHARED_LIBS OFF)

include(CPM.cmake)

CPMAddPackage(
    NAME raylib
    GIT_TAG 5.0
    GIT_REPOSITORY "https://github.com/raysan5/raylib"
)

CPMAddPackage(
    NAME ctensor
    GIT_TAG main
    GIT_REPOSITORY "https://github.com/am1tyadav/ctensor"
)

include_directories(${ctensor_SOURCE_DIR}/src)

add_executable(mnist_example main.c)

target_link_libraries(mnist_example raylib ctensor)
