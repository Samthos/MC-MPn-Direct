cmake_minimum_required(VERSION 3.5)
project(MC_MP3_Direct LANGUAGES CXX)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 14)

find_package(CUDA)
list(APPEND CUDA_NVCC_FLAGS "-std=c++14;-O3;-DVERBOSE")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

find_package(MPI)
find_package(Armadillo REQUIRED)
find_package(BLAS)

execute_process(
        COMMAND git log -1 --format=%h
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_COMMIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
add_definitions("-DVERSION=\"${GIT_COMMIT_HASH}\"")


option(NDEBUG "NDEBUG" OFF)
if(${NDEBUG})
    add_definitions(-DNDEBUG)
endif()

#add_subdirectory(test)
add_subdirectory(src)
