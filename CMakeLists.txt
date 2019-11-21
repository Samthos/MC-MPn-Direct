cmake_minimum_required(VERSION 3.5)
project(MC_MPn LANGUAGES CXX)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 14)

execute_process(
    COMMAND git log -1 --format=%h
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
add_definitions("-DVERSION=\"${GIT_COMMIT_HASH}\"")

#find_package(CUDA)
#list(APPEND CUDA_NVCC_FLAGS "-std=c++14;-O3;-DVERBOSE")
#set(CUDA_PROPAGATE_HOST_FLAGS OFF)

find_package(Armadillo REQUIRED)
find_package(BLAS REQUIRED)

option(Enable_MPI "Enables MPI Sampling" Off)
if (Enable_MPI)
    find_package(MPI REQUIRED)
endif()

option(NDEBUG "enables NDEBUG macro" OFF)
if(${NDEBUG})
    add_definitions(-DNDEBUG)
endif()

option(Enable_Full_Print "Enables Printing of the full energy sequence" Off)
if (Enable_Full_Print)
    add_definitions(-DFULL_PRINTING)
endif()

set(MP2CV 0 CACHE STRING "Sets MP2CV level")
set(MP3CV 0 CACHE STRING "Sets MP3CV level")
set(MP4CV 0 CACHE STRING "Sets MP4CV level")

add_subdirectory(src)
add_subdirectory(util)
