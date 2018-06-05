cmake_minimum_required(VERSION 3.5)
project(MC_MP3_Direct LANGUAGES CXX CUDA)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

#find_package(MPI)
find_package(Armadillo REQUIRED)

add_subdirectory(test)
add_subdirectory(src)
