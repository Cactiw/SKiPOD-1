cmake_minimum_required(VERSION 3.15)
project(1)
include_directories("/usr/include/mpi/")
set(CMAKE_C_COMPILER "/usr/bin/mpicc")
set(CMAKE_CXX_COMPILER "/usr/bin/mpic++")

set(CMAKE_CXX_STANDARD 17)

# add_executable(1 main_openMP.cpp main_MPI.cpp)
add_executable(1 main_MPI.cpp)
