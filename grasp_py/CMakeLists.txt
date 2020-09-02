cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(grasp_gen)

enable_language(CUDA)

find_package(PCL 1.2 REQUIRED)
find_package(CUDA    REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

pybind11_add_module(AntiGrasp src/PointCloud.cu src/PointCloud.h src/module.cpp)
add_executable(test src/PointCloud.cu src/PointCloud.h src/test.cpp)