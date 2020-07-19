cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(grasp_gen)

enable_language(CUDA)

find_package(PCL 1.2 REQUIRED)
find_package(CUDA    REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
# list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

add_executable (grasp_gen grasp_gen.cpp PointCloud.cu )
target_link_libraries (grasp_gen ${PCL_LIBRARIES})