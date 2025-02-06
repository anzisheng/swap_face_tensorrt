cmake_minimum_required(VERSION 3.12)
set(CMAKE_BUILD_TYPE  Debug)

project(swap_face_tensorrt)
set(TensorRT_DIR /usr/local/TensorRT-8.6.1.6/)

if (NOT TensorRT_DIR)
    set(TensorRT_DIR /usr/local/TensorRT-8.6.1.6/)
endif()

# For finding FindTensorRT.cmake
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)

#add_executable(swap_face src/main.cpp)
target_include_directories(swap_face ${TensorRT_INCLUDE_DIRS}
/usr/local/TensorRT-8.6.1.6/include
/usr/local/TensorRT-8.6.1.6/samples/common/)
add_executable(swap_face src/main.cpp)

