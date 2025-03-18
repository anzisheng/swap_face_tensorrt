cmake_minimum_required(VERSION 3.18)
project(tensorrt_cpp_api)
add_definitions(-DSHOW) #定义宏：SHOW
set(cmake_build_type debug)
# Use ccache to speed up rebuilds
include(cmake/ccache.cmake)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG -Wno-deprecated-declarations")

# For finding FindTensorRT.cmake
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# TODO: Specify the path to TensorRT root dir
set(TensorRT_DIR /usr/local/TensorRT-8.6.1.6/)
# Use the correct version of CUDA
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
# We require CUDA, OpenCV, and TensorRT
find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)

add_library(tensorrt_cpp_api SHARED
        src/engine.cpp
        src/yolov8n_face.cpp
        src/Face68Landmarks_trt.cpp
        src/utils.cpp
        src/facerecognizer_trt.cpp
        )

target_include_directories(tensorrt_cpp_api PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} include include/interfaces . )
target_link_libraries(tensorrt_cpp_api PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES} fmt::fmt)

add_executable(swap_face src/main.cpp)
target_link_libraries(swap_face tensorrt_cpp_api fmt::fmt)