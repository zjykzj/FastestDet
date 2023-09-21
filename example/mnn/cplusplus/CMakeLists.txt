cmake_minimum_required(VERSION 3.16)
project(classify)

set(CMAKE_CXX_STANDARD 17)

set(MNN_INFER_ROOT /home/zj/mnn/MNN-2.4.0)
set(MNN_INFER_INCLUDE_DIR ${MNN_INFER_ROOT}/include/)
include_directories(${MNN_INFER_INCLUDE_DIR})
set(INFER_LIB ${MNN_INFER_ROOT}/build/libMNN.so)

add_executable(classify pictureRecognition.cpp)
target_link_libraries(classify PRIVATE ${INFER_LIB})

add_executable(classify pictureRecognition.cpp)
target_link_libraries(classify PRIVATE ${INFER_LIB})