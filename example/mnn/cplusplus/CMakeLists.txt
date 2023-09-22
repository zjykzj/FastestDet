cmake_minimum_required(VERSION 3.16)
project(cplusplus)

set(CMAKE_CXX_STANDARD 17)

# set OpenCV
set(OpenCV_DIR /home/zj/software/opencv/install_4_5_2/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

MESSAGE("OpenCV version: ${OpenCV_VERSION}")
MESSAGE("OpenCV OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
MESSAGE("OpenCV OpenCV_LIBS: ${OpenCV_LIBS}")

# set MNN
set(MNN_INFER_ROOT /home/zj/mnn/MNN-2.4.0)
set(MNN_INFER_INCLUDE_DIR ${MNN_INFER_ROOT}/include/)
include_directories(${MNN_INFER_INCLUDE_DIR})
set(INFER_LIB ${MNN_INFER_ROOT}/build/libMNN.so)

# Compile
add_executable(classify-mnn-stb classify-mnn-stb.cpp)
target_link_libraries(classify-mnn-stb PRIVATE ${INFER_LIB})

add_executable(classify-mnn-stb2 classify-mnn-stb2.cpp)
target_link_libraries(classify-mnn-stb2 PRIVATE ${INFER_LIB} ${OpenCV_LIBS})

add_executable(classify-mnn-opencv classify-mnn-opencv.cpp)
target_link_libraries(classify-mnn-opencv PRIVATE ${INFER_LIB} ${OpenCV_LIBS})

add_executable(detect detect.cpp)
target_link_libraries(detect PRIVATE ${INFER_LIB} ${OpenCV_LIBS})