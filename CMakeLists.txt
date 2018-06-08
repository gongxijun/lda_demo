cmake_minimum_required(VERSION 3.8)
project(lda_demo)

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/thirdparty/cppjieba/deps
        ${PROJECT_SOURCE_DIR}/thirdparty/cppjieba/include)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
ADD_SUBDIRECTORY(thirdparty/cppjieba)
set(SOURCE_FILES main.cpp)
add_executable(lda_demo ${SOURCE_FILES})