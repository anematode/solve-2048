cmake_minimum_required(VERSION 3.29)
project(solve_2048)

set(CMAKE_CXX_STANDARD 20)

set(SOURCES StupidHashMap.h
        Position.h
        Position.cpp
        MoveLUT.h
        MoveLUT.cpp
        Compressor.h
        doctest.h
        Test.cpp
        libdivide.h)

add_library(solve_2048_lib ${SOURCES})
add_executable(solve_2048 main.cpp)
add_executable(test Test.cpp)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native -fopenmp -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fopenmp -g")

find_package(TBB REQUIRED)
find_library(ZSTD_LIB zstd REQUIRED)
find_path(ZSTD_INCLUDE_DIR zstd.h REQUIRED)

target_link_libraries(solve_2048_lib PRIVATE TBB::tbb)
target_include_directories(solve_2048_lib PRIVATE ${ZSTD_INCLUDE_DIR})
target_link_libraries(solve_2048_lib PRIVATE ${ZSTD_LIB})

target_link_libraries(solve_2048 solve_2048_lib)
target_link_libraries(test solve_2048_lib)