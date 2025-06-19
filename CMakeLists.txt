cmake_minimum_required(VERSION 3.29)
project(solve_2048)

set(CMAKE_CXX_STANDARD 20)

add_executable(solve_2048 main.cpp
        StupidHashMap.h
        Position.h
        Position.cpp
        MoveLUT.h
        MoveLUT.cpp)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native -fopenmp -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fopenmp -g")

find_package(TBB REQUIRED)
find_library(ZSTD_LIB zstd REQUIRED)
find_path(ZSTD_INCLUDE_DIR zstd.h REQUIRED)

target_link_libraries(solve_2048 PRIVATE TBB::tbb)
target_include_directories(solve_2048 PRIVATE ${ZSTD_INCLUDE_DIR})
target_link_libraries(solve_2048 PRIVATE ${ZSTD_LIB})