cmake_minimum_required(VERSION 3.29)
project(solve_2048)

set(CMAKE_CXX_STANDARD 20)

add_executable(solve_2048 main.cpp
        conc_hash_map.cc
        conc_hash_map.h)
