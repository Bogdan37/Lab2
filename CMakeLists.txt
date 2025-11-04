cmake_minimum_required(VERSION 3.16)
project(Lab2)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Додаємо Release оптимізації
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

add_executable(Lab2 main.cpp)