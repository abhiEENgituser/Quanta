cmake_minimum_required(VERSION 3.16)
project(quanta_shim CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(vendor/llama.cpp)

add_executable(quanta_shim
    src/main.cpp
)

target_link_libraries(quanta_shim PRIVATE llama)

add_executable(probe
    src/probe.cpp
)

target_link_libraries(probe PRIVATE llama)
