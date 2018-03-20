cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(kalmanfilter)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")

# Eigen library
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(
    ./
    )

add_executable(
    example
    main.cpp
    )

target_link_libraries(
    example
    Eigen3::Eigen
    )

# Run unit tests
add_custom_target(run ./example -c DEPENDS example)
