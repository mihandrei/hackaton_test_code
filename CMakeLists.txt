
cmake_minimum_required(VERSION 2.8.4)
project(hackatron)

if ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -g -O3 -march=native -Wall -fopenmp -Wno-unknown-pragmas")
elseif("${CMAKE_C_COMPILER_ID}" STREQUAL "PGI")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -acc -Minfo=accel -ta=nvidia -mp")
endif()

set(SOURCE_FILES
        
integrator.c
setup.c
sim.c

)

add_executable(hackatron ${SOURCE_FILES})
target_link_libraries(hackatron m)
