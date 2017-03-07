
cmake_minimum_required(VERSION 2.8.4)
project(hackatron)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -lm")

set(SOURCE_FILES
        
integrator.c
modelepi.c
rk_mt.c
setup.c
sim.c

)

add_executable(hackatron ${SOURCE_FILES})
