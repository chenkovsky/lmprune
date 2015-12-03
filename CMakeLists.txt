cmake_minimum_required(VERSION 2.8.4)
project(lmprune)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O3 -Wall -std=c++11")

file(GLOB DOCOPT_SRC docopt.cpp/docopt.cpp)
add_library(docopt ${DOCOPT_SRC})

add_executable(lmprune gram.cpp lm.cpp lm_io.cpp scissors.cpp lm_prune.cpp)
target_link_libraries(lmprune boost_iostreams z docopt)
