cmake_minimum_required(VERSION 3.17)
project(pv021_FashionMNIST_project)

set(CMAKE_CXX_STANDARD 20)

add_executable(pv021_FashionMNIST_project src/main.cpp src/matrix.hpp src/matrix.cpp  src/IOUtils.hpp src/IOUtils.cpp src/network.hpp src/network.cpp src/tests.hpp src/tests.cpp)

set(CMAKE_CXX_FLAGS "-Wall -Werror -Ofast -ffp-contract=fast -funsafe-math-optimizations -march=native -msse2 -mfpmath=sse -ftree-vectorize -fopenmp")