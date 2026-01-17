cmake_minimum_required(VERSION 3.16)
project(mcts_cpp LANGUAGES CXX)

execute_process(COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
                OUTPUT_VARIABLE TORCH_PREFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE)

list(APPEND CMAKE_PREFIX_PATH "${TORCH_PREFIX}")
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Debug)

find_package(Python3 3.13 EXACT REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

set(NSHOGI_INCLUDE_DIR /usr/local/include)
find_path(NSHOGI_INCLUDE_DIR /usr/local/include)


set(SOURCES
    src/mcts.cc
    src/nn.cc
    src/encoder.cc
    src/types.cc
    src/bindings.cc
)

pybind11_add_module(mcts_cpp ${SOURCES})

target_include_directories(mcts_cpp PRIVATE
    ${NSHOGI_INCLUDE_DIR}
    ${Python3_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(mcts_cpp PRIVATE
    -O2
    -Wall
    -Wextra
    -Wno-unused-parameter
    -g
)

target_link_libraries(mcts_cpp PRIVATE
    pybind11::module
    Python3::Module
    "${TORCH_LIBRARIES}"
    nshogi
)
