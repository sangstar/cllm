cmake_minimum_required(VERSION 4.0)
project(cllm C)

set(CMAKE_C_STANDARD 99)

add_executable(cllm main.c deserialize.c
)
