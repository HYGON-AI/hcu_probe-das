cmake_minimum_required(VERSION 3.22.2)
project(MindStudio-Probe)

execute_process(
    COMMAND uname -m
    OUTPUT_VARIABLE machine_arch
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if (DEFINED ARCH_TYPE AND NOT "${ARCH_TYPE}" STREQUAL "${machine_arch}")
    message(FATAL_ERROR "Cross-compilation is not supported currently. (compile ${ARCH_TYPE} on ${machine_arch})")
endif()


set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
set(ENV{PROJECT_ROOT_PATH} "${CMAKE_SOURCE_DIR}")
include(utils)
add_subdirectory(ccsrc)
