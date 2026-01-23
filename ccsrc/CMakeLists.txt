cmake_minimum_required(VERSION 3.22.2)
project(ccsrc)

if ($ENV{MSPROBE_INCLUDE_MOD} STREQUAL "adump")
    add_subdirectory(adump)
endif()

if ($ENV{MSPROBE_INCLUDE_MOD} STREQUAL "atb_probe")
    add_subdirectory(atb_probe)
endif()
