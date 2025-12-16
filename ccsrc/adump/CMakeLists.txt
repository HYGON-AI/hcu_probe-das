project(msprobe VERSION 1.0.0 LANGUAGES CXX C)
cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(ASCEND_TOOLKIT_PATH $ENV{ASCEND_TOOLKIT_HOME})

find_package(cpython MODULE REQUIRED)
find_package(openssl MODULE REQUIRED)
find_package(nlohmannjson MODULE REQUIRED)
find_package(protobuf MODULE REQUIRED)
find_package(re2 MODULE REQUIRED)

find_library(adump_server NAMES adump_server PATHS ${ASCEND_TOOLKIT_PATH}/lib64 REQUIRED)

if (DEFINED CANN_PATH AND NOT "${CANN_PATH}" STREQUAL "")
    file(GLOB_RECURSE DUMP_DATA_PROTOS "${CANN_PATH}/**/dump_data.proto")
    if (DUMP_DATA_PROTOS)
        list(GET DUMP_DATA_PROTOS 0 DUMP_DATA_PROTO)
        file(COPY "${DUMP_DATA_PROTO}" DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ACL/AclDumpMsg.proto")
    else()
        message("Warning: File dump_data.proto not found.")
    endif()
endif()

set(PROTO_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/proto)
file(GLOB_RECURSE PROTO_SRC "*.proto")
compile_protobuf_file(
    ${PROTO_PATH}
    ${PROTO_SRC}
)

set(CMAKE_SKIP_RPATH TRUE)

add_library(_msprobe_c SHARED)

target_compile_options(_msprobe_c PRIVATE "-Wall")
target_compile_options(_msprobe_c PRIVATE "-fPIC")
target_compile_options(_msprobe_c PRIVATE "-fstack-protector-all")
target_compile_options(_msprobe_c PRIVATE "-ftrapv")
target_compile_options(_msprobe_c PRIVATE "-fstack-check")
target_compile_options(_msprobe_c PRIVATE "-D_FORTIFY_SOURCE=2")

target_link_options(_msprobe_c PRIVATE "-Wl,-z,relro")
target_link_options(_msprobe_c PRIVATE "-Wl,-z,now")
target_link_options(_msprobe_c PRIVATE "-Wl,-z,noexecstack")

target_link_libraries(_msprobe_c PRIVATE ${adump_server})
target_link_libraries(_msprobe_c PUBLIC dl)
target_link_libraries(_msprobe_c PUBLIC pthread)
target_link_libraries(_msprobe_c PUBLIC ${cpython_LIBRARIES})
target_link_libraries(_msprobe_c PUBLIC ${openssl_LIBRARIES})
target_link_libraries(_msprobe_c PUBLIC ${protobuf_LIBRARIES})
target_link_libraries(_msprobe_c PUBLIC ${re2_LIBRARIES})

if(DEFINED BUILD_TYPE AND "${BUILD_TYPE}" STREQUAL "debug")
    target_compile_options(_msprobe_c PRIVATE "-O0")
    target_compile_options(_msprobe_c PRIVATE "-g")
    target_compile_definitions(_msprobe_c PRIVATE __DEBUG__)
else()
    target_compile_options(_msprobe_c PRIVATE "-O2")
    target_link_options(_msprobe_c PRIVATE "-s")
endif()

target_include_directories(_msprobe_c PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB_RECURSE SOURCES "*.cpp" "*.cc")
target_sources(_msprobe_c PRIVATE ${SOURCES})

install(TARGETS _msprobe_c LIBRARY DESTINATION lib)
