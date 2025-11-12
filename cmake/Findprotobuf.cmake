set(PACKAGE_VERSION 3.15.0)

set(PKG_NAME protobuf)
set(SHA256_VALUE "a1ce078c369f46a3277fdc7ce462ac73cb7cb0edec8bc9d90d23fdb34491c575")
set(DOWNLOAD_PATH "$ENV{PROJECT_ROOT_PATH}/third_party")
set(DIR_NAME "${DOWNLOAD_PATH}/protobuf_source-v3.15.0")

if (NOT ${PKG_NAME}_FOUND)

if (DEFINED USE_LOCAL_FIRST AND "${USE_LOCAL_FIRST}" STREQUAL "True")
find_program(PROTOC_EXECUTABLE protoc)
find_package(Protobuf)
if (PROTOC_EXECUTABLE AND Protobuf_FOUND)
execute_process(
    COMMAND ${PROTOC_EXECUTABLE} --version
    OUTPUT_VARIABLE PROTOC_VERSION_OUTPUT
    ERROR_VARIABLE PROTOC_VERSION_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REGEX MATCH "[0-9]+\\.[0-9]+" PROTOC_VERSION ${PROTOC_VERSION_OUTPUT})
if(${PROTOC_VERSION} VERSION_GREATER_EQUAL ${PACKAGE_VERSION})
    message("Found protoc ${PROTOC_VERSION}, witch is equal or greater than the minimum required version ${PACKAGE_VERSION}. Use it instead.")
    set(PACKAGE_VERSION ${PROTOC_VERSION})
    set(${PKG_NAME}_FOUND TRUE)
    set(${PKG_NAME}_LIBRARIES ${Protobuf_LIBRARIES})
    set(PROTOC_EXECUTABLE ${PROTOC_EXECUTABLE})
    include_directories(${Protobuf_INCLUDE_DIRS})
    return()
endif()
endif()
endif()

download_opensource_pkg(${PKG_NAME}
    SHA256 ${SHA256_VALUE}
    DOWNLOAD_PATH ${DOWNLOAD_PATH}
)

include_directories(${DIR_NAME}/src)
set(BUILD_DEPENDENCY_PATH "$ENV{PROJECT_ROOT_PATH}/build_dependency")
file(GLOB PROTOC_EXECUTABLE "${BUILD_DEPENDENCY_PATH}/bin/protoc")
file(GLOB ${PKG_NAME}_LIBRARIES "${BUILD_DEPENDENCY_PATH}/lib/libprotobuf.a")
if (PROTOC_EXECUTABLE AND ${PKG_NAME}_LIBRARIES)
    set(${PKG_NAME}_FOUND TRUE)
    set(PROTOC_EXECUTABLE ${PROTOC_EXECUTABLE})
    set(${PKG_NAME}_LIBRARIES ${${PKG_NAME}_LIBRARIES})
    return()
endif()

execute_process(
    WORKING_DIRECTORY ${DIR_NAME}
    COMMAND ./autogen.sh
    RESULT_VARIABLE RESULT
)
if (NOT RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to build protobuf. ${RESULT}")
endif()

execute_process(
    WORKING_DIRECTORY ${DIR_NAME}
    COMMAND ./configure CFLAGS=-fPIC CXXFLAGS=-fPIC --prefix=${BUILD_DEPENDENCY_PATH} --enable-cpp
    RESULT_VARIABLE RESULT
)
if (NOT RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to build protobuf. ${RESULT}")
endif()

execute_process(
    WORKING_DIRECTORY ${DIR_NAME}
    COMMAND make -j16
    RESULT_VARIABLE RESULT
)
if (NOT RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to build protobuf. ${RESULT}")
endif()

execute_process(
    WORKING_DIRECTORY ${DIR_NAME}
    COMMAND make install
)

file(GLOB PROTOC_EXECUTABLE "${BUILD_DEPENDENCY_PATH}/bin/protoc")
file(GLOB ${PKG_NAME}_LIBRARIES "${BUILD_DEPENDENCY_PATH}/lib/libprotobuf.a")
if (NOT PROTOC_EXECUTABLE OR NOT ${PKG_NAME}_LIBRARIES)
    message(FATAL_ERROR "Failed to build protobuf.")
endif()

set(PROTOC_EXECUTABLE ${PROTOC_EXECUTABLE})
set(${PKG_NAME}_LIBRARIES ${${PKG_NAME}_LIBRARIES})
set(${PKG_NAME}_FOUND TRUE)

endif()
