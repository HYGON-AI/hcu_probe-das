set(PKG_NAME re2)
set(SHA256_VALUE "7268e1b4254d9ffa5ccf010fee954150dbb788fd9705234442e7d9f0ee5a42d3")
set(DOWNLOAD_PATH "$ENV{PROJECT_ROOT_PATH}/third_party")
set(DIR_NAME "${DOWNLOAD_PATH}/re2-2019-12-01")
set(BUILD_DIR "${DIR_NAME}/build")
file(MAKE_DIRECTORY "${BUILD_DIR}")
set(BUILD_DEPENDENCY_PATH "$ENV{PROJECT_ROOT_PATH}/build_dependency/${PKG_NAME}")

if (NOT ${PKG_NAME}_FOUND)

file(GLOB RE2_INCLUDE "${BUILD_DEPENDENCY_PATH}/include/${PKG_NAME}/re2.h")
file(GLOB_RECURSE RE2_LIB "${BUILD_DEPENDENCY_PATH}/*libre2.a")
if (RE2_INCLUDE AND RE2_LIB)
    include_directories(${BUILD_DEPENDENCY_PATH}/include)
    set(${PKG_NAME}_LIBRARIES "${RE2_LIB}")
    set(${PKG_NAME}_FOUND TRUE)
    return()
endif()

download_opensource_pkg(${PKG_NAME}
    SHA256 ${SHA256_VALUE}
    DOWNLOAD_PATH ${DOWNLOAD_PATH}
)

execute_process(
    WORKING_DIRECTORY ${BUILD_DIR}
    COMMAND cmake -DCMAKE_INSTALL_PREFIX=${BUILD_DEPENDENCY_PATH} -DCMAKE_C_FLAGS=-fPIC -DCMAKE_CXX_FLAGS=-fPIC ..
    RESULT_VARIABLE RESULT
)
if (NOT RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to build re2. ${RESULT}")
endif()

execute_process(
    WORKING_DIRECTORY ${BUILD_DIR}
    COMMAND make -j16
    RESULT_VARIABLE RESULT
)
if (NOT RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to build re2. ${RESULT}")
endif()

execute_process(
    WORKING_DIRECTORY ${BUILD_DIR}
    COMMAND make install
)

file(GLOB RE2_INCLUDE "${BUILD_DEPENDENCY_PATH}/include/${PKG_NAME}/re2.h")
file(GLOB_RECURSE RE2_LIB "${BUILD_DEPENDENCY_PATH}/*libre2.a")
if (NOT RE2_INCLUDE OR NOT RE2_LIB)
    message(FATAL_ERROR "Failed to build re2.")
endif()

include_directories(${BUILD_DEPENDENCY_PATH}/include)
set(${PKG_NAME}_LIBRARIES "${RE2_LIB}")
set(${PKG_NAME}_FOUND TRUE)

endif()
