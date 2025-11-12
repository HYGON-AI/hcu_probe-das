set(PACKAGE_VERSION 1.12.1)

set(PKG_NAME gtest)
set(SHA256_VALUE "81964fe578e9bd7c94dfdb09c8e4d6e6759e19967e397dbea48d1c10e45d0df2")
set(DOWNLOAD_PATH "$ENV{PROJECT_ROOT_PATH}/third_party")
set(DIR_NAME "${DOWNLOAD_PATH}/googletest-release-1.12.1")

if (NOT ${PKG_NAME}_FOUND)

download_opensource_pkg(${PKG_NAME}
    SHA256 ${SHA256_VALUE}
    DOWNLOAD_PATH ${DOWNLOAD_PATH}
)

include_directories(${DIR_NAME}/googletest/include)
include_directories(${DIR_NAME}/googlemock/include)

set(BUILD_DEPENDENCY_PATH "$ENV{PROJECT_ROOT_PATH}/build_dependency")
execute_process(
    WORKING_DIRECTORY ${DIR_NAME}
    COMMAND cmake . -DBUILD_SHARED_LIBS=ON
    RESULT_VARIABLE RESULT
)
if (NOT RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to build gtest. ${RESULT}")
endif()
execute_process(
    WORKING_DIRECTORY ${DIR_NAME}
    COMMAND make -j16 
    RESULT_VARIABLE RESULT
)
if (NOT RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to build gtest. ${RESULT}")
endif()

file(GLOB GTEST_SO "${DIR_NAME}/lib/libgtest.so")
file(GLOB GMOCK_SO "${DIR_NAME}/lib/libgmock.so")
file(GLOB GTEST_MAIN_SO "${DIR_NAME}/lib/libgtest_main.so")
file(GLOB GMOCK_MAIN_SO "${DIR_NAME}/lib/libgmock_main.so")
if (NOT GTEST_SO OR NOT GMOCK_SO OR NOT GTEST_MAIN_SO OR NOT GMOCK_MAIN_SO)
    message(FATAL_ERROR "Failed to build gtest.")
endif()

set(${PKG_NAME}_LIBRARIES "${GTEST_SO};${GMOCK_SO};${GTEST_MAIN_SO};${GMOCK_MAIN_SO}")
set(${PKG_NAME}_FOUND TRUE)

endif()