set(PACKAGE_VERSION 1.1.1)

set(PKG_NAME openssl)
set(SHA256_VALUE "b92f9d3d12043c02860e5e602e50a73ed21a69947bcc74d391f41148e9f6aa95")
set(DOWNLOAD_PATH "$ENV{PROJECT_ROOT_PATH}/third_party")
set(DIR_NAME "${DOWNLOAD_PATH}/openssl-OpenSSL_1_1_1k")

if (NOT ${PKG_NAME}_FOUND)

if (DEFINED USE_LOCAL_FIRST AND "${USE_LOCAL_FIRST}" STREQUAL "True")
find_package(OpenSSL)
if (OpenSSL_FOUND AND OPENSSL_INCLUDE_DIR AND OPENSSL_LIBRARIES)
    if (${OPENSSL_VERSION} VERSION_GREATER_EQUAL ${PACKAGE_VERSION})
        message("Found openssl ${OPENSSL_VERSION}, witch is equal or greater than the minimum required version ${PACKAGE_VERSION}. Use it instead.")
        set(PACKAGE_VERSION ${PACKAGE_VERSION})
        set(${PKG_NAME}_FOUND TRUE)
        include_directories(${OPENSSL_INCLUDE_DIR})
        set(${PKG_NAME}_LIBRARIES ${OPENSSL_LIBRARIES})
        return()
    endif()
endif()
endif()

download_opensource_pkg(${PKG_NAME}
    SHA256 ${SHA256_VALUE}
    DOWNLOAD_PATH ${DOWNLOAD_PATH}
)

include_directories(${DIR_NAME}/include)
set(BUILD_DEPENDENCY_PATH "$ENV{PROJECT_ROOT_PATH}/build_dependency")
file(GLOB OPENSSL_LIB "${BUILD_DEPENDENCY_PATH}/lib/libssl.a")
file(GLOB CRYPTO_LIB "${BUILD_DEPENDENCY_PATH}/lib/libcrypto.a")
if (OPENSSL_LIB AND CRYPTO_LIB)
    set(${PKG_NAME}_FOUND TRUE)
    set(${PKG_NAME}_LIBRARIES "${OPENSSL_LIB};${CRYPTO_LIB}")
    return()
endif()

execute_process(
    WORKING_DIRECTORY ${DIR_NAME}
    COMMAND ./config -fPIC no-shared --prefix=${BUILD_DEPENDENCY_PATH}
    RESULT_VARIABLE RESULT
)
if (NOT RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to build openssl. ${RESULT}")
endif()

execute_process(
    WORKING_DIRECTORY ${DIR_NAME}
    COMMAND make -j16
    RESULT_VARIABLE RESULT
)
if (NOT RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to build openssl. ${RESULT}")
endif()

execute_process(
    WORKING_DIRECTORY ${DIR_NAME}
    COMMAND make install
)

file(GLOB OPENSSL_LIB "${BUILD_DEPENDENCY_PATH}/lib/libssl.a")
file(GLOB CRYPTO_LIB "${BUILD_DEPENDENCY_PATH}/lib/libcrypto.a")
if (NOT OPENSSL_LIB OR NOT CRYPTO_LIB)
    message(FATAL_ERROR "Failed to build openssl.")
endif()

set(${PKG_NAME}_LIBRARIES "${OPENSSL_LIB};${CRYPTO_LIB}")
set(${PKG_NAME}_FOUND TRUE)

endif()
