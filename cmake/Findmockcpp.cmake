set(PACKAGE_VERSION 2.7)

set(PKG_NAME mockcpp)
set(SHA256_VALUE "0dc7111c5be9785d0550ed3b68db7e12fd5d7802b7bc6548c52ac7b9e727fcc1")
set(DOWNLOAD_PATH "$ENV{PROJECT_ROOT_PATH}/third_party")
set(DIR_NAME "${DOWNLOAD_PATH}/mockcpp-v2.7")

if (NOT ${PKG_NAME}_FOUND)

download_opensource_pkg(${PKG_NAME}
    SHA256 ${SHA256_VALUE}
    DOWNLOAD_PATH ${DOWNLOAD_PATH}
)

include_directories(${DIR_NAME}/include)
include_directories(${DIR_NAME}/3rdparty)

execute_process(
    WORKING_DIRECTORY ${DIR_NAME}
    COMMAND cmake .
    RESULT_VARIABLE RESULT
)
if (NOT RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to build mockcpp. ${RESULT}")
endif()
execute_process(
    WORKING_DIRECTORY ${DIR_NAME}
    COMMAND make -j16 
    RESULT_VARIABLE RESULT
)
if (NOT RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to build mockcpp. ${RESULT}")
endif()

file(GLOB MOCKCPP_LIB "${DIR_NAME}/src/libmockcpp.a")
if (NOT MOCKCPP_LIB)
    message(FATAL_ERROR "Failed to build mockcpp.")
endif()

set(${PKG_NAME}_LIBRARIES "${MOCKCPP_LIB}")
set(${PKG_NAME}_FOUND TRUE)

endif()