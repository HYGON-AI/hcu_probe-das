set(PACKAGE_VERSION 3.10.1)

set(PKG_NAME nlohmannjson)
set(SHA256_VALUE "5c7d0a0542431fef628f8dc4c34fd022fe8747ccb577012d58f38672d8747e0d")
set(DOWNLOAD_PATH "$ENV{PROJECT_ROOT_PATH}/third_party")
set(DIR_NAME "${DOWNLOAD_PATH}/JSON-for-Modern-CPP-v3.10.1")

if (NOT ${PKG_NAME}_FOUND)

download_opensource_pkg(${PKG_NAME}
    SHA256 ${SHA256_VALUE}
    DOWNLOAD_PATH ${DOWNLOAD_PATH}
)

include_directories(${DIR_NAME}/include)
set(${PKG_NAME}_FOUND TRUE)

endif()
