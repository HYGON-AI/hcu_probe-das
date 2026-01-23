set(PKG_NAME securec)
set(SHA256_VALUE "1ae16ab92de4884eacea211dcd1989af95dae79b")
set(TAG "v1.1.16")
set(DOWNLOAD_PATH "$ENV{PROJECT_ROOT_PATH}/third_party")
set(DIR_NAME "${DOWNLOAD_PATH}/libboundscheck")

if (NOT ${PKG_NAME}_FOUND)

download_opensource_pkg(${PKG_NAME}
    SHA256 ${SHA256_VALUE}
    GIT_TAG ${TAG}
    DOWNLOAD_PATH ${DOWNLOAD_PATH}
)

include_directories(${DIR_NAME}/include)
set(SECUREC_DIR "${DIR_NAME}")
set(${PKG_NAME}_FOUND TRUE)

endif()
