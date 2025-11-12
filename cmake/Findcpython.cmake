set(PKG_NAME cpython)

if (NOT ${PKG_NAME}_FOUND)

find_package(Python3 ${PYTHON_VERSION} EXACT COMPONENTS Development)
if (NOT Python3_FOUND)
    message(FATAL_ERROR "${Python3} is not found.")
endif()

set(PACKAGE_VERSION ${Python3_VERSION})

include_directories(${Python3_INCLUDE_DIRS})
set(${PKG_NAME}_LIBRARIES ${Python3_LIBRARIES})
set(${PKG_NAME}_FOUND TRUE)

endif()
