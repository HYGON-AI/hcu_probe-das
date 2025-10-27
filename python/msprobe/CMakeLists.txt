add_subdirectory(ccsrc)

if (DEFINED BUILD_TEST_CASE AND "${BUILD_TEST_CASE}" STREQUAL "True")
add_subdirectory(test)
endif()