#
# Copyright 2022, 2024 Johannes Demel
#
# This file is part of VOLK.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
#

if(NOT ENABLE_TESTING)
    return()
endif(NOT ENABLE_TESTING)

include(FetchContent)

find_package(fmt QUIET)
    find_package(fmt QUIET)
    find_package(GTest QUIET)
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 10.2.1
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(fmt)
endif()

if(NOT GTest_FOUND)
  message(STATUS "GTest not found: fetching googletest via FetchContent")
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(googletest)
  find_package(GTest REQUIRED)
endif()

file(GLOB volk_test_files "test_*.cc")

add_executable(
  volk_tests
  volk_test.cc
  ${volk_test_files}
)

target_compile_features(volk_tests PUBLIC cxx_std_20)

target_link_libraries(volk_tests
  PRIVATE
    GTest::gtest_main
    volk
    fmt::fmt
)

include(GoogleTest)
gtest_discover_tests(volk_tests)


target_include_directories(volk_tests
    PRIVATE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    PRIVATE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/lib>
    PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib>
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)
