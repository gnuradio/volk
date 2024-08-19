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

find_package(fmt)
find_package(GTest)

if(NOT fmt_FOUND OR NOT GTest_FOUND)
    message(warning "Missing fmtlib and/or googletest for this test suite")
    return()
endif(NOT fmt_FOUND OR NOT GTest_FOUND)

file(GLOB volk_test_files "test_*.cc")

add_executable(
  volk_tests
  volk_test.cc
  ${volk_test_files}
)

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
