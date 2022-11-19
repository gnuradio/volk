#
# Copyright 2022, 2024 Johannes Demel
#
# This file is part of VOLK.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
#

add_subdirectory(googletest)

find_package(fmt REQUIRED)

file(GLOB volk_test_files "test_*.cc")

add_executable(
  volk_tests
  volk_test.cc
  ${volk_test_files}
)

target_link_libraries(
  volk_tests
  gtest_main
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

# if(ENABLE_STATIC_LIBS)
#     target_link_libraries(volk_tests PRIVATE volk_static)
#     set_target_properties(volk_tests PROPERTIES LINK_FLAGS "-static")
# else()
#     target_link_libraries(volk_tests PRIVATE volk)
# endif()

# install(
#     TARGETS volk_tests
#     DESTINATION bin
#     COMPONENT "volk"
# )