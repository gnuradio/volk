#
# Copyright 2021 Free Software Foundation, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

add_subdirectory(googletest)

add_executable(
  volk_tests
  test_volk_32fc_x2_multiply_32fc.cc
)
target_link_libraries(
  volk_tests
  gtest_main
  volk
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