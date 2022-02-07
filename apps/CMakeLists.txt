#
# Copyright 2011-2013 Free Software Foundation, Inc.
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

########################################################################
# Setup profiler
########################################################################

# POSIX_MEMALIGN: If we have to fall back to `posix_memalign`.
if(HAVE_POSIX_MEMALIGN)
    message(STATUS "Use `posix_memalign` for aligned malloc!")
    add_definitions(-DHAVE_POSIX_MEMALIGN)
endif(HAVE_POSIX_MEMALIGN)

# MAKE volk_profile
add_executable(volk_profile
    ${CMAKE_CURRENT_SOURCE_DIR}/volk_profile.cc
    ${PROJECT_SOURCE_DIR}/lib/qa_utils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/volk_option_helpers.cc
)

if(MSVC)
    target_include_directories(volk_profile
        PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cmake/msvc>
    )
endif(MSVC)

target_include_directories(volk_profile
    PRIVATE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    PRIVATE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/lib>
    PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib>
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

add_definitions(-DHAS_STD_FILESYSTEM=1)
if(${find_experimental})
    add_definitions(-DHAS_STD_FILESYSTEM_EXPERIMENTAL=1)
endif()
target_link_libraries(volk_profile PRIVATE std::filesystem)

if(ENABLE_STATIC_LIBS)
    target_link_libraries(volk_profile PRIVATE volk_static)
    set_target_properties(volk_profile PROPERTIES LINK_FLAGS "-static")
else()
    target_link_libraries(volk_profile PRIVATE volk)
endif()

install(
    TARGETS volk_profile
    DESTINATION bin
    COMPONENT "volk"
)

# MAKE volk-config-info
add_executable(volk-config-info volk-config-info.cc ${CMAKE_CURRENT_SOURCE_DIR}/volk_option_helpers.cc
        )

if(ENABLE_STATIC_LIBS)
    target_link_libraries(volk-config-info volk_static)
    set_target_properties(volk-config-info PROPERTIES LINK_FLAGS "-static")
else()
    target_link_libraries(volk-config-info volk)
endif()

install(
    TARGETS volk-config-info
    DESTINATION bin
    COMPONENT "volk"
)

# Launch volk_profile if requested to do so
if(ENABLE_PROFILING)
   if(DEFINED VOLK_CONFIGPATH)
        set( VOLK_CONFIG_ARG "-p${VOLK_CONFIGPATH}" )
        set( VOLK_CONFIG "${VOLK_CONFIGPATH}/volk_config" )
   endif()

   add_custom_command(OUTPUT ${VOLK_CONFIG}
        COMMAND volk_profile "${VOLK_CONFIG_ARG}"
        DEPENDS volk_profile
        COMMENT "Launching profiler, this may take a few minutes..."
    )
    add_custom_target(volk-profile-run ALL DEPENDS ${VOLK_CONFIG})

endif()
