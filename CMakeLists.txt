#
# Copyright 2011 Free Software Foundation, Inc.
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
# Project setup
########################################################################
cmake_minimum_required(VERSION 2.6)
set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose build type: None Debug Release RelWithDebInfo MinSizeRel")
project(volk)
enable_language(CXX)
enable_language(C)
enable_testing()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

set(CMAKE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}) #allows this to be a sub-project
set(CMAKE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}) #allows this to be a sub-project
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules) #location for custom "Modules"

include(VolkBuildTypes)
#select the release build type by default to get optimization flags
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
    message(STATUS "Build type not specified: defaulting to release.")
endif()
VOLK_CHECK_BUILD_TYPE(${CMAKE_BUILD_TYPE})
set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "")
message(STATUS "Build type set to ${CMAKE_BUILD_TYPE}.")

set(VERSION_INFO_MAJOR_VERSION 1)
set(VERSION_INFO_MINOR_VERSION 4)
set(VERSION_INFO_MAINT_VERSION 0)
include(VolkVersion) #setup version info

########################################################################
# Environment setup
########################################################################
IF(NOT DEFINED BOOST_ROOT AND NOT DEFINED ENV{BOOST_ROOT})
    SET(BOOST_ROOT ${CMAKE_INSTALL_PREFIX})
ENDIF()

IF(NOT DEFINED CROSSCOMPILE_MULTILIB)
    SET(CROSSCOMPILE_MULTILIB "")
ENDIF()
SET(CROSSCOMPILE_MULTILIB ${CROSSCOMPILE_MULTILIB} CACHE STRING "Define \"true\" if you have and want to use multiple C development libs installed for cross compile")

if(MSVC)
    add_definitions(-D_USE_MATH_DEFINES) #enables math constants on all supported versions of MSVC
    add_compile_options(/W1) #reduce warnings
    add_compile_options(/wo4309)
    add_compile_options(/wd4752)
    add_compile_options(/wo4273)
    add_compile_options(/wo4838)
endif(MSVC)

########################################################################
# Dependencies setup
########################################################################

# Python
include(VolkPython) #sets PYTHON_EXECUTABLE and PYTHON_DASH_B
VOLK_PYTHON_CHECK_MODULE("python >= 2.7" sys "sys.version.split()[0] >= '2.7'" PYTHON_MIN_VER_FOUND)
VOLK_PYTHON_CHECK_MODULE("mako >= 0.4.2" mako "mako.__version__ >= '0.4.2'" MAKO_FOUND)
VOLK_PYTHON_CHECK_MODULE("six - python 2 and 3 compatibility library" six "True" SIX_FOUND)

if(NOT PYTHON_MIN_VER_FOUND)
    message(FATAL_ERROR "Python 2.7 or greater required to build VOLK")
endif()

# Mako
if(NOT MAKO_FOUND)
    message(FATAL_ERROR "Mako templates required to build VOLK")
endif()

# Six
if(NOT SIX_FOUND)
    message(FATAL_ERROR "six - python 2 and 3 compatibility library required to build VOLK")
endif()

# Boost
if(MSVC)
    if (NOT DEFINED BOOST_ALL_DYN_LINK)
        set(BOOST_ALL_DYN_LINK TRUE)
    endif()
    set(BOOST_ALL_DYN_LINK "${BOOST_ALL_DYN_LINK}" CACHE BOOL "boost enable dynamic linking")
    if(BOOST_ALL_DYN_LINK)
        add_definitions(-DBOOST_ALL_DYN_LINK) #setup boost auto-linking in msvc
    else(BOOST_ALL_DYN_LINK)
        unset(BOOST_REQUIRED_COMPONENTS) #empty components list for static link
    endif(BOOST_ALL_DYN_LINK)
endif(MSVC)

include(VolkBoost)

if(NOT Boost_FOUND)
    message(FATAL_ERROR "VOLK Requires boost to build")
endif()

# Orc
option(ENABLE_ORC "Enable Orc" True)
if(ENABLE_ORC)
  find_package(ORC)
else(ENABLE_ORC)
  message(STATUS "Disabling use of ORC")
endif(ENABLE_ORC)

########################################################################
# Setup doxygen
########################################################################
find_package(Doxygen)
if(DOXYGEN_FOUND)
    configure_file(
        ${CMAKE_SOURCE_DIR}/Doxyfile.in
        ${CMAKE_BINARY_DIR}/Doxyfile
    @ONLY)

    add_custom_target(volk_doc
        ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating documentation with Doxygen" VERBATIM
    )
endif(DOXYGEN_FOUND)

########################################################################
# Setup the package config file
########################################################################
#set variables found in the pc.in file
set(prefix ${CMAKE_INSTALL_PREFIX})
set(exec_prefix "\${prefix}")
set(libdir "\${exec_prefix}/lib${LIB_SUFFIX}")
set(includedir "\${prefix}/include")

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/volk.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/volk.pc
@ONLY)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/volk.pc
    DESTINATION lib${LIB_SUFFIX}/pkgconfig
    COMPONENT "volk_devel"
)

########################################################################
# Install all headers in the include directories
########################################################################
set(VOLK_RUNTIME_DIR   bin)
set(VOLK_LIBRARY_DIR   lib${LIB_SUFFIX})
set(VOLK_INCLUDE_DIR   include)

install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/kernels/volk
    DESTINATION include COMPONENT "volk_devel"
    FILES_MATCHING PATTERN "*.h"
)

install(FILES
    ${CMAKE_SOURCE_DIR}/include/volk/volk_prefs.h
    ${CMAKE_SOURCE_DIR}/include/volk/volk_complex.h
    ${CMAKE_SOURCE_DIR}/include/volk/volk_common.h
    ${CMAKE_SOURCE_DIR}/include/volk/saturation_arithmetic.h
    ${CMAKE_SOURCE_DIR}/include/volk/volk_avx_intrinsics.h
    ${CMAKE_SOURCE_DIR}/include/volk/volk_sse_intrinsics.h
    ${CMAKE_SOURCE_DIR}/include/volk/volk_sse3_intrinsics.h
    ${CMAKE_SOURCE_DIR}/include/volk/volk_neon_intrinsics.h
    ${CMAKE_BINARY_DIR}/include/volk/volk.h
    ${CMAKE_BINARY_DIR}/include/volk/volk_cpu.h
    ${CMAKE_BINARY_DIR}/include/volk/volk_config_fixed.h
    ${CMAKE_BINARY_DIR}/include/volk/volk_typedefs.h
    ${CMAKE_SOURCE_DIR}/include/volk/volk_malloc.h
    DESTINATION include/volk
    COMPONENT "volk_devel"
)

########################################################################
# On Apple only, set install name and use rpath correctly, if not already set
########################################################################
if(APPLE)
    if(NOT CMAKE_INSTALL_NAME_DIR)
        set(CMAKE_INSTALL_NAME_DIR
            ${CMAKE_INSTALL_PREFIX}/${VOLK_LIBRARY_DIR} CACHE
            PATH "Library Install Name Destination Directory" FORCE)
    endif(NOT CMAKE_INSTALL_NAME_DIR)
    if(NOT CMAKE_INSTALL_RPATH)
        set(CMAKE_INSTALL_RPATH
            ${CMAKE_INSTALL_PREFIX}/${VOLK_LIBRARY_DIR} CACHE
            PATH "Library Install RPath" FORCE)
    endif(NOT CMAKE_INSTALL_RPATH)
    if(NOT CMAKE_BUILD_WITH_INSTALL_RPATH)
        set(CMAKE_BUILD_WITH_INSTALL_RPATH ON CACHE
            BOOL "Do Build Using Library Install RPath" FORCE)
    endif(NOT CMAKE_BUILD_WITH_INSTALL_RPATH)
endif(APPLE)

########################################################################
# Create uninstall target
########################################################################
configure_file(
    ${CMAKE_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
@ONLY)

# Only add the target if there isn't one defined already
if(NOT TARGET uninstall)
    add_custom_target(uninstall
        ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
    )
endif()


########################################################################
# Install our Cmake modules into $prefix/lib/cmake/volk
# See "Package Configuration Files" on page:
#    http://www.cmake.org/Wiki/CMake/Tutorials/Packaging
########################################################################

configure_file(
  ${CMAKE_SOURCE_DIR}/cmake/Modules/VolkConfig.cmake.in
  ${CMAKE_BINARY_DIR}/cmake/Modules/VolkConfig.cmake
@ONLY)

configure_file(
  ${CMAKE_SOURCE_DIR}/cmake/Modules/VolkConfigVersion.cmake.in
  ${CMAKE_BINARY_DIR}/cmake/Modules/VolkConfigVersion.cmake
@ONLY)


########################################################################
# Install cmake search routine for external use
########################################################################

if(NOT CMAKE_MODULES_DIR)
    set(CMAKE_MODULES_DIR lib${LIB_SUFFIX}/cmake)
endif(NOT CMAKE_MODULES_DIR)

install(
    FILES
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/Modules/VolkConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/Modules/VolkConfigVersion.cmake
    DESTINATION ${CMAKE_MODULES_DIR}/volk
    COMPONENT "volk_devel"
)

########################################################################
# Option to enable QA testing, on by default
########################################################################
OPTION(ENABLE_TESTING "Enable QA testing" ON)
if(ENABLE_TESTING)
  message(STATUS "QA Testing is enabled.")
else()
  message(STATUS "QA Testing is disabled.")
endif()
message(STATUS "  Modify using: -DENABLE_TESTING=ON/OFF")

########################################################################
# Option to enable post-build profiling using volk_profile, off by default
########################################################################
OPTION(ENABLE_PROFILING "Launch system profiler after build" OFF)
if(ENABLE_PROFILING)
  if(DEFINED VOLK_CONFIGPATH)
    get_filename_component(VOLK_CONFIGPATH ${VOLK_CONFIGPATH} ABSOLUTE)
    set(VOLK_CONFIGPATH "${VOLK_CONFIGPATH}/volk")
    message(STATUS "System profiling is enabled, using path: ${VOLK_CONFIGPATH}")
  elseif(DEFINED ENV{VOLK_CONFIGPATH})
    set(VOLK_CONFIGPATH "$ENV{VOLK_CONFIGPATH}/volk")
    message(STATUS "System profiling is enabled, using env path: $ENV{VOLK_CONFIGPATH}")
  else()
    message(STATUS "System profiling is enabled with default paths.")
    if(DEFINED ENV{HOME})
        set(VOLK_CONFIGPATH "$ENV{HOME}/.volk")
    elseif(DEFINED ENV{APPDATA})
        set(VOLK_CONFIGPATH "$ENV{APPDATA}/.volk")
    endif()
  endif()
else()
  message(STATUS "System profiling is disabled.")
endif()
message(STATUS "  Modify using: -DENABLE_PROFILING=ON/OFF")

########################################################################
# Setup the library
########################################################################
add_subdirectory(lib)

########################################################################
# And the utility apps
########################################################################
add_subdirectory(apps)
add_subdirectory(python/volk_modtool)

########################################################################
# Print summary
########################################################################
message(STATUS "Using install prefix: ${CMAKE_INSTALL_PREFIX}")
