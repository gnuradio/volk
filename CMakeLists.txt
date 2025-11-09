#
# Copyright 2011-2020 Free Software Foundation, Inc.
# Copyright 2023 Magnus Lundmark <magnuslundmark@gmail.com>
#
# This file is part of VOLK
#
# SPDX-License-Identifier: LGPL-3.0-or-later
#

########################################################################
# Project setup
########################################################################
# We use `IS_64BIT now: https://cmake.org/cmake/help/latest/command/cmake_host_system_information.html
cmake_minimum_required(VERSION 3.22)

set(CMAKE_BUILD_TYPE
    ${CMAKE_BUILD_TYPE}
    CACHE STRING "Choose build type: None Debug Release RelWithDebInfo MinSizeRel")
project(volk)

enable_language(CXX)
enable_language(C)

enable_testing()

########################################################################
# Common compile flags
########################################################################

# Disable complex math NaN/INFO range checking for performance

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    # Abort compilation if kernel implementations have inconsistent function
    # prototypes, i.e. if
    #
    #     kernel_foo_sse(uint32_t *dst, lv32fc_t *src)
    #     kernel_foo_avx(uint16_t *dst, lv32fc_t *src)
    #
    # are defined. Note the different data type of the first argument). By
    # default 'incompatible-pointer-types' is a warning only and 'pointer-sign'
    # is a warning enabled by '-Wall'. These warnings are only applicable to C.
    set(CMAKE_C_FLAGS
        "${CMAKE_C_FLAGS} -Werror=incompatible-pointer-types -Werror=pointer-sign")
endif()

list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules
)#location for custom "Modules"

include(VolkBuildTypes)
#select the release build type by default to get optimization flags
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
    message(STATUS "Build type not specified: defaulting to release.")
endif()
volk_check_build_type(${CMAKE_BUILD_TYPE})
set(CMAKE_BUILD_TYPE
    ${CMAKE_BUILD_TYPE}
    CACHE STRING "")
message(STATUS "Build type set to ${CMAKE_BUILD_TYPE}.")

########################################################################
# Version setup
########################################################################

set(VERSION_INFO_MAJOR_VERSION 3)
set(VERSION_INFO_MINOR_VERSION 2)
set(VERSION_INFO_MAINT_VERSION 0)
include(VolkVersion) #setup version info

math(EXPR VOLK_VERSION_DECIMAL "${VERSION_INFO_MAJOR_VERSION} * 10000
    + ${VERSION_INFO_MINOR_VERSION} * 100
    + ${VERSION_INFO_MAINT_VERSION}")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/volk/volk_version.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/include/volk/volk_version.h @ONLY)

########################################################################
# Environment setup
########################################################################
if(NOT DEFINED CROSSCOMPILE_MULTILIB)
    set(CROSSCOMPILE_MULTILIB "")
endif()
set(CROSSCOMPILE_MULTILIB
    ${CROSSCOMPILE_MULTILIB}
    CACHE
        STRING
        "Define \"true\" if you have and want to use multiple C development libs installed for cross compile"
)

if(MSVC)
    add_definitions(-D_USE_MATH_DEFINES
    )#enables math constants on all supported versions of MSVC
    add_compile_options(/W1) #reduce warnings
    add_compile_options(/wo4309)
    add_compile_options(/wd4752)
    add_compile_options(/wo4273)
    add_compile_options(/wo4838)
endif(MSVC)

########################################################################
# Dependencies setup
########################################################################

# cpu_features - sensible defaults, user settable option
if(CMAKE_SYSTEM_PROCESSOR MATCHES
   "(^mips)|(^arm)|(^aarch64)|(x86_64)|(AMD64|amd64)|(^i.86$)|(^powerpc)|(^ppc)|(^riscv)")
    option(VOLK_CPU_FEATURES "Volk uses cpu_features" ON)
else()
    option(VOLK_CPU_FEATURES "Volk uses cpu_features" OFF)
endif()

if(VOLK_CPU_FEATURES)
    find_package(CpuFeatures QUIET)
    if(NOT CpuFeatures_FOUND)
        message(
            STATUS "cpu_features package not found. Requiring cpu_features submodule ...")
        if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cpu_features/CMakeLists.txt")
            message(
                FATAL_ERROR
                    "cpu_features/CMakeLists.txt not found. Did you forget to git clone recursively?\nFix with: git submodule update --init"
            )
        endif()
        message(STATUS "Building Volk with cpu_features")
        set(BUILD_TESTING
            OFF
            CACHE BOOL "Build cpu_features without tests." FORCE)
        set(BUILD_PIC
            ON
            CACHE BOOL "Build cpu_features with Position Independent Code (PIC)." FORCE)
        set(CMAKE_POSITION_INDEPENDENT_CODE
            ON
            CACHE BOOL "Build cpu_features with Position Independent Code (PIC)." FORCE)
        include(CheckCCompilerFlag)
        set(CMAKE_C_FLAGS_SAVED "${CMAKE_C_FLAGS}")
        check_c_compiler_flag(-Wno-unused-function HAVE_WNO_UNUSED_FUNCTION_CMD_LINE_ARG)
        if(HAVE_WNO_UNUSED_FUNCTION_CMD_LINE_ARG)
            set(CMAKE_C_FLAGS
                "${CMAKE_C_FLAGS} -Wno-unused-function -Wno-unused-variable")
        endif(HAVE_WNO_UNUSED_FUNCTION_CMD_LINE_ARG)

        set(BUILD_SHARED_LIBS OFF)
        set(BUILD_SHARED_LIBS_SAVED "${BUILD_SHARED_LIBS}")
        set(BUILD_SHARED_LIBS OFF)
        set(ENABLE_INSTALL OFF)
        add_subdirectory(cpu_features)
        set(BUILD_SHARED_LIBS "${BUILD_SHARED_LIBS_SAVED}")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS_SAVED}")
    endif()
else()
    message(STATUS "Building Volk without cpu_features")
endif()

# Python
include(VolkPython) #sets PYTHON_EXECUTABLE and PYTHON_DASH_B
volk_python_check_module("python >= 3.4" sys "sys.version_info >= (3, 4)"
                         PYTHON_MIN_VER_FOUND)
volk_python_check_module("mako >= 0.4.2" mako "mako.__version__ >= '0.4.2'" MAKO_FOUND)

if(NOT PYTHON_MIN_VER_FOUND)
    message(FATAL_ERROR "Python 3.4 or greater required to build VOLK")
endif()

# Mako
if(NOT MAKO_FOUND)
    message(FATAL_ERROR "Mako templates required to build VOLK")
endif()

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

########################################################################
# check for aligned_alloc, since some compilers lack this C11 feature.
# For Apple-clang use `posix_memalign`
# For MSVC use `_aligned_malloc`.
########################################################################
include(CheckSymbolExists)
if(NOT (${CMAKE_SYSTEM_NAME} MATCHES "Darwin"))
    check_symbol_exists(aligned_alloc stdlib.h USE_ALIGNED_ALLOC)
endif()
if(NOT USE_ALIGNED_ALLOC)
    check_symbol_exists(posix_memalign stdlib.h HAVE_POSIX_MEMALIGN)
endif()

########################################################################
# Check if Orc is available
########################################################################
option(ENABLE_ORC "Enable Orc" True)
if(ENABLE_ORC)
    find_package(ORC)
else(ENABLE_ORC)
    message(STATUS "Disabling use of ORC")
endif(ENABLE_ORC)

########################################################################
# Setup doxygen
########################################################################
add_subdirectory(docs)

########################################################################
# Detect /lib versus /lib64
########################################################################
include(GNUInstallDirs)

########################################################################
# Setup the package config file
########################################################################
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tmpl/volk.pc.in
               ${CMAKE_CURRENT_BINARY_DIR}/volk.pc @ONLY)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/volk.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
    COMPONENT "volk_devel")

########################################################################
# Install all headers in the include directories
########################################################################
set(VOLK_RUNTIME_DIR bin)
set(VOLK_LIBRARY_DIR ${CMAKE_INSTALL_LIBDIR})
set(VOLK_INCLUDE_DIR include)

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/kernels/volk
    DESTINATION include
    COMPONENT "volk_devel"
    FILES_MATCHING
    PATTERN "*.h")

install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/volk/volk_prefs.h
          ${CMAKE_CURRENT_SOURCE_DIR}/include/volk/volk_alloc.hh
          ${CMAKE_CURRENT_SOURCE_DIR}/include/volk/volk_complex.h
          ${CMAKE_CURRENT_SOURCE_DIR}/include/volk/volk_common.h
          ${CMAKE_CURRENT_SOURCE_DIR}/include/volk/saturation_arithmetic.h
          ${CMAKE_CURRENT_SOURCE_DIR}/include/volk/volk_avx_intrinsics.h
          ${CMAKE_CURRENT_SOURCE_DIR}/include/volk/volk_avx2_intrinsics.h
          ${CMAKE_CURRENT_SOURCE_DIR}/include/volk/volk_avx2_fma_intrinsics.h
          ${CMAKE_CURRENT_SOURCE_DIR}/include/volk/volk_avx512_intrinsics.h
          ${CMAKE_CURRENT_SOURCE_DIR}/include/volk/volk_sse_intrinsics.h
          ${CMAKE_CURRENT_SOURCE_DIR}/include/volk/volk_sse3_intrinsics.h
          ${CMAKE_CURRENT_SOURCE_DIR}/include/volk/volk_neon_intrinsics.h
          ${CMAKE_CURRENT_BINARY_DIR}/include/volk/volk.h
          ${CMAKE_CURRENT_BINARY_DIR}/include/volk/volk_cpu.h
          ${CMAKE_CURRENT_BINARY_DIR}/include/volk/volk_config_fixed.h
          ${CMAKE_CURRENT_BINARY_DIR}/include/volk/volk_typedefs.h
          ${CMAKE_CURRENT_SOURCE_DIR}/include/volk/volk_malloc.h
          ${CMAKE_CURRENT_BINARY_DIR}/include/volk/volk_version.h
          ${CMAKE_CURRENT_SOURCE_DIR}/include/volk/constants.h
    DESTINATION include/volk
    COMPONENT "volk_devel")

########################################################################
# On Apple only, set install name and use rpath correctly, if not already set
########################################################################
if(APPLE)
    if(NOT CMAKE_INSTALL_NAME_DIR)
        set(CMAKE_INSTALL_NAME_DIR
            ${CMAKE_INSTALL_PREFIX}/${VOLK_LIBRARY_DIR}
            CACHE PATH "Library Install Name Destination Directory" FORCE)
    endif(NOT CMAKE_INSTALL_NAME_DIR)
    if(NOT CMAKE_INSTALL_RPATH)
        set(CMAKE_INSTALL_RPATH
            ${CMAKE_INSTALL_PREFIX}/${VOLK_LIBRARY_DIR}
            CACHE PATH "Library Install RPath" FORCE)
    endif(NOT CMAKE_INSTALL_RPATH)
    if(NOT CMAKE_BUILD_WITH_INSTALL_RPATH)
        set(CMAKE_BUILD_WITH_INSTALL_RPATH
            ON
            CACHE BOOL "Do Build Using Library Install RPath" FORCE)
    endif(NOT CMAKE_BUILD_WITH_INSTALL_RPATH)
endif(APPLE)

########################################################################
# Create uninstall target
########################################################################
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in
               ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake @ONLY)

# Only add the target if there isn't one defined already
if(NOT TARGET uninstall)
    add_custom_target(uninstall ${CMAKE_COMMAND} -P
                                ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()

########################################################################
# Install our Cmake modules into $prefix/lib/cmake/volk
# See "Package Configuration Files" on page:
# https://gitlab.kitware.com/cmake/community/-/wikis/doc/tutorials/Packaging
########################################################################

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/VolkConfig.cmake.in
               ${CMAKE_CURRENT_BINARY_DIR}/cmake/Modules/VolkConfig.cmake @ONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/VolkConfigVersion.cmake.in
               ${CMAKE_CURRENT_BINARY_DIR}/cmake/Modules/VolkConfigVersion.cmake @ONLY)

########################################################################
# Install cmake search routine for external use
########################################################################

if(NOT CMAKE_MODULES_DIR)
    set(CMAKE_MODULES_DIR ${CMAKE_INSTALL_LIBDIR}/cmake)
endif(NOT CMAKE_MODULES_DIR)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/cmake/Modules/VolkConfig.cmake
          ${CMAKE_CURRENT_BINARY_DIR}/cmake/Modules/VolkConfigVersion.cmake
    DESTINATION ${CMAKE_MODULES_DIR}/volk
    COMPONENT "volk_devel")

install(
    EXPORT VOLK-export
    FILE VolkTargets.cmake
    NAMESPACE Volk::
    DESTINATION ${CMAKE_MODULES_DIR}/volk)

########################################################################
# Option to enable QA testing, on by default
########################################################################
option(ENABLE_TESTING "Enable QA testing" ON)
if(ENABLE_TESTING)
    message(STATUS "QA Testing is enabled.")
else()
    message(STATUS "QA Testing is disabled.")
endif()
message(STATUS "  Modify using: -DENABLE_TESTING=ON/OFF")

########################################################################
# Option to enable post-build profiling using volk_profile, off by default
########################################################################
option(ENABLE_PROFILING "Launch system profiler after build" OFF)
if(ENABLE_PROFILING)
    if(DEFINED VOLK_CONFIGPATH)
        get_filename_component(VOLK_CONFIGPATH ${VOLK_CONFIGPATH} ABSOLUTE)
        set(VOLK_CONFIGPATH "${VOLK_CONFIGPATH}/volk")
        message(STATUS "System profiling is enabled, using path: ${VOLK_CONFIGPATH}")
    elseif(DEFINED ENV{VOLK_CONFIGPATH})
        set(VOLK_CONFIGPATH "$ENV{VOLK_CONFIGPATH}/volk")
        message(
            STATUS "System profiling is enabled, using env path: $ENV{VOLK_CONFIGPATH}")
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
# Add tests
########################################################################
add_subdirectory(tests)

########################################################################
# And the utility apps
########################################################################
add_subdirectory(apps)
option(ENABLE_MODTOOL "Enable volk_modtool python utility" True)
if(ENABLE_MODTOOL)
    add_subdirectory(python/volk_modtool)
endif()

########################################################################
# Print summary
########################################################################
message(STATUS "Using install prefix: ${CMAKE_INSTALL_PREFIX}")
