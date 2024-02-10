# Copyright 2014, 2019, 2020 Free Software Foundation, Inc.
#
# This file is part of VOLK.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
#

find_package(PkgConfig)
pkg_check_modules(PC_ORC "orc-0.4 > 0.4.11")

include(GNUInstallDirs)

find_program(
    ORCC_EXECUTABLE orcc
    HINTS ${PC_ORC_TOOLSDIR}
    PATHS ${ORC_ROOT}/bin ${CMAKE_INSTALL_PREFIX}/bin)

find_path(
    ORC_INCLUDE_DIR
    NAMES orc/orc.h
    HINTS ${PC_ORC_INCLUDEDIR}
    PATHS ${ORC_ROOT}/include ${CMAKE_INSTALL_PREFIX}/include
    PATH_SUFFIXES orc-0.4)

find_path(
    ORC_LIBRARY_DIR
    NAMES ${CMAKE_SHARED_LIBRARY_PREFIX}orc-0.4${CMAKE_SHARED_LIBRARY_SUFFIX}
    HINTS ${PC_ORC_LIBDIR}
    PATHS ${ORC_ROOT}/${CMAKE_INSTALL_LIBDIR}
          ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})

find_library(
    ORC_LIB orc-0.4
    HINTS ${PC_ORC_LIBRARY_DIRS}
    PATHS ${ORC_ROOT}/${CMAKE_INSTALL_LIBDIR}
          ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})

find_library(
    ORC_LIBRARY_STATIC liborc-0.4.a
    HINTS ${PC_ORC_LIBRARY_DIRS}
    PATHS ${ORC_ROOT}/${CMAKE_INSTALL_LIBDIR}
          ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})

list(APPEND ORC_LIBRARY ${ORC_LIB})

set(ORC_INCLUDE_DIRS ${ORC_INCLUDE_DIR})
set(ORC_LIBRARIES ${ORC_LIBRARY})
set(ORC_LIBRARY_DIRS ${ORC_LIBRARY_DIR})
set(ORC_LIBRARIES_STATIC ${ORC_LIBRARY_STATIC})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ORC "orc files" ORC_LIBRARY ORC_INCLUDE_DIR
                                  ORCC_EXECUTABLE)

mark_as_advanced(ORC_INCLUDE_DIR ORC_LIBRARY ORCC_EXECUTABLE)
