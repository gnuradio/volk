#
# Copyright 2018, 2020 Free Software Foundation, Inc.
#
# This file is part of VOLK
#
# SPDX-License-Identifier: LGPL-3.0-or-later
#

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

if(MINGW
   OR CYGWIN
   OR WIN32)
    set(UTIL_SEARCH_CMD where)
elseif(UNIX OR APPLE)
    set(UTIL_SEARCH_CMD which)
endif()

set(TOOLCHAIN_PREFIX aarch64-linux-gnu-)

execute_process(
    COMMAND ${UTIL_SEARCH_CMD} ${TOOLCHAIN_PREFIX}gcc
    OUTPUT_VARIABLE BINUTILS_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE)

get_filename_component(ARM_TOOLCHAIN_DIR ${BINUTILS_PATH} DIRECTORY)

# The following is not needed on debian
# Without that flag CMake is not able to pass test compilation check
#set(CMAKE_EXE_LINKER_FLAGS_INIT "--specs=nosys.specs")

set(CMAKE_C_COMPILER ${TOOLCHAIN_PREFIX}gcc)
set(CMAKE_ASM_COMPILER ${CMAKE_C_COMPILER})
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}g++)

set(CMAKE_OBJCOPY
    ${ARM_TOOLCHAIN_DIR}/${TOOLCHAIN_PREFIX}objcopy
    CACHE INTERNAL "objcopy tool")
set(CMAKE_SIZE_UTIL
    ${ARM_TOOLCHAIN_DIR}/${TOOLCHAIN_PREFIX}size
    CACHE INTERNAL "size tool")

execute_process(
    COMMAND ${CMAKE_C_COMPILER} -print-file-name=libc.so
    OUTPUT_VARIABLE AARCH64_LIBC
    OUTPUT_STRIP_TRAILING_WHITESPACE)
get_filename_component(AARCH64_LIBC ${AARCH64_LIBC} REALPATH)
get_filename_component(AARCH64_SYSROOT ${AARCH64_LIBC} DIRECTORY)
get_filename_component(AARCH64_SYSROOT ${AARCH64_SYSROOT} DIRECTORY)
get_filename_component(AARCH64_SYSROOT_PARENT ${AARCH64_SYSROOT} DIRECTORY)
string(REGEX REPLACE "-$" "" AARCH64_TRIPLE ${TOOLCHAIN_PREFIX})

set(CMAKE_FIND_ROOT_PATH ${AARCH64_SYSROOT} ${AARCH64_SYSROOT_PARENT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Keep pkg-config from resolving host libraries when cross-compiling.
set(ENV{PKG_CONFIG_LIBDIR}
    "${AARCH64_SYSROOT}/lib/pkgconfig:${AARCH64_SYSROOT}/usr/lib/pkgconfig:${AARCH64_SYSROOT_PARENT}/lib/${AARCH64_TRIPLE}/pkgconfig")

set(CMAKE_CROSSCOMPILING_EMULATOR qemu-aarch64 -L ${AARCH64_SYSROOT})
