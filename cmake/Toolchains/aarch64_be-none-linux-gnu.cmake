#
# Copyright 2026 Zetier Inc
#
# This file is part of VOLK
#
# SPDX-License-Identifier: LGPL-3.0-or-later
#

# Big-endian AArch64 with the Arm GNU Toolchain (aarch64_be-none-linux-gnu),
# whose bin directory must be on the PATH. Tests run under qemu-aarch64_be.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64_be)

set(TOOLCHAIN_PREFIX aarch64_be-none-linux-gnu-)

set(CMAKE_C_COMPILER ${TOOLCHAIN_PREFIX}gcc)
set(CMAKE_ASM_COMPILER ${CMAKE_C_COMPILER})
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}g++)

# The toolchain ships its own glibc sysroot; libstdc++ and libgcc_s sit
# outside it, next to the sysroot in the toolchain's target directory.
execute_process(
    COMMAND ${CMAKE_C_COMPILER} -print-sysroot
    RESULT_VARIABLE TOOLCHAIN_SYSROOT_RESULT
    OUTPUT_VARIABLE TOOLCHAIN_SYSROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT TOOLCHAIN_SYSROOT_RESULT EQUAL 0 OR TOOLCHAIN_SYSROOT STREQUAL "")
    message(FATAL_ERROR "Could not run ${CMAKE_C_COMPILER} -print-sysroot; "
                        "put the toolchain's bin directory on the PATH")
endif()
get_filename_component(TOOLCHAIN_SYSROOT "${TOOLCHAIN_SYSROOT}" REALPATH)
get_filename_component(TOOLCHAIN_TARGET_DIR "${TOOLCHAIN_SYSROOT}" DIRECTORY)

set(CMAKE_FIND_ROOT_PATH ${TOOLCHAIN_SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# One string, as VolkAddTest pastes it into each test's shell script. The -E
# replaces the scripts' LD_LIBRARY_PATH; the build tree's RPATH still finds
# libvolk.
set(CMAKE_CROSSCOMPILING_EMULATOR
    "qemu-aarch64_be -L ${TOOLCHAIN_SYSROOT} -E LD_LIBRARY_PATH=${TOOLCHAIN_TARGET_DIR}/lib64")
