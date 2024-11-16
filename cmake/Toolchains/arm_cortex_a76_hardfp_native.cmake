#
# This file is part of VOLK
#
# SPDX-License-Identifier: LGPL-3.0-or-later
#

########################################################################
# Toolchain file for building native on a ARM Cortex A76 w/ NEON
# Usage: cmake -DCMAKE_TOOLCHAIN_FILE=<this file> <source directory>
########################################################################
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_FLAGS
    "-march=armv8.2-a -mtune=cortex-a76 -mfpu=neon-fp-armv8 -mfloat-abi=hard"
    CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS
    ${CMAKE_CXX_FLAGS}
    CACHE STRING "" FORCE) #same flags for C sources
set(CMAKE_ASM_FLAGS
    "${CMAKE_CXX_FLAGS} -mthumb -g"
    CACHE STRING "" FORCE) #same flags for asm sources
