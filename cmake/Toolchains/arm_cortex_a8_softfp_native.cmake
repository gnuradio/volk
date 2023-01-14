#
# Copyright 2014, 2019 Free Software Foundation, Inc.
#
# This file is part of VOLK
#
# SPDX-License-Identifier: LGPL-3.0-or-later
#

########################################################################
# Toolchain file for building native on a ARM Cortex A8 w/ NEON
# Usage: cmake -DCMAKE_TOOLCHAIN_FILE=<this file> <source directory>
########################################################################
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_C_COMPILER  gcc)
set(CMAKE_CXX_FLAGS "-march=armv7-a -mtune=cortex-a8 -mfpu=neon -mfloat-abi=softfp" CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS ${CMAKE_CXX_FLAGS} CACHE STRING "" FORCE) #same flags for C sources
set(CMAKE_ASM_FLAGS "${CMAKE_CXX_FLAGS} -g" CACHE STRING "" FORCE) #same flags for asm sources