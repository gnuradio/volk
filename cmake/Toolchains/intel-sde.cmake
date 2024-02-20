#
# Copyright 2019 Free Software Foundation, Inc.
#
# This file is part of VOLK
#
# SPDX-License-Identifier: LGPL-3.0-or-later
#

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=knl")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=knl")
set(CMAKE_CROSSCOMPILING_EMULATOR
    "$ENV{TRAVIS_BUILD_DIR}/cache/$ENV{SDE_VERSION}/sde64 -knl --")
