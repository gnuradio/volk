[![Build Status](https://travis-ci.com/gnuradio/volk.svg?branch=master)](https://travis-ci.com/gnuradio/volk) [![Build status](https://ci.appveyor.com/api/projects/status/5o56mgw0do20jlh3/branch/master?svg=true)](https://ci.appveyor.com/project/gnuradio/volk/branch/master)
![Check PR Formatting](https://github.com/gnuradio/volk/workflows/Check%20PR%20Formatting/badge.svg)
![Run VOLK tests](https://github.com/gnuradio/volk/workflows/Run%20VOLK%20tests/badge.svg)

![VOLK Logo](/docs/volk_logo.png)

# Welcome to VOLK!

VOLK is a sub-project of GNU Radio. Please see http://libvolk.org for bug
tracking, documentation, source code, and contact information about VOLK.
See https://www.gnuradio.org/ for information about GNU Radio.

VOLK is the Vector-Optimized Library of Kernels. It is a library that contains kernels of hand-written SIMD code for different mathematical operations. Since each SIMD architecture can be very different and no compiler has yet come along to handle vectorization properly or highly efficiently, VOLK approaches the problem differently.

For each architecture or platform that a developer wishes to vectorize for, a new proto-kernel is added to VOLK. At runtime, VOLK will select the correct proto-kernel. In this way, the users of VOLK call a kernel for performing the operation that is platform/architecture agnostic. This allows us to write portable SIMD code.

Bleeding edge code can be found in our git repository at
https://www.gnuradio.org/git/volk.git/.

## How to use VOLK:

For detailed instructions see http://libvolk.org/doxygen/using_volk.html

See these steps for a quick build guide.

### Building on most x86 (32-bit and 64-bit) platforms

```bash
$ mkdir build
$ cd build
$ cmake ..
$ make
$ make test
$ sudo make install

# You might want to explore "make -j[SOMEVALUE]" options for your multicore CPU.

# Perform post-installation steps

# Linux OS: Link and cache shared library
$ sudo ldconfig

# macOS/Windows: Update PATH environment variable to point to lib install location

# volk_profile will profile your system so that the best kernel is used
$ volk_profile
```

#### Missing submodule
We use [cpu_features](https://github.com/google/cpu_features) as a git submodule to detect CPU features, e.g. AVX.
There are two options to get the required code:
```bash
git clone --recursive https://github.com/gnuradio/volk.git
```
will automatically clone submodules as well.
In case you missed that, you can just run:
```bash
git submodule update --init --recursive
```
that'll pull in missing submodules.


### Building on Raspberry Pi and other ARM boards (32 bit)

To build for these boards you need specify the correct cmake toolchain file for best performance.

_Note: There is no need for adding extra options to the compiler for 64 bit._

* Raspberry Pi 4 `arm_cortex_a72_hardfp_native.cmake`
* Raspberry Pi 3 `arm_cortex_a53_hardfp_native.cmake`

```bash
$ mkdir build && cd build
$ cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/Toolchains/arm_cortex_a72_hardfp_native.cmake ..
# make -j4 might be faster
$ make
$ make test
$ sudo make install

# volk_profile will profile your system so that the best kernel is used
$ volk_profile
```

## Supported platforms
VOLK aims to be portable to as many platforms as possible. We can only run tests on some platforms.

### Hardware architectures
Currently VOLK aims to run with optimized kernels on x86 with SSE/AVX and ARM with NEON.

### OS / Distro
We run tests on a variety of Ubuntu versions and aim to support as many current distros as possible.
The same goal applies to different OSes. Although this does only rarely happen, it might occur that VOLK does not work on obsolete distros, e.g. Ubuntu 12.04.

### Compilers
We want to make sure VOLK works with C/C++ standard compliant compilers. Of course, as an open source project we focus on open source compilers, most notably GCC and Clang.
We want to make sure VOLK compiles on a wide variety of compilers. Thus, we target AppleClang and MSVC as well. Mind that MSVC lacks `aligned_alloc` support for aligned arrays. We use MSVC specific instructions in this case which cannot be `free`'d with `free`.


## License

>
> Copyright 2015 Free Software Foundation, Inc.
>
> This file is part of VOLK
>
> VOLK is free software; you can redistribute it and/or modify
> it under the terms of the GNU General Public License as published by
> the Free Software Foundation; either version 3, or (at your option)
> any later version.
>
> VOLK is distributed in the hope that it will be useful,
> but WITHOUT ANY WARRANTY; without even the implied warranty of
> MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
> GNU General Public License for more details.
>
> You should have received a copy of the GNU General Public License
> along with GNU Radio; see the file COPYING.  If not, write to
> the Free Software Foundation, Inc., 51 Franklin Street,
> Boston, MA 02110-1301, USA.
>
