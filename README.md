[![Build Status](https://travis-ci.org/gnuradio/volk.svg?branch=master)](https://travis-ci.org/gnuradio/volk) [![Build status](https://ci.appveyor.com/api/projects/status/5o56mgw0do20jlh3/branch/master?svg=true)](https://ci.appveyor.com/project/gnuradio/volk/branch/master)

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

# volk_profile will profile your system so that the best kernel is used
$ volk_profile
```

### Building on Raspberry Pi and other ARM boards

To build for these boards you need specify the correct cmake toolchain file for best performace.

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
