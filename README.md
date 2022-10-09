[![Build Status](https://travis-ci.com/gnuradio/volk.svg?branch=master)](https://travis-ci.com/gnuradio/volk) [![Build status](https://ci.appveyor.com/api/projects/status/5o56mgw0do20jlh3/branch/master?svg=true)](https://ci.appveyor.com/project/gnuradio/volk/branch/master)
![Check PR Formatting](https://github.com/gnuradio/volk/workflows/Check%20PR%20Formatting/badge.svg)
![Run VOLK tests](https://github.com/gnuradio/volk/workflows/Run%20VOLK%20tests/badge.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3360942.svg)](https://doi.org/10.5281/zenodo.3360942)

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

## Code of Conduct
We want to make sure everyone feels welcome. Thus, we follow our [Code of Conduct](docs/CODE_OF_CONDUCT.md).

## Contributing
We are happy to accept contributions. [Please refer to our contributing guide for further details](docs/CONTRIBUTING.md).
Also, make sure to read the [Developer's Certificate of Origin](docs/DCO.txt) and make sure to sign every commit with `git commit -s`.

## Releases and development
We maintain a [CHANGELOG](docs/CHANGELOG.md) for every release. Please refer to this file for more detailed information.
We follow semantic versioning as outlined in [our versioning guide](docs/versioning.md).

## Supported platforms
VOLK aims to be portable to as many platforms as possible. We can only run tests on some platforms.

### Hardware architectures
Currently VOLK aims to run with optimized kernels on x86 with SSE/AVX and ARM with NEON.
Support for MIPS and RISC-V is experimental; some kernels are known not to work on these architectures.

### OS / Distro
We run tests on a variety of Ubuntu versions and aim to support as many current distros as possible.
The same goal applies to different OSes. Although this does only rarely happen, it might occur that VOLK does not work on obsolete distros, e.g. Ubuntu 12.04.

### Compilers
We want to make sure VOLK works with C/C++ standard compliant compilers. Of course, as an open source project we focus on open source compilers, most notably GCC and Clang.
We want to make sure VOLK compiles on a wide variety of compilers. Thus, we target AppleClang and MSVC as well. Mind that MSVC lacks `aligned_alloc` support for aligned arrays. We use MSVC specific instructions in this case which cannot be `free`'d with `free`.


## License

**VOLK 3.0 and later are licensed under the GNU Lesser General Public License v3.0 or later (LGPL-3.0-or-later).**

### Previous VOLK version license

Earlier versions of VOLK (before VOLK 3.0) were licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later).
Since then, VOLK migrated to the LGPL-3.0-or-later.

Being technical: There are 3 people left (out of 74) who we haven't been able to get in contact with (at all), for a total of 4 (out of 1092) commits, 13 (of 282822) additions, and 7 (of 170421) deletions. We have reviewed these commits and all are simple, trivial changes (e.g., 1 line change) and most are no longer relevant (e.g., to a file that no longer exists). Volk maintainers (@michaelld and @jdemel) are in agreement that the combination -- small numbers of changes per committer, simple changes per commit, commits no longer relevant -- means that we can proceed with relicensing without the approval of the folks. We will try reaching out periodically to these folks, but we believe it unlikely we will get a reply.
We kindly request them to re-submit their GPL-3.0-or-later license code contributions to LGPL-3.0-or-later by adding their name, GitHub handle, and email address(es) used for VOLK commits
to the file [AUTHORS_RESUBMITTING_UNDER_LGPL_LICENSE.md](docs/AUTHORS_RESUBMITTING_UNDER_LGPL_LICENSE.md).

### Legal Matters

Some files have been changed many times throughout the years. Copyright
notices at the top of source files list which years changes have been
made. For some files, changes have occurred in many consecutive years.
These files may often have the format of a year range (e.g., "2006 - 2011"),
which indicates that these files have had copyrightable changes made
during each year in the range, inclusive.