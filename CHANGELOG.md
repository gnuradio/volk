# Changelog
All notable changes to VOLK will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html), starting with version 2.0.0.


## [2.0.0] - 2019-08-06

This is the first version to use semantic versioning. It starts the logging of changes.


## [2.1.0] - 2019-12-22

Hi everyone,

we would like to announce that Michael Dickens and Johannes Demel are the new VOLK maintainers. We want to review and merge PRs in a timely manner as well as commenting on issues in order to resolve them.

We want to thank all contributors. This release wouldn't have been possible without them.

We're curious about VOLK users. Especially we'd like to learn about VOLK users who use VOLK outside GNU Radio.

If you have ideas for VOLK enhancements, let us know. Start with an issue to discuss your idea. We'll be happy to see new features get merged into VOLK.

### Highlights

VOLK v2.1.0 is a collection of really cool changes. We'd like to highlight some of them.

- The AVX FMA rotator bug is fixed
- VOLK offers `volk::vector<>` for C++ to follow RAII
- Move towards modern dependencies
    - CMake 3.8
    - Prefer Python3
        - We will drop Python2 support in a future release!
    - Use C++17 `std::filesystem`
        - This enables VOLK to be built without Boost if available!
- more stable CI
- lots of bugfixes
- more optimized kernels, especially more NEON versions

### Contributors

*  Albin Stigo <albin.stigo@gmail.com>
*  Amr Bekhit <amr@helmpcb.com>
*  Andrej Rode <mail@andrejro.de>
*  Carles Fernandez <carles.fernandez@gmail.com>
*  Christoph Mayer <hcab14@gmail.com>
*  Clayton Smith <argilo@gmail.com>
*  Damian Miralles <dmiralles2009@gmail.com>
*  Johannes Demel <demel@ant.uni-bremen.de> <demel@uni-bremen.de>
*  Marcus Müller <marcus@hostalia.de>
*  Michael Dickens <michael.dickens@ettus.com>
*  Philip Balister <philip@balister.org>
*  Takehiro Sekine <takehiro.sekine@ps23.jp>
*  Vasil Velichkov <vvvelichkov@gmail.com>
*  luz.paz <luzpaz@users.noreply.github.com>


### Changes

* Usage
    - Update README to reflect how to build on Raspberry Pi and the importance of running volk_profile

* Toolchain
    -  Add toolchain file for Raspberry Pi 3
    -  Update Raspberry 4 toolchain file

* Kernels
    - Add neonv7 to volk_16ic_magnitude_16i
    - Add neonv7 to volk_32fc_index_max_32u
    - Add neonv7 to volk_32fc_s32f_power_spectrum_32f
    - Add NEONv8 to volk_32f_64f_add_64f
    - Add Neonv8 to volk_32fc_deinterleave_64f_x2
    - Add volk_32fc_x2_s32fc_multiply_conjugate_add_32fc
    - Add NEONv8 to volk_32fc_convert_16ic

* CI
    - Fix AVX FMA rotator
    - appveyor: Enable testing on windows
    - Fixes for flaky kernels for more reliable CI
        - volk_32f_log2_32f
        - volk_32f_x3_sum_of_poly_32f
        - volk_32f_index_max_{16,32}u
        - volk_32f_8u_polarbutterflypuppet_32f
        - volk_8u_conv_k7_r2puppet_8u
        - volk_32fc_convert_16ic
        - volk_32fc_s32f_magnitude_16i
        - volk_32f_s32f_convert_{8,16,32}i
        - volk_16ic_magnitude_16i
        - volk_32f_64f_add_64f
    - Use Intel SDE to test all kernels
    - TravisCI
        - Add native tests on arm64
        - Add native tests on s390x and ppc64le (allow failure)

* Build
    - Build Volk without Boost if C++17 std::filesystem or std::experimental::filesystem is available
    - Update to more modern CMake
    - Prevent CMake to choose previously installed VOLK headers
    - CMake
        - bump minimum version to 3.8
        - Use sha256 instead of md5 for unique target name hash
    - Python: Prefer Python3 over Python2 if available

* C++
    - VOLK C++ allocator and C++11 std::vector type alias added
\n
## [2.2.0] - 2020-02-16

Hi everyone,

we have a new VOLK release v2.2.0!

We want to thank all contributors. This release wouldn't have been possible without them.

We're curious about VOLK users. Especially we'd like to learn about VOLK users who use VOLK outside GNU Radio.

If you have ideas for VOLK enhancements, let us know. Start with an issue to discuss your idea. We'll be happy to see new features get merged into VOLK.

The v2.1.0 release was rather large because we had a lot of backlog. We aim for more incremental releases in order to get new features out there.

### Highlights

VOLK v2.2.0 updates our build tools and adds support functionality to make it easier to use VOLK in your projects.

* Dropped Python 2 build support
    - Removed Python six module dependency
* Use C11 aligned_alloc whenever possible
    - MacOS `posix_memalign` fall-back
    - MSVC `_aligned_malloc` fall-back
* Add VOLK version in `volk_version.h` (included in `volk.h`)
* Improved CMake code
* Improved code with lots of refactoring and performance tweaks

### Contributors

*  Carles Fernandez <carles.fernandez@gmail.com>
*  Gwenhael Goavec-Merou <gwenhael.goavec-merou@trabucayre.com>
*  Albin Stigo <albin.stigo@gmail.com>
*  Johannes Demel <demel@ant.uni-bremen.de> <demel@uni-bremen.de>
*  Michael Dickens <michael.dickens@ettus.com>
*  Valerii Zapodovnikov <val.zapod.vz@gmail.com>
*  Vasil Velichkov <vvvelichkov@gmail.com>
*  ghostop14 <ghostop14@gmail.com>
*  rear1019 <rear1019@posteo.de>

### Changes

* CMake
    - Fix detection of AVX and NEON
    - Fix for macOS
    - lib/CMakeLists: use __asm__ instead of asm for ARM tests
    - lib/CMakeLists: fix detection when compiler support NEON but nor neonv7 nor neonv8
    - lib/CMakeLists.txt: use __VOLK_ASM instead of __asm__
    - lib/CMakeLists.txt: let VOLK choose preferred neon version when both are supported
    - lib/CMakeLists.txt: simplify neon test support. Unset neon version if not supported
    - For attribute, change from clang to "clang but not MSC"
* Readme
    - logo: Add logo at top of README.md
* Build dependencies
    - python: Drop Python2 support
    - python: Reduce six usage
    - python: Move to Python3 syntax and modules
    - six: Remove build dependency on python six
* Allocation
    - alloc: Use C11 aligned_alloc
    - alloc: Implement fall backs for C11 aligned_alloc
    - alloc: Fix for incomplete MSVC standard compliance
    - alloc: update to reflect alloc changes
* Library usage
    - Fixup VolkConfigVersion
    - add volk_version.h
* Refactoring
    - qa_utils.cc: fix always false expression
    - volk_prefs.c: check null realloc and use temporary pointer
    - volk_profile.cc: double assignment and return 0
    - volk_32f_x2_pow_32f.h: do not need to _mm256_setzero_ps()
    - volk_8u_conv_k7_r2puppet_8u.h: d_polys[i] is positive
    - kernels: change one iteration for's to if's
    - kernels: get rid of some assignments
    - qa_utils.cc: actually throw something
    - qa_utils.cc: fix always true code
    - rotator: Refactor AVX kernels
    - rotator: Remove unnecessary variable
    - kernel: Refactor square_dist_scalar_mult
    - square_dist_scalar_mult: Speed-Up AVX, Add unaligned
    - square_dist_scalar_mult: refactor AVX2 kernel
    - kernel: create AVX2 meta intrinsics
* CI
    - appveyor: Test with python 3.4 and 3.8
    - appveyor: Add job names
    - appveyor: Make ctest more verbose
* Performance
    - Improve performance of generic kernels with complex multiply
    - square_dist_scalar_mult: Add SSE version
    - Adds NEON versions of cos, sin and tan
