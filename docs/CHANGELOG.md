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

## [2.2.1] - 2020-02-24

Hi everyone,

with VOLK 2.2.0, we introduced another AVX rotator bug which is fixed with this release.
In the process 2 more bugs were identified and fixed. Further, we saw some documentation improvements.


### Contributors

*  Clayton Smith <argilo@gmail.com>
*  Michael Dickens <michael.dickens@ettus.com>


### Changes


* Fix loop bound in AVX rotator
* Fix out-of-bounds read in AVX2 square dist kernel
* Fix length checks in AVX2 index max kernels
* includes: rearrange attributes to simplify macros Whitespace
* kernels: fix usage in header comments
\n
## [2.3.0] - 2020-05-09

Hi everyone!

VOLK 2.3 is out! We want to thank all contributors. This release wouldn't have been possible without them. We saw lots of great improvements.

On GNU Radio 'master' VOLK was finally removed as a submodule.

Currently we see ongoing discussions on how to improve CPU feature detection because VOLK is not as reliable as we'd like it to be in that department. We'd like to benefit from other open source projects and don't want to reinvent the wheel. Thus, one approach would be to include `cpu_features` as a submodule.

### Highlights

* Better reproducible builds
* CMake improvements
    - ORC is removed from the public interface where it was never supposed to be.
    - CMake fixes for better usability
* Updated and new CI chain
    - TravisCI moved to new distro and does more tests for newer GCC/Clang
    - Github Actions
        - Add Action to check proper code formatting.
        - Add CI that also runs on MacOS with XCode.
* Enforce C/C++ coding style via clang-format
* Kernel fixes
    - Add puppet for `power_spectral_density` kernel
    - Treat the `mod_range` puppet as a puppet for correct use with `volk_profile`
    - Fix `index_max` kernels
    - Fix `rotator`. We hope that we finally found the root cause of the issue.
* Kernel optimizations
    - Updated log10 calcs to use faster log2 approach
    - Optimize `complexmultiplyconjugate`
* New kernels
    - accurate exp kernel is finally merged after years
    - Add 32f_s32f_add_32f kernel to perform vector + scalar float operation

### Contributors

* Bernhard M. Wiedemann <bwiedemann@suse.de>
* Clayton Smith <argilo@gmail.com>
* Johannes Demel <demel@ant.uni-bremen.de> <demel@uni-bremen.de>
* Michael Dickens <michael.dickens@ettus.com>
* Tom Rondeau <tom@trondeau.com>
* Vasil Velichkov <vvvelichkov@gmail.com>
* ghostop14 <ghostop14@gmail.com>

### Changes

* Reproducible builds
    - Drop compile-time CPU detection
    - Drop another instance of compile-time CPU detection
* CI updates
    - ci: Add Github CI Action
    - ci: Remove Ubuntu 16.04 GCC5 test on TravisCI
    - TravisCI: Update CI to bionic distro
    - TravisCI: Add GCC 8 test
    - TravisCI: Structure CI file
    - TravisCI: Clean-up .travis.yml
* Enforce C/C++ coding style
    - clang-format: Apply clang-format
    - clang-format: Update PR with GitHub Action
    - clang-format: Rebase onto current master
* Fix compiler warnings
    - shadow: rebase kernel fixes
    - shadow: rebase volk_profile fix
* CMake
    - cmake: Remove the ORC from the VOLK public link interface
    - versioning: Remove .Maint from libvolk version
    - Fix endif macro name in comment
* Kernels
    - volk: accurate exp kernel
        - exp: Rename SSE4.1 to SSE2 kernel
    - Add 32f_s32f_add_32f kernel
        - This kernel adds in vector + scalar functionality
    - Fix the broken index max kernels
    - Treat the mod_range puppet as such
    - Add puppet for power spectral density kernel
    - Updated log10 calcs to use faster log2 approach
    - fix: Use unaligned load
    - divide: Optimize complexmultiplyconjugate


## [2.4.0] - 2020-11-22

Hi everyone!

We have another VOLK release! We're happy to announce VOLK v2.4.0! We want to thank all contributors. This release wouldn't have been possible without them.

We introduce `cpu_features` as a private submodule in this release because we use it to detect CPU features during runtime now. This should enable more reliable feature detection. Further, it takes away the burden to implement platform specific code. As a result, optimized VOLK kernels build for MacOS and Windows with AppleClang/MSVC out-of-the-box now.


### Highlights

* Documentation
    - Update README to be more verbose and to improve usefulness.

* Compilers
    - MSVC: Handle compiler flags and thus architecture specific kernels correctly. This enables optimized kernels with MSVC builds.
    - AppleClang: Treat AppleClang as Clang.
    - Paired with the `cpu_features` introduction, this enables us to use architecture specific kernels on a broader set of platforms.
* CI
    - Update to newest version of the Intel SDE
    - Test the rotator kernel with a realistic scalar
    - Introduce more test cases with newer GCC and newer Clang versions.
* CMake
    - Enable to not install `volk_modtool`.
    - Remove "find_package_handle_standard_args" warning.
* cpu_features
    - Use `cpu_features` v0.6.0 as a private submodule to detect available CPU features.
    - Fix incorrect feature detection for newer AVX versions.
    - Circumvent platform specific feature detection.
    - Enable more architecture specific kernels on more platforms.
* Kernels
    - Disable slow and broken SSE4.1 kernel in `volk_32fc_x2_dot_prod_32fc`
    - Adjust min/max for `32f_s32f_convert_8i` kernel
    - Use `INT8_*` instead of `CHAR_*`


### Contributors

* Adam Thompson <adamt@nvidia.com>
* Andrej Rode <mail@andrejro.de>
* Christoph Mayer <hcab14@gmail.com>
* Clayton Smith <argilo@gmail.com>
* Doron Behar <doron.behar@gmail.com>
* Johannes Demel <demel@ant.uni-bremen.de>, <demel@uni-bremen.de>
* Martin Kaesberger <git@skipfish.de>
* Michael Dickens <michael.dickens@ettus.com>
* Ron Economos <w6rz@comcast.net>


### Changes

* Documentation
    - Update README to include ldconfig upon volk build and install completion
    - Update README based on review
    - readme: Fix wording
    - docs: Fix conversion inaccuracy

* MSVC
    - archs: MSVC 2013 and greater don't have a SSE flag

* CI
    - update to newest version of the Intel SDE
    - Test the rotator kernel with a realistic scalar

* CMake
    - build: Enable to not install volk_modtool
    - cmake: Remove "find_package_handle_standard_args" warning.
    - cmake: Ensure that cpu_features is built as a static library.

* cpu_features
    - readme: Add section on supported platforms
    - readme: Make supported compiler section more specific
    - travis: Add GCC 9 test on focal
    - travis: Add tests for clang 8, 9, 10
    - travis: Fix incorrect CXX compiler assignment
    - cpu_features: Remove unused feature checks
    - ci: Update TravisCI for cpu_features
    - cpu_features: Fix MSVC build
    - pic: Fix BUILD_PIC issue
    - ci: Update CI system configuration
    - cpu_features: Bump submodule pointer to v0.6.0
    - docs: Add hints how to handle required submodules
    - cpu_features: Switch to cpu_features
    - ci: Update CI system for cpu_features
    - cpu_features: Force PIC build flag
    - appveyor: Add recursive clone command
    - cpu_features: Remove xgetbv checks
    - pic: Cache and force BUILD_PIC
    - ci: Remove explicit BUILD_PIC from cmake args
    - ci: Remove BUILD_PIC from CI cmake args
    - cpu_features: Remove commented code
    - cpu_features: Assume AppleClang == Clang
    - cpu_features: Remove obsolete code in archs.xml
    - fix for ARM cross-compiling CI
    - ARM CI: remove unneeded environment variables

* Housekeeping
    - structure: Move release scripts to scripts folder

* Kernels
    - emit an emms instruction after using the mmx extension
    - volk_32fc_x2_dot_prod_32fc: disable slow & broken SSE4.1 kernel
    - fix: Adjust min/max for 32f_s32f_convert_8i kernel
    - fix: Use INT8_* instead of CHAR_*


## [2.4.1] - 2020-12-17

Hi everyone!

We have a new VOLK bugfix release! We are happy to announce VOLK v2.4.1! We want to thank all contributors. This release wouldn't have been possible without them.

Our v2.4.0 release introduced quite a lot of changes under the hood. With this bugfix release, we want to make sure that everything works as expected again.


### Contributors

* A. Maitland Bottoms <bottoms@debian.org>
* Johannes Demel <demel@uni-bremen.de>
* Michael Dickens <michael.dickens@ettus.com>
* Philip Balister <philip@balister.org>
* Ron Economos <w6rz@comcast.net>
* Ryan Volz <ryan.volz@gmail.com>


### Changes

* Build
    - cpu_features CMake option
    - Add cpu_features to static library build.
    - Use static liborc-0.4 library for static library build.
    - cmake: Detect if cpu_features submodule is present.

* Install
    - Check for lib64 versus lib and set LIB_SUFFIX accordingly.

* CI
    - Add CI test for static library build.

* Releases
    - project: Include git submodules (i.e. cpu_features) in release tarball.
    - scripts: Add GPG signature to release script

* Other
    - readme: Update TravisCI status badge


## [2.5.0] - 2021-06-05

Hi everyone!

We have a new VOLK release! We are happy to announce VOLK v2.5.0! We want to thank all contributors. This release wouldn't have been possible without them.

This release adds new kernel implementations and fixes. Some of these were longstanding PRs that could only be merged recently thanks to our switch from CLA to DCO.

### Announcements

I would like to point out one upcoming change. After this release, we will rename our development branch to `main` as discussed in [issue #461](https://github.com/gnuradio/volk/issues/461).


I'd like to point the community to this [VOLK relicensing GREP](https://github.com/gnuradio/greps/pull/33).
This is an ongoing effort to relicense VOLK under LGPLv3.
We're looking for people and organizations that are interested in leading this effort.

### Contributors

* Aang23 <qwerty15@gmx.fr>
* Carles Fernandez <carles.fernandez@gmail.com>
* Florian Ritterhoff <ritterho@hm.edu>
* Jam M. Hernandez Quiceno <jamarck96@gmail.com>, <jam_quiceno@partech.com>
* Jaroslav Škarvada <jskarvad@redhat.com>
* Johannes Demel <demel@uni-bremen.de>
* Magnus Lundmark <magnus@skysense.io>
* Michael Dickens <michael.dickens@ettus.com>
* Steven Behnke <steven_behnke@me.com>
* alesha72003 <alesha72003@ya.ru>
* dernasherbrezon <rodionovamp@mail.ru>
* rear1019 <rear1019@posteo.de>


### Changes

* Kernels
    - volk_32f_stddev_and_mean_32f_x2: implemented Young and Cramer's algorithm
    - volk_32fc_accumulator_s32fc: Add new kernel
    - volk_16ic_x2_dot_prod_16ic_u_avx2: Fix Typo, was `_axv2`.
    - Remove _mm256_zeroupper() calls
    - Enforce consistent function prototypes
    - 32fc_index_max: Improve speed of AVX2 version
    - conv_k7_r2: Disable broken AVX2 code
    - improve volk_8i_s32f_convert_32f for ARM NEON
    - Calculate cos in AVX512F
    - Calculate sin using AVX512F


* Compilers
    - MSVC
        - Fix MSVC builds
    - GCC
        - Fix segmentation fault when using GCC 8
    - MinGW
        - add support and test for MinGW/MSYS2

* The README has received several improvements

* Build
    - Fix python version detection
    - cmake: Check that 'distutils' is available
    - c11: Remove pre-C11 preprocessor instructions

* CI
    - Add more CI to GitHub Actions
    - Remove redundant tests from TravisCI
    - Add non-x86 GitHub Actions
    - Update compiler names in CI
    - Disable fail-fast CI
    - Add more debug output to tests

* Contributing
    - contributing: Add CONTRIBUTING.md and DCO.txt


## [2.5.1] - 2022-02-12

Hi everyone!

We have a new VOLK release! We are happy to announce VOLK v2.5.1! We want to thank all contributors. This release wouldn't have been possible without them.

The list of contributors is pretty long this time due to a lot of support to relicense VOLK under LGPL. Currently, we are "almost there" but need a few more approvals, please support us. We thank everyone for their support in this effort.

We use `cpu_features` for a while now. This maintainance release should make it easier for package maintainers, FreeBSD users, and M1 users to use VOLK. Package maintainers should be able to use an external `cpu_features` module. For everyone else, `cpu_features` received support for M1 and FreeBSD.

You can find [VOLK on Zenodo DOI: 10.5281/zenodo.3360942](https://doi.org/10.5281/zenodo.3360942).
We started to actively support Zenodo now via a `.zenodo.json` file. This might come in handy for people who use VOLK in publications. As a contributor, if you want more information about yourself added to VOLK, feel free to add your ORCiD and affiliation.

In the past, we relied on Boost for several tasks in `volk_profile`. For years, we minimized Boost usage to `boost::filesystem`. We mostly switched to C++17 `std::filesystem` years ago. The last distribution in our CI system that required Boost to build VOLK, was Ubuntu 14.04. Thus, now is the time to remove the Boost dependency completely and rely on C++17 features.

Some VOLK kernels are untested for years. We decided to deprecate these kernels but assume that nobody uses them anyways. If your compiler spits out a warning that you use a deprecated kernel, get in touch. Besides, we received fixes for various kernels. Especially FEC kernels are notoriously difficult to debug because issues often pop up as performance regressions.

Finally, we saw a lot of housekeeping in different areas. Scripts to support us in our LGPL relicensing effort, updated docs, and updated our code of conduct. We could remove some double entries in our QA system and fixed a `volk_malloc` bug that ASAN reported.
Finally, we switched to the Python `sysconfig` module in our build system to ensure Python 3.12+ does not break our builds.



### Contributors

* A. Maitland Bottoms <bottoms@debian.org>
* Aang23 <qwerty15@gmx.fr>
* AlexandreRouma <alexandre.rouma@gmail.com>
* Andrej Rode <mail@andrejro.de>
* Ben Hilburn <ben@hilburn.dev>
* Bernhard M. Wiedemann <bwiedemann@suse.de>
* Brennan Ashton <bashton@brennanashton.com>
* Carles Fernandez <carles.fernandez@gmail.com>
* Clayton Smith <argilo@gmail.com>
* Doug <douggeiger@users.noreply.github.com>
* Douglas Anderson <djanderson@users.noreply.github.com>
* Florian Ritterhoff <ritterho@hm.edu>
* Jaroslav Škarvada <jskarvad@redhat.com>
* Johannes Demel <demel@uni-bremen.de>
* Josh Blum <josh@joshknows.com>
* Kyle A Logue <kyle.a.logue@aero.org>
* Luigi Cruz <luigifcruz@gmail.com>
* Magnus Lundmark <magnus@skysense.io>
* Marc L <marcll@vt.edu>
* Marcus Müller <marcus@hostalia.de>
* Martin Kaesberger <git@skipfish.de>
* Michael Dickens <michael.dickens@ettus.com>
* Nathan West <nwest@deepsig.io>
* Paul Cercueil <paul.cercueil@analog.com>
* Philip Balister <philip@balister.org>
* Ron Economos <w6rz@comcast.net>
* Ryan Volz <ryan.volz@gmail.com>
* Sylvain Munaut <tnt@246tNt.com>
* Takehiro Sekine <takehiro.sekine@ps23.jp>
* Vanya Sergeev <vsergeev@gmail.com>
* Vasil Velichkov <vvvelichkov@gmail.com>
* Zlika <zlika_ese@hotmail.com>
* namccart <namccart@gmail.com>
* dernasherbrezon <rodionovamp@mail.ru>
* rear1019 <rear1019@posteo.de>


### Changes

* Kernels
    - Fixup underperforming GENERIC kernel for volk_8u_x4_conv_k7_r2_8u
    - volk_32fc_x2_conjugate_dot_prod_32fc: New generic implementation
    - Add volk_32f(c)_index_min_16/32u
    - Fix volk_32fc_index_min_32u_neon
    - Fix volk_32fc_index_min_32u_neon

* Misc
    - Fix volk_malloc alignment bug
    - qa: Remove repeating tests
    - python: Switch to sysconfig module
    - deprecate: Add attribute deprecated
    - deprecate: Exclude warnings on Windows
    - docs: Update docs
    - Add the list of contributors agreeing to LGPL licensing
    - Add a script to count the lines that are pending resubmission
    - Testing: Add test for LGPL licensing
    - Update CODE_OF_CONDUCT file

* Boost
    - boost: Remove boost dependency
    - c++: Require C++17 for std::filesystem

* cpu_features
      cpu_features: Update submodule pointer
      cpu_features: Make cpu_features submodule optional

* Zenodo
      zenodo: Add metadata file
      zenodo: Re-organize .zenodo.json
