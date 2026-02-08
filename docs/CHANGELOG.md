# Changelog
All notable changes to VOLK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), starting with version 2.0.0.


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

## [2.5.2] - 2022-09-04

Hi everyone!

We have a new VOLK release! We are happy to announce VOLK v2.5.2! We want to thank all contributors. This release wouldn't have been possible without them.

We are happy to announce that our re-licensing effort is complete. This has been a long and challenging journey. Being technical: There are 3 people left (out of 74) who we haven't been able to get in contact with (at all), for a total of 4 (out of 1092) commits, 13 (of 282822) additions, and 7 (of 170421) deletions. We have reviewed these commits and all are simple changes (e.g., 1 line change) and most are no longer relevant (e.g., to a file that no longer exists). VOLK maintainers are in agreement that the combination -- small numbers of changes per committer, simple changes per commit, commits no longer relevant -- means that we can proceed with re-licensing without the approval of the folks. We will try reaching out periodically to these folks, but we believe it unlikely we will get a reply.

This maintainance release is intended to be the last VOLK 2.x release. After we completed our re-licensing effort, we want to make a VOLK 3.0 release soon. VOLK 3.0 will be fully compatible with VOLK 2.x on a technical level. However, VOLK 3+ will be released under LGPL. We are convinced a license change justifies a major release.

### Contributors

* Aang23 <qwerty15@gmx.fr>
* Clayton Smith <argilo@gmail.com>
* Johannes Demel <demel@ant.uni-bremen.de>, <demel@uni-bremen.de>
* Michael Dickens <michael.dickens@ettus.com>
* Michael Roe <michael-roe@users.noreply.github.com>

### Changes

* Android
    - Add Android CI
    - Fix armeabi-v7a on Android
* CI
    - Update all test jobs to more recent actions
* volk_8u_x4_conv_k7_r2_8u
    - Add NEON implementation `neonspiral` via `sse2neon.h`
* Fixes
    - Fix out-of-bounds reads
    - Fix broken neon kernels
    - Fix float to int conversion
* CMake
    - Suppress superfluous warning
    - Fix Python install path calculation and documentation
* cpu_features
    - Update submodule pointer
* VOLK 3.0 release preparations
    - Use SPDX license identifiers everywhere
    - Re-arrange files in top-level folder
    - Update Doxygen and all Doxygen related tasks into `docs`

## [3.0.0] - 2023-01-14

Hi everyone!

This is the VOLK v3.0.0 major release! This release marks the conclusion of a long lasting effort to complete [GREP 23](https://github.com/gnuradio/greps/blob/main/grep-0023-relicense-volk.md) that proposes to change the VOLK license to LGPLv3+. We would like to thank all VOLK contributors that they allowed this re-licensing effort to complete. This release wouldn't have been possible without them.

For VOLK users it is important to not that the VOLK API does NOT change in this major release. After a series of discussion we are convinced a license change justifies a major release. Thus, you may switch to VOLK 3 and enjoy the additional freedom the LGPL offers.

### Motivation for the switch to LGPLv3+

We want to remove usage entry barriers from VOLK. As a result, we expect greater adoption and a growing user and contributor base of VOLK. This move helps to spread the value of Free and Open Source Software in the SIMD community, which so far is dominated by non-FOSS software. Moreover, we recognize the desire of our long term contributors to be able to use VOLK with their contributions in their projects. This may explicitly include proprietary projects. We want to enable all contributors to be able to use VOLK wherever they want. At the same time, we want to make sure that improvements to VOLK itself are easily obtained by everyone, i.e. strike a balance between permissiveness and strict copyleft.

Since VOLK is a library it should choose a fitting license. If we see greater adoption of VOLK in more projects, we are confident that we will receive more contributions. May it be bug fixes, new kernels or even contributions to core features.

Historically, VOLK was part of GNU Radio. Thus, it made total sense to use GPLv3+ just like GNU Radio. Since then, VOLK became its own project with its own repository and leadership. While it is still a core dependency of GNU Radio and considers GNU Radio as its main user, others may want to use it too. Especially, long term VOLK contributors may be able to use VOLK in a broader set of projects now.

After a fruitful series of discussions we settled on the LGPLv3+. We believe this license strikes a good balance between permissiveness and strict copyleft for VOLK. We hope to foster contributions to VOLK. Furthermore, we hope to see VOLK usage in a wider set of projects.

### Contributors

The VOLK 3.0.0 release is only possible because all contributors helped to make it possible. Thus, we omit a list of contributors that contributed since the last release.
Instead we want to thank everyone again!

### Changes

* License switch to LGPLv3+
* Fix build for 32 bit arm with neon
* Add experimental support for MIPS and RISC-V


## [3.1.0] - 2023-12-05

Hi everyone!

This is the VOLK v3.1.0 release! We want to thank all contributors. This release wouldn't have been possible without them.

This release introduces new kernels, fixes a lot of subtle bugs, and introduces an updated API that allows VOLK to run on PowerPC and MIPS platforms without issues. Namely, complex scalar values are passed to kernels by pointer instead of by value. The old API is still around and will be for the whole VOLK v3 release cycle. However, it is recommended to switch to the new API for improved compatibility. Besides, we saw improvements to our `cpu_features` usage that should make it easier for package maintainers. Finally, a lot of tests received fixes that allow our CI to run without hiccups.

### Contributors

* A. Maitland Bottoms <bottoms@debian.org>
* Andrej Rode <mail@andrejro.de>
* Ashley Brighthope <ashley.b@reddegrees.com>
* Clayton Smith <argilo@gmail.com>
* Daniel Estévez <daniel@destevez.net>
* Johannes Demel <demel@uni-bremen.de>, <jdemel@gnuradio.org>
* John Sallay <jasallay@gmail.com>
* Magnus Lundmark <magnus@skysense.io>, <magnuslundmark@gmail.com>
* Marcus Müller <mmueller@gnuradio.org>
* Michael Roe <michael-roe@users.noreply.github.com>
* Thomas Habets <thomas@habets.se>


### Changes

- Build and dependency updates
      - omit build path
      - cmake: Link to cpu_features only in BUILD_INTERFACE
      - cpu_features: Update submodule pointer and new CMake target name
      - cmake: Removed duplicated logic
      - cmake: Do not install cpu_features with volk
      - Use CMake target in one more place
      - Fix typo in the CMake target name
      - Use CpuFeatures target
      - Use cpu_features on RISC-V platforms
      - cpu_features: Update submodule pointer
      - Add UBSAN to ASAN builds
      
- CI fixes
      - Add length checks to volk_8u_x2_encodeframepolar_8u
      - Fix flaky qa_volk_32f_s32f_convertpuppet_8u
      - Use absolute tolerance for stddev_and_mean
      - Use absolute tolerance for sum_of_poly kernel
      - Add length checks to conv_k7 kernels
      - Fix variable name in dot product kernels
      - Fix buffer overflow in volk_32fc_x2_square_dist_32f_a_sse3
      - Increase tolerance for volk_32f_log2_32f
      - Re-enable tests on aarch64 clang-14
      - Fix undefined behaviour in volk_8u_x4_conv_k7_r2_8u
      - Fix undefined behaviour in volk_32u_reverse_32u
      - Fix aligned loads and stores in unaligned kernels
      - Fix register size warnings in reverse kernel
      - Fix undefined behaviour in dot product kernels
      - Use an absolute tolerance to test the dot product kernels
      - Always initialize returnValue
      - Add length checks to puppets
      - Add carriage return to error message
      - Include ORC in neonv8 machine definition
      - Add back volk_32f_exp_32f test
      - Generate random integers with uniform_int_distribution
      - Fix puppet master name for volk_32u_popcnt
      - Avoid integer overflow in volk_8ic_x2_multiply_conjugate_16ic corner case
      - Use a reasonable scalar and tolerance for spectral_noise_floor
      - Increase volk_32f_x2_dot_prod_16i tolerance to 1
      - Increase tolerance for the power_spectrum test
      - Decrease the range for signed 16-bit integer testing
      - Use a puppet to pass positive values to volk_32f_x2_pow_32f
      - Use absolute tolerances for accumulator and dot product
      - Fix AppVeyor git checkout

- new kernel API
      - Use pointers to pass in s32fc arguments
        - The old API is deprecated but will be available for the foreseeable future

- updated kernels
      - Remove unused ORC code
      - Prefer NEON kernels over ORC
      - Require all kernels to have a generic implementation
      - Remove redundant a_generic kernels
      - Remove ORC kernels that use sqrtf
      - reverse: Rename dword_shuffle to generic
      - volk_32f_s32f_convert_8i: code style
      - volk_32fc_x2_divide_32fc: add documentation about numerical accuracy
      - kernel: Refactor 32f_s32f_multiply_32f kernel
      - kernel: Refactor 32f_x2_subtract_32f kernel
      - convert 32f->32i: fix compiler warnings about loss of int precision
      - 64u_ byteswape: remove buggy Neonv8 protokernel
      - 64u_ byteswape: remove buggy Neon protokernel
      - Remove broken volk_16i_max_star_16i_neon protokernel
      - Fix truncate-toward-zero distortion
      - Fix encodepolar documentation
      

- new kernels
      - add volk_32f_s32f_x2_convert_8u kernel
      - Fix documentation for the clamp kernel
      - added new kernel: volk_32f_s32f_x2_clamp
      - new kernels for atan2
      - Add 32f_s32f_multiply_32f RISC-V manually optimized assembly
      - Add .size to volk_32f_s32f_multiply_32f_sifive_u74
      - Add volk_32fc_x2_dot_prod_32fc_sifive_u74

## [3.1.1] - 2024-01-29

Hi everyone!

This is the VOLK v3.1.1 release! We want to thank all contributors. This release wouldn't have been possible without them.

This is a maintenance release to fix subtle bugs in many areas and to improve our tests where possible. All in all, our CI is more stable now and catches more errors.

### Contributors

Clayton Smith <argilo@gmail.com>
Johannes Demel <demel@uni-bremen.de>, <jdemel@gnuradio.org>
Kenji Rikitake <kenji.rikitake@acm.org>
Philip Balister <philip@opensdr.com>

### Changes

- CI fixes
  - Allow for rounding error in float-to-int conversions
  - Allow for rounding error in `volk_32fc_s32f_magnitude_16i`
  - Allow for rounding error in float-to-int interleave
  - Add missing `volk_16_byteswap_u_orc` to puppet
  - Fix 64-bit integer testing
  - Build and test neonv7 protokernels on armv7

- kernels
  - Remove broken sse32 kernels
  - Fix flaky `fm_detect` test
  - Fix flaky `mod_range` test
  - Remove unnecessary volatiles from `volk_32fc_s32f_magnitude_16i`
  - Remove SSE protokernels written in assembly
  - Remove inline assembler from `volk_32fc_convert_16ic_neon`
  - Use bit shifts in generic and `byte_shuffle` reverse
  - Remove disabled SSE4.1 dot product
  - Fix `conv_k7_r2` kernel and puppet
  - Remove unused argument from renormalize
  - Align types in ORC function signatures
  - Uncomment AVX2 implementation
  - Renormalize in every iteration on AVX2
  - Remove extraneous permutations
  - Compute the minimum over both register lanes
  - `volk_32fc_s32f_atan2_32f`: Add NaN tests for avx2 and avx2fma code

- fixes
  - Express version information in decimal
  - Remove `__VOLK_VOLATILE`
  - Remove references to simdmath library
  - cmake: Switch to GNUInstallDirs
  - fprintf: Remove fprintf statements from `volk_malloc`
  - release: Prepare release with updated files
  - Get the sse2neon.h file to a git submodule to avoid random copies.


## [3.1.2] - 2024-02-25

Hi everyone!

This is the VOLK v3.1.2 release! We want to thank all contributors.
This release wouldn't have been possible without them.

The last maintenance release revealed issues in areas that are difficult to test. 
While the changes to the library should be minimal, usability should be improved. 
Most notably, we build and deploy [the VOLK documentation](https://www.libvolk.org/docs) 
automatically now.

### Contributors

- Andrej Rode <mail@andrejro.de>
- Clayton Smith <argilo@gmail.com>
- Johannes Demel <demel@uni-bremen.de>, <jdemel@gnuradio.org>
- Marcus Müller <mmueller@gnuradio.org>
- Rick Farina (Zero_Chaos) <zerochaos@gentoo.org>

### Changes

- Documentation improvements, and automatically generate and publish
    - docs: Add VOLK doc build to CI
    - docs: Add upload to GitHub actions
    - cpu_features: Update hints in README
- Remove sse2neon with a native NEON implementation
    - Replace sse2neon with native NEON
    - Remove loop unrolling
    - Simplify Spiral-generated code
- Improve CI pipeline with new runner
    - flyci: Test CI service with M2 instance
    - actions: Update GH Actions checkout
- Auto-format CMake files
    - cmake: Add .cmake-format.py
    - cmake: Apply .cmake-format.py
- Release script fixes
    - scripts/release: fix multi-concatenation of submodule tars
    - shellcheck fixes
    - bash negative exit codes are not portable, let's be positive




## [3.2.0] - 2025-02-03

Hi everyone!

This is the VOLK v3.2.0 release! We want to thank all contributors.
This release wouldn't have been possible without them.

Thanks to Olaf Bernstein, VOLK received well optimized RiscV implementations for almost every kernel.
Together with the appropriate CI, this contribution makes VOLK way more powerful on a whole new architecture.

We started to use gtest as an additional test framework. The current "one kinda test fits all" approach is often insufficient to test kernels where they really should not fail.
Now, this approach should allow us to implement more powerful tests more easily.

Besides the x86 platform, we see more and more ARM activity. The corresponding kernels can now be tested natively on Linux and MacOS.
This approach is way faster than before with QEMU. A single job runs in ~1min instead of ~12min now.

### Contributors

- Doron Behar <doron.behar@gmail.com>
- Johannes Demel <jdemel@gnuradio.org>
- John Sallay <jasallay@gmail.com>
- Magnus Lundmark <magnuslundmark@gmail.com>
- Olaf Bernstein <camel-cdr@protonmail.com>
- Ron Economos <w6rz@comcast.net>
- Sam Lane <sl01172@surrey.ac.uk>
- Suleyman Poyraz <zaryob.dev@gmail.com>
- tinyboxvk <13696594+tinyboxvk@users.noreply.github.com>

### Changes

- New and improved kernels
    - add RISC-V Vector extension (RVV) kernels
    - New AVX512F implementation
- Improved and modernized CI
    - ci: Add first native Linux ARM runners
    - macos: Fix CI dependency error
    - appveyor: Update to VS 2022/Python 3.12
    - Update android_build.yml
- Improved builds
    - cmake: Fix 64bit host CPU detection
    - cmake: Suppress invalid escape sequence warnings with Python 3.12
    - cmake/pkgconfig: use CMAKE_INSTALL_FULL_* variables
    - cmake: Fix VOLK as a submodule build issue
    - Adds toolchain file for Raspberry Pi 5
- New and improved tests
    - gtest: Start work on new test infrastructure
    - tests: Add a log info print test
    - gtest: Make gtest an install dependency
    - gtest: Enable GTests in CI workflows
    - tests: Beautify test output
- Documentation
    - cpu_features: Update hints in README
- Code quality
    - Add const to several args
- Usability features
    - feature: add env variable kernel override


## [3.3.0] - 2026-02-08

Hi everyone!

This is the VOLK v3.3.0 release! We want to thank all contributors.
This release wouldn't have been possible without them.

We received a lot of improvements to existing kernels, new kernels,
and optimized support for a lot of existing kernels. 
Moreover, a lot more implementations make use of AVX512 now, as well as 
more optimizations for RiscV, and more NEON implementations.
Thus, overall this is a very exciting release!

Additionally, we received updates all over the code base to improve code quality.
Obsolete code was removed, we get closer and closer to being able to remove the 
cpu_features submodule and rely on the distribution package everywhere.
Besides, our throughput test output received a face lift to make it easier to digest.

Finally, over the years, we discussed in-place kernel operations repeatedly.
While we don't test correct in-place operation, e.g., GNU Radio relies on it for 
multiple kernels. Finding a way to check and document this behavior is an ongoing effort.

### Contributors

- Anil Gurses <anilgurses98@gmail.com>
- Johannes Sterz Demel <jdemel@gnuradio.org>
- Magnus Lundmark <magnuslundmark@gmail.com>
- Marcus Müller <marcus@hostalia.de>
- Olaf Bernstein <camel-cdr@protonmail.com>

### Changes

- New kernels
	- volk_16i_x2_add_saturated_16i
	- volk_16u_x2_add_saturated_16u
	- volk_32f_sincos_32f_x2.h
	- volk_64f_x2_dot_prod_64f.h
	- volk_8i_x2_add_saturated_8i.h
	- volk_8u_x2_add_saturated_8u.h
- Improvements to a lot of kernels
	- RiscV kernels are further improved and fixed
	- RVV index_max/min kernels always return the correct (first) index now
	- New AVX512 implementations for a lot of kernels
	- Add more NEON kernels with better accuracy
- Documentation
	- Working on auto-publishing latest docs
	- More clarification on our software library dependencies policy
	- Improved documentation on the underlying algorithms that are used
- Code quality
	- cx-limited-range: Reduce scope of compile feature
	- Fully rely on std::filesystem (we used to have a boost::filesystem fallback)
	- Align CMake auto-format with GNU Radio
	- Update to modern PIC enablement
	- Fix NEON compile checks
	- Update code style in more places
	- tighter 
- CI
	- Add -Werror flag to CI for C compilation
	- Remove obsolete CI, add new CI
	- Fix obsolete MacOS Intel CI
- Tests
	- Add specialized test suite for the rotator kernel
	- Improved usability with gtest
	- Tighter error bounds for a lot of implementations      
- Usability
	- new performance test output
	- fastest implementation is marked with a star
	- speed up vs. generic implementation is printed
	- test "heat up" added
 
