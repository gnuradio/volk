#
# Copyright 2011-2012,2014,2018 Free Software Foundation, Inc.
#
# This file is part of VOLK.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
#

########################################################################
# header file detection
########################################################################
include(CheckIncludeFile)
check_include_file(cpuid.h HAVE_CPUID_H)
if(HAVE_CPUID_H)
    add_definitions(-DHAVE_CPUID_H)
endif()

check_include_file(intrin.h HAVE_INTRIN_H)
if(HAVE_INTRIN_H)
    add_definitions(-DHAVE_INTRIN_H)
endif()

check_include_file(fenv.h HAVE_FENV_H)
if(HAVE_FENV_H)
    add_definitions(-DHAVE_FENV_H)
endif()

check_include_file(dlfcn.h HAVE_DLFCN_H)
if(HAVE_DLFCN_H)
    add_definitions(-DHAVE_DLFCN_H)
    list(APPEND volk_libraries ${CMAKE_DL_LIBS})
endif()

########################################################################
# Setup the compiler name
########################################################################
set(COMPILER_NAME ${CMAKE_C_COMPILER_ID})
if(MSVC) #its not set otherwise
    set(COMPILER_NAME MSVC)
endif()

# Assume "AppleClang == Clang".
string(TOLOWER ${COMPILER_NAME} COMPILER_NAME_LOWER)
string(REGEX MATCH "clang" COMPILER_NAME_LOWER ${COMPILER_NAME_LOWER})
if(${COMPILER_NAME_LOWER} MATCHES "clang")
    set(COMPILER_NAME "Clang")
endif()

message(STATUS "Compiler name: ${COMPILER_NAME}")

if(NOT DEFINED COMPILER_NAME)
    message(
        FATAL_ERROR "COMPILER_NAME undefined. Volk build may not support this compiler.")
endif()

########################################################################
# Special clang flag so flag checks can fail
########################################################################
if(COMPILER_NAME MATCHES "GNU")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-Werror=unused-command-line-argument"
                            HAVE_WERROR_UNUSED_CMD_LINE_ARG)
    if(HAVE_WERROR_UNUSED_CMD_LINE_ARG)
        set(VOLK_FLAG_CHECK_FLAGS "-Werror=unused-command-line-argument")
    endif()
endif()

########################################################################
# POSIX_MEMALIGN: If we have to fall back to `posix_memalign`,
# make it known to the compiler.
########################################################################
if(HAVE_POSIX_MEMALIGN)
    message(STATUS "Use `posix_memalign` for aligned malloc!")
    add_definitions(-DHAVE_POSIX_MEMALIGN)
endif(HAVE_POSIX_MEMALIGN)

########################################################################
# detect x86 flavor of CPU
########################################################################
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "^(i.86|x86|x86_64|amd64|AMD64)$")
    message(STATUS "x86* CPU detected")
    set(CPU_IS_x86 TRUE)
endif()

########################################################################
# determine passing architectures based on compile flag tests
########################################################################
execute_process(
    COMMAND
        ${PYTHON_EXECUTABLE} ${PYTHON_DASH_B}
        ${PROJECT_SOURCE_DIR}/gen/volk_compile_utils.py --mode "arch_flags" --compiler
        "${COMPILER_NAME}"
    OUTPUT_VARIABLE arch_flag_lines
    OUTPUT_STRIP_TRAILING_WHITESPACE)

try_compile(HAVE_RVV_INTRINSICS ${CMAKE_BINARY_DIR}
            ${CMAKE_SOURCE_DIR}/cmake/Checks/check-rvv-intrinsics.c)
if(HAVE_RVV_INTRINSICS)
    message(STATUS "Checking RVV intrinsics - found")
else()
    message(STATUS "Checking RVV intrinsics - not found")
endif()

macro(check_arch arch_name)
    set(flags ${ARGN})
    set(have_${arch_name} TRUE)

    string(SUBSTRING "${arch_name}" 0 2 arch_prefix)
    foreach(flag ${flags})
        if(MSVC AND (${flag} STREQUAL "/arch:SSE2" OR ${flag} STREQUAL "/arch:SSE"))
            # SSE/SSE2 is supported in MSVC since VS 2005 but flag not available when compiling 64-bit so do not check
        elseif("${arch_prefix}" STREQUAL "rv" AND NOT HAVE_RVV_INTRINSICS)
            message(STATUS "Skipping ${arch_name} due to missing RVV intrinsics support")
            set(have_${arch_name} FALSE)
        else()
            include(CheckCXXCompilerFlag)
            set(have_flag have${flag})
            #make the have_flag have nice alphanum chars (just for looks/not necessary)
            execute_process(
                COMMAND ${PYTHON_EXECUTABLE} -c
                        "import re; print(re.sub(r'\\W', '_', '${have_flag}'))"
                OUTPUT_VARIABLE have_flag
                OUTPUT_STRIP_TRAILING_WHITESPACE)
            if(VOLK_FLAG_CHECK_FLAGS)
                set(CMAKE_REQUIRED_FLAGS ${VOLK_FLAG_CHECK_FLAGS})
            endif()
            check_cxx_compiler_flag(${flag} ${have_flag})
            unset(CMAKE_REQUIRED_FLAGS)
            if(NOT ${have_flag})
                set(have_${arch_name} FALSE)
            endif()
        endif()
    endforeach()
    if(have_${arch_name})
        list(APPEND available_archs ${arch_name})
    endif()
endmacro(check_arch)

foreach(line ${arch_flag_lines})
    string(REGEX REPLACE "," ";" arch_flags ${line})
    check_arch(${arch_flags})
endforeach(line)

macro(OVERRULE_ARCH arch reason)
    message(STATUS "${reason}, Overruled arch ${arch}")
    list(REMOVE_ITEM available_archs ${arch})
endmacro(OVERRULE_ARCH)

macro(FORCE_ARCH arch reason)
    message(STATUS "${reason}, Forced arch ${arch}")
    list(APPEND available_archs ${arch})
endmacro(FORCE_ARCH)

########################################################################
# eliminate AVX on if not on x86, or if the compiler does not accept
# the xgetbv instruction, or {if not cross-compiling and the xgetbv
# executable does not function correctly}.
########################################################################
set(HAVE_AVX_CVTPI32_PS 0)
if(CPU_IS_x86)
    #########################################################################
    # eliminate AVX if cvtpi32_ps intrinsic fails like some versions of clang
    #########################################################################

    # check to see if the compiler/linker works with cvtpi32_ps instrinsic when using AVX
    if(CMAKE_SIZEOF_VOID_P EQUAL 4)
        file(
            WRITE ${CMAKE_CURRENT_BINARY_DIR}/test_cvtpi32_ps.c
            "#include <immintrin.h>\nint main (void) {__m128 __a; __m64 __b; __m128 foo = _mm_cvtpi32_ps(__a, __b); return (0); }"
        )
        execute_process(
            COMMAND
                ${CMAKE_C_COMPILER} -mavx -o ${CMAKE_CURRENT_BINARY_DIR}/test_cvtpi32_ps
                ${CMAKE_CURRENT_BINARY_DIR}/test_cvtpi32_ps.c
            OUTPUT_QUIET ERROR_QUIET
            RESULT_VARIABLE avx_compile_result)
        if(NOT ${avx_compile_result} EQUAL 0)
            overrule_arch(avx "Compiler missing cvtpi32_ps instrinsic")
        else()
            set(HAVE_AVX_CVTPI32_PS 1)
        endif()
        file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/test_cvtpi32_ps
             ${CMAKE_CURRENT_BINARY_DIR}/test_cvtpi32_ps.c)
    else(CMAKE_SIZEOF_VOID_P EQUAL 4)
        # 64-bit compilations won't need this command so don't overrule AVX
        set(HAVE_AVX_CVTPI32_PS 0)
    endif(CMAKE_SIZEOF_VOID_P EQUAL 4)

    # Disable SSE4a if Clang is less than version 3.2
    if("${CMAKE_C_COMPILER_ID}" MATCHES "Clang")
        # Figure out the version of Clang

        if(CMAKE_C_COMPILER_VERSION VERSION_LESS "3.2")
            overrule_arch(sse4_a "Clang >= 3.2 required for SSE4a")
        endif(CMAKE_C_COMPILER_VERSION VERSION_LESS "3.2")
    endif("${CMAKE_C_COMPILER_ID}" MATCHES "Clang")

endif(CPU_IS_x86)

if(${HAVE_AVX_CVTPI32_PS})
    add_definitions(-DHAVE_AVX_CVTPI32_PS)
endif()

########################################################################
# if the CPU is not x86, eliminate all Intel SIMD
########################################################################

if(NOT CPU_IS_x86)
    overrule_arch(3dnow "Architecture is not x86 or x86_64")
    overrule_arch(mmx "Architecture is not x86 or x86_64")
    overrule_arch(sse "Architecture is not x86 or x86_64")
    overrule_arch(sse2 "Architecture is not x86 or x86_64")
    overrule_arch(sse3 "Architecture is not x86 or x86_64")
    overrule_arch(ssse3 "Architecture is not x86 or x86_64")
    overrule_arch(sse4_a "Architecture is not x86 or x86_64")
    overrule_arch(sse4_1 "Architecture is not x86 or x86_64")
    overrule_arch(sse4_2 "Architecture is not x86 or x86_64")
    overrule_arch(avx "Architecture is not x86 or x86_64")
    overrule_arch(avx512f "Architecture is not x86 or x86_64")
    overrule_arch(avx512cd "Architecture is not x86 or x86_64")
endif(NOT CPU_IS_x86)

########################################################################
# Select neon based on ARM ISA version
########################################################################

# First, compile a test program to see if compiler supports neon.

include(CheckCSourceCompiles)

set(CMAKE_C_FLAGS_SAVED "${CMAKE_C_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-uninitialized")
check_c_source_compiles("#include <arm_neon.h>
int main(){ uint8_t *dest; uint8x8_t res; vst1_u8(dest, res); }" neon_compile_result)

if(neon_compile_result)
    set(CMAKE_REQUIRED_INCLUDES ${PROJECT_SOURCE_DIR}/include)
    check_c_source_compiles("#include <volk/volk_common.h>
        int main(){__VOLK_ASM(\"vrev32.8 q0, q0\");}" have_neonv7_result)
    check_c_source_compiles("#include <volk/volk_common.h>
        int main(){__VOLK_ASM(\"sub v1.4s,v1.4s,v1.4s\");}" have_neonv8_result)

    if(NOT have_neonv7_result)
        overrule_arch(neonv7 "Compiler doesn't support neonv7")
    endif()

    if(NOT have_neonv8_result)
        overrule_arch(neonv8 "Compiler doesn't support neonv8")
    endif()
else(neon_compile_result)
    overrule_arch(neon "Compiler doesn't support NEON")
    overrule_arch(neonv7 "Compiler doesn't support NEON")
    overrule_arch(neonv8 "Compiler doesn't support NEON")
endif(neon_compile_result)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS_SAVED}")
########################################################################
# implement overruling in the ORC case,
# since ORC always passes flag detection
########################################################################
if(NOT ORC_FOUND)
    overrule_arch(orc "ORC support not found")
endif()

########################################################################
# implement overruling in the non-multilib case
# this makes things work when both -m32 and -m64 pass
########################################################################
if(NOT CROSSCOMPILE_MULTILIB AND CPU_IS_x86)
    include(CheckTypeSize)
    cmake_host_system_information(RESULT ASSUME_64BIT_HOST QUERY IS_64BIT)
    if(ASSUME_64BIT_HOST)
        overrule_arch(32 "CPU width is 64 bits")
    else()
        overrule_arch(64 "CPU width is 32 bits")
    endif()

    #MSVC 64 bit does not have MMX, overrule it
    if(MSVC)
        if(ASSUME_64BIT_HOST)
            overrule_arch(mmx "No MMX for Win64")
        endif()
        force_arch(sse "Built-in for MSVC > 2013")
        force_arch(sse2 "Built-in for MSVC > 2013")
    endif()

endif()

if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "^riscv64$")
    message("---- Adding RISC-V ASM files")
    message("DEBUG: looking for ASM files in ${CMAKE_SOURCE_DIR}/kernels/volk/asm/riscv")
    include_directories(${CMAKE_SOURCE_DIR}/kernels/volk/asm/riscv)
    file(GLOB asm_files ${CMAKE_SOURCE_DIR}/kernels/volk/asm/riscv/*.s)
    foreach(asm_file ${asm_files})
        list(APPEND volk_sources ${asm_file})
        message(STATUS "Adding source file: ${asm_file}")
    endforeach(asm_file)
else()
    overrule_arch(riscv64 "machine is not riscv64")
endif()

########################################################################
# done overrules! print the result
########################################################################
message(STATUS "Available architectures: ${available_archs}")

########################################################################
# determine available machines given the available architectures
########################################################################
execute_process(
    COMMAND
        ${PYTHON_EXECUTABLE} ${PYTHON_DASH_B}
        ${PROJECT_SOURCE_DIR}/gen/volk_compile_utils.py --mode "machines" --archs
        "${available_archs}"
    OUTPUT_VARIABLE available_machines
    OUTPUT_STRIP_TRAILING_WHITESPACE)

########################################################################
# Implement machine overruling for redundant machines:
# A machine is redundant when expansion rules occur,
# and the arch superset passes configuration checks.
# When this occurs, eliminate the redundant machines
# to avoid unnecessary compilation of subset machines.
########################################################################
foreach(arch mmx orc 64 32)
    foreach(machine_name ${available_machines})
        string(REPLACE "_${arch}" "" machine_name_no_arch ${machine_name})
        if(${machine_name} STREQUAL ${machine_name_no_arch})

        else()
            list(REMOVE_ITEM available_machines ${machine_name_no_arch})
        endif()
    endforeach(machine_name)
endforeach(arch)

########################################################################
# done overrules! print the result
########################################################################
message(STATUS "Available machines: ${available_machines}")

########################################################################
# Create rules to run the volk generator
########################################################################

#dependencies are all python, xml, and header implementation files
file(GLOB xml_files ${PROJECT_SOURCE_DIR}/gen/*.xml)
file(GLOB py_files ${PROJECT_SOURCE_DIR}/gen/*.py)
file(GLOB h_files ${PROJECT_SOURCE_DIR}/kernels/volk/*.h)

macro(gen_template tmpl output)
    list(APPEND volk_gen_sources ${output})
    add_custom_command(
        OUTPUT ${output}
        DEPENDS ${xml_files} ${py_files} ${h_files} ${tmpl}
        COMMAND
            ${PYTHON_EXECUTABLE} ${PYTHON_DASH_B}
            ${PROJECT_SOURCE_DIR}/gen/volk_tmpl_utils.py --input ${tmpl} --output
            ${output} ${ARGN})
endmacro(gen_template)

make_directory(${PROJECT_BINARY_DIR}/include/volk)

gen_template(${PROJECT_SOURCE_DIR}/tmpl/volk.tmpl.h
             ${PROJECT_BINARY_DIR}/include/volk/volk.h)
gen_template(${PROJECT_SOURCE_DIR}/tmpl/volk.tmpl.c ${PROJECT_BINARY_DIR}/lib/volk.c)
gen_template(${PROJECT_SOURCE_DIR}/tmpl/volk_typedefs.tmpl.h
             ${PROJECT_BINARY_DIR}/include/volk/volk_typedefs.h)
gen_template(${PROJECT_SOURCE_DIR}/tmpl/volk_cpu.tmpl.h
             ${PROJECT_BINARY_DIR}/include/volk/volk_cpu.h)
gen_template(${PROJECT_SOURCE_DIR}/tmpl/volk_cpu.tmpl.c
             ${PROJECT_BINARY_DIR}/lib/volk_cpu.c)
gen_template(${PROJECT_SOURCE_DIR}/tmpl/volk_config_fixed.tmpl.h
             ${PROJECT_BINARY_DIR}/include/volk/volk_config_fixed.h)
gen_template(${PROJECT_SOURCE_DIR}/tmpl/volk_machines.tmpl.h
             ${PROJECT_BINARY_DIR}/lib/volk_machines.h)
gen_template(${PROJECT_SOURCE_DIR}/tmpl/volk_machines.tmpl.c
             ${PROJECT_BINARY_DIR}/lib/volk_machines.c)

set(BASE_CFLAGS NONE)
string(TOUPPER ${CMAKE_BUILD_TYPE} CBTU)
message(STATUS "BUILD TYPE = ${CBTU}")
message(STATUS "Base cflags = ${CMAKE_C_FLAGS_${CBTU}} ${CMAKE_C_FLAGS}")
set(COMPILER_INFO "")
if(MSVC)
    if(MSVC90) #Visual Studio 9
        set(cmake_c_compiler_version "Microsoft Visual Studio 9.0")
    elseif(MSVC10) #Visual Studio 10
        set(cmake_c_compiler_version "Microsoft Visual Studio 10.0")
    elseif(MSVC11) #Visual Studio 11
        set(cmake_c_compiler_version "Microsoft Visual Studio 11.0")
    elseif(MSVC12) #Visual Studio 12
        set(cmake_c_compiler_version "Microsoft Visual Studio 12.0")
    elseif(MSVC14) #Visual Studio 14
        set(cmake_c_compiler_version "Microsoft Visual Studio 14.0")
    endif()
else()
    execute_process(COMMAND ${CMAKE_C_COMPILER} --version
                    OUTPUT_VARIABLE cmake_c_compiler_version)
endif(MSVC)
set(COMPILER_INFO
    "${CMAKE_C_COMPILER}:::${CMAKE_C_FLAGS_${GRCBTU}} ${CMAKE_C_FLAGS}\n${CMAKE_CXX_COMPILER}:::${CMAKE_CXX_FLAGS_${GRCBTU}} ${CMAKE_CXX_FLAGS}\n"
)

foreach(machine_name ${available_machines})
    #generate machine source
    set(machine_source ${CMAKE_CURRENT_BINARY_DIR}/volk_machine_${machine_name}.c)
    gen_template(${PROJECT_SOURCE_DIR}/tmpl/volk_machine_xxx.tmpl.c ${machine_source}
                 ${machine_name})

    #determine machine flags
    execute_process(
        COMMAND
            ${PYTHON_EXECUTABLE} ${PYTHON_DASH_B}
            ${PROJECT_SOURCE_DIR}/gen/volk_compile_utils.py --mode "machine_flags"
            --machine "${machine_name}" --compiler "${COMPILER_NAME}"
        OUTPUT_VARIABLE ${machine_name}_flags
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    message(
        STATUS
            "BUILD INFO ::: ${machine_name} ::: ${COMPILER_NAME} ::: ${CMAKE_C_FLAGS_${CBTU}} ${CMAKE_C_FLAGS} ${${machine_name}_flags}"
    )
    set(COMPILER_INFO
        "${COMPILER_INFO}${machine_name}:::${COMPILER_NAME}:::${CMAKE_C_FLAGS_${CBTU}} ${CMAKE_C_FLAGS} ${${machine_name}_flags}\n"
    )
    if(${machine_name}_flags AND NOT MSVC)
        set_source_files_properties(${machine_source}
                                    PROPERTIES COMPILE_FLAGS "${${machine_name}_flags}")
    endif()

    #add to available machine defs
    string(TOUPPER LV_MACHINE_${machine_name} machine_def)
    list(APPEND machine_defs ${machine_def})
endforeach(machine_name)

# Convert to a C string to compile and display properly
string(STRIP "${cmake_c_compiler_version}" cmake_c_compiler_version)
string(STRIP ${COMPILER_INFO} COMPILER_INFO)
message(STATUS "Compiler Version: ${cmake_c_compiler_version}")
string(REPLACE "\n" " \\n" cmake_c_compiler_version ${cmake_c_compiler_version})
string(REPLACE "\n" " \\n" COMPILER_INFO ${COMPILER_INFO})

########################################################################
# Handle ASM support
#  on by default, but let users turn it off
########################################################################
set(ASM_ARCHS_AVAILABLE "neonv7" "neonv8")

set(FULL_C_FLAGS "${CMAKE_C_FLAGS}" "${CMAKE_CXX_COMPILER_ARG1}")

# sort through a list of all architectures we have ASM for
# if we find one that matches our current system architecture
# set up the assembler flags and include the source files
foreach(ARCH ${ASM_ARCHS_AVAILABLE})
    string(REGEX MATCH "${ARCH}" ASM_ARCH "${available_archs}")
    if(ASM_ARCH STREQUAL "neonv7")
        message(STATUS "---- Adding ASM files") # we always use ATT syntax
        message(STATUS "-- Detected neon architecture; enabling ASM")
        # architecture specific assembler flags are now set in the cmake toolchain file
        # then add the files
        include_directories(${PROJECT_SOURCE_DIR}/kernels/volk/asm/neon)
        file(GLOB asm_files ${PROJECT_SOURCE_DIR}/kernels/volk/asm/neon/*.s)
        foreach(asm_file ${asm_files})
            list(APPEND volk_sources ${asm_file})
            message(STATUS "Adding source file: ${asm_file}")
        endforeach(asm_file)
    endif()
    enable_language(ASM)
    message(STATUS "c flags: ${FULL_C_FLAGS}")
    message(STATUS "asm flags: ${CMAKE_ASM_FLAGS}")
endforeach(ARCH)

########################################################################
# Handle orc support
########################################################################
if(ORC_FOUND)
    #setup orc library usage
    include_directories(${ORC_INCLUDE_DIRS})
    link_directories(${ORC_LIBRARY_DIRS})

    #setup orc functions
    file(GLOB orc_files ${PROJECT_SOURCE_DIR}/kernels/volk/asm/orc/*.orc)
    foreach(orc_file ${orc_files})

        #extract the name for the generated c source from the orc file
        get_filename_component(orc_file_name_we ${orc_file} NAME_WE)
        set(orcc_gen ${CMAKE_CURRENT_BINARY_DIR}/${orc_file_name_we}.c)

        #create a rule to generate the source and add to the list of sources
        add_custom_command(
            COMMAND ${ORCC_EXECUTABLE} --include math.h --implementation --include
                    volk/volk_complex.h -o ${orcc_gen} ${orc_file}
            DEPENDS ${orc_file}
            OUTPUT ${orcc_gen})
        list(APPEND volk_sources ${orcc_gen})

    endforeach(orc_file)
else()
    message(STATUS "Did not find liborc and orcc, disabling orc support...")
endif()

########################################################################
# Handle the generated constants
########################################################################

message(STATUS "Loading version ${VERSION} into constants...")

#double escape for windows backslash path separators
string(REPLACE "\\" "\\\\" prefix "${prefix}")
string(REPLACE "${CMAKE_SOURCE_DIR}" "$BUILD_DIR" COMPILER_INFO "${COMPILER_INFO}")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/constants.c.in
               ${CMAKE_CURRENT_BINARY_DIR}/constants.c @ONLY)

list(APPEND volk_sources ${CMAKE_CURRENT_BINARY_DIR}/constants.c)

########################################################################
# Setup the volk sources list and library
########################################################################
if(NOT WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden -Wno-deprecated-declarations")
endif()

list(APPEND volk_sources ${CMAKE_CURRENT_SOURCE_DIR}/volk_prefs.c
     ${CMAKE_CURRENT_SOURCE_DIR}/volk_rank_archs.c
     ${CMAKE_CURRENT_SOURCE_DIR}/volk_malloc.c ${volk_gen_sources})

#set the machine definitions where applicable
set_source_files_properties(
    ${CMAKE_CURRENT_BINARY_DIR}/volk.c ${CMAKE_CURRENT_BINARY_DIR}/volk_machines.c
    PROPERTIES COMPILE_DEFINITIONS "${machine_defs}")

if(MSVC)
    #add compatibility includes for stdint types
    include_directories(${PROJECT_SOURCE_DIR}/cmake/msvc)
    add_definitions(-DHAVE_CONFIG_H)
    #compile the sources as C++ due to the lack of complex.h under MSVC
    set_source_files_properties(${volk_sources} PROPERTIES LANGUAGE CXX)
endif()

#Create an object to reference Volk source and object files.
#
#NOTE: This object cannot be used to propagate include directories or
#library linkage, but we have to define them here for compiling to
#work. There are options starting with CMake 3.13 for using the OBJECT
#to propagate this information.
add_library(volk_obj OBJECT ${volk_sources})
target_include_directories(
    volk_obj
    PRIVATE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/kernels>
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(VOLK_CPU_FEATURES)
    set_source_files_properties(volk_cpu.c PROPERTIES COMPILE_DEFINITIONS
                                                      "VOLK_CPU_FEATURES=1")
    target_include_directories(
        volk_obj
        PRIVATE $<TARGET_PROPERTY:CpuFeatures::cpu_features,INTERFACE_INCLUDE_DIRECTORIES>
    )
endif()
target_compile_features(volk_obj PUBLIC c_std_17)

#Configure object target properties
if(NOT MSVC)
    set_target_properties(volk_obj PROPERTIES COMPILE_FLAGS "-fPIC")
endif()

# Disable complex math NaN/INFO range checking for performance
include(CheckCCompilerFlag)
check_c_compiler_flag(-fcx-limited-range HAVE_C_LIMITED_RANGE)
if(HAVE_C_LIMITED_RANGE)
    set_target_properties(volk_obj PROPERTIES COMPILE_FLAGS "-fcx-limited-range")
    check_c_compiler_flag(-Wno-unused-command-line-argument HAVE_WNO_UNUSED_CMD_LINE_ARG)
    if(HAVE_WNO_UNUSED_CMD_LINE_ARG)
        set_target_properties(volk_obj PROPERTIES COMPILE_FLAGS
                                                  "-Wno-unused-command-line-argument")
    endif(HAVE_WNO_UNUSED_CMD_LINE_ARG)
endif(HAVE_C_LIMITED_RANGE)

#Add dynamic library
#
#NOTE: The PUBLIC include directories and library linkage will be
#propagated when this target is used to build other targets. Also, the
#PUBLIC library linkage and include directories INSTALL_INTERFACE will
#be used to create the target import information. Ordering of the
#include directories is taken as provided; it -might- matter, but
#probably doesn't.
add_library(volk SHARED $<TARGET_OBJECTS:volk_obj>)
target_link_libraries(volk PUBLIC ${volk_libraries})
if(VOLK_CPU_FEATURES)
    target_link_libraries(volk PRIVATE $<BUILD_INTERFACE:CpuFeatures::cpu_features>)
endif()
target_include_directories(
    volk
    PUBLIC $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/kernels>
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC $<INSTALL_INTERFACE:include>)

#Configure target properties
if(ORC_FOUND)
    target_link_libraries(volk PRIVATE ${ORC_LIBRARIES})
endif()
if(NOT MSVC)
    target_link_libraries(volk PUBLIC m)
endif()
set_target_properties(volk PROPERTIES VERSION ${VERSION})
set_target_properties(volk PROPERTIES SOVERSION ${SOVERSION})
set_target_properties(volk PROPERTIES DEFINE_SYMBOL "volk_EXPORTS")

#Install locations
install(
    TARGETS volk
    EXPORT VOLK-export
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT "volk_runtime" # .so file
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT "volk_devel" # .lib file
    RUNTIME DESTINATION bin COMPONENT "volk_runtime" # .dll file
)

#Configure static library
#
#NOTE: The PUBLIC include directories and library linkage will be
#propagated when this target is used to build other targets. Also, the
#PUBLIC library linkage and include directories INSTALL_INTERFACE will
#be used to create the target import information. Ordering of the
#include directories is taken as provided; it -might- matter, but
#probably doesn't.
if(ENABLE_STATIC_LIBS)
    add_library(volk_static STATIC $<TARGET_OBJECTS:volk_obj>)
    target_link_libraries(volk_static PUBLIC ${volk_libraries})
    if(VOLK_CPU_FEATURES)
        target_link_libraries(volk_static
                              PRIVATE $<BUILD_INTERFACE:CpuFeatures::cpu_features>)
    endif()
    if(ORC_FOUND)
        target_link_libraries(volk_static PUBLIC ${ORC_LIBRARIES_STATIC})
    endif()
    if(NOT ANDROID)
        target_link_libraries(volk_static PRIVATE pthread)
    endif()
    if(NOT MSVC)
        target_link_libraries(volk_static PUBLIC m)
    endif()
    target_include_directories(
        volk_static
        PUBLIC $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/kernels>
        PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
        PUBLIC $<INSTALL_INTERFACE:include>)

    set_target_properties(volk_static PROPERTIES OUTPUT_NAME volk)

    install(
        TARGETS volk_static
        EXPORT VOLK-export
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT "volk_devel")
endif(ENABLE_STATIC_LIBS)

########################################################################
# Build the QA test application
########################################################################
if(ENABLE_TESTING)

    make_directory(${CMAKE_CURRENT_BINARY_DIR}/.unittest)
    include(VolkAddTest)
    if(ENABLE_STATIC_LIBS)
        volk_gen_test(volk_test_all SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/testqa.cc
                      ${CMAKE_CURRENT_SOURCE_DIR}/qa_utils.cc TARGET_DEPS volk_static)
    else()
        volk_gen_test(volk_test_all SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/testqa.cc
                      ${CMAKE_CURRENT_SOURCE_DIR}/qa_utils.cc TARGET_DEPS volk)
    endif()
    foreach(kernel ${h_files})
        get_filename_component(kernel ${kernel} NAME)
        string(REPLACE ".h" "" kernel ${kernel})
        volk_add_test(${kernel} volk_test_all)
    endforeach()

endif(ENABLE_TESTING)
