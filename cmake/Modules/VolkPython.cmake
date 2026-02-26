# Copyright 2010-2011,2013 Free Software Foundation, Inc.
#
# This file is part of VOLK.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
#

if(DEFINED __INCLUDED_VOLK_PYTHON_CMAKE)
    return()
endif()
set(__INCLUDED_VOLK_PYTHON_CMAKE TRUE)

########################################################################
# Setup the python interpreter:
# This allows the user to specify a specific interpreter,
# or finds the interpreter via the built-in cmake module.
########################################################################
# Allow users/CI to override the interpreter explicitly.
# FindPython3 uses Python3_EXECUTABLE as its override input.
if(PYTHON_EXECUTABLE)
    set(Python3_EXECUTABLE "${PYTHON_EXECUTABLE}")
endif()
find_package(Python3 3.4 COMPONENTS Interpreter REQUIRED)

# Expose the resolved interpreter path in the CMake cache/GUI.
set(PYTHON_EXECUTABLE
    ${Python3_EXECUTABLE}
    CACHE FILEPATH "python interpreter")

########################################################################
# Check for the existence of a python module:
# - desc a string description of the check
# - mod the name of the module to import
# - cmd an additional command to run
# - have the result variable to set
########################################################################
macro(VOLK_PYTHON_CHECK_MODULE desc mod cmd have)
    message(STATUS "")
    message(STATUS "Python checking for ${desc}")
    execute_process(
        COMMAND
            ${PYTHON_EXECUTABLE} -c "
#########################################
try: import ${mod}
except:
    try: ${mod}
    except: exit(-1)
try: assert ${cmd}
except: exit(-1)
#########################################"
        RESULT_VARIABLE ${have})
    if(${have} EQUAL 0)
        message(STATUS "Python checking for ${desc} - found")
        set(${have} TRUE)
    else(${have} EQUAL 0)
        message(STATUS "Python checking for ${desc} - not found")
        set(${have} FALSE)
    endif(${have} EQUAL 0)
endmacro(VOLK_PYTHON_CHECK_MODULE)

########################################################################
# Sets the python installation directory VOLK_PYTHON_DIR
# cf. https://github.com/gnuradio/gnuradio/blob/master/cmake/Modules/GrPython.cmake
# From https://github.com/pothosware/SoapySDR/tree/master/python
# https://github.com/pothosware/SoapySDR/blob/master/LICENSE_1_0.txt
########################################################################
if(NOT DEFINED VOLK_PYTHON_DIR)
    execute_process(
        COMMAND
            ${PYTHON_EXECUTABLE} -c "import os
import sysconfig
import site

install_dir = None
# The next line passes a CMake variable into our script.
prefix = '${CMAKE_INSTALL_PREFIX}'

# We use `site` to identify if our chosen prefix is a default one.
# https://docs.python.org/3/library/site.html
try:
    # https://docs.python.org/3/library/site.html#site.getsitepackages
    paths = [p for p in site.getsitepackages() if p.startswith(prefix)]
    if len(paths) == 1: install_dir = paths[0]
except AttributeError: pass

# If we found a default install path, `install_dir` is set.
if not install_dir:
    # We use a custom install prefix!
    # Determine the correct install path in that prefix on the current platform.
    # For Python 3.11+, we could use the 'venv' scheme for all platforms
    # https://docs.python.org/3.11/library/sysconfig.html#installation-paths
    if os.name == 'nt':
        scheme = 'nt'
    else:
        scheme = 'posix_prefix'
    install_dir = sysconfig.get_path('platlib', scheme)
    prefix = sysconfig.get_path('data', scheme)

#strip the prefix to return a relative path
print(os.path.relpath(install_dir, prefix))"
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE VOLK_PYTHON_DIR)
endif()
file(TO_CMAKE_PATH ${VOLK_PYTHON_DIR} VOLK_PYTHON_DIR)

########################################################################
# Create an always-built target with a unique name
# Usage: VOLK_UNIQUE_TARGET(<description> <dependencies list>)
########################################################################
function(VOLK_UNIQUE_TARGET desc)
    file(RELATIVE_PATH reldir ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "import re, hashlib
unique = hashlib.sha256(b'${reldir}${ARGN}').hexdigest()[:5]
print(re.sub(r'\\W', '_', '${desc} ${reldir} ' + unique))"
        OUTPUT_VARIABLE _target
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    add_custom_target(${_target} ALL DEPENDS ${ARGN})
endfunction(VOLK_UNIQUE_TARGET)

########################################################################
# Install python sources (also builds and installs byte-compiled python)
########################################################################
function(VOLK_PYTHON_INSTALL)
    cmake_parse_arguments(VOLK_PYTHON_INSTALL "" "DESTINATION;COMPONENT" "FILES;PROGRAMS"
                          ${ARGN})

    ####################################################################
    if(VOLK_PYTHON_INSTALL_FILES)
        ####################################################################
        install(${ARGN}) #installs regular python files

        #create a list of all generated files
        unset(pysrcfiles)
        unset(pycfiles)
        unset(pyofiles)
        foreach(pyfile ${VOLK_PYTHON_INSTALL_FILES})
            get_filename_component(pyfile ${pyfile} ABSOLUTE)
            list(APPEND pysrcfiles ${pyfile})

            #determine if this file is in the source or binary directory
            file(RELATIVE_PATH source_rel_path ${CMAKE_CURRENT_SOURCE_DIR} ${pyfile})
            string(LENGTH "${source_rel_path}" source_rel_path_len)
            file(RELATIVE_PATH binary_rel_path ${CMAKE_CURRENT_BINARY_DIR} ${pyfile})
            string(LENGTH "${binary_rel_path}" binary_rel_path_len)

            #and set the generated path appropriately
            if(${source_rel_path_len} GREATER ${binary_rel_path_len})
                set(pygenfile ${CMAKE_CURRENT_BINARY_DIR}/${binary_rel_path})
            else()
                set(pygenfile ${CMAKE_CURRENT_BINARY_DIR}/${source_rel_path})
            endif()
            list(APPEND pycfiles ${pygenfile}c)
            list(APPEND pyofiles ${pygenfile}o)

            #ensure generation path exists
            get_filename_component(pygen_path ${pygenfile} PATH)
            file(MAKE_DIRECTORY ${pygen_path})

        endforeach(pyfile)

        #the command to generate the pyc files
        add_custom_command(
            DEPENDS ${pysrcfiles}
            OUTPUT ${pycfiles}
            COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_BINARY_DIR}/python_compile_helper.py
                    ${pysrcfiles} ${pycfiles})

        #the command to generate the pyo files
        add_custom_command(
            DEPENDS ${pysrcfiles}
            OUTPUT ${pyofiles}
            COMMAND ${PYTHON_EXECUTABLE} -O ${PROJECT_BINARY_DIR}/python_compile_helper.py
                    ${pysrcfiles} ${pyofiles})

        #create install rule and add generated files to target list
        set(python_install_gen_targets ${pycfiles} ${pyofiles})
        install(
            FILES ${python_install_gen_targets}
            DESTINATION ${VOLK_PYTHON_INSTALL_DESTINATION}
            COMPONENT ${VOLK_PYTHON_INSTALL_COMPONENT})

        ####################################################################
    elseif(VOLK_PYTHON_INSTALL_PROGRAMS)
        ####################################################################
        file(TO_NATIVE_PATH ${PYTHON_EXECUTABLE} pyexe_native)

        if(CMAKE_CROSSCOMPILING)
            set(pyexe_native "/usr/bin/env python")
        endif()

        foreach(pyfile ${VOLK_PYTHON_INSTALL_PROGRAMS})
            get_filename_component(pyfile_name ${pyfile} NAME)
            get_filename_component(pyfile ${pyfile} ABSOLUTE)
            string(REPLACE "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}" pyexefile
                           "${pyfile}.exe")
            list(APPEND python_install_gen_targets ${pyexefile})

            get_filename_component(pyexefile_path ${pyexefile} PATH)
            file(MAKE_DIRECTORY ${pyexefile_path})

            add_custom_command(
                OUTPUT ${pyexefile}
                DEPENDS ${pyfile}
                COMMAND
                    ${PYTHON_EXECUTABLE} -c
                    "open('${pyexefile}','w').write(r'\#!${pyexe_native}'+'\\n'+open('${pyfile}').read())"
                COMMENT "Shebangin ${pyfile_name}"
                VERBATIM)

            #on windows, python files need an extension to execute
            get_filename_component(pyfile_ext ${pyfile} EXT)
            if(WIN32 AND NOT pyfile_ext)
                set(pyfile_name "${pyfile_name}.py")
            endif()

            install(
                PROGRAMS ${pyexefile}
                RENAME ${pyfile_name}
                DESTINATION ${VOLK_PYTHON_INSTALL_DESTINATION}
                COMPONENT ${VOLK_PYTHON_INSTALL_COMPONENT})
        endforeach(pyfile)

    endif()

    volk_unique_target("pygen" ${python_install_gen_targets})

endfunction(VOLK_PYTHON_INSTALL)

########################################################################
# Write the python helper script that generates byte code files
########################################################################
file(
    WRITE ${PROJECT_BINARY_DIR}/python_compile_helper.py
    "
import sys, py_compile
files = sys.argv[1:]
srcs, gens = files[:len(files)//2], files[len(files)//2:]
for src, gen in zip(srcs, gens):
    py_compile.compile(file=src, cfile=gen, doraise=True)
")
