# -*- coding: utf-8 -*-
#
# Copyright 2022 Johannes Demel.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
#

find_package(Doxygen)
if(DOXYGEN_FOUND)

    message(STATUS "Doxygen found. Building docs ...")

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${PROJECT_BINARY_DIR}/Doxyfile
                   @ONLY)

    add_custom_target(
        volk_doc
        ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
        COMMENT "Generating documentation with Doxygen"
        VERBATIM)

endif(DOXYGEN_FOUND)
