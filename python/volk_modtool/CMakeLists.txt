#
# Copyright 2013, 2014 Free Software Foundation, Inc.
#
# This file is part of VOLK
#
# SPDX-License-Identifier: LGPL-3.0-or-later
#

########################################################################
# Install python files and apps
########################################################################
include(VolkPython)

VOLK_PYTHON_INSTALL(
    FILES
    __init__.py
    cfg.py
    volk_modtool_generate.py
    DESTINATION ${VOLK_PYTHON_DIR}/volk_modtool
    COMPONENT "volk"
)

VOLK_PYTHON_INSTALL(
    PROGRAMS
    volk_modtool
    DESTINATION ${VOLK_RUNTIME_DIR}
    COMPONENT "volk"
)
