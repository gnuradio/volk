/* -*- c++ -*- */
/*
 * Copyright 2006,2009,2013 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#ifndef INCLUDED_VOLK_CONSTANTS_H
#define INCLUDED_VOLK_CONSTANTS_H

#include <volk/volk_common.h>

__VOLK_DECL_BEGIN

VOLK_API const char* volk_prefix();
VOLK_API const char* volk_version();
VOLK_API const char* volk_c_compiler();
VOLK_API const char* volk_compiler_flags();
VOLK_API const char* volk_available_machines();

__VOLK_DECL_END

#endif /* INCLUDED_VOLK_CONSTANTS_H */
