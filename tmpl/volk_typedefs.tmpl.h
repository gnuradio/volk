/* -*- c++ -*- */
/*
 * Copyright 2011-2012 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_VOLK_TYPEDEFS
#define INCLUDED_VOLK_TYPEDEFS

#include <inttypes.h>
#include <volk/volk_complex.h>

%for kern in kernels:
typedef void (*${kern.pname})(${kern.arglist_types});
%endfor

#endif /*INCLUDED_VOLK_TYPEDEFS*/
