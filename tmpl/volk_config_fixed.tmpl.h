/* -*- c++ -*- */
/*
 * Copyright 2011-2012 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#ifndef INCLUDED_VOLK_CONFIG_FIXED_H
#define INCLUDED_VOLK_CONFIG_FIXED_H

%for i, arch in enumerate(archs):
#define LV_${arch.name.upper()} ${i}
%endfor

#endif /*INCLUDED_VOLK_CONFIG_FIXED*/
