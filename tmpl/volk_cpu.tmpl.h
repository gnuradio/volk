/* -*- c++ -*- */
/*
 * Copyright 2011-2012 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_VOLK_CPU_H
#define INCLUDED_VOLK_CPU_H

#include <volk/volk_common.h>

__VOLK_DECL_BEGIN

struct VOLK_CPU {
    %for arch in archs:
    int (*has_${arch.name}) ();
    %endfor
};

extern struct VOLK_CPU volk_cpu;

void volk_cpu_init ();
unsigned int volk_get_lvarch ();

__VOLK_DECL_END

#endif /*INCLUDED_VOLK_CPU_H*/
