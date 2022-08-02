/* -*- c++ -*- */
/*
 * Copyright 2011-2012 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_LIBVOLK_MACHINES_H
#define INCLUDED_LIBVOLK_MACHINES_H

#include <volk/volk_common.h>
#include <volk/volk_typedefs.h>

#include <stdbool.h>
#include <stdlib.h>

__VOLK_DECL_BEGIN

struct volk_machine {
    const unsigned int caps; //capabilities (i.e., archs compiled into this machine, in the volk_get_lvarch format)
    const char *name;
    const size_t alignment; //the maximum byte alignment required for functions in this library
    %for kern in kernels:
    const char *${kern.name}_name;
    const char *${kern.name}_impl_names[<%len_archs=len(archs)%>${len_archs}];
    const int ${kern.name}_impl_deps[${len_archs}];
    const bool ${kern.name}_impl_alignment[${len_archs}];
    const ${kern.pname} ${kern.name}_impls[${len_archs}];
    const size_t ${kern.name}_n_impls;
    %endfor
};

%for machine in machines:
#ifdef LV_MACHINE_${machine.name.upper()}
extern struct volk_machine volk_machine_${machine.name};
#endif
%endfor

__VOLK_DECL_END

#endif //INCLUDED_LIBVOLK_MACHINES_H
