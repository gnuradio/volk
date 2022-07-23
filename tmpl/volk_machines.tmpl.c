/* -*- c++ -*- */
/*
 * Copyright 2011-2012 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <volk/volk_common.h>
#include <volk/volk_typedefs.h>
#include "volk_machines.h"

struct volk_machine *volk_machines[] = {
%for machine in machines:
#ifdef LV_MACHINE_${machine.name.upper()}
&volk_machine_${machine.name},
#endif
%endfor
};

unsigned int n_volk_machines = sizeof(volk_machines)/sizeof(*volk_machines);
