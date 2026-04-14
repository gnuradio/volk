/* -*- c++ -*- */
/*
 * Copyright 2011-2020 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

typedef struct volk_func_desc
{
    const char **impl_names;
    const int *impl_deps;
    const bool *impl_alignment;
    size_t n_impls;
} volk_func_desc_t;

//! Prints a list of machines available
VOLK_API void volk_list_machines(void);

//! Returns the name of the machine this instance will use
VOLK_API const char* volk_get_machine(void);

//! Get the machine alignment in bytes
VOLK_API size_t volk_get_alignment(void);

//! Is the pointer on a machine alignment boundary?
VOLK_API bool volk_is_aligned(const void *ptr);

// Just drop the deprecated attribute in case we are on Windows. Clang and GCC support `__attribute__`.
// We just assume the compiler and the system are tight together as far as Mako templates are concerned.
<%
deprecated_kernels = ('volk_16i_x5_add_quad_16i_x4', 'volk_16i_branch_4_state_8',
                      'volk_16i_max_star_16i', 'volk_16i_max_star_horizontal_16i',
                      'volk_16i_permute_and_scalar_add', 'volk_16i_x4_quad_max_star_16i',
                      'volk_32fc_s32fc_multiply_32fc', 'volk_32fc_s32fc_x2_rotator_32fc',
                      'volk_32fc_x2_s32fc_multiply_conjugate_add_32fc')
from platform import system
if system() == 'Windows':
    deprecated_kernels = ()
%>
%for kern in kernels:

% if kern.name in deprecated_kernels:
//! A function pointer to the dispatcher implementation
extern VOLK_API ${kern.pname} ${kern.name} __attribute__((deprecated));

//! A function pointer to the fastest aligned implementation
extern VOLK_API ${kern.pname} ${kern.name}_a __attribute__((deprecated));

//! A function pointer to the fastest unaligned implementation
extern VOLK_API ${kern.pname} ${kern.name}_u __attribute__((deprecated));

//! Call into a specific implementation given by name
extern VOLK_API void ${kern.name}_manual(${kern.arglist_full}, const char* impl_name) __attribute__((deprecated));

//! Get description parameters for this kernel
extern VOLK_API volk_func_desc_t ${kern.name}_get_func_desc(void) __attribute__((deprecated));
% else:
//! A function pointer to the dispatcher implementation
extern VOLK_API ${kern.pname} ${kern.name};

//! A function pointer to the fastest aligned implementation
extern VOLK_API ${kern.pname} ${kern.name}_a;

//! A function pointer to the fastest unaligned implementation
extern VOLK_API ${kern.pname} ${kern.name}_u;

//! Call into a specific implementation given by name
extern VOLK_API void ${kern.name}_manual(${kern.arglist_full}, const char* impl_name);

//! Get description parameters for this kernel
extern VOLK_API volk_func_desc_t ${kern.name}_get_func_desc(void);
% endif

%endfor
