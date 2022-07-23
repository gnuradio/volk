/* -*- c++ -*- */
/*
 * Copyright 2011-2020 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_VOLK_RUNTIME
#define INCLUDED_VOLK_RUNTIME

#include <volk/volk_typedefs.h>
#include <volk/volk_config_fixed.h>
#include <volk/volk_common.h>
#include <volk/volk_complex.h>
#include <volk/volk_malloc.h>
#include <volk/volk_version.h>

#include <stdlib.h>
#include <stdbool.h>

__VOLK_DECL_BEGIN

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

/*!
 * The VOLK_OR_PTR macro is a convenience macro
 * for checking the alignment of a set of pointers.
 * Example usage:
 * volk_is_aligned(VOLK_OR_PTR((VOLK_OR_PTR(p0, p1), p2)))
 */
#define VOLK_OR_PTR(ptr0, ptr1) \
    (const void *)(((intptr_t)(ptr0)) | ((intptr_t)(ptr1)))

/*!
 * Is the pointer on a machine alignment boundary?
 *
 * Note: for performance reasons, this function
 * is not usable until another volk API call is made
 * which will perform certain initialization tasks.
 *
 * \param ptr the pointer to some memory buffer
 * \return 1 for alignment boundary, else 0
 */
VOLK_API bool volk_is_aligned(const void *ptr);

// Just drop the deprecated attribute in case we are on Windows. Clang and GCC support `__attribute__`.
// We just assume the compiler and the system are tight together as far as Mako templates are concerned.
<%
deprecated_kernels = ('volk_16i_x5_add_quad_16i_x4', 'volk_16i_branch_4_state_8', 
                      'volk_16i_max_star_16i', 'volk_16i_max_star_horizontal_16i', 
                      'volk_16i_permute_and_scalar_add', 'volk_16i_x4_quad_max_star_16i')
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

__VOLK_DECL_END

#endif /*INCLUDED_VOLK_RUNTIME*/
