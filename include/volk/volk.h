/* -*- c++ -*- */
/*
 * Copyright 2011-2020 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#ifndef INCLUDED_VOLK_RUNTIME
#define INCLUDED_VOLK_RUNTIME

#include <volk/volk_common.h>
#include <volk/volk_complex.h>
#include <volk/volk_config_fixed.h>
#include <volk/volk_malloc.h>
#include <volk/volk_typedefs.h>
#include <volk/volk_version.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

__VOLK_DECL_BEGIN

#define VOLK_OR_PTR(ptr0, ptr1) (const void*)(((intptr_t)(ptr0)) | ((intptr_t)(ptr1)))

#include <volk/volk_dispatch.h>

__VOLK_DECL_END

#endif /*INCLUDED_VOLK_RUNTIME*/
