/* -*- c++ -*- */
/*
 * Copyright 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#ifndef INCLUDED_volk_32u_popcntpuppet_32u_H
#define INCLUDED_volk_32u_popcntpuppet_32u_H

#include <stdint.h>
#include <volk/volk_32u_popcnt.h>

#ifdef LV_HAVE_GENERIC
static inline void volk_32u_popcntpuppet_32u_generic(uint32_t* outVector,
                                                     const uint32_t* inVector,
                                                     unsigned int num_points)
{
    unsigned int ii;
    for (ii = 0; ii < num_points; ++ii) {
        volk_32u_popcnt_generic(outVector + ii, *(inVector + ii));
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_SSE4_2
static inline void volk_32u_popcntpuppet_32u_a_sse4_2(uint32_t* outVector,
                                                      const uint32_t* inVector,
                                                      unsigned int num_points)
{
    unsigned int ii;
    for (ii = 0; ii < num_points; ++ii) {
        volk_32u_popcnt_a_sse4_2(outVector + ii, *(inVector + ii));
    }
}
#endif /* LV_HAVE_SSE4_2 */

#endif /* INCLUDED_volk_32fc_s32fc_rotatorpuppet_32fc_a_H */
