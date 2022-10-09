/* -*- c++ -*- */
/*
 * Copyright 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#ifndef INCLUDED_volk_64u_popcntpuppet_64u_H
#define INCLUDED_volk_64u_popcntpuppet_64u_H

#include <stdint.h>
#include <string.h>
#include <volk/volk_64u_popcnt.h>

#ifdef LV_HAVE_GENERIC
static inline void volk_64u_popcntpuppet_64u_generic(uint64_t* outVector,
                                                     const uint64_t* inVector,
                                                     unsigned int num_points)
{
    unsigned int ii;
    for (ii = 0; ii < num_points; ++ii) {
        volk_64u_popcnt_generic(outVector + ii, num_points);
    }
    memcpy((void*)outVector, (void*)inVector, num_points * sizeof(uint64_t));
}
#endif /* LV_HAVE_GENERIC */

#if LV_HAVE_SSE4_2 && LV_HAVE_64
static inline void volk_64u_popcntpuppet_64u_a_sse4_2(uint64_t* outVector,
                                                      const uint64_t* inVector,
                                                      unsigned int num_points)
{
    unsigned int ii;
    for (ii = 0; ii < num_points; ++ii) {
        volk_64u_popcnt_a_sse4_2(outVector + ii, num_points);
    }
    memcpy((void*)outVector, (void*)inVector, num_points * sizeof(uint64_t));
}
#endif /* LV_HAVE_SSE4_2 */

#ifdef LV_HAVE_NEON
static inline void volk_64u_popcntpuppet_64u_neon(uint64_t* outVector,
                                                  const uint64_t* inVector,
                                                  unsigned int num_points)
{
    unsigned int ii;
    for (ii = 0; ii < num_points; ++ii) {
        volk_64u_popcnt_neon(outVector + ii, num_points);
    }
    memcpy((void*)outVector, (void*)inVector, num_points * sizeof(uint64_t));
}
#endif /* LV_HAVE_NEON */

#endif /* INCLUDED_volk_32fc_s32fc_rotatorpuppet_32fc_a_H */
