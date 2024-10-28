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
    for (size_t i = 0; i < num_points; ++i) {
        volk_64u_popcnt_generic(outVector + i, inVector[i]);
    }
}
#endif /* LV_HAVE_GENERIC */

#if LV_HAVE_SSE4_2 && LV_HAVE_64
static inline void volk_64u_popcntpuppet_64u_a_sse4_2(uint64_t* outVector,
                                                      const uint64_t* inVector,
                                                      unsigned int num_points)
{
    for (size_t i = 0; i < num_points; ++i) {
        volk_64u_popcnt_a_sse4_2(outVector + i, inVector[i]);
    }
}
#endif /* LV_HAVE_SSE4_2 */

#ifdef LV_HAVE_NEON
static inline void volk_64u_popcntpuppet_64u_neon(uint64_t* outVector,
                                                  const uint64_t* inVector,
                                                  unsigned int num_points)
{
    for (size_t i = 0; i < num_points; ++i) {
        volk_64u_popcnt_neon(outVector + i, inVector[i]);
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_RVV
static inline void volk_64u_popcntpuppet_64u_rvv(uint64_t* outVector,
                                                 const uint64_t* inVector,
                                                 unsigned int num_points)
{
    for (size_t i = 0; i < num_points; ++i) {
        volk_64u_popcnt_rvv(outVector + i, inVector[i]);
    }
}
#endif /* LV_HAVE_RVV */

#ifdef LV_HAVE_RVA22V
static inline void volk_64u_popcntpuppet_64u_rva22(uint64_t* outVector,
                                                   const uint64_t* inVector,
                                                   unsigned int num_points)
{
    for (size_t i = 0; i < num_points; ++i) {
        volk_64u_popcnt_rva22(outVector + i, inVector[i]);
    }
}
#endif /* LV_HAVE_RVA22V */

#endif /* INCLUDED_volk_32fc_s32fc_rotatorpuppet_32fc_a_H */
