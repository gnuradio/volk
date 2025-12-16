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
    for (size_t i = 0; i < num_points; ++i) {
        volk_32u_popcnt_generic(outVector + i, inVector[i]);
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_SSE4_2
static inline void volk_32u_popcntpuppet_32u_a_sse4_2(uint32_t* outVector,
                                                      const uint32_t* inVector,
                                                      unsigned int num_points)
{
    for (size_t i = 0; i < num_points; ++i) {
        volk_32u_popcnt_a_sse4_2(outVector + i, inVector[i]);
    }
}
#endif /* LV_HAVE_SSE4_2 */

#ifdef LV_HAVE_NEON
static inline void volk_32u_popcntpuppet_32u_neon(uint32_t* outVector,
                                                  const uint32_t* inVector,
                                                  unsigned int num_points)
{
    for (size_t i = 0; i < num_points; ++i) {
        volk_32u_popcnt_neon(outVector + i, inVector[i]);
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_RVV
static inline void volk_32u_popcntpuppet_32u_rvv(uint32_t* outVector,
                                                 const uint32_t* inVector,
                                                 unsigned int num_points)
{
    for (size_t i = 0; i < num_points; ++i) {
        volk_32u_popcnt_rvv(outVector + i, inVector[i]);
    }
}
#endif /* LV_HAVE_RVV */

#ifdef LV_HAVE_RVA22V
static inline void volk_32u_popcntpuppet_32u_rva22(uint32_t* outVector,
                                                   const uint32_t* inVector,
                                                   unsigned int num_points)
{
    for (size_t i = 0; i < num_points; ++i) {
        volk_32u_popcnt_rva22(outVector + i, inVector[i]);
    }
}
#endif /* LV_HAVE_RVA22V */

#endif /* INCLUDED_volk_32fc_s32fc_rotatorpuppet_32fc_a_H */
