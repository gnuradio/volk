/* -*- c++ -*- */
/*
 * Copyright 2014, 2015, 2018, 2020 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#ifndef INCLUDED_volk_32u_byteswappuppet_32u_H
#define INCLUDED_volk_32u_byteswappuppet_32u_H

#include <stdint.h>
#include <string.h>
#include <volk/volk_32u_byteswap.h>

#ifdef LV_HAVE_GENERIC
static inline void volk_32u_byteswappuppet_32u_generic(uint32_t* output,
                                                       uint32_t* intsToSwap,
                                                       unsigned int num_points)
{

    volk_32u_byteswap_generic((uint32_t*)intsToSwap, num_points);
    memcpy((void*)output, (void*)intsToSwap, num_points * sizeof(uint32_t));
}
#endif

#ifdef LV_HAVE_NEON
static inline void volk_32u_byteswappuppet_32u_neon(uint32_t* output,
                                                    uint32_t* intsToSwap,
                                                    unsigned int num_points)
{

    volk_32u_byteswap_neon((uint32_t*)intsToSwap, num_points);
    memcpy((void*)output, (void*)intsToSwap, num_points * sizeof(uint32_t));
}
#endif

#ifdef LV_HAVE_NEONV8
static inline void volk_32u_byteswappuppet_32u_neonv8(uint32_t* output,
                                                      uint32_t* intsToSwap,
                                                      unsigned int num_points)
{

    volk_32u_byteswap_neonv8((uint32_t*)intsToSwap, num_points);
    memcpy((void*)output, (void*)intsToSwap, num_points * sizeof(uint32_t));
}
#endif

#ifdef LV_HAVE_SSE2
static inline void volk_32u_byteswappuppet_32u_u_sse2(uint32_t* output,
                                                      uint32_t* intsToSwap,
                                                      unsigned int num_points)
{

    volk_32u_byteswap_u_sse2((uint32_t*)intsToSwap, num_points);
    memcpy((void*)output, (void*)intsToSwap, num_points * sizeof(uint32_t));
}
#endif

#ifdef LV_HAVE_SSE2
static inline void volk_32u_byteswappuppet_32u_a_sse2(uint32_t* output,
                                                      uint32_t* intsToSwap,
                                                      unsigned int num_points)
{

    volk_32u_byteswap_a_sse2((uint32_t*)intsToSwap, num_points);
    memcpy((void*)output, (void*)intsToSwap, num_points * sizeof(uint32_t));
}
#endif

#ifdef LV_HAVE_AVX2
static inline void volk_32u_byteswappuppet_32u_u_avx2(uint32_t* output,
                                                      uint32_t* intsToSwap,
                                                      unsigned int num_points)
{

    volk_32u_byteswap_u_avx2((uint32_t*)intsToSwap, num_points);
    memcpy((void*)output, (void*)intsToSwap, num_points * sizeof(uint32_t));
}
#endif

#ifdef LV_HAVE_AVX2
static inline void volk_32u_byteswappuppet_32u_a_avx2(uint32_t* output,
                                                      uint32_t* intsToSwap,
                                                      unsigned int num_points)
{

    volk_32u_byteswap_a_avx2((uint32_t*)intsToSwap, num_points);
    memcpy((void*)output, (void*)intsToSwap, num_points * sizeof(uint32_t));
}
#endif

#ifdef LV_HAVE_RVV
static inline void volk_32u_byteswappuppet_32u_rvv(uint32_t* output,
                                                   uint32_t* intsToSwap,
                                                   unsigned int num_points)
{

    volk_32u_byteswap_rvv((uint32_t*)intsToSwap, num_points);
    memcpy((void*)output, (void*)intsToSwap, num_points * sizeof(uint32_t));
}
#endif

#ifdef LV_HAVE_RVA23
static inline void volk_32u_byteswappuppet_32u_rva23(uint32_t* output,
                                                     uint32_t* intsToSwap,
                                                     unsigned int num_points)
{

    volk_32u_byteswap_rva23((uint32_t*)intsToSwap, num_points);
    memcpy((void*)output, (void*)intsToSwap, num_points * sizeof(uint32_t));
}
#endif

#endif
