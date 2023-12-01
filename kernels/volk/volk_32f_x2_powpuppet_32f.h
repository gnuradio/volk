/* -*- c++ -*- */
/*
 * Copyright 2023 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */


#ifndef INCLUDED_volk_32f_x2_powpuppet_32f_H
#define INCLUDED_volk_32f_x2_powpuppet_32f_H

#include <math.h>
#include <volk/volk.h>
#include <volk/volk_32f_x2_pow_32f.h>

static inline float* make_positive(const float* input, unsigned int num_points)
{
    float* output = (float*)volk_malloc(num_points * sizeof(float), volk_get_alignment());
    for (unsigned int i = 0; i < num_points; i++) {
        output[i] = fabsf(input[i]);
        if (output[i] == 0) {
            output[i] = 2.0f;
        }
    }
    return output;
}

#if LV_HAVE_AVX2 && LV_HAVE_FMA
static inline void volk_32f_x2_powpuppet_32f_a_avx2_fma(float* cVector,
                                                        const float* bVector,
                                                        const float* aVector,
                                                        unsigned int num_points)
{
    float* aVectorPos = make_positive(aVector, num_points);
    volk_32f_x2_pow_32f_a_avx2_fma(cVector, bVector, aVectorPos, num_points);
    volk_free(aVectorPos);
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA for aligned */

#ifdef LV_HAVE_AVX2
static inline void volk_32f_x2_powpuppet_32f_a_avx2(float* cVector,
                                                    const float* bVector,
                                                    const float* aVector,
                                                    unsigned int num_points)
{
    float* aVectorPos = make_positive(aVector, num_points);
    volk_32f_x2_pow_32f_a_avx2(cVector, bVector, aVectorPos, num_points);
    volk_free(aVectorPos);
}
#endif /* LV_HAVE_AVX2 for aligned */

#ifdef LV_HAVE_SSE4_1
static inline void volk_32f_x2_powpuppet_32f_a_sse4_1(float* cVector,
                                                      const float* bVector,
                                                      const float* aVector,
                                                      unsigned int num_points)
{
    float* aVectorPos = make_positive(aVector, num_points);
    volk_32f_x2_pow_32f_a_sse4_1(cVector, bVector, aVectorPos, num_points);
    volk_free(aVectorPos);
}
#endif /* LV_HAVE_SSE4_1 for aligned */

#ifdef LV_HAVE_GENERIC
static inline void volk_32f_x2_powpuppet_32f_generic(float* cVector,
                                                     const float* bVector,
                                                     const float* aVector,
                                                     unsigned int num_points)
{
    float* aVectorPos = make_positive(aVector, num_points);
    volk_32f_x2_pow_32f_generic(cVector, bVector, aVectorPos, num_points);
    volk_free(aVectorPos);
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_SSE4_1
static inline void volk_32f_x2_powpuppet_32f_u_sse4_1(float* cVector,
                                                      const float* bVector,
                                                      const float* aVector,
                                                      unsigned int num_points)
{
    float* aVectorPos = make_positive(aVector, num_points);
    volk_32f_x2_pow_32f_u_sse4_1(cVector, bVector, aVectorPos, num_points);
    volk_free(aVectorPos);
}
#endif /* LV_HAVE_SSE4_1 for unaligned */

#if LV_HAVE_AVX2 && LV_HAVE_FMA
static inline void volk_32f_x2_powpuppet_32f_u_avx2_fma(float* cVector,
                                                        const float* bVector,
                                                        const float* aVector,
                                                        unsigned int num_points)
{
    float* aVectorPos = make_positive(aVector, num_points);
    volk_32f_x2_pow_32f_u_avx2_fma(cVector, bVector, aVectorPos, num_points);
    volk_free(aVectorPos);
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA for unaligned */

#ifdef LV_HAVE_AVX2
static inline void volk_32f_x2_powpuppet_32f_u_avx2(float* cVector,
                                                    const float* bVector,
                                                    const float* aVector,
                                                    unsigned int num_points)
{
    float* aVectorPos = make_positive(aVector, num_points);
    volk_32f_x2_pow_32f_u_avx2(cVector, bVector, aVectorPos, num_points);
    volk_free(aVectorPos);
}
#endif /* LV_HAVE_AVX2 for unaligned */

#endif /* INCLUDED_volk_32f_x2_powpuppet_32f_H */
