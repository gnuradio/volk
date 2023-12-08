/* -*- c++ -*- */
/*
 * Copyright 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#ifndef INCLUDED_volk_32f_x2_fm_detectpuppet_32f_a_H
#define INCLUDED_volk_32f_x2_fm_detectpuppet_32f_a_H

#include "volk_32f_s32f_32f_fm_detect_32f.h"

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_x2_fm_detectpuppet_32f_a_avx(float* outputVector,
                                                         const float* inputVector,
                                                         float* saveValue,
                                                         unsigned int num_points)
{
    const float bound = 2.0f;

    volk_32f_s32f_32f_fm_detect_32f_a_avx(
        outputVector, inputVector, bound, saveValue, num_points);
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_x2_fm_detectpuppet_32f_a_sse(float* outputVector,
                                                         const float* inputVector,
                                                         float* saveValue,
                                                         unsigned int num_points)
{
    const float bound = 2.0f;

    volk_32f_s32f_32f_fm_detect_32f_a_sse(
        outputVector, inputVector, bound, saveValue, num_points);
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_GENERIC

static inline void volk_32f_x2_fm_detectpuppet_32f_generic(float* outputVector,
                                                           const float* inputVector,
                                                           float* saveValue,
                                                           unsigned int num_points)
{
    const float bound = 2.0f;

    volk_32f_s32f_32f_fm_detect_32f_generic(
        outputVector, inputVector, bound, saveValue, num_points);
}
#endif /* LV_HAVE_GENERIC */


#endif /* INCLUDED_volk_32f_x2_fm_detectpuppet_32f_a_H */


#ifndef INCLUDED_volk_32f_x2_fm_detectpuppet_32f_u_H
#define INCLUDED_volk_32f_x2_fm_detectpuppet_32f_u_H

#include "volk_32f_s32f_32f_fm_detect_32f.h"

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_x2_fm_detectpuppet_32f_u_avx(float* outputVector,
                                                         const float* inputVector,
                                                         float* saveValue,
                                                         unsigned int num_points)
{
    const float bound = 2.0f;

    volk_32f_s32f_32f_fm_detect_32f_u_avx(
        outputVector, inputVector, bound, saveValue, num_points);
}
#endif /* LV_HAVE_AVX */
#endif /* INCLUDED_volk_32f_x2_fm_detectpuppet_32f_u_H */
