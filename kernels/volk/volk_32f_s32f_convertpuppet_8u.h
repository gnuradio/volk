/* -*- c++ -*- */
/*
 * Copyright 2023 Daniel Estevez <daniel@destevez.net>
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#ifndef INCLUDED_VOLK_32F_S32F_MOD_CONVERTPUPPET_8U_H
#define INCLUDED_VOLK_32F_S32F_MOD_CONVERTPUPPET_8U_H

#include <inttypes.h>
#include <volk/volk_32f_s32f_x2_convert_8u.h>

#ifdef LV_HAVE_GENERIC
static inline void volk_32f_s32f_convertpuppet_8u_generic(uint8_t* output,
                                                          const float* input,
                                                          float scale,
                                                          unsigned int num_points)
{
    volk_32f_s32f_x2_convert_8u_generic(output, input, scale, 128.0, num_points);
}
#endif

#if LV_HAVE_AVX2 && LV_HAVE_FMA
static inline void volk_32f_s32f_convertpuppet_8u_u_avx2_fma(uint8_t* output,
                                                             const float* input,
                                                             float scale,
                                                             unsigned int num_points)
{
    volk_32f_s32f_x2_convert_8u_u_avx2_fma(output, input, scale, 128.0, num_points);
}
#endif

#if LV_HAVE_AVX2 && LV_HAVE_FMA
static inline void volk_32f_s32f_convertpuppet_8u_a_avx2_fma(uint8_t* output,
                                                             const float* input,
                                                             float scale,
                                                             unsigned int num_points)
{
    volk_32f_s32f_x2_convert_8u_a_avx2_fma(output, input, scale, 128.0, num_points);
}
#endif

#ifdef LV_HAVE_AVX2
static inline void volk_32f_s32f_convertpuppet_8u_u_avx2(uint8_t* output,
                                                         const float* input,
                                                         float scale,
                                                         unsigned int num_points)
{
    volk_32f_s32f_x2_convert_8u_u_avx2(output, input, scale, 128.0, num_points);
}
#endif

#ifdef LV_HAVE_AVX2
static inline void volk_32f_s32f_convertpuppet_8u_a_avx2(uint8_t* output,
                                                         const float* input,
                                                         float scale,
                                                         unsigned int num_points)
{
    volk_32f_s32f_x2_convert_8u_a_avx2(output, input, scale, 128.0, num_points);
}
#endif

#ifdef LV_HAVE_SSE2
static inline void volk_32f_s32f_convertpuppet_8u_u_sse2(uint8_t* output,
                                                         const float* input,
                                                         float scale,
                                                         unsigned int num_points)
{
    volk_32f_s32f_x2_convert_8u_u_sse2(output, input, scale, 128.0, num_points);
}
#endif

#ifdef LV_HAVE_SSE2
static inline void volk_32f_s32f_convertpuppet_8u_a_sse2(uint8_t* output,
                                                         const float* input,
                                                         float scale,
                                                         unsigned int num_points)
{
    volk_32f_s32f_x2_convert_8u_a_sse2(output, input, scale, 128.0, num_points);
}
#endif

#ifdef LV_HAVE_SSE
static inline void volk_32f_s32f_convertpuppet_8u_u_sse(uint8_t* output,
                                                        const float* input,
                                                        float scale,
                                                        unsigned int num_points)
{
    volk_32f_s32f_x2_convert_8u_u_sse(output, input, scale, 128.0, num_points);
}
#endif

#ifdef LV_HAVE_SSE
static inline void volk_32f_s32f_convertpuppet_8u_a_sse(uint8_t* output,
                                                        const float* input,
                                                        float scale,
                                                        unsigned int num_points)
{
    volk_32f_s32f_x2_convert_8u_a_sse(output, input, scale, 128.0, num_points);
}
#endif
#endif
