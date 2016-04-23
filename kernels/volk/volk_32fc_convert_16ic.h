/* -*- c++ -*- */
/*
 * Copyright 2016 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * GNU Radio is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * GNU Radio is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See thegit
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Radio; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

/*!
 * \page volk_32fc_convert_16ic
 *
 * \b Overview
 *
 * Converts a complex vector of 32-bits float each component into
 * a complex vector of 16-bits integer each component.
 * Values are saturated to the limit values of the output data type.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_convert_16ic(lv_16sc_t* outputVector, const lv_32fc_t* inputVector, unsigned int num_points);
 * \endcode
 *
 * \b Inputs
 * \li inputVector:  The complex 32-bit float input data buffer.
 * \li num_points:   The number of data values to be converted.
 *
 * \b Outputs
 * \li outputVector: The complex 16-bit integer output data buffer.
 *
 */

#ifndef INCLUDED_volk_32fc_convert_16ic_H
#define INCLUDED_volk_32fc_convert_16ic_H

#include <limits.h>
#include <math.h>
#include "volk/volk_complex.h"

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_32fc_convert_16ic_u_sse2(lv_16sc_t* outputVector, const lv_32fc_t* inputVector, unsigned int num_points)
{
    const unsigned int sse_iters = num_points / 4;

    float* inputVectorPtr = (float*)inputVector;
    int16_t* outputVectorPtr = (int16_t*)outputVector;
    float aux;

    const float min_val = (float)SHRT_MIN;
    const float max_val = (float)SHRT_MAX;

    __m128 inputVal1, inputVal2;
    __m128i intInputVal1, intInputVal2;
    __m128 ret1, ret2;
    const __m128 vmin_val = _mm_set_ps1(min_val);
    const __m128 vmax_val = _mm_set_ps1(max_val);

    for(unsigned int i = 0; i < sse_iters; i++)
        {
            inputVal1 = _mm_loadu_ps((float*)inputVectorPtr); inputVectorPtr += 4;
            inputVal2 = _mm_loadu_ps((float*)inputVectorPtr); inputVectorPtr += 4;
            __builtin_prefetch(inputVectorPtr + 8);

            // Clip
            ret1 = _mm_max_ps(_mm_min_ps(inputVal1, vmax_val), vmin_val);
            ret2 = _mm_max_ps(_mm_min_ps(inputVal2, vmax_val), vmin_val);

            intInputVal1 = _mm_cvtps_epi32(ret1);
            intInputVal2 = _mm_cvtps_epi32(ret2);

            intInputVal1 = _mm_packs_epi32(intInputVal1, intInputVal2);

            _mm_storeu_si128((__m128i*)outputVectorPtr, intInputVal1);
            outputVectorPtr += 8;
        }

    for(unsigned int i = sse_iters * 8; i < num_points * 2; i++)
        {
            aux = *inputVectorPtr++;
            if(aux > max_val)
                aux = max_val;
            else if(aux < min_val)
                aux = min_val;
            *outputVectorPtr++ = (int16_t)rintf(aux);
        }
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_32fc_convert_16ic_a_sse2(lv_16sc_t* outputVector, const lv_32fc_t* inputVector, unsigned int num_points)
{
    const unsigned int sse_iters = num_points / 4;

    float* inputVectorPtr = (float*)inputVector;
    int16_t* outputVectorPtr = (int16_t*)outputVector;
    float aux;

    const float min_val = (float)SHRT_MIN;
    const float max_val = (float)SHRT_MAX;

    __m128 inputVal1, inputVal2;
    __m128i intInputVal1, intInputVal2;
    __m128 ret1, ret2;
    const __m128 vmin_val = _mm_set_ps1(min_val);
    const __m128 vmax_val = _mm_set_ps1(max_val);

    for(unsigned int i = 0; i < sse_iters; i++)
        {
            inputVal1 = _mm_load_ps((float*)inputVectorPtr); inputVectorPtr += 4;
            inputVal2 = _mm_load_ps((float*)inputVectorPtr); inputVectorPtr += 4;
            __builtin_prefetch(inputVectorPtr + 8);

            // Clip
            ret1 = _mm_max_ps(_mm_min_ps(inputVal1, vmax_val), vmin_val);
            ret2 = _mm_max_ps(_mm_min_ps(inputVal2, vmax_val), vmin_val);

            intInputVal1 = _mm_cvtps_epi32(ret1);
            intInputVal2 = _mm_cvtps_epi32(ret2);

            intInputVal1 = _mm_packs_epi32(intInputVal1, intInputVal2);

            _mm_store_si128((__m128i*)outputVectorPtr, intInputVal1);
            outputVectorPtr += 8;
        }

    for(unsigned int i = sse_iters * 8; i < num_points * 2; i++)
        {
            aux = *inputVectorPtr++;
            if(aux > max_val)
                aux = max_val;
            else if(aux < min_val)
                aux = min_val;
            *outputVectorPtr++ = (int16_t)rintf(aux);
        }
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32fc_convert_16ic_neon(lv_16sc_t* outputVector, const lv_32fc_t* inputVector, unsigned int num_points)
{
    const unsigned int neon_iters = num_points / 4;

    float32_t* inputVectorPtr = (float32_t*)inputVector;
    int16_t* outputVectorPtr = (int16_t*)outputVector;

    const float min_val_f = (float)SHRT_MIN;
    const float max_val_f = (float)SHRT_MAX;
    float32_t aux;

    const float32x4_t min_val = vmovq_n_f32(min_val_f);
    const float32x4_t max_val = vmovq_n_f32(max_val_f);
    float32x4_t half = vdupq_n_f32(0.5f);
    float32x4_t ret1, ret2, a, b, sign, PlusHalf, Round;

    int32x4_t toint_a, toint_b;
    int16x4_t intInputVal1, intInputVal2;
    int16x8_t res;

    for(unsigned int i = 0; i < neon_iters; i++)
        {
            a = vld1q_f32((const float32_t*)(inputVectorPtr)); inputVectorPtr += 4;
            b = vld1q_f32((const float32_t*)(inputVectorPtr)); inputVectorPtr += 4;
            __builtin_prefetch(inputVectorPtr + 8);

            ret1 = vmaxq_f32(vminq_f32(a, max_val), min_val);
            ret2 = vmaxq_f32(vminq_f32(b, max_val), min_val);

            /* in __aarch64__ we can do that with vcvtaq_s32_f32(ret1); vcvtaq_s32_f32(ret2); */
            sign = vcvtq_f32_u32((vshrq_n_u32(vreinterpretq_u32_f32(ret1), 31)));
            PlusHalf = vaddq_f32(ret1, half);
            Round = vsubq_f32(PlusHalf, sign);
            toint_a = vcvtq_s32_f32(Round);

            sign = vcvtq_f32_u32((vshrq_n_u32(vreinterpretq_u32_f32(ret2), 31)));
            PlusHalf = vaddq_f32(ret2, half);
            Round = vsubq_f32(PlusHalf, sign);
            toint_b = vcvtq_s32_f32(Round);

            intInputVal1 = vqmovn_s32(toint_a);
            intInputVal2 = vqmovn_s32(toint_b);

            res = vcombine_s16(intInputVal1, intInputVal2);
            vst1q_s16((int16_t*)outputVectorPtr, res);
            outputVectorPtr += 8;
        }

    for(unsigned int i = neon_iters * 8; i < num_points * 2; i++)
        {
            aux = *inputVectorPtr++;
            if(aux > max_val_f)
                aux = max_val_f;
            else if(aux < min_val_f)
                aux = min_val_f;
            *outputVectorPtr++ = (int16_t)rintf(aux);
        }
}

#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_GENERIC

static inline void volk_32fc_convert_16ic_generic(lv_16sc_t* outputVector, const lv_32fc_t* inputVector, unsigned int num_points)
{
    float* inputVectorPtr = (float*)inputVector;
    int16_t* outputVectorPtr = (int16_t*)outputVector;
    const float min_val = (float)SHRT_MIN;
    const float max_val = (float)SHRT_MAX;
    float aux;

    for(unsigned int i = 0; i < num_points * 2; i++)
        {
            aux = *inputVectorPtr++;
            if(aux > max_val)
                aux = max_val;
            else if(aux < min_val)
                aux = min_val;
           *outputVectorPtr++ = (int16_t)rintf(aux);
        }
}
#endif /* LV_HAVE_GENERIC */

#endif /* INCLUDED_volk_32fc_convert_16ic_H */
