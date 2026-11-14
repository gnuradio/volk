/* -*- c++ -*- */
/*
 * Copyright 2016 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
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
 * void volk_32fc_convert_16ic(lv_16sc_t* outputVector, const lv_32fc_t* inputVector,
 * unsigned int num_points); \endcode
 *
 * \b Inputs
 * \li inputVector:  The complex 32-bit float input data buffer.
 * \li num_points:   The number of data values to be converted.
 *
 * \b Outputs
 * \li outputVector: The complex 16-bit integer output data buffer.
 *
 */

#ifndef INCLUDED_volk_32fc_convert_16ic_a_H
#define INCLUDED_volk_32fc_convert_16ic_a_H

#include "volk/volk_complex.h"
#include <limits.h>
#include <math.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_32fc_convert_16ic_generic(lv_16sc_t* outputVector,
                                                  const lv_32fc_t* inputVector,
                                                  unsigned int num_points)
{
    const float min_val = (float)SHRT_MIN;
    const float max_val = (float)SHRT_MAX;
    const float* input = (const float*)inputVector;
    int16_t* output = (int16_t*)outputVector;

    for (unsigned int number = 0; number < num_points * 2; ++number) {
        float value = input[number];
        if (value > max_val)
            value = max_val;
        else if (value < min_val)
            value = min_val;
        output[number] = (int16_t)rintf(value);
    }
}

#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32fc_convert_16ic_a_avx2(lv_16sc_t* outputVector,
                                                 const lv_32fc_t* inputVector,
                                                 unsigned int num_points)
{
    const unsigned int avx_iters = num_points / 8;

    const __m256 vmin_val = _mm256_set1_ps((float)SHRT_MIN);
    const __m256 vmax_val = _mm256_set1_ps((float)SHRT_MAX);

    for (unsigned int number = 0; number < avx_iters; ++number) {
        const __m256 inputVal1 = _mm256_load_ps((const float*)inputVector);
        const __m256 inputVal2 = _mm256_load_ps((const float*)inputVector + 8);
        __VOLK_PREFETCH(inputVector + 16);
        const __m256 ret1 = _mm256_max_ps(_mm256_min_ps(inputVal1, vmax_val), vmin_val);
        const __m256 ret2 = _mm256_max_ps(_mm256_min_ps(inputVal2, vmax_val), vmin_val);
        const __m256i packed = _mm256_permute4x64_epi64(
            _mm256_packs_epi32(_mm256_cvtps_epi32(ret1), _mm256_cvtps_epi32(ret2)), 0xd8);
        _mm256_store_si256((__m256i*)outputVector, packed);
        inputVector += 8;
        outputVector += 8;
    }

    volk_32fc_convert_16ic_generic(outputVector, inputVector, num_points - avx_iters * 8);
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32fc_convert_16ic_a_avx512(lv_16sc_t* outputVector,
                                                   const lv_32fc_t* inputVector,
                                                   unsigned int num_points)
{
    const unsigned int avx512_iters = num_points / 8;

    const __m512 vmin_val = _mm512_set1_ps((float)SHRT_MIN);
    const __m512 vmax_val = _mm512_set1_ps((float)SHRT_MAX);

    for (unsigned int number = 0; number < avx512_iters; ++number) {
        const __m512 inputVal = _mm512_load_ps((const float*)inputVector);
        __VOLK_PREFETCH((const float*)inputVector + 16);
        const __m512 ret = _mm512_max_ps(_mm512_min_ps(inputVal, vmax_val), vmin_val);
        const __m256i output = _mm512_cvtsepi32_epi16(_mm512_cvtps_epi32(ret));
        _mm256_store_si256((__m256i*)outputVector, output);
        inputVector += 8;
        outputVector += 8;
    }

    volk_32fc_convert_16ic_generic(
        outputVector, inputVector, num_points - avx512_iters * 8);
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_32fc_convert_16ic_a_sse2(lv_16sc_t* outputVector,
                                                 const lv_32fc_t* inputVector,
                                                 unsigned int num_points)
{
    const unsigned int sse_iters = num_points / 4;

    const __m128 vmin_val = _mm_set1_ps((float)SHRT_MIN);
    const __m128 vmax_val = _mm_set1_ps((float)SHRT_MAX);

    for (unsigned int number = 0; number < sse_iters; ++number) {
        const __m128 inputVal1 = _mm_load_ps((const float*)inputVector);
        const __m128 inputVal2 = _mm_load_ps((const float*)inputVector + 4);
        __VOLK_PREFETCH((const float*)inputVector + 8);
        const __m128 ret1 = _mm_max_ps(_mm_min_ps(inputVal1, vmax_val), vmin_val);
        const __m128 ret2 = _mm_max_ps(_mm_min_ps(inputVal2, vmax_val), vmin_val);
        const __m128i output =
            _mm_packs_epi32(_mm_cvtps_epi32(ret1), _mm_cvtps_epi32(ret2));
        _mm_store_si128((__m128i*)outputVector, output);
        inputVector += 4;
        outputVector += 4;
    }

    volk_32fc_convert_16ic_generic(outputVector, inputVector, num_points - sse_iters * 4);
}
#endif /* LV_HAVE_SSE2 */


#if LV_HAVE_NEONV7
#include <arm_neon.h>

static inline void volk_32fc_convert_16ic_neon(lv_16sc_t* outputVector,
                                               const lv_32fc_t* inputVector,
                                               unsigned int num_points)
{

    const unsigned int neon_iters = num_points / 4;

    const float32x4_t min_val = vdupq_n_f32((float)SHRT_MIN);
    const float32x4_t max_val = vdupq_n_f32((float)SHRT_MAX);
    const float32x4_t half = vdupq_n_f32(0.5f);
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const int32x4_t one = vdupq_n_s32(1);

    for (unsigned int number = 0; number < neon_iters; ++number) {
        const float32x4_t inputVal1 = vld1q_f32((const float*)inputVector);
        const float32x4_t inputVal2 = vld1q_f32((const float*)inputVector + 4);
        __VOLK_PREFETCH((const float*)inputVector + 8);
        const float32x4_t ret1 = vmaxq_f32(vminq_f32(inputVal1, max_val), min_val);
        const float32x4_t ret2 = vmaxq_f32(vminq_f32(inputVal2, max_val), min_val);
        int32x4_t output1 = vcvtq_s32_f32(
            vsubq_f32(vaddq_f32(ret1, half),
                      vcvtq_f32_u32(vshrq_n_u32(vreinterpretq_u32_f32(ret1), 31))));
        int32x4_t output2 = vcvtq_s32_f32(
            vsubq_f32(vaddq_f32(ret2, half),
                      vcvtq_f32_u32(vshrq_n_u32(vreinterpretq_u32_f32(ret2), 31))));
        const uint32x4_t correction_mask1 =
            vandq_u32(vceqq_f32(vabsq_f32(vsubq_f32(ret1, vcvtq_f32_s32(output1))), half),
                      vtstq_s32(output1, one));
        const uint32x4_t correction_mask2 =
            vandq_u32(vceqq_f32(vabsq_f32(vsubq_f32(ret2, vcvtq_f32_s32(output2))), half),
                      vtstq_s32(output2, one));
        output1 =
            vsubq_s32(output1,
                      vbslq_s32(correction_mask1,
                                vbslq_s32(vcltq_f32(ret1, zero), vdupq_n_s32(-1), one),
                                vdupq_n_s32(0)));
        output2 =
            vsubq_s32(output2,
                      vbslq_s32(correction_mask2,
                                vbslq_s32(vcltq_f32(ret2, zero), vdupq_n_s32(-1), one),
                                vdupq_n_s32(0)));
        vst1q_s16((int16_t*)outputVector,
                  vcombine_s16(vqmovn_s32(output1), vqmovn_s32(output2)));
        inputVector += 4;
        outputVector += 4;
    }

    volk_32fc_convert_16ic_generic(
        outputVector, inputVector, num_points - neon_iters * 4);
}

#endif /* LV_HAVE_NEONV7 */

#if LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32fc_convert_16ic_neonv8(lv_16sc_t* outputVector,
                                                 const lv_32fc_t* inputVector,
                                                 unsigned int num_points)
{
    const unsigned int neon_iters = num_points / 4;

    const float32x4_t min_val = vdupq_n_f32((float)SHRT_MIN);
    const float32x4_t max_val = vdupq_n_f32((float)SHRT_MAX);

    for (unsigned int number = 0; number < neon_iters; ++number) {
        const float32x4_t inputVal1 = vld1q_f32((const float*)inputVector);
        const float32x4_t inputVal2 = vld1q_f32((const float*)inputVector + 4);
        __VOLK_PREFETCH((const float*)inputVector + 8);
        const float32x4_t ret1 = vmaxq_f32(vminq_f32(inputVal1, max_val), min_val);
        const float32x4_t ret2 = vmaxq_f32(vminq_f32(inputVal2, max_val), min_val);
        const int32x4_t output1 = vcvtq_s32_f32(vrndiq_f32(ret1));
        const int32x4_t output2 = vcvtq_s32_f32(vrndiq_f32(ret2));
        vst1q_s16((int16_t*)outputVector,
                  vcombine_s16(vqmovn_s32(output1), vqmovn_s32(output2)));
        inputVector += 4;
        outputVector += 4;
    }

    volk_32fc_convert_16ic_generic(
        outputVector, inputVector, num_points - neon_iters * 4);
}
#endif /* LV_HAVE_NEONV8 */


#endif /* INCLUDED_volk_32fc_convert_16ic_a_H */

#ifndef INCLUDED_volk_32fc_convert_16ic_u_H
#define INCLUDED_volk_32fc_convert_16ic_u_H

#include "volk/volk_complex.h"
#include <limits.h>
#include <math.h>


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32fc_convert_16ic_u_avx2(lv_16sc_t* outputVector,
                                                 const lv_32fc_t* inputVector,
                                                 unsigned int num_points)
{
    const unsigned int avx_iters = num_points / 8;

    const __m256 vmin_val = _mm256_set1_ps((float)SHRT_MIN);
    const __m256 vmax_val = _mm256_set1_ps((float)SHRT_MAX);

    for (unsigned int number = 0; number < avx_iters; ++number) {
        const __m256 inputVal1 = _mm256_loadu_ps((const float*)inputVector);
        const __m256 inputVal2 = _mm256_loadu_ps((const float*)inputVector + 8);
        __VOLK_PREFETCH((const float*)inputVector + 16);
        const __m256 ret1 = _mm256_max_ps(_mm256_min_ps(inputVal1, vmax_val), vmin_val);
        const __m256 ret2 = _mm256_max_ps(_mm256_min_ps(inputVal2, vmax_val), vmin_val);
        const __m256i output = _mm256_permute4x64_epi64(
            _mm256_packs_epi32(_mm256_cvtps_epi32(ret1), _mm256_cvtps_epi32(ret2)), 0xd8);
        _mm256_storeu_si256((__m256i*)outputVector, output);
        inputVector += 8;
        outputVector += 8;
    }

    volk_32fc_convert_16ic_generic(outputVector, inputVector, num_points - avx_iters * 8);
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32fc_convert_16ic_u_avx512(lv_16sc_t* outputVector,
                                                   const lv_32fc_t* inputVector,
                                                   unsigned int num_points)
{
    const unsigned int avx512_iters = num_points / 8;

    const __m512 vmin_val = _mm512_set1_ps((float)SHRT_MIN);
    const __m512 vmax_val = _mm512_set1_ps((float)SHRT_MAX);

    for (unsigned int number = 0; number < avx512_iters; ++number) {
        const __m512 inputVal = _mm512_loadu_ps((const float*)inputVector);
        __VOLK_PREFETCH((const float*)inputVector + 16);
        const __m512 ret = _mm512_max_ps(_mm512_min_ps(inputVal, vmax_val), vmin_val);
        const __m256i output = _mm512_cvtsepi32_epi16(_mm512_cvtps_epi32(ret));
        _mm256_storeu_si256((__m256i*)outputVector, output);
        inputVector += 8;
        outputVector += 8;
    }

    volk_32fc_convert_16ic_generic(
        outputVector, inputVector, num_points - avx512_iters * 8);
}
#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_32fc_convert_16ic_u_sse2(lv_16sc_t* outputVector,
                                                 const lv_32fc_t* inputVector,
                                                 unsigned int num_points)
{
    const unsigned int sse_iters = num_points / 4;

    const __m128 vmin_val = _mm_set1_ps((float)SHRT_MIN);
    const __m128 vmax_val = _mm_set1_ps((float)SHRT_MAX);

    for (unsigned int number = 0; number < sse_iters; ++number) {
        const __m128 inputVal1 = _mm_loadu_ps((const float*)inputVector);
        const __m128 inputVal2 = _mm_loadu_ps((const float*)inputVector + 4);
        __VOLK_PREFETCH((const float*)inputVector + 8);
        const __m128 ret1 = _mm_max_ps(_mm_min_ps(inputVal1, vmax_val), vmin_val);
        const __m128 ret2 = _mm_max_ps(_mm_min_ps(inputVal2, vmax_val), vmin_val);
        const __m128i output =
            _mm_packs_epi32(_mm_cvtps_epi32(ret1), _mm_cvtps_epi32(ret2));
        _mm_storeu_si128((__m128i*)outputVector, output);
        inputVector += 4;
        outputVector += 4;
    }

    volk_32fc_convert_16ic_generic(outputVector, inputVector, num_points - sse_iters * 4);
}
#endif /* LV_HAVE_SSE2 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32fc_convert_16ic_rvv(lv_16sc_t* outputVector,
                                              const lv_32fc_t* inputVector,
                                              unsigned int num_points)
{
    int16_t* out = (int16_t*)outputVector;
    float* in = (float*)inputVector;
    size_t n = num_points * 2;
    for (size_t vl; n > 0; n -= vl, in += vl, out += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t v = __riscv_vle32_v_f32m8(in, vl);
        __riscv_vse16(out, __riscv_vfncvt_x(v, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32fc_convert_16ic_u_H */
