/* -*- c++ -*- */
/*
 * Copyright 2023 Magnus Lundmark <magnuslundmark@gmail.com>
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_s32f_x2_clamp_32f
 *
 * \b Overview
 *
 * Clamps the values to an upper and a lower bound.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_s32f_x2_clamp_32f(float* out,
 *              const float* in,
 *              const float min,
 *              const float max,
 *              unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li in: Pointer to float values.
 * \li min: Minimum value to clamp to. \li max: Maximum value to clamp to. \li
 * num_points: The number of points in the vector.
 *
 * \b Outputs
 * \li out: Pointer to output values.
 *
 * \b Example
 * \code
 * float x[4] = {-2.f, -1.f, 1.f, 2.f};
 * float y[4];
 *
 * volk_32f_s32f_x2_clamp_32f(y, x, -1.5f, 1.5f, 4);
 * // Expect y = {-1.5f, -1.f, 1.f, 1.5f}
 *
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_s32f_x2_clamp_32f_a_H
#define INCLUDED_volk_32fc_s32f_x2_clamp_32f_a_H

#ifdef LV_HAVE_GENERIC
static inline void volk_32f_s32f_x2_clamp_32f_generic(float* out,
                                                      const float* in,
                                                      const float min,
                                                      const float max,
                                                      unsigned int num_points)
{
    unsigned int number = 0;
    for (; number < num_points; number++) {
        if (*in > max) {
            *out = max;
        } else if (*in < min) {
            *out = min;
        } else {
            *out = *in;
        }
        in++;
        out++;
    }
}
#endif /* LV_HAVE_GENERIC */

#if LV_HAVE_AVX2
#include <immintrin.h>
static inline void volk_32f_s32f_x2_clamp_32f_a_avx2(float* out,
                                                     const float* in,
                                                     const float min,
                                                     const float max,
                                                     unsigned int num_points)
{
    const __m256 vmin = _mm256_set1_ps(min);
    const __m256 vmax = _mm256_set1_ps(max);

    unsigned int number = 0;
    unsigned int eighth_points = num_points / 8;
    for (; number < eighth_points; number++) {
        __m256 res = _mm256_load_ps(in);
        __m256 max_mask = _mm256_cmp_ps(vmax, res, _CMP_LT_OS);
        __m256 min_mask = _mm256_cmp_ps(res, vmin, _CMP_LT_OS);
        res = _mm256_blendv_ps(res, vmax, max_mask);
        res = _mm256_blendv_ps(res, vmin, min_mask);
        _mm256_store_ps(out, res);
        in += 8;
        out += 8;
    }

    number = eighth_points * 8;
    volk_32f_s32f_x2_clamp_32f_generic(out, in, min, max, num_points - number);
}
#endif /* LV_HAVE_AVX2 */

#if LV_HAVE_SSE4_1
#include <immintrin.h>
static inline void volk_32f_s32f_x2_clamp_32f_a_sse4_1(float* out,
                                                       const float* in,
                                                       const float min,
                                                       const float max,
                                                       unsigned int num_points)
{
    const __m128 vmin = _mm_set1_ps(min);
    const __m128 vmax = _mm_set1_ps(max);

    unsigned int number = 0;
    unsigned int quarter_points = num_points / 4;
    for (; number < quarter_points; number++) {
        __m128 res = _mm_load_ps(in);
        __m128 max_mask = _mm_cmplt_ps(vmax, res);
        __m128 min_mask = _mm_cmplt_ps(res, vmin);
        res = _mm_blendv_ps(res, vmax, max_mask);
        res = _mm_blendv_ps(res, vmin, min_mask);
        _mm_store_ps(out, res);
        in += 4;
        out += 4;
    }

    number = quarter_points * 4;
    volk_32f_s32f_x2_clamp_32f_generic(out, in, min, max, num_points - number);
}
#endif /* LV_HAVE_SSE4_1 */

#endif /* INCLUDED_volk_32fc_s32f_x2_clamp_32f_a_H */

#ifndef INCLUDED_volk_32fc_s32f_x2_clamp_32f_u_H
#define INCLUDED_volk_32fc_s32f_x2_clamp_32f_u_H

#if LV_HAVE_AVX2
#include <immintrin.h>
static inline void volk_32f_s32f_x2_clamp_32f_u_avx2(float* out,
                                                     const float* in,
                                                     const float min,
                                                     const float max,
                                                     unsigned int num_points)
{
    const __m256 vmin = _mm256_set1_ps(min);
    const __m256 vmax = _mm256_set1_ps(max);

    unsigned int number = 0;
    unsigned int eighth_points = num_points / 8;
    for (; number < eighth_points; number++) {
        __m256 res = _mm256_loadu_ps(in);
        __m256 max_mask = _mm256_cmp_ps(vmax, res, _CMP_LT_OS);
        __m256 min_mask = _mm256_cmp_ps(res, vmin, _CMP_LT_OS);
        res = _mm256_blendv_ps(res, vmax, max_mask);
        res = _mm256_blendv_ps(res, vmin, min_mask);
        _mm256_storeu_ps(out, res);
        in += 8;
        out += 8;
    }

    number = eighth_points * 8;
    volk_32f_s32f_x2_clamp_32f_generic(out, in, min, max, num_points - number);
}
#endif /* LV_HAVE_AVX2 */

#if LV_HAVE_SSE4_1
#include <immintrin.h>
static inline void volk_32f_s32f_x2_clamp_32f_u_sse4_1(float* out,
                                                       const float* in,
                                                       const float min,
                                                       const float max,
                                                       unsigned int num_points)
{
    const __m128 vmin = _mm_set1_ps(min);
    const __m128 vmax = _mm_set1_ps(max);

    unsigned int number = 0;
    unsigned int quarter_points = num_points / 4;
    for (; number < quarter_points; number++) {
        __m128 res = _mm_loadu_ps(in);
        __m128 max_mask = _mm_cmplt_ps(vmax, res);
        __m128 min_mask = _mm_cmplt_ps(res, vmin);
        res = _mm_blendv_ps(res, vmax, max_mask);
        res = _mm_blendv_ps(res, vmin, min_mask);
        _mm_storeu_ps(out, res);
        in += 4;
        out += 4;
    }

    number = quarter_points * 4;
    volk_32f_s32f_x2_clamp_32f_generic(out, in, min, max, num_points - number);
}
#endif /* LV_HAVE_SSE4_1 */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32f_s32f_x2_clamp_32f_neon(float* out,
                                                   const float* in,
                                                   const float min,
                                                   const float max,
                                                   unsigned int num_points)
{
    const float32x4_t vmin = vdupq_n_f32(min);
    const float32x4_t vmax = vdupq_n_f32(max);

    unsigned int number = 0;
    const unsigned int quarter_points = num_points / 4;

    for (; number < quarter_points; number++) {
        float32x4_t val = vld1q_f32(in);
        val = vmaxq_f32(val, vmin);
        val = vminq_f32(val, vmax);
        vst1q_f32(out, val);
        in += 4;
        out += 4;
    }

    number = quarter_points * 4;
    for (; number < num_points; number++) {
        float val = *in++;
        if (val < min)
            val = min;
        else if (val > max)
            val = max;
        *out++ = val;
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32f_s32f_x2_clamp_32f_neonv8(float* out,
                                                     const float* in,
                                                     const float min,
                                                     const float max,
                                                     unsigned int num_points)
{
    const float32x4_t vmin = vdupq_n_f32(min);
    const float32x4_t vmax = vdupq_n_f32(max);

    unsigned int number = 0;
    const unsigned int eighth_points = num_points / 8;

    for (; number < eighth_points; number++) {
        float32x4_t val0 = vld1q_f32(in);
        float32x4_t val1 = vld1q_f32(in + 4);
        __VOLK_PREFETCH(in + 8);

        val0 = vmaxq_f32(val0, vmin);
        val1 = vmaxq_f32(val1, vmin);
        val0 = vminq_f32(val0, vmax);
        val1 = vminq_f32(val1, vmax);

        vst1q_f32(out, val0);
        vst1q_f32(out + 4, val1);
        in += 8;
        out += 8;
    }

    number = eighth_points * 8;
    for (; number < num_points; number++) {
        float val = *in++;
        if (val < min)
            val = min;
        else if (val > max)
            val = max;
        *out++ = val;
    }
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32f_s32f_x2_clamp_32f_rvv(float* out,
                                                  const float* in,
                                                  const float min,
                                                  const float max,
                                                  unsigned int num_points)
{
    vfloat32m8_t vmin = __riscv_vfmv_v_f_f32m8(min, __riscv_vsetvlmax_e32m8());
    vfloat32m8_t vmax = __riscv_vfmv_v_f_f32m8(max, __riscv_vsetvlmax_e32m8());
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, in += vl, out += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t v = __riscv_vle32_v_f32m8(in, vl);
        v = __riscv_vfmin(__riscv_vfmax(v, vmin, vl), vmax, vl);
        __riscv_vse32(out, v, vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32fc_s32f_x2_clamp_32f_u_H */
