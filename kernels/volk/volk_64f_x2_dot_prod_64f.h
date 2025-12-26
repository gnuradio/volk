/* -*- c++ -*- */
/*
 * Copyright 2025 Magnus Lundmark <magnuslundmark@gmail.com>
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_64f_x2_dot_prod_64f
 *
 * \b Overview
 *
 * Computes the dot product (inner product) of two double-precision vectors.
 * Returns the sum of element-wise products: result = sum(input[i] * taps[i]).
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_64f_x2_dot_prod_64f(double* result, const double* input, const double* taps,
 * unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li input: First input vector.
 * \li taps: Second input vector (filter coefficients).
 * \li num_points: Vector length.
 *
 * \b Outputs
 * \li result: Pointer to store the scalar dot product result.
 *
 * \b Example
 * \code
 *   unsigned int N = 10;
 *   unsigned int align = volk_get_alignment();
 *   double* a = (double*)volk_malloc(N * sizeof(double), align);
 *   double* b = (double*)volk_malloc(N * sizeof(double), align);
 *   double result;
 *
 *   // Compute dot product of [0,1,2,...,9] with [1,1,1,...,1]
 *   for (unsigned int i = 0; i < N; i++) {
 *       a[i] = (double)i;
 *       b[i] = 1.0;
 *   }
 *
 *   volk_64f_x2_dot_prod_64f(&result, a, b, N);
 *   // result == 45.0 (sum of 0 through 9)
 *
 *   volk_free(a);
 *   volk_free(b);
 * \endcode
 */

#ifndef INCLUDED_volk_64f_x2_dot_prod_64f_u_H
#define INCLUDED_volk_64f_x2_dot_prod_64f_u_H

#include <volk/volk_common.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_64f_x2_dot_prod_64f_generic(double* result,
                                                    const double* input,
                                                    const double* taps,
                                                    unsigned int num_points)
{
    double dot = 0.0;
    for (unsigned int i = 0; i < num_points; i++) {
        dot += input[i] * taps[i];
    }
    *result = dot;
}

#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_64f_x2_dot_prod_64f_u_sse2(double* result,
                                                   const double* input,
                                                   const double* taps,
                                                   unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;
    unsigned int number = 0;

    __m128d acc0 = _mm_setzero_pd();
    __m128d acc1 = _mm_setzero_pd();
    __m128d acc2 = _mm_setzero_pd();
    __m128d acc3 = _mm_setzero_pd();

    for (; number < eighthPoints; number++) {
        acc0 = _mm_add_pd(acc0, _mm_mul_pd(_mm_loadu_pd(input), _mm_loadu_pd(taps)));
        acc1 =
            _mm_add_pd(acc1, _mm_mul_pd(_mm_loadu_pd(input + 2), _mm_loadu_pd(taps + 2)));
        acc2 =
            _mm_add_pd(acc2, _mm_mul_pd(_mm_loadu_pd(input + 4), _mm_loadu_pd(taps + 4)));
        acc3 =
            _mm_add_pd(acc3, _mm_mul_pd(_mm_loadu_pd(input + 6), _mm_loadu_pd(taps + 6)));
        input += 8;
        taps += 8;
    }

    acc0 = _mm_add_pd(acc0, acc1);
    acc2 = _mm_add_pd(acc2, acc3);
    acc0 = _mm_add_pd(acc0, acc2);

    __VOLK_ATTR_ALIGNED(16) double tmp[2];
    _mm_store_pd(tmp, acc0);
    double dot = tmp[0] + tmp[1];

    for (number = eighthPoints * 8; number < num_points; number++) {
        dot += (*input++) * (*taps++);
    }
    *result = dot;
}

#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_64f_x2_dot_prod_64f_u_avx(double* result,
                                                  const double* input,
                                                  const double* taps,
                                                  unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;
    unsigned int number = 0;

    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();

    for (; number < eighthPoints; number++) {
        acc0 = _mm256_add_pd(
            acc0, _mm256_mul_pd(_mm256_loadu_pd(input), _mm256_loadu_pd(taps)));
        acc1 = _mm256_add_pd(
            acc1, _mm256_mul_pd(_mm256_loadu_pd(input + 4), _mm256_loadu_pd(taps + 4)));
        input += 8;
        taps += 8;
    }

    acc0 = _mm256_add_pd(acc0, acc1);

    __VOLK_ATTR_ALIGNED(32) double tmp[4];
    _mm256_storeu_pd(tmp, acc0);
    double dot = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    for (number = eighthPoints * 8; number < num_points; number++) {
        dot += (*input++) * (*taps++);
    }
    *result = dot;
}

#endif /* LV_HAVE_AVX */


#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_64f_x2_dot_prod_64f_u_avx2_fma(double* result,
                                                       const double* input,
                                                       const double* taps,
                                                       unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;
    unsigned int number = 0;

    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();

    for (; number < eighthPoints; number++) {
        acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(input), _mm256_loadu_pd(taps), acc0);
        acc1 =
            _mm256_fmadd_pd(_mm256_loadu_pd(input + 4), _mm256_loadu_pd(taps + 4), acc1);
        input += 8;
        taps += 8;
    }

    acc0 = _mm256_add_pd(acc0, acc1);

    __VOLK_ATTR_ALIGNED(32) double tmp[4];
    _mm256_storeu_pd(tmp, acc0);
    double dot = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    for (number = eighthPoints * 8; number < num_points; number++) {
        dot += (*input++) * (*taps++);
    }
    *result = dot;
}

#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */


#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_64f_x2_dot_prod_64f_u_avx512f(double* result,
                                                      const double* input,
                                                      const double* taps,
                                                      unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;
    unsigned int number = 0;

    __m512d acc = _mm512_setzero_pd();

    for (; number < eighthPoints; number++) {
        acc = _mm512_fmadd_pd(_mm512_loadu_pd(input), _mm512_loadu_pd(taps), acc);
        input += 8;
        taps += 8;
    }

    double dot = _mm512_reduce_add_pd(acc);

    for (number = eighthPoints * 8; number < num_points; number++) {
        dot += (*input++) * (*taps++);
    }
    *result = dot;
}

#endif /* LV_HAVE_AVX512F */


#endif /* INCLUDED_volk_64f_x2_dot_prod_64f_u_H */


#ifndef INCLUDED_volk_64f_x2_dot_prod_64f_a_H
#define INCLUDED_volk_64f_x2_dot_prod_64f_a_H

#include <volk/volk_common.h>

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_64f_x2_dot_prod_64f_a_sse2(double* result,
                                                   const double* input,
                                                   const double* taps,
                                                   unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;
    unsigned int number = 0;

    __m128d acc0 = _mm_setzero_pd();
    __m128d acc1 = _mm_setzero_pd();
    __m128d acc2 = _mm_setzero_pd();
    __m128d acc3 = _mm_setzero_pd();

    for (; number < eighthPoints; number++) {
        acc0 = _mm_add_pd(acc0, _mm_mul_pd(_mm_load_pd(input), _mm_load_pd(taps)));
        acc1 =
            _mm_add_pd(acc1, _mm_mul_pd(_mm_load_pd(input + 2), _mm_load_pd(taps + 2)));
        acc2 =
            _mm_add_pd(acc2, _mm_mul_pd(_mm_load_pd(input + 4), _mm_load_pd(taps + 4)));
        acc3 =
            _mm_add_pd(acc3, _mm_mul_pd(_mm_load_pd(input + 6), _mm_load_pd(taps + 6)));
        input += 8;
        taps += 8;
    }

    acc0 = _mm_add_pd(acc0, acc1);
    acc2 = _mm_add_pd(acc2, acc3);
    acc0 = _mm_add_pd(acc0, acc2);

    __VOLK_ATTR_ALIGNED(16) double tmp[2];
    _mm_store_pd(tmp, acc0);
    double dot = tmp[0] + tmp[1];

    for (number = eighthPoints * 8; number < num_points; number++) {
        dot += (*input++) * (*taps++);
    }
    *result = dot;
}

#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_64f_x2_dot_prod_64f_a_avx(double* result,
                                                  const double* input,
                                                  const double* taps,
                                                  unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;
    unsigned int number = 0;

    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();

    for (; number < eighthPoints; number++) {
        acc0 = _mm256_add_pd(acc0,
                             _mm256_mul_pd(_mm256_load_pd(input), _mm256_load_pd(taps)));
        acc1 = _mm256_add_pd(
            acc1, _mm256_mul_pd(_mm256_load_pd(input + 4), _mm256_load_pd(taps + 4)));
        input += 8;
        taps += 8;
    }

    acc0 = _mm256_add_pd(acc0, acc1);

    __VOLK_ATTR_ALIGNED(32) double tmp[4];
    _mm256_store_pd(tmp, acc0);
    double dot = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    for (number = eighthPoints * 8; number < num_points; number++) {
        dot += (*input++) * (*taps++);
    }
    *result = dot;
}

#endif /* LV_HAVE_AVX */


#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>

static inline void volk_64f_x2_dot_prod_64f_a_avx2_fma(double* result,
                                                       const double* input,
                                                       const double* taps,
                                                       unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;
    unsigned int number = 0;

    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();

    for (; number < eighthPoints; number++) {
        acc0 = _mm256_fmadd_pd(_mm256_load_pd(input), _mm256_load_pd(taps), acc0);
        acc1 = _mm256_fmadd_pd(_mm256_load_pd(input + 4), _mm256_load_pd(taps + 4), acc1);
        input += 8;
        taps += 8;
    }

    acc0 = _mm256_add_pd(acc0, acc1);

    __VOLK_ATTR_ALIGNED(32) double tmp[4];
    _mm256_store_pd(tmp, acc0);
    double dot = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    for (number = eighthPoints * 8; number < num_points; number++) {
        dot += (*input++) * (*taps++);
    }
    *result = dot;
}

#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */


#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_64f_x2_dot_prod_64f_a_avx512f(double* result,
                                                      const double* input,
                                                      const double* taps,
                                                      unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;
    unsigned int number = 0;

    __m512d acc = _mm512_setzero_pd();

    for (; number < eighthPoints; number++) {
        acc = _mm512_fmadd_pd(_mm512_load_pd(input), _mm512_load_pd(taps), acc);
        input += 8;
        taps += 8;
    }

    double dot = _mm512_reduce_add_pd(acc);

    for (number = eighthPoints * 8; number < num_points; number++) {
        dot += (*input++) * (*taps++);
    }
    *result = dot;
}

#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_64f_x2_dot_prod_64f_neon(double* result,
                                                 const double* input,
                                                 const double* taps,
                                                 unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;
    unsigned int number = 0;

    float64x2_t acc0 = vdupq_n_f64(0.0);
    float64x2_t acc1 = vdupq_n_f64(0.0);
    float64x2_t acc2 = vdupq_n_f64(0.0);
    float64x2_t acc3 = vdupq_n_f64(0.0);

    for (; number < eighthPoints; number++) {
        acc0 = vmlaq_f64(acc0, vld1q_f64(input), vld1q_f64(taps));
        acc1 = vmlaq_f64(acc1, vld1q_f64(input + 2), vld1q_f64(taps + 2));
        acc2 = vmlaq_f64(acc2, vld1q_f64(input + 4), vld1q_f64(taps + 4));
        acc3 = vmlaq_f64(acc3, vld1q_f64(input + 6), vld1q_f64(taps + 6));
        input += 8;
        taps += 8;
    }

    acc0 = vaddq_f64(acc0, acc1);
    acc2 = vaddq_f64(acc2, acc3);
    acc0 = vaddq_f64(acc0, acc2);

    double dot = vgetq_lane_f64(acc0, 0) + vgetq_lane_f64(acc0, 1);

    for (number = eighthPoints * 8; number < num_points; number++) {
        dot += (*input++) * (*taps++);
    }
    *result = dot;
}

#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_64f_x2_dot_prod_64f_neonv8(double* result,
                                                   const double* input,
                                                   const double* taps,
                                                   unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;
    unsigned int number = 0;

    float64x2_t acc0 = vdupq_n_f64(0.0);
    float64x2_t acc1 = vdupq_n_f64(0.0);
    float64x2_t acc2 = vdupq_n_f64(0.0);
    float64x2_t acc3 = vdupq_n_f64(0.0);

    for (; number < eighthPoints; number++) {
        __VOLK_PREFETCH(input + 16);
        __VOLK_PREFETCH(taps + 16);

        acc0 = vfmaq_f64(acc0, vld1q_f64(input), vld1q_f64(taps));
        acc1 = vfmaq_f64(acc1, vld1q_f64(input + 2), vld1q_f64(taps + 2));
        acc2 = vfmaq_f64(acc2, vld1q_f64(input + 4), vld1q_f64(taps + 4));
        acc3 = vfmaq_f64(acc3, vld1q_f64(input + 6), vld1q_f64(taps + 6));
        input += 8;
        taps += 8;
    }

    acc0 = vaddq_f64(acc0, acc1);
    acc2 = vaddq_f64(acc2, acc3);
    acc0 = vaddq_f64(acc0, acc2);

    double dot = vaddvq_f64(acc0);

    for (number = eighthPoints * 8; number < num_points; number++) {
        dot += (*input++) * (*taps++);
    }
    *result = dot;
}

#endif /* LV_HAVE_NEONV8 */


#ifdef LV_HAVE_RVV
#include <riscv_vector.h>
#include <volk/volk_rvv_intrinsics.h>

static inline void volk_64f_x2_dot_prod_64f_rvv(double* result,
                                                const double* input,
                                                const double* taps,
                                                unsigned int num_points)
{
    vfloat64m8_t vsum = __riscv_vfmv_v_f_f64m8(0, __riscv_vsetvlmax_e64m8());
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, input += vl, taps += vl) {
        vl = __riscv_vsetvl_e64m8(n);
        vfloat64m8_t v0 = __riscv_vle64_v_f64m8(input, vl);
        vfloat64m8_t v1 = __riscv_vle64_v_f64m8(taps, vl);
        vsum = __riscv_vfmacc_tu(vsum, v0, v1, vl);
    }
    size_t vl = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t v = RISCV_SHRINK8(vfadd, f, 64, vsum);
    v = __riscv_vfredusum(v, __riscv_vfmv_s_f_f64m1(0, vl), vl);
    *result = __riscv_vfmv_f(v);
}

#endif /* LV_HAVE_RVV */


#endif /* INCLUDED_volk_64f_x2_dot_prod_64f_a_H */
