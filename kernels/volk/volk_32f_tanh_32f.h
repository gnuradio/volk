/* -*- c++ -*- */
/*
 * Copyright 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_tanh_32f
 *
 * \b Overview
 *
 * Computes the hyperbolic tangent of each element of the aVector:
 *
 * c[i] = tanh(a[i])
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_tanh_32f(float* cVector, const float* aVector, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li aVector: The buffer of points.
 * \li num_points: The number of values in input buffer.
 *
 * \b Outputs
 * \li cVector: The output buffer.
 *
 * \b Example
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   float* in = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* out = (float*)volk_malloc(sizeof(float)*N, alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       // the approximate artanh(x) for x<1
 *       float x = (float)ii / (float)N;
 *       in[ii] = 0.5 * std::log((1.f+x)/(1.f-x));
 *   }
 *
 *   volk_32f_tanh_32f(out, in, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("out(%i) = %f\n", ii, out[ii]);
 *   }
 *
 *   volk_free(in);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_tanh_32f_a_H
#define INCLUDED_volk_32f_tanh_32f_a_H

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <string.h>


#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_tanh_32f_generic(float* cVector, const float* aVector, unsigned int num_points)
{
    unsigned int number = 0;
    float* cPtr = cVector;
    const float* aPtr = aVector;
    for (; number < num_points; number++) {
        *cPtr++ = tanhf(*aPtr++);
    }
}

#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_tanh_32f_series(float* cVector, const float* aVector, unsigned int num_points)
{
    float* cPtr = cVector;
    const float* aPtr = aVector;
    for (unsigned int number = 0; number < num_points; number++) {
        if (*aPtr > 4.97)
            *cPtr++ = 1;
        else if (*aPtr <= -4.97)
            *cPtr++ = -1;
        else {
            float x2 = (*aPtr) * (*aPtr);
            float a = (*aPtr) * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
            float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
            *cPtr++ = a / b;
            aPtr++;
        }
    }
}

#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void
volk_32f_tanh_32f_a_sse(float* cVector, const float* aVector, unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    float* cPtr = cVector;
    const float* aPtr = aVector;

    __m128 aVal, cVal, x2, a, b;
    __m128 const1, const2, const3, const4, const5, const6;
    const1 = _mm_set_ps1(135135.0f);
    const2 = _mm_set_ps1(17325.0f);
    const3 = _mm_set_ps1(378.0f);
    const4 = _mm_set_ps1(62370.0f);
    const5 = _mm_set_ps1(3150.0f);
    const6 = _mm_set_ps1(28.0f);
    for (; number < quarterPoints; number++) {

        aVal = _mm_load_ps(aPtr);
        x2 = _mm_mul_ps(aVal, aVal);
        a = _mm_mul_ps(
            aVal,
            _mm_add_ps(
                const1,
                _mm_mul_ps(x2,
                           _mm_add_ps(const2, _mm_mul_ps(x2, _mm_add_ps(const3, x2))))));
        b = _mm_add_ps(
            const1,
            _mm_mul_ps(
                x2,
                _mm_add_ps(const4,
                           _mm_mul_ps(x2, _mm_add_ps(const5, _mm_mul_ps(x2, const6))))));

        cVal = _mm_div_ps(a, b);

        _mm_store_ps(cPtr, cVal); // Store the results back into the C container

        aPtr += 4;
        cPtr += 4;
    }

    number = quarterPoints * 4;
    volk_32f_tanh_32f_series(cPtr, aPtr, num_points - number);
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32f_tanh_32f_a_avx(float* cVector, const float* aVector, unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    float* cPtr = cVector;
    const float* aPtr = aVector;

    __m256 aVal, cVal, x2, a, b;
    __m256 const1, const2, const3, const4, const5, const6;
    const1 = _mm256_set1_ps(135135.0f);
    const2 = _mm256_set1_ps(17325.0f);
    const3 = _mm256_set1_ps(378.0f);
    const4 = _mm256_set1_ps(62370.0f);
    const5 = _mm256_set1_ps(3150.0f);
    const6 = _mm256_set1_ps(28.0f);
    for (; number < eighthPoints; number++) {

        aVal = _mm256_load_ps(aPtr);
        x2 = _mm256_mul_ps(aVal, aVal);
        a = _mm256_mul_ps(
            aVal,
            _mm256_add_ps(
                const1,
                _mm256_mul_ps(
                    x2,
                    _mm256_add_ps(const2,
                                  _mm256_mul_ps(x2, _mm256_add_ps(const3, x2))))));
        b = _mm256_add_ps(
            const1,
            _mm256_mul_ps(
                x2,
                _mm256_add_ps(
                    const4,
                    _mm256_mul_ps(x2,
                                  _mm256_add_ps(const5, _mm256_mul_ps(x2, const6))))));

        cVal = _mm256_div_ps(a, b);

        _mm256_store_ps(cPtr, cVal); // Store the results back into the C container

        aPtr += 8;
        cPtr += 8;
    }

    number = eighthPoints * 8;
    volk_32f_tanh_32f_series(cPtr, aPtr, num_points - number);
}
#endif /* LV_HAVE_AVX */

#if LV_HAVE_AVX && LV_HAVE_FMA
#include <immintrin.h>

static inline void
volk_32f_tanh_32f_a_avx_fma(float* cVector, const float* aVector, unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    float* cPtr = cVector;
    const float* aPtr = aVector;

    __m256 aVal, cVal, x2, a, b;
    __m256 const1, const2, const3, const4, const5, const6;
    const1 = _mm256_set1_ps(135135.0f);
    const2 = _mm256_set1_ps(17325.0f);
    const3 = _mm256_set1_ps(378.0f);
    const4 = _mm256_set1_ps(62370.0f);
    const5 = _mm256_set1_ps(3150.0f);
    const6 = _mm256_set1_ps(28.0f);
    for (; number < eighthPoints; number++) {

        aVal = _mm256_load_ps(aPtr);
        x2 = _mm256_mul_ps(aVal, aVal);
        a = _mm256_mul_ps(
            aVal,
            _mm256_fmadd_ps(
                x2, _mm256_fmadd_ps(x2, _mm256_add_ps(const3, x2), const2), const1));
        b = _mm256_fmadd_ps(
            x2, _mm256_fmadd_ps(x2, _mm256_fmadd_ps(x2, const6, const5), const4), const1);

        cVal = _mm256_div_ps(a, b);

        _mm256_store_ps(cPtr, cVal); // Store the results back into the C container

        aPtr += 8;
        cPtr += 8;
    }

    number = eighthPoints * 8;
    volk_32f_tanh_32f_series(cPtr, aPtr, num_points - number);
}
#endif /* LV_HAVE_AVX && LV_HAVE_FMA */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void
volk_32f_tanh_32f_neon(float* cVector, const float* aVector, unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    float* cPtr = cVector;
    const float* aPtr = aVector;

    // Polynomial coefficients for tanh approximation
    const float32x4_t const1 = vdupq_n_f32(135135.0f);
    const float32x4_t const2 = vdupq_n_f32(17325.0f);
    const float32x4_t const3 = vdupq_n_f32(378.0f);
    const float32x4_t const4 = vdupq_n_f32(62370.0f);
    const float32x4_t const5 = vdupq_n_f32(3150.0f);
    const float32x4_t const6 = vdupq_n_f32(28.0f);

    for (; number < quarterPoints; number++) {
        float32x4_t aVal = vld1q_f32(aPtr);
        float32x4_t x2 = vmulq_f32(aVal, aVal);

        // a = x * (135135 + x2 * (17325 + x2 * (378 + x2)))
        float32x4_t inner_a = vaddq_f32(const3, x2);
        inner_a = vmlaq_f32(const2, x2, inner_a);
        inner_a = vmlaq_f32(const1, x2, inner_a);
        float32x4_t a = vmulq_f32(aVal, inner_a);

        // b = 135135 + x2 * (62370 + x2 * (3150 + x2 * 28))
        float32x4_t inner_b = vmlaq_f32(const5, x2, const6);
        inner_b = vmlaq_f32(const4, x2, inner_b);
        float32x4_t b = vmlaq_f32(const1, x2, inner_b);

        // c = a / b (use reciprocal approximation)
        float32x4_t b_recip = vrecpeq_f32(b);
        b_recip = vmulq_f32(b_recip, vrecpsq_f32(b, b_recip));
        b_recip = vmulq_f32(b_recip, vrecpsq_f32(b, b_recip));
        float32x4_t cVal = vmulq_f32(a, b_recip);

        vst1q_f32(cPtr, cVal);
        aPtr += 4;
        cPtr += 4;
    }

    number = quarterPoints * 4;
    volk_32f_tanh_32f_series(cPtr, aPtr, num_points - number);
}

#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void
volk_32f_tanh_32f_neonv8(float* cVector, const float* aVector, unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    float* cPtr = cVector;
    const float* aPtr = aVector;

    // Polynomial coefficients for tanh approximation
    const float32x4_t const1 = vdupq_n_f32(135135.0f);
    const float32x4_t const2 = vdupq_n_f32(17325.0f);
    const float32x4_t const3 = vdupq_n_f32(378.0f);
    const float32x4_t const4 = vdupq_n_f32(62370.0f);
    const float32x4_t const5 = vdupq_n_f32(3150.0f);
    const float32x4_t const6 = vdupq_n_f32(28.0f);

    for (; number < eighthPoints; number++) {
        __VOLK_PREFETCH(aPtr + 16);

        float32x4_t aVal0 = vld1q_f32(aPtr);
        float32x4_t aVal1 = vld1q_f32(aPtr + 4);
        float32x4_t x2_0 = vmulq_f32(aVal0, aVal0);
        float32x4_t x2_1 = vmulq_f32(aVal1, aVal1);

        // a = x * (135135 + x2 * (17325 + x2 * (378 + x2))) using FMA
        float32x4_t inner_a0 = vaddq_f32(const3, x2_0);
        float32x4_t inner_a1 = vaddq_f32(const3, x2_1);
        inner_a0 = vfmaq_f32(const2, x2_0, inner_a0);
        inner_a1 = vfmaq_f32(const2, x2_1, inner_a1);
        inner_a0 = vfmaq_f32(const1, x2_0, inner_a0);
        inner_a1 = vfmaq_f32(const1, x2_1, inner_a1);
        float32x4_t a0 = vmulq_f32(aVal0, inner_a0);
        float32x4_t a1 = vmulq_f32(aVal1, inner_a1);

        // b = 135135 + x2 * (62370 + x2 * (3150 + x2 * 28)) using FMA
        float32x4_t inner_b0 = vfmaq_f32(const5, x2_0, const6);
        float32x4_t inner_b1 = vfmaq_f32(const5, x2_1, const6);
        inner_b0 = vfmaq_f32(const4, x2_0, inner_b0);
        inner_b1 = vfmaq_f32(const4, x2_1, inner_b1);
        float32x4_t b0 = vfmaq_f32(const1, x2_0, inner_b0);
        float32x4_t b1 = vfmaq_f32(const1, x2_1, inner_b1);

        // c = a / b using native division
        float32x4_t cVal0 = vdivq_f32(a0, b0);
        float32x4_t cVal1 = vdivq_f32(a1, b1);

        vst1q_f32(cPtr, cVal0);
        vst1q_f32(cPtr + 4, cVal1);
        aPtr += 8;
        cPtr += 8;
    }

    number = eighthPoints * 8;
    volk_32f_tanh_32f_series(cPtr, aPtr, num_points - number);
}

#endif /* LV_HAVE_NEONV8 */

#endif /* INCLUDED_volk_32f_tanh_32f_a_H */


#ifndef INCLUDED_volk_32f_tanh_32f_u_H
#define INCLUDED_volk_32f_tanh_32f_u_H

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <string.h>


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void
volk_32f_tanh_32f_u_sse(float* cVector, const float* aVector, unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    float* cPtr = cVector;
    const float* aPtr = aVector;

    __m128 aVal, cVal, x2, a, b;
    __m128 const1, const2, const3, const4, const5, const6;
    const1 = _mm_set_ps1(135135.0f);
    const2 = _mm_set_ps1(17325.0f);
    const3 = _mm_set_ps1(378.0f);
    const4 = _mm_set_ps1(62370.0f);
    const5 = _mm_set_ps1(3150.0f);
    const6 = _mm_set_ps1(28.0f);
    for (; number < quarterPoints; number++) {

        aVal = _mm_loadu_ps(aPtr);
        x2 = _mm_mul_ps(aVal, aVal);
        a = _mm_mul_ps(
            aVal,
            _mm_add_ps(
                const1,
                _mm_mul_ps(x2,
                           _mm_add_ps(const2, _mm_mul_ps(x2, _mm_add_ps(const3, x2))))));
        b = _mm_add_ps(
            const1,
            _mm_mul_ps(
                x2,
                _mm_add_ps(const4,
                           _mm_mul_ps(x2, _mm_add_ps(const5, _mm_mul_ps(x2, const6))))));

        cVal = _mm_div_ps(a, b);

        _mm_storeu_ps(cPtr, cVal); // Store the results back into the C container

        aPtr += 4;
        cPtr += 4;
    }

    number = quarterPoints * 4;
    volk_32f_tanh_32f_series(cPtr, aPtr, num_points - number);
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32f_tanh_32f_u_avx(float* cVector, const float* aVector, unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    float* cPtr = cVector;
    const float* aPtr = aVector;

    __m256 aVal, cVal, x2, a, b;
    __m256 const1, const2, const3, const4, const5, const6;
    const1 = _mm256_set1_ps(135135.0f);
    const2 = _mm256_set1_ps(17325.0f);
    const3 = _mm256_set1_ps(378.0f);
    const4 = _mm256_set1_ps(62370.0f);
    const5 = _mm256_set1_ps(3150.0f);
    const6 = _mm256_set1_ps(28.0f);
    for (; number < eighthPoints; number++) {

        aVal = _mm256_loadu_ps(aPtr);
        x2 = _mm256_mul_ps(aVal, aVal);
        a = _mm256_mul_ps(
            aVal,
            _mm256_add_ps(
                const1,
                _mm256_mul_ps(
                    x2,
                    _mm256_add_ps(const2,
                                  _mm256_mul_ps(x2, _mm256_add_ps(const3, x2))))));
        b = _mm256_add_ps(
            const1,
            _mm256_mul_ps(
                x2,
                _mm256_add_ps(
                    const4,
                    _mm256_mul_ps(x2,
                                  _mm256_add_ps(const5, _mm256_mul_ps(x2, const6))))));

        cVal = _mm256_div_ps(a, b);

        _mm256_storeu_ps(cPtr, cVal); // Store the results back into the C container

        aPtr += 8;
        cPtr += 8;
    }

    number = eighthPoints * 8;
    volk_32f_tanh_32f_series(cPtr, aPtr, num_points - number);
}
#endif /* LV_HAVE_AVX */

#if LV_HAVE_AVX && LV_HAVE_FMA
#include <immintrin.h>

static inline void
volk_32f_tanh_32f_u_avx_fma(float* cVector, const float* aVector, unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    float* cPtr = cVector;
    const float* aPtr = aVector;

    __m256 aVal, cVal, x2, a, b;
    __m256 const1, const2, const3, const4, const5, const6;
    const1 = _mm256_set1_ps(135135.0f);
    const2 = _mm256_set1_ps(17325.0f);
    const3 = _mm256_set1_ps(378.0f);
    const4 = _mm256_set1_ps(62370.0f);
    const5 = _mm256_set1_ps(3150.0f);
    const6 = _mm256_set1_ps(28.0f);
    for (; number < eighthPoints; number++) {

        aVal = _mm256_loadu_ps(aPtr);
        x2 = _mm256_mul_ps(aVal, aVal);
        a = _mm256_mul_ps(
            aVal,
            _mm256_fmadd_ps(
                x2, _mm256_fmadd_ps(x2, _mm256_add_ps(const3, x2), const2), const1));
        b = _mm256_fmadd_ps(
            x2, _mm256_fmadd_ps(x2, _mm256_fmadd_ps(x2, const6, const5), const4), const1);

        cVal = _mm256_div_ps(a, b);

        _mm256_storeu_ps(cPtr, cVal); // Store the results back into the C container

        aPtr += 8;
        cPtr += 8;
    }

    number = eighthPoints * 8;
    volk_32f_tanh_32f_series(cPtr, aPtr, num_points - number);
}
#endif /* LV_HAVE_AVX && LV_HAVE_FMA */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void
volk_32f_tanh_32f_rvv(float* bVector, const float* aVector, unsigned int num_points)
{
    size_t vlmax = __riscv_vsetvlmax_e32m2();

    const vfloat32m2_t c1 = __riscv_vfmv_v_f_f32m2(135135.0f, vlmax);
    const vfloat32m2_t c2 = __riscv_vfmv_v_f_f32m2(17325.0f, vlmax);
    const vfloat32m2_t c3 = __riscv_vfmv_v_f_f32m2(378.0f, vlmax);
    const vfloat32m2_t c4 = __riscv_vfmv_v_f_f32m2(62370.0f, vlmax);
    const vfloat32m2_t c5 = __riscv_vfmv_v_f_f32m2(3150.0f, vlmax);
    const vfloat32m2_t c6 = __riscv_vfmv_v_f_f32m2(28.0f, vlmax);

    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, aVector += vl, bVector += vl) {
        vl = __riscv_vsetvl_e32m2(n);
        vfloat32m2_t x = __riscv_vle32_v_f32m2(aVector, vl);
        vfloat32m2_t xx = __riscv_vfmul(x, x, vl);
        vfloat32m2_t a, b;
        a = __riscv_vfadd(xx, c3, vl);
        a = __riscv_vfmadd(a, xx, c2, vl);
        a = __riscv_vfmadd(a, xx, c1, vl);
        a = __riscv_vfmul(a, x, vl);
        b = c6;
        b = __riscv_vfmadd(b, xx, c5, vl);
        b = __riscv_vfmadd(b, xx, c4, vl);
        b = __riscv_vfmadd(b, xx, c1, vl);
        __riscv_vse32(bVector, __riscv_vfdiv(a, b, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32f_tanh_32f_u_H */
