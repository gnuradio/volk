/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_magnitude_squared_32f
 *
 * \b Overview
 *
 * Calculates the magnitude squared of the complexVector and stores
 * the results in the magnitudeVector.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_magnitude_squared_32f(float* magnitudeVector, const lv_32fc_t*
 * complexVector, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li complexVector: The complex input vector.
 * \li num_points: The number of samples.
 *
 * \b Outputs
 * \li magnitudeVector: The output value.
 *
 * \b Example
 * Calculate the magnitude squared of \f$x^2 + x\f$ for points around the unit circle.
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   lv_32fc_t* in  = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *   float* magnitude = (float*)volk_malloc(sizeof(float)*N, alignment);
 *
 *   for(unsigned int ii = 0; ii < N/2; ++ii){
 *       float real = 2.f * ((float)ii / (float)N) - 1.f;
 *       float imag = std::sqrt(1.f - real * real);
 *       in[ii] = lv_cmake(real, imag);
 *       in[ii] = in[ii] * in[ii] + in[ii];
 *       in[N-ii] = lv_cmake(real, imag);
 *       in[N-ii] = in[N-ii] * in[N-ii] + in[N-ii];
 *   }
 *
 *   volk_32fc_magnitude_32f(magnitude, in, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("out(%i) = %+.1f\n", ii, magnitude[ii]);
 *   }
 *
 *   volk_free(in);
 *   volk_free(magnitude);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_magnitude_squared_32f_u_H
#define INCLUDED_volk_32fc_magnitude_squared_32f_u_H

#include <inttypes.h>
#include <math.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX
#include <immintrin.h>
#include <volk/volk_avx_intrinsics.h>

static inline void volk_32fc_magnitude_squared_32f_u_avx(float* magnitudeVector,
                                                         const lv_32fc_t* complexVector,
                                                         unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float* complexVectorPtr = (float*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    __m256 cplxValue1, cplxValue2, result;

    for (; number < eighthPoints; number++) {
        cplxValue1 = _mm256_loadu_ps(complexVectorPtr);
        cplxValue2 = _mm256_loadu_ps(complexVectorPtr + 8);
        result = _mm256_magnitudesquared_ps(cplxValue1, cplxValue2);
        _mm256_storeu_ps(magnitudeVectorPtr, result);

        complexVectorPtr += 16;
        magnitudeVectorPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        float val1Real = *complexVectorPtr++;
        float val1Imag = *complexVectorPtr++;
        *magnitudeVectorPtr++ = (val1Real * val1Real) + (val1Imag * val1Imag);
    }
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>
#include <volk/volk_sse3_intrinsics.h>

static inline void volk_32fc_magnitude_squared_32f_u_sse3(float* magnitudeVector,
                                                          const lv_32fc_t* complexVector,
                                                          unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const float* complexVectorPtr = (float*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    __m128 cplxValue1, cplxValue2, result;
    for (; number < quarterPoints; number++) {
        cplxValue1 = _mm_loadu_ps(complexVectorPtr);
        complexVectorPtr += 4;

        cplxValue2 = _mm_loadu_ps(complexVectorPtr);
        complexVectorPtr += 4;

        result = _mm_magnitudesquared_ps_sse3(cplxValue1, cplxValue2);
        _mm_storeu_ps(magnitudeVectorPtr, result);
        magnitudeVectorPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        float val1Real = *complexVectorPtr++;
        float val1Imag = *complexVectorPtr++;
        *magnitudeVectorPtr++ = (val1Real * val1Real) + (val1Imag * val1Imag);
    }
}
#endif /* LV_HAVE_SSE3 */


#ifdef LV_HAVE_SSE
#include <volk/volk_sse_intrinsics.h>
#include <xmmintrin.h>

static inline void volk_32fc_magnitude_squared_32f_u_sse(float* magnitudeVector,
                                                         const lv_32fc_t* complexVector,
                                                         unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const float* complexVectorPtr = (float*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    __m128 cplxValue1, cplxValue2, result;

    for (; number < quarterPoints; number++) {
        cplxValue1 = _mm_loadu_ps(complexVectorPtr);
        complexVectorPtr += 4;

        cplxValue2 = _mm_loadu_ps(complexVectorPtr);
        complexVectorPtr += 4;

        result = _mm_magnitudesquared_ps(cplxValue1, cplxValue2);
        _mm_storeu_ps(magnitudeVectorPtr, result);
        magnitudeVectorPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        float val1Real = *complexVectorPtr++;
        float val1Imag = *complexVectorPtr++;
        *magnitudeVectorPtr++ = (val1Real * val1Real) + (val1Imag * val1Imag);
    }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_GENERIC

static inline void volk_32fc_magnitude_squared_32f_generic(float* magnitudeVector,
                                                           const lv_32fc_t* complexVector,
                                                           unsigned int num_points)
{
    const float* complexVectorPtr = (float*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;
    unsigned int number = 0;
    for (number = 0; number < num_points; number++) {
        const float real = *complexVectorPtr++;
        const float imag = *complexVectorPtr++;
        *magnitudeVectorPtr++ = (real * real) + (imag * imag);
    }
}
#endif /* LV_HAVE_GENERIC */


#endif /* INCLUDED_volk_32fc_magnitude_32f_u_H */
#ifndef INCLUDED_volk_32fc_magnitude_squared_32f_a_H
#define INCLUDED_volk_32fc_magnitude_squared_32f_a_H

#include <inttypes.h>
#include <math.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX
#include <immintrin.h>
#include <volk/volk_avx_intrinsics.h>

static inline void volk_32fc_magnitude_squared_32f_a_avx(float* magnitudeVector,
                                                         const lv_32fc_t* complexVector,
                                                         unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float* complexVectorPtr = (float*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    __m256 cplxValue1, cplxValue2, result;
    for (; number < eighthPoints; number++) {
        cplxValue1 = _mm256_load_ps(complexVectorPtr);
        complexVectorPtr += 8;

        cplxValue2 = _mm256_load_ps(complexVectorPtr);
        complexVectorPtr += 8;

        result = _mm256_magnitudesquared_ps(cplxValue1, cplxValue2);
        _mm256_store_ps(magnitudeVectorPtr, result);
        magnitudeVectorPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        float val1Real = *complexVectorPtr++;
        float val1Imag = *complexVectorPtr++;
        *magnitudeVectorPtr++ = (val1Real * val1Real) + (val1Imag * val1Imag);
    }
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>
#include <volk/volk_sse3_intrinsics.h>

static inline void volk_32fc_magnitude_squared_32f_a_sse3(float* magnitudeVector,
                                                          const lv_32fc_t* complexVector,
                                                          unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const float* complexVectorPtr = (float*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    __m128 cplxValue1, cplxValue2, result;
    for (; number < quarterPoints; number++) {
        cplxValue1 = _mm_load_ps(complexVectorPtr);
        complexVectorPtr += 4;

        cplxValue2 = _mm_load_ps(complexVectorPtr);
        complexVectorPtr += 4;

        result = _mm_magnitudesquared_ps_sse3(cplxValue1, cplxValue2);
        _mm_store_ps(magnitudeVectorPtr, result);
        magnitudeVectorPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        float val1Real = *complexVectorPtr++;
        float val1Imag = *complexVectorPtr++;
        *magnitudeVectorPtr++ = (val1Real * val1Real) + (val1Imag * val1Imag);
    }
}
#endif /* LV_HAVE_SSE3 */


#ifdef LV_HAVE_SSE
#include <volk/volk_sse_intrinsics.h>
#include <xmmintrin.h>

static inline void volk_32fc_magnitude_squared_32f_a_sse(float* magnitudeVector,
                                                         const lv_32fc_t* complexVector,
                                                         unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const float* complexVectorPtr = (float*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    __m128 cplxValue1, cplxValue2, result;
    for (; number < quarterPoints; number++) {
        cplxValue1 = _mm_load_ps(complexVectorPtr);
        complexVectorPtr += 4;

        cplxValue2 = _mm_load_ps(complexVectorPtr);
        complexVectorPtr += 4;

        result = _mm_magnitudesquared_ps(cplxValue1, cplxValue2);
        _mm_store_ps(magnitudeVectorPtr, result);
        magnitudeVectorPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        float val1Real = *complexVectorPtr++;
        float val1Imag = *complexVectorPtr++;
        *magnitudeVectorPtr++ = (val1Real * val1Real) + (val1Imag * val1Imag);
    }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32fc_magnitude_squared_32f_neon(float* magnitudeVector,
                                                        const lv_32fc_t* complexVector,
                                                        unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const float* complexVectorPtr = (float*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    float32x4x2_t cmplx_val;
    float32x4_t result;
    for (; number < quarterPoints; number++) {
        cmplx_val = vld2q_f32(complexVectorPtr);
        complexVectorPtr += 8;

        cmplx_val.val[0] =
            vmulq_f32(cmplx_val.val[0], cmplx_val.val[0]); // Square the values
        cmplx_val.val[1] =
            vmulq_f32(cmplx_val.val[1], cmplx_val.val[1]); // Square the values

        result =
            vaddq_f32(cmplx_val.val[0], cmplx_val.val[1]); // Add the I2 and Q2 values

        vst1q_f32(magnitudeVectorPtr, result);
        magnitudeVectorPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        float val1Real = *complexVectorPtr++;
        float val1Imag = *complexVectorPtr++;
        *magnitudeVectorPtr++ = (val1Real * val1Real) + (val1Imag * val1Imag);
    }
}
#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32fc_magnitude_squared_32f_neonv8(float* magnitudeVector,
                                                          const lv_32fc_t* complexVector,
                                                          unsigned int num_points)
{
    unsigned int n = num_points;
    const float* in = (const float*)complexVector;
    float* out = magnitudeVector;

    /* Process 4 complex numbers per iteration using interleaved loads + pairwise add
     * Load: [r0,i0,r1,i1] [r2,i2,r3,i3]
     * Square: [r0²,i0²,r1²,i1²] [r2²,i2²,r3²,i3²]
     * Pairwise add: [r0²+i0²,r1²+i1²,r2²+i2²,r3²+i3²]
     */
    while (n >= 4) {
        float32x4_t v0 = vld1q_f32(in);     /* r0,i0,r1,i1 */
        float32x4_t v1 = vld1q_f32(in + 4); /* r2,i2,r3,i3 */
        __VOLK_PREFETCH(in + 16);

        /* Square all elements */
        v0 = vmulq_f32(v0, v0); /* r0²,i0²,r1²,i1² */
        v1 = vmulq_f32(v1, v1); /* r2²,i2²,r3²,i3² */

        /* Pairwise add: vpaddq adds adjacent pairs */
        float32x4_t mag = vpaddq_f32(v0, v1); /* r0²+i0²,r1²+i1²,r2²+i2²,r3²+i3² */

        vst1q_f32(out, mag);

        in += 8;
        out += 4;
        n -= 4;
    }

    /* Process remaining 2 complex numbers */
    if (n >= 2) {
        float32x4_t v0 = vld1q_f32(in); /* r0,i0,r1,i1 */
        v0 = vmulq_f32(v0, v0);
        float32x2_t mag = vpadd_f32(vget_low_f32(v0), vget_high_f32(v0));
        vst1_f32(out, mag);
        in += 4;
        out += 2;
        n -= 2;
    }

    /* Scalar tail */
    if (n > 0) {
        float re = *in++;
        float im = *in++;
        *out++ = (re * re) + (im * im);
    }
}

#endif /* LV_HAVE_NEONV8 */


#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32fc_magnitude_squared_32f_rvv(float* magnitudeVector,
                                                       const lv_32fc_t* complexVector,
                                                       unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, complexVector += vl, magnitudeVector += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vuint64m8_t vc = __riscv_vle64_v_u64m8((const uint64_t*)complexVector, vl);
        vfloat32m4_t vr = __riscv_vreinterpret_f32m4(__riscv_vnsrl(vc, 0, vl));
        vfloat32m4_t vi = __riscv_vreinterpret_f32m4(__riscv_vnsrl(vc, 32, vl));
        vfloat32m4_t v = __riscv_vfmacc(__riscv_vfmul(vi, vi, vl), vr, vr, vl);
        __riscv_vse32(magnitudeVector, v, vl);
    }
}
#endif /*LV_HAVE_RVV*/

#ifdef LV_HAVE_RVVSEG
#include <riscv_vector.h>

static inline void volk_32fc_magnitude_squared_32f_rvvseg(float* magnitudeVector,
                                                          const lv_32fc_t* complexVector,
                                                          unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, complexVector += vl, magnitudeVector += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4x2_t vc = __riscv_vlseg2e32_v_f32m4x2((const float*)complexVector, vl);
        vfloat32m4_t vr = __riscv_vget_f32m4(vc, 0);
        vfloat32m4_t vi = __riscv_vget_f32m4(vc, 1);
        vfloat32m4_t v = __riscv_vfmacc(__riscv_vfmul(vi, vi, vl), vr, vr, vl);
        __riscv_vse32(magnitudeVector, v, vl);
    }
}
#endif /*LV_HAVE_RVVSEG*/

#endif /* INCLUDED_volk_32fc_magnitude_32f_a_H */
