/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_magnitude_32f
 *
 * \b Overview
 *
 * Calculates the magnitude of the complexVector and stores the
 * results in the magnitudeVector.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_magnitude_32f(float* magnitudeVector, const lv_32fc_t* complexVector,
 * unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li complexVector: The complex input vector.
 * \li num_points: The number of samples.
 *
 * \b Outputs
 * \li magnitudeVector: The output value.
 *
 * \b Example
 * Calculate the magnitude of \f$x^2 + x\f$ for points around the unit circle.
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

#ifndef INCLUDED_volk_32fc_magnitude_32f_u_H
#define INCLUDED_volk_32fc_magnitude_32f_u_H

#include <inttypes.h>
#include <math.h>
#include <stdio.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_32fc_magnitude_32f_generic(float* magnitudeVector,
                                                   const lv_32fc_t* complexVector,
                                                   unsigned int num_points)
{
    const float* complexVectorPtr = (float*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;
    unsigned int number = 0;
    for (number = 0; number < num_points; number++) {
        const float real = *complexVectorPtr++;
        const float imag = *complexVectorPtr++;
        *magnitudeVectorPtr++ = sqrtf((real * real) + (imag * imag));
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32fc_magnitude_32f_u_avx512f(float* magnitudeVector,
                                                     const lv_32fc_t* complexVector,
                                                     unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float* complexVectorPtr = (float*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    __m512 cplxValue;
    __m512 iValues, qValues;
    __m512 result;

    for (; number < eighthPoints; number++) {
        // Load 8 complex numbers (16 floats): I0,Q0,I1,Q1,...,I7,Q7
        cplxValue = _mm512_loadu_ps(complexVectorPtr);

        // Separate I and Q values using permute
        // Extract I values (even indices: 0,2,4,6,8,10,12,14)
        iValues = _mm512_permutexvar_ps(
            _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0),
            cplxValue);
        // Extract Q values (odd indices: 1,3,5,7,9,11,13,15)
        qValues = _mm512_permutexvar_ps(
            _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 0, 0, 0, 0, 0, 0, 0, 0),
            cplxValue);

        // Square and add: I^2 + Q^2
        result = _mm512_fmadd_ps(iValues, iValues, _mm512_mul_ps(qValues, qValues));

        // Square root
        result = _mm512_sqrt_ps(result);

        // Store only first 8 results
        _mm256_storeu_ps(magnitudeVectorPtr, _mm512_castps512_ps256(result));

        complexVectorPtr += 16;
        magnitudeVectorPtr += 8;
    }

    number = eighthPoints * 8;
    volk_32fc_magnitude_32f_generic(
        magnitudeVectorPtr, (const lv_32fc_t*)complexVectorPtr, num_points - number);
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_AVX
#include <immintrin.h>
#include <volk/volk_avx_intrinsics.h>

static inline void volk_32fc_magnitude_32f_u_avx(float* magnitudeVector,
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
        result = _mm256_magnitude_ps(cplxValue1, cplxValue2);
        _mm256_storeu_ps(magnitudeVectorPtr, result);

        complexVectorPtr += 16;
        magnitudeVectorPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        float val1Real = *complexVectorPtr++;
        float val1Imag = *complexVectorPtr++;
        *magnitudeVectorPtr++ = sqrtf((val1Real * val1Real) + (val1Imag * val1Imag));
    }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>
#include <volk/volk_sse3_intrinsics.h>

static inline void volk_32fc_magnitude_32f_u_sse3(float* magnitudeVector,
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

        result = _mm_magnitude_ps_sse3(cplxValue1, cplxValue2);

        _mm_storeu_ps(magnitudeVectorPtr, result);
        magnitudeVectorPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        float val1Real = *complexVectorPtr++;
        float val1Imag = *complexVectorPtr++;
        *magnitudeVectorPtr++ = sqrtf((val1Real * val1Real) + (val1Imag * val1Imag));
    }
}
#endif /* LV_HAVE_SSE3 */


#ifdef LV_HAVE_SSE
#include <volk/volk_sse_intrinsics.h>
#include <xmmintrin.h>

static inline void volk_32fc_magnitude_32f_u_sse(float* magnitudeVector,
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

        result = _mm_magnitude_ps(cplxValue1, cplxValue2);
        _mm_storeu_ps(magnitudeVectorPtr, result);
        magnitudeVectorPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        float val1Real = *complexVectorPtr++;
        float val1Imag = *complexVectorPtr++;
        *magnitudeVectorPtr++ = sqrtf((val1Real * val1Real) + (val1Imag * val1Imag));
    }
}
#endif /* LV_HAVE_SSE */

#endif /* INCLUDED_volk_32fc_magnitude_32f_u_H */
#ifndef INCLUDED_volk_32fc_magnitude_32f_a_H
#define INCLUDED_volk_32fc_magnitude_32f_a_H

#include <inttypes.h>
#include <math.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32fc_magnitude_32f_a_avx512f(float* magnitudeVector,
                                                     const lv_32fc_t* complexVector,
                                                     unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const float* complexVectorPtr = (float*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    __m512 cplxValue;
    __m512 iValues, qValues;
    __m512 result;

    for (; number < eighthPoints; number++) {
        // Load 8 complex numbers (16 floats): I0,Q0,I1,Q1,...,I7,Q7
        cplxValue = _mm512_load_ps(complexVectorPtr);

        // Separate I and Q values using permute
        iValues = _mm512_permutexvar_ps(
            _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0),
            cplxValue);
        qValues = _mm512_permutexvar_ps(
            _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 0, 0, 0, 0, 0, 0, 0, 0),
            cplxValue);

        // Square and add: I^2 + Q^2
        result = _mm512_fmadd_ps(iValues, iValues, _mm512_mul_ps(qValues, qValues));

        // Square root
        result = _mm512_sqrt_ps(result);

        // Store only first 8 results
        _mm256_store_ps(magnitudeVectorPtr, _mm512_castps512_ps256(result));

        complexVectorPtr += 16;
        magnitudeVectorPtr += 8;
    }

    number = eighthPoints * 8;
    volk_32fc_magnitude_32f_generic(
        magnitudeVectorPtr, (const lv_32fc_t*)complexVectorPtr, num_points - number);
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_AVX
#include <immintrin.h>
#include <volk/volk_avx_intrinsics.h>

static inline void volk_32fc_magnitude_32f_a_avx(float* magnitudeVector,
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

        result = _mm256_magnitude_ps(cplxValue1, cplxValue2);
        _mm256_store_ps(magnitudeVectorPtr, result);
        magnitudeVectorPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        float val1Real = *complexVectorPtr++;
        float val1Imag = *complexVectorPtr++;
        *magnitudeVectorPtr++ = sqrtf((val1Real * val1Real) + (val1Imag * val1Imag));
    }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>
#include <volk/volk_sse3_intrinsics.h>

static inline void volk_32fc_magnitude_32f_a_sse3(float* magnitudeVector,
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

        result = _mm_magnitude_ps_sse3(cplxValue1, cplxValue2);
        _mm_store_ps(magnitudeVectorPtr, result);
        magnitudeVectorPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        float val1Real = *complexVectorPtr++;
        float val1Imag = *complexVectorPtr++;
        *magnitudeVectorPtr++ = sqrtf((val1Real * val1Real) + (val1Imag * val1Imag));
    }
}
#endif /* LV_HAVE_SSE3 */

#ifdef LV_HAVE_SSE
#include <volk/volk_sse_intrinsics.h>
#include <xmmintrin.h>

static inline void volk_32fc_magnitude_32f_a_sse(float* magnitudeVector,
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

        result = _mm_magnitude_ps(cplxValue1, cplxValue2);
        _mm_store_ps(magnitudeVectorPtr, result);
        magnitudeVectorPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        float val1Real = *complexVectorPtr++;
        float val1Imag = *complexVectorPtr++;
        *magnitudeVectorPtr++ = sqrtf((val1Real * val1Real) + (val1Imag * val1Imag));
    }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32fc_magnitude_32f_neon(float* magnitudeVector,
                                                const lv_32fc_t* complexVector,
                                                unsigned int num_points)
{
    unsigned int number;
    unsigned int quarter_points = num_points / 4;
    const float* complexVectorPtr = (float*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    float32x4x2_t complex_vec;
    float32x4_t magnitude_vec;
    for (number = 0; number < quarter_points; number++) {
        complex_vec = vld2q_f32(complexVectorPtr);
        complex_vec.val[0] = vmulq_f32(complex_vec.val[0], complex_vec.val[0]);
        magnitude_vec =
            vmlaq_f32(complex_vec.val[0], complex_vec.val[1], complex_vec.val[1]);
        magnitude_vec = vrsqrteq_f32(magnitude_vec);
        magnitude_vec = vrecpeq_f32(magnitude_vec); // no plain ol' sqrt
        vst1q_f32(magnitudeVectorPtr, magnitude_vec);

        complexVectorPtr += 8;
        magnitudeVectorPtr += 4;
    }

    for (number = quarter_points * 4; number < num_points; number++) {
        const float real = *complexVectorPtr++;
        const float imag = *complexVectorPtr++;
        *magnitudeVectorPtr++ = sqrtf((real * real) + (imag * imag));
    }
}
#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_NEON
/*!
  \brief Calculates the magnitude of the complexVector and stores the results in the
  magnitudeVector

  This is an approximation from "Streamlining Digital Signal Processing" by
  Richard Lyons. Apparently max error is about 1% and mean error is about 0.6%.
  The basic idea is to do a weighted sum of the abs. value of imag and real parts
  where weight A is always assigned to max(imag, real) and B is always min(imag,real).
  There are two pairs of cofficients chosen based on whether min <= 0.4142 max.
  This method is called equiripple-error magnitude estimation proposed by Filip in '73

  \param complexVector The vector containing the complex input values
  \param magnitudeVector The vector containing the real output values
  \param num_points The number of complex values in complexVector to be calculated and
  stored into cVector
*/
static inline void volk_32fc_magnitude_32f_neon_fancy_sweet(
    float* magnitudeVector, const lv_32fc_t* complexVector, unsigned int num_points)
{
    unsigned int number;
    unsigned int quarter_points = num_points / 4;
    const float* complexVectorPtr = (float*)complexVector;
    float* magnitudeVectorPtr = magnitudeVector;

    const float threshold = 0.4142135;

    float32x4_t a_vec, b_vec, a_high, a_low, b_high, b_low;
    a_high = vdupq_n_f32(0.84);
    b_high = vdupq_n_f32(0.561);
    a_low = vdupq_n_f32(0.99);
    b_low = vdupq_n_f32(0.197);

    uint32x4_t comp0, comp1;

    float32x4x2_t complex_vec;
    float32x4_t min_vec, max_vec, magnitude_vec;
    float32x4_t real_abs, imag_abs;
    for (number = 0; number < quarter_points; number++) {
        complex_vec = vld2q_f32(complexVectorPtr);

        real_abs = vabsq_f32(complex_vec.val[0]);
        imag_abs = vabsq_f32(complex_vec.val[1]);

        min_vec = vminq_f32(real_abs, imag_abs);
        max_vec = vmaxq_f32(real_abs, imag_abs);

        // effective branch to choose coefficient pair.
        comp0 = vcgtq_f32(min_vec, vmulq_n_f32(max_vec, threshold));
        comp1 = vcleq_f32(min_vec, vmulq_n_f32(max_vec, threshold));

        // and 0s or 1s with coefficients from previous effective branch
        a_vec = (float32x4_t)vaddq_s32(vandq_s32((int32x4_t)comp0, (int32x4_t)a_high),
                                       vandq_s32((int32x4_t)comp1, (int32x4_t)a_low));
        b_vec = (float32x4_t)vaddq_s32(vandq_s32((int32x4_t)comp0, (int32x4_t)b_high),
                                       vandq_s32((int32x4_t)comp1, (int32x4_t)b_low));

        // coefficients chosen, do the weighted sum
        min_vec = vmulq_f32(min_vec, b_vec);
        max_vec = vmulq_f32(max_vec, a_vec);

        magnitude_vec = vaddq_f32(min_vec, max_vec);
        vst1q_f32(magnitudeVectorPtr, magnitude_vec);

        complexVectorPtr += 8;
        magnitudeVectorPtr += 4;
    }

    for (number = quarter_points * 4; number < num_points; number++) {
        const float real = *complexVectorPtr++;
        const float imag = *complexVectorPtr++;
        *magnitudeVectorPtr++ = sqrtf((real * real) + (imag * imag));
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32fc_magnitude_32f_rvv(float* magnitudeVector,
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
        __riscv_vse32(magnitudeVector, __riscv_vfsqrt(v, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#ifdef LV_HAVE_RVVSEG
#include <riscv_vector.h>

static inline void volk_32fc_magnitude_32f_rvvseg(float* magnitudeVector,
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
        __riscv_vse32(magnitudeVector, __riscv_vfsqrt(v, vl), vl);
    }
}
#endif /*LV_HAVE_RVVSEG*/

#endif /* INCLUDED_volk_32fc_magnitude_32f_a_H */
