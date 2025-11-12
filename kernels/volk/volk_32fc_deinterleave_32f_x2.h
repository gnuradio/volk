/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_deinterleave_32f_x2
 *
 * \b Overview
 *
 * Deinterleaves the complex floating point vector into I & Q vector
 * data.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_deinterleave_32f_x2(float* iBuffer, float* qBuffer, const lv_32fc_t*
 * complexVector, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li complexVector: The complex input vector.
 * \li num_points: The number of complex data values to be deinterleaved.
 *
 * \b Outputs
 * \li iBuffer: The I buffer output data.
 * \li qBuffer: The Q buffer output data.
 *
 * \b Example
 * Generate complex numbers around the top half of the unit circle and
 * deinterleave in to real and imaginary buffers.
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   lv_32fc_t* in  = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *   float* re = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* im = (float*)volk_malloc(sizeof(float)*N, alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       float real = 2.f * ((float)ii / (float)N) - 1.f;
 *       float imag = std::sqrt(1.f - real * real);
 *       in[ii] = lv_cmake(real, imag);
 *   }
 *
 *   volk_32fc_deinterleave_32f_x2(re, im, in, N);
 *
 *   printf("          re  | im\n");
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("out(%i) = %+.1f | %+.1f\n", ii, re[ii], im[ii]);
 *   }
 *
 *   volk_free(in);
 *   volk_free(re);
 *   volk_free(im);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_deinterleave_32f_x2_a_H
#define INCLUDED_volk_32fc_deinterleave_32f_x2_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_32fc_deinterleave_32f_x2_generic(float* iBuffer,
                                                         float* qBuffer,
                                                         const lv_32fc_t* complexVector,
                                                         unsigned int num_points)
{
    const float* complexVectorPtr = (float*)complexVector;
    float* iBufferPtr = iBuffer;
    float* qBufferPtr = qBuffer;
    unsigned int number;
    for (number = 0; number < num_points; number++) {
        *iBufferPtr++ = *complexVectorPtr++;
        *qBufferPtr++ = *complexVectorPtr++;
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32fc_deinterleave_32f_x2_a_avx512f(float* iBuffer,
                                                           float* qBuffer,
                                                           const lv_32fc_t* complexVector,
                                                           unsigned int num_points)
{
    const float* complexVectorPtr = (float*)complexVector;
    float* iBufferPtr = iBuffer;
    float* qBufferPtr = qBuffer;

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    __m512 cplxValue;
    __m512 iValue, qValue;

    for (; number < eighthPoints; number++) {
        // Load 8 complex numbers (16 floats): I0,Q0,I1,Q1,...,I7,Q7
        cplxValue = _mm512_load_ps(complexVectorPtr);

        // Deinterleave using permute
        // Extract all I values (even indices: 0,2,4,6,8,10,12,14)
        iValue = _mm512_permutexvar_ps(
            _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0),
            cplxValue);

        // Extract all Q values (odd indices: 1,3,5,7,9,11,13,15)
        qValue = _mm512_permutexvar_ps(
            _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 0, 0, 0, 0, 0, 0, 0, 0),
            cplxValue);

        // Store only the first 8 results (lower 256 bits)
        _mm256_store_ps(iBufferPtr, _mm512_castps512_ps256(iValue));
        _mm256_store_ps(qBufferPtr, _mm512_castps512_ps256(qValue));

        complexVectorPtr += 16;
        iBufferPtr += 8;
        qBufferPtr += 8;
    }

    number = eighthPoints * 8;
    volk_32fc_deinterleave_32f_x2_generic(
        iBufferPtr, qBufferPtr, (const lv_32fc_t*)complexVectorPtr, num_points - number);
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_AVX
#include <immintrin.h>
static inline void volk_32fc_deinterleave_32f_x2_a_avx(float* iBuffer,
                                                       float* qBuffer,
                                                       const lv_32fc_t* complexVector,
                                                       unsigned int num_points)
{
    const float* complexVectorPtr = (float*)complexVector;
    float* iBufferPtr = iBuffer;
    float* qBufferPtr = qBuffer;

    unsigned int number = 0;
    // Mask for real and imaginary parts
    const unsigned int eighthPoints = num_points / 8;
    __m256 cplxValue1, cplxValue2, complex1, complex2, iValue, qValue;
    for (; number < eighthPoints; number++) {
        cplxValue1 = _mm256_load_ps(complexVectorPtr);
        complexVectorPtr += 8;

        cplxValue2 = _mm256_load_ps(complexVectorPtr);
        complexVectorPtr += 8;

        complex1 = _mm256_permute2f128_ps(cplxValue1, cplxValue2, 0x20);
        complex2 = _mm256_permute2f128_ps(cplxValue1, cplxValue2, 0x31);

        // Arrange in i1i2i3i4 format
        iValue = _mm256_shuffle_ps(complex1, complex2, 0x88);
        // Arrange in q1q2q3q4 format
        qValue = _mm256_shuffle_ps(complex1, complex2, 0xdd);

        _mm256_store_ps(iBufferPtr, iValue);
        _mm256_store_ps(qBufferPtr, qValue);

        iBufferPtr += 8;
        qBufferPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *iBufferPtr++ = *complexVectorPtr++;
        *qBufferPtr++ = *complexVectorPtr++;
    }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32fc_deinterleave_32f_x2_a_sse(float* iBuffer,
                                                       float* qBuffer,
                                                       const lv_32fc_t* complexVector,
                                                       unsigned int num_points)
{
    const float* complexVectorPtr = (float*)complexVector;
    float* iBufferPtr = iBuffer;
    float* qBufferPtr = qBuffer;

    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;
    __m128 cplxValue1, cplxValue2, iValue, qValue;
    for (; number < quarterPoints; number++) {
        cplxValue1 = _mm_load_ps(complexVectorPtr);
        complexVectorPtr += 4;

        cplxValue2 = _mm_load_ps(complexVectorPtr);
        complexVectorPtr += 4;

        // Arrange in i1i2i3i4 format
        iValue = _mm_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(2, 0, 2, 0));
        // Arrange in q1q2q3q4 format
        qValue = _mm_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(3, 1, 3, 1));

        _mm_store_ps(iBufferPtr, iValue);
        _mm_store_ps(qBufferPtr, qValue);

        iBufferPtr += 4;
        qBufferPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *iBufferPtr++ = *complexVectorPtr++;
        *qBufferPtr++ = *complexVectorPtr++;
    }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32fc_deinterleave_32f_x2_neon(float* iBuffer,
                                                      float* qBuffer,
                                                      const lv_32fc_t* complexVector,
                                                      unsigned int num_points)
{
    unsigned int number = 0;
    unsigned int quarter_points = num_points / 4;
    const float* complexVectorPtr = (float*)complexVector;
    float* iBufferPtr = iBuffer;
    float* qBufferPtr = qBuffer;
    float32x4x2_t complexInput;

    for (number = 0; number < quarter_points; number++) {
        complexInput = vld2q_f32(complexVectorPtr);
        vst1q_f32(iBufferPtr, complexInput.val[0]);
        vst1q_f32(qBufferPtr, complexInput.val[1]);
        complexVectorPtr += 8;
        iBufferPtr += 4;
        qBufferPtr += 4;
    }

    for (number = quarter_points * 4; number < num_points; number++) {
        *iBufferPtr++ = *complexVectorPtr++;
        *qBufferPtr++ = *complexVectorPtr++;
    }
}
#endif /* LV_HAVE_NEON */

#endif /* INCLUDED_volk_32fc_deinterleave_32f_x2_a_H */


#ifndef INCLUDED_volk_32fc_deinterleave_32f_x2_u_H
#define INCLUDED_volk_32fc_deinterleave_32f_x2_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32fc_deinterleave_32f_x2_u_avx512f(float* iBuffer,
                                                           float* qBuffer,
                                                           const lv_32fc_t* complexVector,
                                                           unsigned int num_points)
{
    const float* complexVectorPtr = (float*)complexVector;
    float* iBufferPtr = iBuffer;
    float* qBufferPtr = qBuffer;

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    __m512 cplxValue;
    __m512 iValue, qValue;

    for (; number < eighthPoints; number++) {
        // Load 8 complex numbers (16 floats): I0,Q0,I1,Q1,...,I7,Q7 - unaligned
        cplxValue = _mm512_loadu_ps(complexVectorPtr);

        // Deinterleave using permute
        // Extract all I values (even indices: 0,2,4,6,8,10,12,14)
        iValue = _mm512_permutexvar_ps(
            _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0),
            cplxValue);

        // Extract all Q values (odd indices: 1,3,5,7,9,11,13,15)
        qValue = _mm512_permutexvar_ps(
            _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 0, 0, 0, 0, 0, 0, 0, 0),
            cplxValue);

        // Store only the first 8 results (lower 256 bits) - unaligned
        _mm256_storeu_ps(iBufferPtr, _mm512_castps512_ps256(iValue));
        _mm256_storeu_ps(qBufferPtr, _mm512_castps512_ps256(qValue));

        complexVectorPtr += 16;
        iBufferPtr += 8;
        qBufferPtr += 8;
    }

    number = eighthPoints * 8;
    volk_32fc_deinterleave_32f_x2_generic(
        iBufferPtr, qBufferPtr, (const lv_32fc_t*)complexVectorPtr, num_points - number);
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_AVX
#include <immintrin.h>
static inline void volk_32fc_deinterleave_32f_x2_u_avx(float* iBuffer,
                                                       float* qBuffer,
                                                       const lv_32fc_t* complexVector,
                                                       unsigned int num_points)
{
    const float* complexVectorPtr = (float*)complexVector;
    float* iBufferPtr = iBuffer;
    float* qBufferPtr = qBuffer;

    unsigned int number = 0;
    // Mask for real and imaginary parts
    const unsigned int eighthPoints = num_points / 8;
    __m256 cplxValue1, cplxValue2, complex1, complex2, iValue, qValue;
    for (; number < eighthPoints; number++) {
        cplxValue1 = _mm256_loadu_ps(complexVectorPtr);
        complexVectorPtr += 8;

        cplxValue2 = _mm256_loadu_ps(complexVectorPtr);
        complexVectorPtr += 8;

        complex1 = _mm256_permute2f128_ps(cplxValue1, cplxValue2, 0x20);
        complex2 = _mm256_permute2f128_ps(cplxValue1, cplxValue2, 0x31);

        // Arrange in i1i2i3i4 format
        iValue = _mm256_shuffle_ps(complex1, complex2, 0x88);
        // Arrange in q1q2q3q4 format
        qValue = _mm256_shuffle_ps(complex1, complex2, 0xdd);

        _mm256_storeu_ps(iBufferPtr, iValue);
        _mm256_storeu_ps(qBufferPtr, qValue);

        iBufferPtr += 8;
        qBufferPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *iBufferPtr++ = *complexVectorPtr++;
        *qBufferPtr++ = *complexVectorPtr++;
    }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32fc_deinterleave_32f_x2_rvv(float* iBuffer,
                                                     float* qBuffer,
                                                     const lv_32fc_t* complexVector,
                                                     unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, complexVector += vl, iBuffer += vl, qBuffer += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vuint64m8_t vc = __riscv_vle64_v_u64m8((const uint64_t*)complexVector, vl);
        vuint32m4_t vr = __riscv_vnsrl(vc, 0, vl);
        vuint32m4_t vi = __riscv_vnsrl(vc, 32, vl);
        __riscv_vse32((uint32_t*)iBuffer, vr, vl);
        __riscv_vse32((uint32_t*)qBuffer, vi, vl);
    }
}
#endif /*LV_HAVE_RVV*/

#ifdef LV_HAVE_RVVSEG
#include <riscv_vector.h>

static inline void volk_32fc_deinterleave_32f_x2_rvvseg(float* iBuffer,
                                                        float* qBuffer,
                                                        const lv_32fc_t* complexVector,
                                                        unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, complexVector += vl, iBuffer += vl, qBuffer += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vuint32m4x2_t vc =
            __riscv_vlseg2e32_v_u32m4x2((const uint32_t*)complexVector, vl);
        vuint32m4_t vr = __riscv_vget_u32m4(vc, 0);
        vuint32m4_t vi = __riscv_vget_u32m4(vc, 1);
        __riscv_vse32((uint32_t*)iBuffer, vr, vl);
        __riscv_vse32((uint32_t*)qBuffer, vi, vl);
    }
}
#endif /*LV_HAVE_RVVSEG*/

#endif /* INCLUDED_volk_32fc_deinterleave_32f_x2_u_H */
