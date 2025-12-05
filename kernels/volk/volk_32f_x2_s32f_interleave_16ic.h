/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_x2_s32f_interleave_16ic
 *
 * \b Overview
 *
 * Takes input vector iBuffer as the real (inphase) part and input
 * vector qBuffer as the imag (quadrature) part and combines them into
 * a complex output vector. The output is scaled by the input scalar
 * value and convert to a 16-bit short comlex number.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_x2_s32f_interleave_16ic(lv_16sc_t* complexVector, const float* iBuffer,
 * const float* qBuffer, const float scalar, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li iBuffer: Input vector of samples for the real part.
 * \li qBuffer: Input vector of samples for the imaginary part.
 * \;i scalar:  The scalar value used to scale the values before converting to shorts.
 * \li num_points: The number of values in both input vectors.
 *
 * \b Outputs
 * \li complexVector: The output vector of complex numbers.
 *
 * \b Example
 * Generate points around the unit circle and convert to complex integers.
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   float* imag = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* real = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   lv_16sc_t* out = (lv_16sc_t*)volk_malloc(sizeof(lv_16sc_t)*N, alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       real[ii] = 2.f * ((float)ii / (float)N) - 1.f;
 *       imag[ii] = std::sqrt(1.f - real[ii] * real[ii]);
 *   }
 *   // Normalize by smallest delta (0.02 in this example)
 *   float scale = 50.f;
 *
 *   volk_32f_x2_s32f_interleave_16ic(out, imag, real, scale, N);
 *
 *  for(unsigned int ii = 0; ii < N; ++ii){
 *      printf("out[%u] = %i + %ij\n", ii, std::real(out[ii]), std::imag(out[ii]));
 *  }
 *
 *   volk_free(imag);
 *   volk_free(real);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_x2_s32f_interleave_16ic_a_H
#define INCLUDED_volk_32f_x2_s32f_interleave_16ic_a_H

#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_common.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32f_x2_s32f_interleave_16ic_a_avx2(lv_16sc_t* complexVector,
                                                           const float* iBuffer,
                                                           const float* qBuffer,
                                                           const float scalar,
                                                           unsigned int num_points)
{
    unsigned int number = 0;
    const float* iBufferPtr = iBuffer;
    const float* qBufferPtr = qBuffer;

    __m256 vScalar = _mm256_set1_ps(scalar);

    const unsigned int eighthPoints = num_points / 8;

    __m256 iValue, qValue, cplxValue1, cplxValue2;
    __m256i intValue1, intValue2;

    int16_t* complexVectorPtr = (int16_t*)complexVector;

    for (; number < eighthPoints; number++) {
        iValue = _mm256_load_ps(iBufferPtr);
        qValue = _mm256_load_ps(qBufferPtr);

        // Interleaves the lower two values in the i and q variables into one buffer
        cplxValue1 = _mm256_unpacklo_ps(iValue, qValue);
        cplxValue1 = _mm256_mul_ps(cplxValue1, vScalar);

        // Interleaves the upper two values in the i and q variables into one buffer
        cplxValue2 = _mm256_unpackhi_ps(iValue, qValue);
        cplxValue2 = _mm256_mul_ps(cplxValue2, vScalar);

        intValue1 = _mm256_cvtps_epi32(cplxValue1);
        intValue2 = _mm256_cvtps_epi32(cplxValue2);

        intValue1 = _mm256_packs_epi32(intValue1, intValue2);

        _mm256_store_si256((__m256i*)complexVectorPtr, intValue1);
        complexVectorPtr += 16;

        iBufferPtr += 8;
        qBufferPtr += 8;
    }

    number = eighthPoints * 8;
    complexVectorPtr = (int16_t*)(&complexVector[number]);
    for (; number < num_points; number++) {
        *complexVectorPtr++ = (int16_t)rintf(*iBufferPtr++ * scalar);
        *complexVectorPtr++ = (int16_t)rintf(*qBufferPtr++ * scalar);
    }
}
#endif /* LV_HAVE_AVX2 */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_32f_x2_s32f_interleave_16ic_a_sse2(lv_16sc_t* complexVector,
                                                           const float* iBuffer,
                                                           const float* qBuffer,
                                                           const float scalar,
                                                           unsigned int num_points)
{
    unsigned int number = 0;
    const float* iBufferPtr = iBuffer;
    const float* qBufferPtr = qBuffer;

    __m128 vScalar = _mm_set_ps1(scalar);

    const unsigned int quarterPoints = num_points / 4;

    __m128 iValue, qValue, cplxValue1, cplxValue2;
    __m128i intValue1, intValue2;

    int16_t* complexVectorPtr = (int16_t*)complexVector;

    for (; number < quarterPoints; number++) {
        iValue = _mm_load_ps(iBufferPtr);
        qValue = _mm_load_ps(qBufferPtr);

        // Interleaves the lower two values in the i and q variables into one buffer
        cplxValue1 = _mm_unpacklo_ps(iValue, qValue);
        cplxValue1 = _mm_mul_ps(cplxValue1, vScalar);

        // Interleaves the upper two values in the i and q variables into one buffer
        cplxValue2 = _mm_unpackhi_ps(iValue, qValue);
        cplxValue2 = _mm_mul_ps(cplxValue2, vScalar);

        intValue1 = _mm_cvtps_epi32(cplxValue1);
        intValue2 = _mm_cvtps_epi32(cplxValue2);

        intValue1 = _mm_packs_epi32(intValue1, intValue2);

        _mm_store_si128((__m128i*)complexVectorPtr, intValue1);
        complexVectorPtr += 8;

        iBufferPtr += 4;
        qBufferPtr += 4;
    }

    number = quarterPoints * 4;
    complexVectorPtr = (int16_t*)(&complexVector[number]);
    for (; number < num_points; number++) {
        *complexVectorPtr++ = (int16_t)rintf(*iBufferPtr++ * scalar);
        *complexVectorPtr++ = (int16_t)rintf(*qBufferPtr++ * scalar);
    }
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_x2_s32f_interleave_16ic_a_sse(lv_16sc_t* complexVector,
                                                          const float* iBuffer,
                                                          const float* qBuffer,
                                                          const float scalar,
                                                          unsigned int num_points)
{
    unsigned int number = 0;
    const float* iBufferPtr = iBuffer;
    const float* qBufferPtr = qBuffer;

    __m128 vScalar = _mm_set_ps1(scalar);

    const unsigned int quarterPoints = num_points / 4;

    __m128 iValue, qValue, cplxValue;

    int16_t* complexVectorPtr = (int16_t*)complexVector;

    __VOLK_ATTR_ALIGNED(16) float floatBuffer[4];

    for (; number < quarterPoints; number++) {
        iValue = _mm_load_ps(iBufferPtr);
        qValue = _mm_load_ps(qBufferPtr);

        // Interleaves the lower two values in the i and q variables into one buffer
        cplxValue = _mm_unpacklo_ps(iValue, qValue);
        cplxValue = _mm_mul_ps(cplxValue, vScalar);

        _mm_store_ps(floatBuffer, cplxValue);

        *complexVectorPtr++ = (int16_t)rintf(floatBuffer[0]);
        *complexVectorPtr++ = (int16_t)rintf(floatBuffer[1]);
        *complexVectorPtr++ = (int16_t)rintf(floatBuffer[2]);
        *complexVectorPtr++ = (int16_t)rintf(floatBuffer[3]);

        // Interleaves the upper two values in the i and q variables into one buffer
        cplxValue = _mm_unpackhi_ps(iValue, qValue);
        cplxValue = _mm_mul_ps(cplxValue, vScalar);

        _mm_store_ps(floatBuffer, cplxValue);

        *complexVectorPtr++ = (int16_t)rintf(floatBuffer[0]);
        *complexVectorPtr++ = (int16_t)rintf(floatBuffer[1]);
        *complexVectorPtr++ = (int16_t)rintf(floatBuffer[2]);
        *complexVectorPtr++ = (int16_t)rintf(floatBuffer[3]);

        iBufferPtr += 4;
        qBufferPtr += 4;
    }

    number = quarterPoints * 4;
    complexVectorPtr = (int16_t*)(&complexVector[number]);
    for (; number < num_points; number++) {
        *complexVectorPtr++ = (int16_t)rintf(*iBufferPtr++ * scalar);
        *complexVectorPtr++ = (int16_t)rintf(*qBufferPtr++ * scalar);
    }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_GENERIC

static inline void volk_32f_x2_s32f_interleave_16ic_generic(lv_16sc_t* complexVector,
                                                            const float* iBuffer,
                                                            const float* qBuffer,
                                                            const float scalar,
                                                            unsigned int num_points)
{
    int16_t* complexVectorPtr = (int16_t*)complexVector;
    const float* iBufferPtr = iBuffer;
    const float* qBufferPtr = qBuffer;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        *complexVectorPtr++ = (int16_t)rintf(*iBufferPtr++ * scalar);
        *complexVectorPtr++ = (int16_t)rintf(*qBufferPtr++ * scalar);
    }
}
#endif /* LV_HAVE_GENERIC */


#endif /* INCLUDED_volk_32f_x2_s32f_interleave_16ic_a_H */

#ifndef INCLUDED_volk_32f_x2_s32f_interleave_16ic_u_H
#define INCLUDED_volk_32f_x2_s32f_interleave_16ic_u_H

#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_common.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32f_x2_s32f_interleave_16ic_u_avx2(lv_16sc_t* complexVector,
                                                           const float* iBuffer,
                                                           const float* qBuffer,
                                                           const float scalar,
                                                           unsigned int num_points)
{
    unsigned int number = 0;
    const float* iBufferPtr = iBuffer;
    const float* qBufferPtr = qBuffer;

    __m256 vScalar = _mm256_set1_ps(scalar);

    const unsigned int eighthPoints = num_points / 8;

    __m256 iValue, qValue, cplxValue1, cplxValue2;
    __m256i intValue1, intValue2;

    int16_t* complexVectorPtr = (int16_t*)complexVector;

    for (; number < eighthPoints; number++) {
        iValue = _mm256_loadu_ps(iBufferPtr);
        qValue = _mm256_loadu_ps(qBufferPtr);

        // Interleaves the lower two values in the i and q variables into one buffer
        cplxValue1 = _mm256_unpacklo_ps(iValue, qValue);
        cplxValue1 = _mm256_mul_ps(cplxValue1, vScalar);

        // Interleaves the upper two values in the i and q variables into one buffer
        cplxValue2 = _mm256_unpackhi_ps(iValue, qValue);
        cplxValue2 = _mm256_mul_ps(cplxValue2, vScalar);

        intValue1 = _mm256_cvtps_epi32(cplxValue1);
        intValue2 = _mm256_cvtps_epi32(cplxValue2);

        intValue1 = _mm256_packs_epi32(intValue1, intValue2);

        _mm256_storeu_si256((__m256i*)complexVectorPtr, intValue1);
        complexVectorPtr += 16;

        iBufferPtr += 8;
        qBufferPtr += 8;
    }

    number = eighthPoints * 8;
    complexVectorPtr = (int16_t*)(&complexVector[number]);
    for (; number < num_points; number++) {
        *complexVectorPtr++ = (int16_t)rintf(*iBufferPtr++ * scalar);
        *complexVectorPtr++ = (int16_t)rintf(*qBufferPtr++ * scalar);
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32f_x2_s32f_interleave_16ic_neon(lv_16sc_t* complexVector,
                                                         const float* iBuffer,
                                                         const float* qBuffer,
                                                         const float scalar,
                                                         unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarter_points = num_points / 4;

    const float* iBufferPtr = iBuffer;
    const float* qBufferPtr = qBuffer;
    int16_t* complexVectorPtr = (int16_t*)complexVector;

    float32x4_t vScalar = vdupq_n_f32(scalar);

    for (; number < quarter_points; number++) {
        float32x4_t iValue = vld1q_f32(iBufferPtr);
        float32x4_t qValue = vld1q_f32(qBufferPtr);

        iValue = vmulq_f32(iValue, vScalar);
        qValue = vmulq_f32(qValue, vScalar);

        int32x4_t iInt = vcvtq_s32_f32(iValue);
        int32x4_t qInt = vcvtq_s32_f32(qValue);

        int16x4_t iShort = vqmovn_s32(iInt);
        int16x4_t qShort = vqmovn_s32(qInt);

        int16x4x2_t interleaved;
        interleaved.val[0] = iShort;
        interleaved.val[1] = qShort;
        vst2_s16(complexVectorPtr, interleaved);

        complexVectorPtr += 8;
        iBufferPtr += 4;
        qBufferPtr += 4;
    }

    number = quarter_points * 4;
    complexVectorPtr = (int16_t*)(&complexVector[number]);
    for (; number < num_points; number++) {
        *complexVectorPtr++ = (int16_t)rintf(*iBufferPtr++ * scalar);
        *complexVectorPtr++ = (int16_t)rintf(*qBufferPtr++ * scalar);
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32f_x2_s32f_interleave_16ic_neonv8(lv_16sc_t* complexVector,
                                                           const float* iBuffer,
                                                           const float* qBuffer,
                                                           const float scalar,
                                                           unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighth_points = num_points / 8;

    const float* iBufferPtr = iBuffer;
    const float* qBufferPtr = qBuffer;
    int16_t* complexVectorPtr = (int16_t*)complexVector;

    float32x4_t vScalar = vdupq_n_f32(scalar);

    for (; number < eighth_points; number++) {
        float32x4_t iValue0 = vld1q_f32(iBufferPtr);
        float32x4_t iValue1 = vld1q_f32(iBufferPtr + 4);
        float32x4_t qValue0 = vld1q_f32(qBufferPtr);
        float32x4_t qValue1 = vld1q_f32(qBufferPtr + 4);
        __VOLK_PREFETCH(iBufferPtr + 8);
        __VOLK_PREFETCH(qBufferPtr + 8);

        iValue0 = vmulq_f32(iValue0, vScalar);
        iValue1 = vmulq_f32(iValue1, vScalar);
        qValue0 = vmulq_f32(qValue0, vScalar);
        qValue1 = vmulq_f32(qValue1, vScalar);

        int32x4_t iInt0 = vcvtnq_s32_f32(iValue0);
        int32x4_t iInt1 = vcvtnq_s32_f32(iValue1);
        int32x4_t qInt0 = vcvtnq_s32_f32(qValue0);
        int32x4_t qInt1 = vcvtnq_s32_f32(qValue1);

        int16x4_t iShort0 = vqmovn_s32(iInt0);
        int16x4_t iShort1 = vqmovn_s32(iInt1);
        int16x4_t qShort0 = vqmovn_s32(qInt0);
        int16x4_t qShort1 = vqmovn_s32(qInt1);

        int16x4x2_t interleaved0, interleaved1;
        interleaved0.val[0] = iShort0;
        interleaved0.val[1] = qShort0;
        interleaved1.val[0] = iShort1;
        interleaved1.val[1] = qShort1;

        vst2_s16(complexVectorPtr, interleaved0);
        vst2_s16(complexVectorPtr + 8, interleaved1);

        complexVectorPtr += 16;
        iBufferPtr += 8;
        qBufferPtr += 8;
    }

    number = eighth_points * 8;
    complexVectorPtr = (int16_t*)(&complexVector[number]);
    for (; number < num_points; number++) {
        *complexVectorPtr++ = (int16_t)rintf(*iBufferPtr++ * scalar);
        *complexVectorPtr++ = (int16_t)rintf(*qBufferPtr++ * scalar);
    }
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32f_x2_s32f_interleave_16ic_rvv(lv_16sc_t* complexVector,
                                                        const float* iBuffer,
                                                        const float* qBuffer,
                                                        const float scalar,
                                                        unsigned int num_points)
{
    uint32_t* out = (uint32_t*)complexVector;
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, out += vl, iBuffer += vl, qBuffer += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t vrf = __riscv_vle32_v_f32m8(iBuffer, vl);
        vfloat32m8_t vif = __riscv_vle32_v_f32m8(qBuffer, vl);
        vint16m4_t vri = __riscv_vfncvt_x(__riscv_vfmul(vrf, scalar, vl), vl);
        vint16m4_t vii = __riscv_vfncvt_x(__riscv_vfmul(vif, scalar, vl), vl);
        vuint16m4_t vr = __riscv_vreinterpret_u16m4(vri);
        vuint16m4_t vi = __riscv_vreinterpret_u16m4(vii);
        vuint32m8_t vc = __riscv_vwmaccu(__riscv_vwaddu_vv(vr, vi, vl), 0xFFFF, vi, vl);
        __riscv_vse32(out, vc, vl);
    }
}
#endif /*LV_HAVE_RVV*/

#ifdef LV_HAVE_RVVSEG
#include <riscv_vector.h>

static inline void volk_32f_x2_s32f_interleave_16ic_rvvseg(lv_16sc_t* complexVector,
                                                           const float* iBuffer,
                                                           const float* qBuffer,
                                                           const float scalar,
                                                           unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, complexVector += vl, iBuffer += vl, qBuffer += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t vrf = __riscv_vle32_v_f32m8(iBuffer, vl);
        vfloat32m8_t vif = __riscv_vle32_v_f32m8(qBuffer, vl);
        vint16m4_t vri = __riscv_vfncvt_x(__riscv_vfmul(vrf, scalar, vl), vl);
        vint16m4_t vii = __riscv_vfncvt_x(__riscv_vfmul(vif, scalar, vl), vl);
        __riscv_vsseg2e16(
            (int16_t*)complexVector, __riscv_vcreate_v_i16m4x2(vri, vii), vl);
    }
}
#endif /*LV_HAVE_RVVSEG*/

#endif /* INCLUDED_volk_32f_x2_s32f_interleave_16ic_u_H */
