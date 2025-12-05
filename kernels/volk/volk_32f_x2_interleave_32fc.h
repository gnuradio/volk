/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_x2_interleave_32fc
 *
 * \b Overview
 *
 * Takes input vector iBuffer as the real (inphase) part and input
 * vector qBuffer as the imag (quadrature) part and combines them into
 * a complex output vector.
 *
 * c[i] = complex(a[i], b[i])
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_x2_interleave_32fc(lv_32fc_t* complexVector, const float* iBuffer, const
 * float* qBuffer, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li iBuffer: Input vector of samples for the real part.
 * \li qBuffer: Input vector of samples for the imaginary part.
 * \li num_points: The number of values in both input vectors.
 *
 * \b Outputs
 * \li complexVector: The output vector of complex numbers.
 *
 * \b Example
 * Generate the top half of the unit circle with real points equally spaced.
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   float* imag = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* real = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   lv_32fc_t* out = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       real[ii] = 2.f * ((float)ii / (float)N) - 1.f;
 *       imag[ii] = std::sqrt(1.f - real[ii] * real[ii]);
 *   }
 *
 *   volk_32f_x2_interleave_32fc(out, imag, real, N);
 *
 *  for(unsigned int ii = 0; ii < N; ++ii){
 *      printf("out[%u] = %1.2f + %1.2fj\n", ii, std::real(out[ii]), std::imag(out[ii]));
 *  }
 *
 *   volk_free(imag);
 *   volk_free(real);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_x2_interleave_32fc_a_H
#define INCLUDED_volk_32f_x2_interleave_32fc_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_x2_interleave_32fc_a_avx(lv_32fc_t* complexVector,
                                                     const float* iBuffer,
                                                     const float* qBuffer,
                                                     unsigned int num_points)
{
    unsigned int number = 0;
    float* complexVectorPtr = (float*)complexVector;
    const float* iBufferPtr = iBuffer;
    const float* qBufferPtr = qBuffer;

    const uint64_t eighthPoints = num_points / 8;

    __m256 iValue, qValue, cplxValue1, cplxValue2, cplxValue;
    for (; number < eighthPoints; number++) {
        iValue = _mm256_load_ps(iBufferPtr);
        qValue = _mm256_load_ps(qBufferPtr);

        // Interleaves the lower two values in the i and q variables into one buffer
        cplxValue1 = _mm256_unpacklo_ps(iValue, qValue);
        // Interleaves the upper two values in the i and q variables into one buffer
        cplxValue2 = _mm256_unpackhi_ps(iValue, qValue);

        cplxValue = _mm256_permute2f128_ps(cplxValue1, cplxValue2, 0x20);
        _mm256_store_ps(complexVectorPtr, cplxValue);
        complexVectorPtr += 8;

        cplxValue = _mm256_permute2f128_ps(cplxValue1, cplxValue2, 0x31);
        _mm256_store_ps(complexVectorPtr, cplxValue);
        complexVectorPtr += 8;

        iBufferPtr += 8;
        qBufferPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *complexVectorPtr++ = *iBufferPtr++;
        *complexVectorPtr++ = *qBufferPtr++;
    }
}

#endif /* LV_HAV_AVX */

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_x2_interleave_32fc_a_sse(lv_32fc_t* complexVector,
                                                     const float* iBuffer,
                                                     const float* qBuffer,
                                                     unsigned int num_points)
{
    unsigned int number = 0;
    float* complexVectorPtr = (float*)complexVector;
    const float* iBufferPtr = iBuffer;
    const float* qBufferPtr = qBuffer;

    const uint64_t quarterPoints = num_points / 4;

    __m128 iValue, qValue, cplxValue;
    for (; number < quarterPoints; number++) {
        iValue = _mm_load_ps(iBufferPtr);
        qValue = _mm_load_ps(qBufferPtr);

        // Interleaves the lower two values in the i and q variables into one buffer
        cplxValue = _mm_unpacklo_ps(iValue, qValue);
        _mm_store_ps(complexVectorPtr, cplxValue);
        complexVectorPtr += 4;

        // Interleaves the upper two values in the i and q variables into one buffer
        cplxValue = _mm_unpackhi_ps(iValue, qValue);
        _mm_store_ps(complexVectorPtr, cplxValue);
        complexVectorPtr += 4;

        iBufferPtr += 4;
        qBufferPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *complexVectorPtr++ = *iBufferPtr++;
        *complexVectorPtr++ = *qBufferPtr++;
    }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32f_x2_interleave_32fc_neon(lv_32fc_t* complexVector,
                                                    const float* iBuffer,
                                                    const float* qBuffer,
                                                    unsigned int num_points)
{
    unsigned int quarter_points = num_points / 4;
    unsigned int number;
    float* complexVectorPtr = (float*)complexVector;

    float32x4x2_t complex_vec;
    for (number = 0; number < quarter_points; ++number) {
        complex_vec.val[0] = vld1q_f32(iBuffer);
        complex_vec.val[1] = vld1q_f32(qBuffer);
        vst2q_f32(complexVectorPtr, complex_vec);
        iBuffer += 4;
        qBuffer += 4;
        complexVectorPtr += 8;
    }

    for (number = quarter_points * 4; number < num_points; ++number) {
        *complexVectorPtr++ = *iBuffer++;
        *complexVectorPtr++ = *qBuffer++;
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32f_x2_interleave_32fc_neonv8(lv_32fc_t* complexVector,
                                                      const float* iBuffer,
                                                      const float* qBuffer,
                                                      unsigned int num_points)
{
    const unsigned int eighthPoints = num_points / 8;

    float* outPtr = (float*)complexVector;
    const float* iPtr = iBuffer;
    const float* qPtr = qBuffer;

    for (unsigned int number = 0; number < eighthPoints; number++) {
        float32x4x2_t cplx0, cplx1;
        cplx0.val[0] = vld1q_f32(iPtr);
        cplx0.val[1] = vld1q_f32(qPtr);
        cplx1.val[0] = vld1q_f32(iPtr + 4);
        cplx1.val[1] = vld1q_f32(qPtr + 4);
        __VOLK_PREFETCH(iPtr + 16);
        __VOLK_PREFETCH(qPtr + 16);

        vst2q_f32(outPtr, cplx0);
        vst2q_f32(outPtr + 8, cplx1);

        iPtr += 8;
        qPtr += 8;
        outPtr += 16;
    }

    for (unsigned int number = eighthPoints * 8; number < num_points; number++) {
        *outPtr++ = *iPtr++;
        *outPtr++ = *qPtr++;
    }
}
#endif /* LV_HAVE_NEONV8 */


#ifdef LV_HAVE_GENERIC

static inline void volk_32f_x2_interleave_32fc_generic(lv_32fc_t* complexVector,
                                                       const float* iBuffer,
                                                       const float* qBuffer,
                                                       unsigned int num_points)
{
    float* complexVectorPtr = (float*)complexVector;
    const float* iBufferPtr = iBuffer;
    const float* qBufferPtr = qBuffer;
    unsigned int number;

    for (number = 0; number < num_points; number++) {
        *complexVectorPtr++ = *iBufferPtr++;
        *complexVectorPtr++ = *qBufferPtr++;
    }
}
#endif /* LV_HAVE_GENERIC */


#endif /* INCLUDED_volk_32f_x2_interleave_32fc_a_H */

#ifndef INCLUDED_volk_32f_x2_interleave_32fc_u_H
#define INCLUDED_volk_32f_x2_interleave_32fc_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_x2_interleave_32fc_u_avx(lv_32fc_t* complexVector,
                                                     const float* iBuffer,
                                                     const float* qBuffer,
                                                     unsigned int num_points)
{
    unsigned int number = 0;
    float* complexVectorPtr = (float*)complexVector;
    const float* iBufferPtr = iBuffer;
    const float* qBufferPtr = qBuffer;

    const uint64_t eighthPoints = num_points / 8;

    __m256 iValue, qValue, cplxValue1, cplxValue2, cplxValue;
    for (; number < eighthPoints; number++) {
        iValue = _mm256_loadu_ps(iBufferPtr);
        qValue = _mm256_loadu_ps(qBufferPtr);

        // Interleaves the lower two values in the i and q variables into one buffer
        cplxValue1 = _mm256_unpacklo_ps(iValue, qValue);
        // Interleaves the upper two values in the i and q variables into one buffer
        cplxValue2 = _mm256_unpackhi_ps(iValue, qValue);

        cplxValue = _mm256_permute2f128_ps(cplxValue1, cplxValue2, 0x20);
        _mm256_storeu_ps(complexVectorPtr, cplxValue);
        complexVectorPtr += 8;

        cplxValue = _mm256_permute2f128_ps(cplxValue1, cplxValue2, 0x31);
        _mm256_storeu_ps(complexVectorPtr, cplxValue);
        complexVectorPtr += 8;

        iBufferPtr += 8;
        qBufferPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *complexVectorPtr++ = *iBufferPtr++;
        *complexVectorPtr++ = *qBufferPtr++;
    }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32f_x2_interleave_32fc_rvv(lv_32fc_t* complexVector,
                                                   const float* iBuffer,
                                                   const float* qBuffer,
                                                   unsigned int num_points)
{
    uint64_t* out = (uint64_t*)complexVector;
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, out += vl, iBuffer += vl, qBuffer += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vuint32m4_t vr = __riscv_vle32_v_u32m4((const uint32_t*)iBuffer, vl);
        vuint32m4_t vi = __riscv_vle32_v_u32m4((const uint32_t*)qBuffer, vl);
        vuint64m8_t vc =
            __riscv_vwmaccu(__riscv_vwaddu_vv(vr, vi, vl), 0xFFFFFFFF, vi, vl);
        __riscv_vse64(out, vc, vl);
    }
}
#endif /*LV_HAVE_RVV*/

#ifdef LV_HAVE_RVVSEG
#include <riscv_vector.h>

static inline void volk_32f_x2_interleave_32fc_rvvseg(lv_32fc_t* complexVector,
                                                      const float* iBuffer,
                                                      const float* qBuffer,
                                                      unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, complexVector += vl, iBuffer += vl, qBuffer += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t vr = __riscv_vle32_v_f32m4(iBuffer, vl);
        vfloat32m4_t vi = __riscv_vle32_v_f32m4(qBuffer, vl);
        __riscv_vsseg2e32((float*)complexVector, __riscv_vcreate_v_f32m4x2(vr, vi), vl);
    }
}
#endif /*LV_HAVE_RVVSEG*/

#endif /* INCLUDED_volk_32f_x2_interleave_32fc_u_H */
