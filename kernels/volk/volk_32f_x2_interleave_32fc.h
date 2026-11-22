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

#ifdef LV_HAVE_GENERIC
static inline void volk_32f_x2_interleave_32fc_generic(lv_32fc_t* complexVector,
                                                       const float* iBuffer,
                                                       const float* qBuffer,
                                                       unsigned int num_points)
{
    for (unsigned int number = 0; number < num_points; ++number) {
        ((float*)complexVector)[0] = *iBuffer++;
        ((float*)complexVector)[1] = *qBuffer++;
        ++complexVector;
    }
}
#endif

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>
static inline void volk_32f_x2_interleave_32fc_a_avx512f(lv_32fc_t* complexVector,
                                                         const float* iBuffer,
                                                         const float* qBuffer,
                                                         unsigned int num_points)
{
    const unsigned int blocks = num_points / 16;
    for (unsigned int number = 0; number < blocks; ++number) {
        const __m512 iValue = _mm512_load_ps(iBuffer);
        const __m512 qValue = _mm512_load_ps(qBuffer);
        const __m512 low = _mm512_unpacklo_ps(iValue, qValue);
        const __m512 high = _mm512_unpackhi_ps(iValue, qValue);
        const __m512 first = _mm512_permutex2var_ps(
            low,
            _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23),
            high);
        const __m512 second = _mm512_permutex2var_ps(
            low,
            _mm512_setr_epi32(
                8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31),
            high);
        _mm512_store_ps((float*)complexVector, first);
        _mm512_store_ps((float*)(complexVector + 8), second);
        complexVector += 16;
        iBuffer += 16;
        qBuffer += 16;
    }
    volk_32f_x2_interleave_32fc_generic(
        complexVector, iBuffer, qBuffer, num_points - blocks * 16);
}
#endif

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>
static inline void volk_32f_x2_interleave_32fc_u_avx512f(lv_32fc_t* complexVector,
                                                         const float* iBuffer,
                                                         const float* qBuffer,
                                                         unsigned int num_points)
{
    const unsigned int blocks = num_points / 16;
    for (unsigned int number = 0; number < blocks; ++number) {
        const __m512 iValue = _mm512_loadu_ps(iBuffer);
        const __m512 qValue = _mm512_loadu_ps(qBuffer);
        const __m512 low = _mm512_unpacklo_ps(iValue, qValue);
        const __m512 high = _mm512_unpackhi_ps(iValue, qValue);
        const __m512 first = _mm512_permutex2var_ps(
            low,
            _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23),
            high);
        const __m512 second = _mm512_permutex2var_ps(
            low,
            _mm512_setr_epi32(
                8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31),
            high);
        _mm512_storeu_ps((float*)complexVector, first);
        _mm512_storeu_ps((float*)(complexVector + 8), second);
        complexVector += 16;
        iBuffer += 16;
        qBuffer += 16;
    }
    volk_32f_x2_interleave_32fc_generic(
        complexVector, iBuffer, qBuffer, num_points - blocks * 16);
}
#endif

#ifdef LV_HAVE_AVX
#include <immintrin.h>
static inline void volk_32f_x2_interleave_32fc_a_avx(lv_32fc_t* complexVector,
                                                     const float* iBuffer,
                                                     const float* qBuffer,
                                                     unsigned int num_points)
{
    const unsigned int blocks = num_points / 8;
    for (unsigned int number = 0; number < blocks; ++number) {
        const __m256 iValue = _mm256_load_ps(iBuffer);
        const __m256 qValue = _mm256_load_ps(qBuffer);
        const __m256 low = _mm256_unpacklo_ps(iValue, qValue);
        const __m256 high = _mm256_unpackhi_ps(iValue, qValue);
        const __m256 first = _mm256_permute2f128_ps(low, high, 0x20);
        const __m256 second = _mm256_permute2f128_ps(low, high, 0x31);
        _mm256_store_ps((float*)complexVector, first);
        _mm256_store_ps((float*)(complexVector + 4), second);
        complexVector += 8;
        iBuffer += 8;
        qBuffer += 8;
    }
    volk_32f_x2_interleave_32fc_generic(
        complexVector, iBuffer, qBuffer, num_points - blocks * 8);
}
#endif

#ifdef LV_HAVE_AVX
#include <immintrin.h>
static inline void volk_32f_x2_interleave_32fc_u_avx(lv_32fc_t* complexVector,
                                                     const float* iBuffer,
                                                     const float* qBuffer,
                                                     unsigned int num_points)
{
    const unsigned int blocks = num_points / 8;
    for (unsigned int number = 0; number < blocks; ++number) {
        const __m256 iValue = _mm256_loadu_ps(iBuffer);
        const __m256 qValue = _mm256_loadu_ps(qBuffer);
        const __m256 low = _mm256_unpacklo_ps(iValue, qValue);
        const __m256 high = _mm256_unpackhi_ps(iValue, qValue);
        const __m256 first = _mm256_permute2f128_ps(low, high, 0x20);
        const __m256 second = _mm256_permute2f128_ps(low, high, 0x31);
        _mm256_storeu_ps((float*)complexVector, first);
        _mm256_storeu_ps((float*)(complexVector + 4), second);
        complexVector += 8;
        iBuffer += 8;
        qBuffer += 8;
    }
    volk_32f_x2_interleave_32fc_generic(
        complexVector, iBuffer, qBuffer, num_points - blocks * 8);
}
#endif

#ifdef LV_HAVE_SSE
#include <immintrin.h>
static inline void volk_32f_x2_interleave_32fc_a_sse(lv_32fc_t* complexVector,
                                                     const float* iBuffer,
                                                     const float* qBuffer,
                                                     unsigned int num_points)
{
    const unsigned int blocks = num_points / 4;
    for (unsigned int number = 0; number < blocks; ++number) {
        const __m128 iValue = _mm_load_ps(iBuffer);
        const __m128 qValue = _mm_load_ps(qBuffer);
        const __m128 low = _mm_unpacklo_ps(iValue, qValue);
        const __m128 high = _mm_unpackhi_ps(iValue, qValue);
        _mm_store_ps((float*)complexVector, low);
        _mm_store_ps((float*)(complexVector + 2), high);
        complexVector += 4;
        iBuffer += 4;
        qBuffer += 4;
    }
    volk_32f_x2_interleave_32fc_generic(
        complexVector, iBuffer, qBuffer, num_points - blocks * 4);
}
#endif

#ifdef LV_HAVE_NEON
#include <arm_neon.h>
static inline void volk_32f_x2_interleave_32fc_neon(lv_32fc_t* complexVector,
                                                    const float* iBuffer,
                                                    const float* qBuffer,
                                                    unsigned int num_points)
{
    const unsigned int blocks = num_points / 4;
    for (unsigned int number = 0; number < blocks; ++number) {
        const float32x4x2_t values = { { vld1q_f32(iBuffer), vld1q_f32(qBuffer) } };
        vst2q_f32((float*)complexVector, values);
        complexVector += 4;
        iBuffer += 4;
        qBuffer += 4;
    }
    volk_32f_x2_interleave_32fc_generic(
        complexVector, iBuffer, qBuffer, num_points - blocks * 4);
}
#endif

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>
static inline void volk_32f_x2_interleave_32fc_neonv8(lv_32fc_t* complexVector,
                                                      const float* iBuffer,
                                                      const float* qBuffer,
                                                      unsigned int num_points)
{
    const unsigned int blocks = num_points / 8;
    for (unsigned int number = 0; number < blocks; ++number) {
        const float32x4x2_t values = { { vld1q_f32(iBuffer), vld1q_f32(qBuffer) } };
        vst2q_f32((float*)complexVector, values);
        const float32x4x2_t values1 = { { vld1q_f32(iBuffer + 4),
                                          vld1q_f32(qBuffer + 4) } };
        __VOLK_PREFETCH(iBuffer + 16);
        __VOLK_PREFETCH(qBuffer + 16);
        vst2q_f32((float*)(complexVector + 4), values1);
        complexVector += 8;
        iBuffer += 8;
        qBuffer += 8;
    }
    volk_32f_x2_interleave_32fc_generic(
        complexVector, iBuffer, qBuffer, num_points - blocks * 8);
}
#endif

#endif /* INCLUDED_volk_32f_x2_interleave_32fc_a_H */
#ifndef INCLUDED_volk_32f_x2_interleave_32fc_u_H
#define INCLUDED_volk_32f_x2_interleave_32fc_u_H
#include <inttypes.h>

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
