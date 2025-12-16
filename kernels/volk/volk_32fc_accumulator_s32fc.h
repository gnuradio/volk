/* -*- c++ -*- */
/*
 * Copyright 2019 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_accumulator_s32fc
 *
 * \b Overview
 *
 * Accumulates the values in the input buffer.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_accumulator_s32fc(lv_32fc_t* result, const lv_32fc_t* inputBuffer,
 * unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li inputBuffer: The buffer of data to be accumulated
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li result: The accumulated result.
 *
 * \b Example
 * Calculate the sum of numbers  0 through 99
 * \code
 *   int N = 100;
 *   unsigned int alignment = volk_get_alignment();
 *   lv_32fc_t* vec = (lv_32fc_t*) volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *   lv_32fc_t* out = (lv_32fc_t*) volk_malloc(sizeof(lv_32fc_t), alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       vec[ii] = lv_cmake( (float) ii, (float) -ii );
 *   }
 *
 *   volk_32fc_accumulator_s32fc(out, vec, N);
 *
 *   printf("sum(0..99)+1j*sum(0..-99) = %1.2f %1.2f \n", lv_creal(*out) , lv_cimag(*out)
 * );
 *
 *   volk_free(vec);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_accumulator_s32fc_a_H
#define INCLUDED_volk_32fc_accumulator_s32fc_a_H

#include <inttypes.h>
#include <volk/volk_common.h>

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32fc_accumulator_s32fc_a_avx512f(lv_32fc_t* result,
                                                         const lv_32fc_t* inputBuffer,
                                                         unsigned int num_points)
{
    lv_32fc_t returnValue = lv_cmake(0.f, 0.f);
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const lv_32fc_t* aPtr = inputBuffer;
    __VOLK_ATTR_ALIGNED(64) float tempBuffer[16];

    __m512 accumulator = _mm512_setzero_ps();
    __m512 aVal = _mm512_setzero_ps();

    for (; number < eighthPoints; number++) {
        aVal = _mm512_load_ps((float*)aPtr);
        accumulator = _mm512_add_ps(accumulator, aVal);
        aPtr += 8;
    }

    _mm512_store_ps(tempBuffer, accumulator);

    // Sum pairs as complex numbers
    returnValue = lv_cmake(tempBuffer[0], tempBuffer[1]);
    returnValue += lv_cmake(tempBuffer[2], tempBuffer[3]);
    returnValue += lv_cmake(tempBuffer[4], tempBuffer[5]);
    returnValue += lv_cmake(tempBuffer[6], tempBuffer[7]);
    returnValue += lv_cmake(tempBuffer[8], tempBuffer[9]);
    returnValue += lv_cmake(tempBuffer[10], tempBuffer[11]);
    returnValue += lv_cmake(tempBuffer[12], tempBuffer[13]);
    returnValue += lv_cmake(tempBuffer[14], tempBuffer[15]);

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        returnValue += (*aPtr++);
    }
    *result = returnValue;
}
#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32fc_accumulator_s32fc_u_avx512f(lv_32fc_t* result,
                                                         const lv_32fc_t* inputBuffer,
                                                         unsigned int num_points)
{
    lv_32fc_t returnValue = lv_cmake(0.f, 0.f);
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    const lv_32fc_t* aPtr = inputBuffer;
    __VOLK_ATTR_ALIGNED(64) float tempBuffer[16];

    __m512 accumulator = _mm512_setzero_ps();
    __m512 aVal = _mm512_setzero_ps();

    for (; number < eighthPoints; number++) {
        aVal = _mm512_loadu_ps((float*)aPtr);
        accumulator = _mm512_add_ps(accumulator, aVal);
        aPtr += 8;
    }

    _mm512_store_ps(tempBuffer, accumulator);

    // Sum pairs as complex numbers
    returnValue = lv_cmake(tempBuffer[0], tempBuffer[1]);
    returnValue += lv_cmake(tempBuffer[2], tempBuffer[3]);
    returnValue += lv_cmake(tempBuffer[4], tempBuffer[5]);
    returnValue += lv_cmake(tempBuffer[6], tempBuffer[7]);
    returnValue += lv_cmake(tempBuffer[8], tempBuffer[9]);
    returnValue += lv_cmake(tempBuffer[10], tempBuffer[11]);
    returnValue += lv_cmake(tempBuffer[12], tempBuffer[13]);
    returnValue += lv_cmake(tempBuffer[14], tempBuffer[15]);

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        returnValue += (*aPtr++);
    }
    *result = returnValue;
}
#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_GENERIC
static inline void volk_32fc_accumulator_s32fc_generic(lv_32fc_t* result,
                                                       const lv_32fc_t* inputBuffer,
                                                       unsigned int num_points)
{
    const lv_32fc_t* aPtr = inputBuffer;
    unsigned int number = 0;
    lv_32fc_t returnValue = lv_cmake(0.f, 0.f);

    for (; number < num_points; number++) {
        returnValue += (*aPtr++);
    }
    *result = returnValue;
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32fc_accumulator_s32fc_u_avx(lv_32fc_t* result,
                                                     const lv_32fc_t* inputBuffer,
                                                     unsigned int num_points)
{
    lv_32fc_t returnValue = lv_cmake(0.f, 0.f);
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const lv_32fc_t* aPtr = inputBuffer;
    __VOLK_ATTR_ALIGNED(32) float tempBuffer[8];

    __m256 accumulator = _mm256_setzero_ps();
    __m256 aVal = _mm256_setzero_ps();

    for (; number < quarterPoints; number++) {
        aVal = _mm256_loadu_ps((float*)aPtr);
        accumulator = _mm256_add_ps(accumulator, aVal);
        aPtr += 4;
    }

    _mm256_store_ps(tempBuffer, accumulator);

    returnValue = lv_cmake(tempBuffer[0], tempBuffer[1]);
    returnValue += lv_cmake(tempBuffer[2], tempBuffer[3]);
    returnValue += lv_cmake(tempBuffer[4], tempBuffer[5]);
    returnValue += lv_cmake(tempBuffer[6], tempBuffer[7]);

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        returnValue += (*aPtr++);
    }
    *result = returnValue;
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32fc_accumulator_s32fc_u_sse(lv_32fc_t* result,
                                                     const lv_32fc_t* inputBuffer,
                                                     unsigned int num_points)
{
    lv_32fc_t returnValue = lv_cmake(0.f, 0.f);
    unsigned int number = 0;
    const unsigned int halfPoints = num_points / 2;

    const lv_32fc_t* aPtr = inputBuffer;
    __VOLK_ATTR_ALIGNED(16) float tempBuffer[4];

    __m128 accumulator = _mm_setzero_ps();
    __m128 aVal = _mm_setzero_ps();

    for (; number < halfPoints; number++) {
        aVal = _mm_loadu_ps((float*)aPtr);
        accumulator = _mm_add_ps(accumulator, aVal);
        aPtr += 2;
    }

    _mm_store_ps(tempBuffer, accumulator);

    returnValue = lv_cmake(tempBuffer[0], tempBuffer[1]);
    returnValue += lv_cmake(tempBuffer[2], tempBuffer[3]);

    number = halfPoints * 2;
    for (; number < num_points; number++) {
        returnValue += (*aPtr++);
    }
    *result = returnValue;
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32fc_accumulator_s32fc_a_avx(lv_32fc_t* result,
                                                     const lv_32fc_t* inputBuffer,
                                                     unsigned int num_points)
{
    lv_32fc_t returnValue = lv_cmake(0.f, 0.f);
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    const lv_32fc_t* aPtr = inputBuffer;
    __VOLK_ATTR_ALIGNED(32) float tempBuffer[8];

    __m256 accumulator = _mm256_setzero_ps();
    __m256 aVal = _mm256_setzero_ps();

    for (; number < quarterPoints; number++) {
        aVal = _mm256_load_ps((float*)aPtr);
        accumulator = _mm256_add_ps(accumulator, aVal);
        aPtr += 4;
    }

    _mm256_store_ps(tempBuffer, accumulator);

    returnValue = lv_cmake(tempBuffer[0], tempBuffer[1]);
    returnValue += lv_cmake(tempBuffer[2], tempBuffer[3]);
    returnValue += lv_cmake(tempBuffer[4], tempBuffer[5]);
    returnValue += lv_cmake(tempBuffer[6], tempBuffer[7]);

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        returnValue += (*aPtr++);
    }
    *result = returnValue;
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32fc_accumulator_s32fc_a_sse(lv_32fc_t* result,
                                                     const lv_32fc_t* inputBuffer,
                                                     unsigned int num_points)
{
    lv_32fc_t returnValue = lv_cmake(0.f, 0.f);
    unsigned int number = 0;
    const unsigned int halfPoints = num_points / 2;

    const lv_32fc_t* aPtr = inputBuffer;
    __VOLK_ATTR_ALIGNED(16) float tempBuffer[4];

    __m128 accumulator = _mm_setzero_ps();
    __m128 aVal = _mm_setzero_ps();

    for (; number < halfPoints; number++) {
        aVal = _mm_load_ps((float*)aPtr);
        accumulator = _mm_add_ps(accumulator, aVal);
        aPtr += 2;
    }

    _mm_store_ps(tempBuffer, accumulator);

    returnValue = lv_cmake(tempBuffer[0], tempBuffer[1]);
    returnValue += lv_cmake(tempBuffer[2], tempBuffer[3]);

    number = halfPoints * 2;
    for (; number < num_points; number++) {
        returnValue += (*aPtr++);
    }
    *result = returnValue;
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>
static inline void volk_32fc_accumulator_s32fc_neon(lv_32fc_t* result,
                                                    const lv_32fc_t* inputBuffer,
                                                    unsigned int num_points)
{
    const lv_32fc_t* aPtr = inputBuffer;
    unsigned int number = 0;
    lv_32fc_t returnValue = lv_cmake(0.f, 0.f);
    unsigned int eighthPoints = num_points / 8;
    float32x4_t in_vec;
    float32x4_t out_vec0 = { 0.f, 0.f, 0.f, 0.f };
    float32x4_t out_vec1 = { 0.f, 0.f, 0.f, 0.f };
    float32x4_t out_vec2 = { 0.f, 0.f, 0.f, 0.f };
    float32x4_t out_vec3 = { 0.f, 0.f, 0.f, 0.f };
    __VOLK_ATTR_ALIGNED(32) float tempBuffer[4];

    for (; number < eighthPoints; number++) {
        in_vec = vld1q_f32((float*)aPtr);
        out_vec0 = vaddq_f32(in_vec, out_vec0);
        aPtr += 2;

        in_vec = vld1q_f32((float*)aPtr);
        out_vec1 = vaddq_f32(in_vec, out_vec1);
        aPtr += 2;

        in_vec = vld1q_f32((float*)aPtr);
        out_vec2 = vaddq_f32(in_vec, out_vec2);
        aPtr += 2;

        in_vec = vld1q_f32((float*)aPtr);
        out_vec3 = vaddq_f32(in_vec, out_vec3);
        aPtr += 2;
    }
    vst1q_f32(tempBuffer, out_vec0);
    returnValue = lv_cmake(tempBuffer[0], tempBuffer[1]);
    returnValue += lv_cmake(tempBuffer[2], tempBuffer[3]);

    vst1q_f32(tempBuffer, out_vec1);
    returnValue += lv_cmake(tempBuffer[0], tempBuffer[1]);
    returnValue += lv_cmake(tempBuffer[2], tempBuffer[3]);

    vst1q_f32(tempBuffer, out_vec2);
    returnValue += lv_cmake(tempBuffer[0], tempBuffer[1]);
    returnValue += lv_cmake(tempBuffer[2], tempBuffer[3]);

    vst1q_f32(tempBuffer, out_vec3);
    returnValue += lv_cmake(tempBuffer[0], tempBuffer[1]);
    returnValue += lv_cmake(tempBuffer[2], tempBuffer[3]);

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        returnValue += (*aPtr++);
    }
    *result = returnValue;
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32fc_accumulator_s32fc_neonv8(lv_32fc_t* result,
                                                      const lv_32fc_t* inputBuffer,
                                                      unsigned int num_points)
{
    const lv_32fc_t* aPtr = inputBuffer;
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    /* Keep interleaved like neon version - vld1q is faster than vld2q */
    float32x4_t in_vec;
    float32x4_t out_vec0 = vdupq_n_f32(0.f);
    float32x4_t out_vec1 = vdupq_n_f32(0.f);
    float32x4_t out_vec2 = vdupq_n_f32(0.f);
    float32x4_t out_vec3 = vdupq_n_f32(0.f);

    for (; number < eighthPoints; number++) {
        in_vec = vld1q_f32((float*)aPtr);
        out_vec0 = vaddq_f32(in_vec, out_vec0);
        aPtr += 2;

        in_vec = vld1q_f32((float*)aPtr);
        out_vec1 = vaddq_f32(in_vec, out_vec1);
        aPtr += 2;

        in_vec = vld1q_f32((float*)aPtr);
        out_vec2 = vaddq_f32(in_vec, out_vec2);
        aPtr += 2;

        in_vec = vld1q_f32((float*)aPtr);
        out_vec3 = vaddq_f32(in_vec, out_vec3);
        aPtr += 2;
    }

    /* Combine the 4 accumulators */
    out_vec0 = vaddq_f32(out_vec0, out_vec1);
    out_vec2 = vaddq_f32(out_vec2, out_vec3);
    out_vec0 = vaddq_f32(out_vec0, out_vec2);

    /* Horizontal reduction: out_vec0 = [sum_r0, sum_i0, sum_r1, sum_i1] */
    /* We need real = sum_r0 + sum_r1, imag = sum_i0 + sum_i1 */
    float32x2_t low = vget_low_f32(out_vec0);   /* [sum_r0, sum_i0] */
    float32x2_t high = vget_high_f32(out_vec0); /* [sum_r1, sum_i1] */
    float32x2_t sum = vadd_f32(low, high);      /* [real_sum, imag_sum] */

    lv_32fc_t returnValue = lv_cmake(vget_lane_f32(sum, 0), vget_lane_f32(sum, 1));

    /* Tail case */
    for (number = eighthPoints * 8; number < num_points; number++) {
        returnValue += (*aPtr++);
    }

    *result = returnValue;
}

#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>
#include <volk/volk_rvv_intrinsics.h>

static inline void volk_32fc_accumulator_s32fc_rvv(lv_32fc_t* result,
                                                   const lv_32fc_t* inputBuffer,
                                                   unsigned int num_points)
{
    size_t vlmax = __riscv_vsetvlmax_e32m8();
    vfloat32m8_t vsum = __riscv_vfmv_v_f_f32m8(0, vlmax);
    const float* in = (const float*)inputBuffer;
    size_t n = num_points * 2;
    for (size_t vl; n > 0; n -= vl, in += vl) {
        vl = __riscv_vsetvl_e32m8(n < vlmax ? n : vlmax); /* force exact vl */
        vfloat32m8_t v = __riscv_vle32_v_f32m8(in, vl);
        vsum = __riscv_vfadd_tu(vsum, vsum, v, vl);
    }
    vuint64m8_t vsumu = __riscv_vreinterpret_u64m8(__riscv_vreinterpret_u32m8(vsum));
    vfloat32m4_t vsum1 = __riscv_vreinterpret_f32m4(__riscv_vnsrl(vsumu, 0, vlmax));
    vfloat32m4_t vsum2 = __riscv_vreinterpret_f32m4(__riscv_vnsrl(vsumu, 32, vlmax));
    vlmax = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t vr = RISCV_SHRINK4(vfadd, f, 32, vsum1);
    vfloat32m1_t vi = RISCV_SHRINK4(vfadd, f, 32, vsum2);
    vfloat32m1_t z = __riscv_vfmv_s_f_f32m1(0, vlmax);
    *result = lv_cmake(__riscv_vfmv_f(__riscv_vfredusum(vr, z, vlmax)),
                       __riscv_vfmv_f(__riscv_vfredusum(vi, z, vlmax)));
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32fc_accumulator_s32fc_a_H */
