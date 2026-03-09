/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_s32f_32f_fm_detect_32f
 *
 * \b Overview
 *
 * Computes the instantaneous frequency of an FM signal by differentiating consecutive
 * phase samples with modular wrapping. Each output sample is the difference between
 * adjacent input samples, wrapped to the interval (-bound, bound]. The saveValue
 * parameter provides continuity across successive calls by storing the last input sample.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_s32f_32f_fm_detect_32f(float* outputVector, const float* inputVector,
 * const float bound, float* saveValue, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li inputVector: The input vector of phase samples as floats.
 * \li bound: The wrapping bound for the phase interval; differences outside
 * (-bound, bound] are wrapped by adding or subtracting 2*bound.
 * \li saveValue: Pointer to a float holding the phase value of the sample before the
 * first input sample (updated to the last input sample on return).
 * \li num_points: The number of input samples.
 *
 * \b Outputs
 * \li outputVector: The vector of wrapped phase differences (instantaneous frequency).
 *
 * \b Example
 * Detect instantaneous frequency from a phase ramp that wraps around PI.
 * \code
 *   #include <volk/volk.h>
 *   #include <stdio.h>
 *   #include <math.h>
 *
 *   unsigned int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   float bound = (float)M_PI;
 *
 *   float* input = (float*)volk_malloc(sizeof(float) * N, alignment);
 *   float* output = (float*)volk_malloc(sizeof(float) * N, alignment);
 *
 *   // Simulate a phase ramp that wraps around PI (as in FM demodulation)
 *   float phase = 0.0f;
 *   float freq = 0.8f; // radians per sample
 *   for (unsigned int i = 0; i < N; i++) {
 *       phase += freq;
 *       // Wrap phase to (-PI, PI]
 *       while (phase > bound) phase -= 2.0f * bound;
 *       while (phase <= -bound) phase += 2.0f * bound;
 *       input[i] = phase;
 *   }
 *
 *   // saveValue holds the "previous" phase sample (0 for the first call)
 *   float saveValue = 0.0f;
 *
 *   volk_32f_s32f_32f_fm_detect_32f(output, input, bound, &saveValue, N);
 *
 *   // Each output should recover the original frequency (~0.8 rad/sample)
 *   for (unsigned int i = 0; i < N; i++) {
 *       printf("output[%u] = %f\n", i, output[i]);
 *   }
 *
 *   volk_free(input);
 *   volk_free(output);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_s32f_32f_fm_detect_32f_a_H
#define INCLUDED_volk_32f_s32f_32f_fm_detect_32f_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_s32f_32f_fm_detect_32f_a_avx(float* outputVector,
                                                         const float* inputVector,
                                                         const float bound,
                                                         float* saveValue,
                                                         unsigned int num_points)
{
    if (num_points < 1) {
        return;
    }
    unsigned int number = 1;
    unsigned int j = 0;
    // num_points-1 keeps Fedora 7's gcc from crashing...
    // num_points won't work.  :(
    const unsigned int eighthPoints = (num_points - 1) / 8;

    float* outPtr = outputVector;
    const float* inPtr = inputVector;
    __m256 upperBound = _mm256_set1_ps(bound);
    __m256 lowerBound = _mm256_set1_ps(-bound);
    __m256 next3old1;
    __m256 next4;
    __m256 boundAdjust;
    __m256 posBoundAdjust = _mm256_set1_ps(-2 * bound); // Subtract when we're above.
    __m256 negBoundAdjust = _mm256_set1_ps(2 * bound);  // Add when we're below.
    // Do the first 8 by hand since we're going in from the saveValue:
    *outPtr = *inPtr - *saveValue;
    if (*outPtr > bound)
        *outPtr -= 2 * bound;
    if (*outPtr < -bound)
        *outPtr += 2 * bound;
    inPtr++;
    outPtr++;
    for (j = 1; j < ((8 < num_points) ? 8 : num_points); j++) {
        *outPtr = *(inPtr) - *(inPtr - 1);
        if (*outPtr > bound)
            *outPtr -= 2 * bound;
        if (*outPtr < -bound)
            *outPtr += 2 * bound;
        inPtr++;
        outPtr++;
    }

    for (; number < eighthPoints; number++) {
        // Load data
        next3old1 = _mm256_loadu_ps((float*)(inPtr - 1));
        next4 = _mm256_load_ps(inPtr);
        inPtr += 8;
        // Subtract and store:
        next3old1 = _mm256_sub_ps(next4, next3old1);
        // Bound:
        boundAdjust = _mm256_cmp_ps(next3old1, upperBound, _CMP_GT_OS);
        boundAdjust = _mm256_and_ps(boundAdjust, posBoundAdjust);
        next4 = _mm256_cmp_ps(next3old1, lowerBound, _CMP_LT_OS);
        next4 = _mm256_and_ps(next4, negBoundAdjust);
        boundAdjust = _mm256_or_ps(next4, boundAdjust);
        // Make sure we're in the bounding interval:
        next3old1 = _mm256_add_ps(next3old1, boundAdjust);
        _mm256_store_ps(outPtr, next3old1); // Store the results back into the output
        outPtr += 8;
    }

    for (number = (8 > (eighthPoints * 8) ? 8 : (8 * eighthPoints)); number < num_points;
         number++) {
        *outPtr = *(inPtr) - *(inPtr - 1);
        if (*outPtr > bound)
            *outPtr -= 2 * bound;
        if (*outPtr < -bound)
            *outPtr += 2 * bound;
        inPtr++;
        outPtr++;
    }

    *saveValue = inputVector[num_points - 1];
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_s32f_32f_fm_detect_32f_a_sse(float* outputVector,
                                                         const float* inputVector,
                                                         const float bound,
                                                         float* saveValue,
                                                         unsigned int num_points)
{
    if (num_points < 1) {
        return;
    }
    unsigned int number = 1;
    unsigned int j = 0;
    // num_points-1 keeps Fedora 7's gcc from crashing...
    // num_points won't work.  :(
    const unsigned int quarterPoints = (num_points - 1) / 4;

    float* outPtr = outputVector;
    const float* inPtr = inputVector;
    __m128 upperBound = _mm_set_ps1(bound);
    __m128 lowerBound = _mm_set_ps1(-bound);
    __m128 next3old1;
    __m128 next4;
    __m128 boundAdjust;
    __m128 posBoundAdjust = _mm_set_ps1(-2 * bound); // Subtract when we're above.
    __m128 negBoundAdjust = _mm_set_ps1(2 * bound);  // Add when we're below.
    // Do the first 4 by hand since we're going in from the saveValue:
    *outPtr = *inPtr - *saveValue;
    if (*outPtr > bound)
        *outPtr -= 2 * bound;
    if (*outPtr < -bound)
        *outPtr += 2 * bound;
    inPtr++;
    outPtr++;
    for (j = 1; j < ((4 < num_points) ? 4 : num_points); j++) {
        *outPtr = *(inPtr) - *(inPtr - 1);
        if (*outPtr > bound)
            *outPtr -= 2 * bound;
        if (*outPtr < -bound)
            *outPtr += 2 * bound;
        inPtr++;
        outPtr++;
    }

    for (; number < quarterPoints; number++) {
        // Load data
        next3old1 = _mm_loadu_ps((float*)(inPtr - 1));
        next4 = _mm_load_ps(inPtr);
        inPtr += 4;
        // Subtract and store:
        next3old1 = _mm_sub_ps(next4, next3old1);
        // Bound:
        boundAdjust = _mm_cmpgt_ps(next3old1, upperBound);
        boundAdjust = _mm_and_ps(boundAdjust, posBoundAdjust);
        next4 = _mm_cmplt_ps(next3old1, lowerBound);
        next4 = _mm_and_ps(next4, negBoundAdjust);
        boundAdjust = _mm_or_ps(next4, boundAdjust);
        // Make sure we're in the bounding interval:
        next3old1 = _mm_add_ps(next3old1, boundAdjust);
        _mm_store_ps(outPtr, next3old1); // Store the results back into the output
        outPtr += 4;
    }

    for (number = (4 > (quarterPoints * 4) ? 4 : (4 * quarterPoints));
         number < num_points;
         number++) {
        *outPtr = *(inPtr) - *(inPtr - 1);
        if (*outPtr > bound)
            *outPtr -= 2 * bound;
        if (*outPtr < -bound)
            *outPtr += 2 * bound;
        inPtr++;
        outPtr++;
    }

    *saveValue = inputVector[num_points - 1];
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_GENERIC

static inline void volk_32f_s32f_32f_fm_detect_32f_generic(float* outputVector,
                                                           const float* inputVector,
                                                           const float bound,
                                                           float* saveValue,
                                                           unsigned int num_points)
{
    if (num_points < 1) {
        return;
    }
    unsigned int number = 0;
    float* outPtr = outputVector;
    const float* inPtr = inputVector;

    // Do the first 1 by hand since we're going in from the saveValue:
    *outPtr = *inPtr - *saveValue;
    if (*outPtr > bound)
        *outPtr -= 2 * bound;
    if (*outPtr < -bound)
        *outPtr += 2 * bound;
    inPtr++;
    outPtr++;

    for (number = 1; number < num_points; number++) {
        *outPtr = *(inPtr) - *(inPtr - 1);
        if (*outPtr > bound)
            *outPtr -= 2 * bound;
        if (*outPtr < -bound)
            *outPtr += 2 * bound;
        inPtr++;
        outPtr++;
    }

    *saveValue = inputVector[num_points - 1];
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32f_s32f_32f_fm_detect_32f_neon(float* outputVector,
                                                        const float* inputVector,
                                                        const float bound,
                                                        float* saveValue,
                                                        unsigned int num_points)
{
    if (num_points < 1) {
        return;
    }

    float* outPtr = outputVector;
    const float* inPtr = inputVector;

    const float32x4_t upperBound = vdupq_n_f32(bound);
    const float32x4_t lowerBound = vdupq_n_f32(-bound);
    const float32x4_t posBoundAdjust = vdupq_n_f32(-2.f * bound);
    const float32x4_t negBoundAdjust = vdupq_n_f32(2.f * bound);

    // Do the first element from saveValue
    *outPtr = *inPtr - *saveValue;
    if (*outPtr > bound)
        *outPtr -= 2 * bound;
    if (*outPtr < -bound)
        *outPtr += 2 * bound;
    inPtr++;
    outPtr++;

    // Do the next 3 elements to align to 4
    for (unsigned int j = 1; j < ((4 < num_points) ? 4 : num_points); j++) {
        *outPtr = *inPtr - *(inPtr - 1);
        if (*outPtr > bound)
            *outPtr -= 2 * bound;
        if (*outPtr < -bound)
            *outPtr += 2 * bound;
        inPtr++;
        outPtr++;
    }

    const unsigned int quarterPoints = (num_points - 1) / 4;
    for (unsigned int number = 1; number < quarterPoints; number++) {
        // Load current and previous (offset by 1)
        float32x4_t curr = vld1q_f32(inPtr);
        float32x4_t prev = vld1q_f32(inPtr - 1);
        inPtr += 4;

        // Compute difference
        float32x4_t diff = vsubq_f32(curr, prev);

        // Apply bound wrapping
        uint32x4_t aboveMask = vcgtq_f32(diff, upperBound);
        uint32x4_t belowMask = vcltq_f32(diff, lowerBound);

        float32x4_t adjust = vbslq_f32(aboveMask, posBoundAdjust, vdupq_n_f32(0));
        adjust = vbslq_f32(belowMask, negBoundAdjust, adjust);

        diff = vaddq_f32(diff, adjust);

        vst1q_f32(outPtr, diff);
        outPtr += 4;
    }

    // Handle remainder
    for (unsigned int number = (4 > (quarterPoints * 4) ? 4 : (4 * quarterPoints));
         number < num_points;
         number++) {
        *outPtr = *inPtr - *(inPtr - 1);
        if (*outPtr > bound)
            *outPtr -= 2 * bound;
        if (*outPtr < -bound)
            *outPtr += 2 * bound;
        inPtr++;
        outPtr++;
    }

    *saveValue = inputVector[num_points - 1];
}
#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32f_s32f_32f_fm_detect_32f_neonv8(float* outputVector,
                                                          const float* inputVector,
                                                          const float bound,
                                                          float* saveValue,
                                                          unsigned int num_points)
{
    if (num_points < 1) {
        return;
    }

    float* outPtr = outputVector;
    const float* inPtr = inputVector;

    const float32x4_t upperBound = vdupq_n_f32(bound);
    const float32x4_t lowerBound = vdupq_n_f32(-bound);
    const float32x4_t posBoundAdjust = vdupq_n_f32(-2.f * bound);
    const float32x4_t negBoundAdjust = vdupq_n_f32(2.f * bound);
    const float32x4_t zeros = vdupq_n_f32(0);

    /* Do the first element from saveValue */
    *outPtr = *inPtr - *saveValue;
    if (*outPtr > bound)
        *outPtr -= 2 * bound;
    if (*outPtr < -bound)
        *outPtr += 2 * bound;
    inPtr++;
    outPtr++;

    /* Do the next 7 elements to align to 8 */
    for (unsigned int j = 1; j < ((8 < num_points) ? 8 : num_points); j++) {
        *outPtr = *inPtr - *(inPtr - 1);
        if (*outPtr > bound)
            *outPtr -= 2 * bound;
        if (*outPtr < -bound)
            *outPtr += 2 * bound;
        inPtr++;
        outPtr++;
    }

    /* Process 8 floats per iteration (2x unroll) */
    const unsigned int eighthPoints = (num_points - 1) / 8;
    for (unsigned int number = 1; number < eighthPoints; number++) {
        /* Load current and previous (offset by 1) */
        float32x4_t curr0 = vld1q_f32(inPtr);
        float32x4_t prev0 = vld1q_f32(inPtr - 1);
        float32x4_t curr1 = vld1q_f32(inPtr + 4);
        float32x4_t prev1 = vld1q_f32(inPtr + 3);
        __VOLK_PREFETCH(inPtr + 16);
        inPtr += 8;

        /* Compute differences */
        float32x4_t diff0 = vsubq_f32(curr0, prev0);
        float32x4_t diff1 = vsubq_f32(curr1, prev1);

        /* Apply bound wrapping for first 4 */
        uint32x4_t above0 = vcgtq_f32(diff0, upperBound);
        uint32x4_t below0 = vcltq_f32(diff0, lowerBound);
        float32x4_t adj0 = vbslq_f32(above0, posBoundAdjust, zeros);
        adj0 = vbslq_f32(below0, negBoundAdjust, adj0);
        diff0 = vaddq_f32(diff0, adj0);

        /* Apply bound wrapping for second 4 */
        uint32x4_t above1 = vcgtq_f32(diff1, upperBound);
        uint32x4_t below1 = vcltq_f32(diff1, lowerBound);
        float32x4_t adj1 = vbslq_f32(above1, posBoundAdjust, zeros);
        adj1 = vbslq_f32(below1, negBoundAdjust, adj1);
        diff1 = vaddq_f32(diff1, adj1);

        vst1q_f32(outPtr, diff0);
        vst1q_f32(outPtr + 4, diff1);
        outPtr += 8;
    }

    /* Handle remainder */
    for (unsigned int number = (8 > (eighthPoints * 8) ? 8 : (8 * eighthPoints));
         number < num_points;
         number++) {
        *outPtr = *inPtr - *(inPtr - 1);
        if (*outPtr > bound)
            *outPtr -= 2 * bound;
        if (*outPtr < -bound)
            *outPtr += 2 * bound;
        inPtr++;
        outPtr++;
    }

    *saveValue = inputVector[num_points - 1];
}
#endif /* LV_HAVE_NEONV8 */

#endif /* INCLUDED_volk_32f_s32f_32f_fm_detect_32f_a_H */


#ifndef INCLUDED_volk_32f_s32f_32f_fm_detect_32f_u_H
#define INCLUDED_volk_32f_s32f_32f_fm_detect_32f_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_s32f_32f_fm_detect_32f_u_avx(float* outputVector,
                                                         const float* inputVector,
                                                         const float bound,
                                                         float* saveValue,
                                                         unsigned int num_points)
{
    if (num_points < 1) {
        return;
    }
    unsigned int number = 1;
    unsigned int j = 0;
    // num_points-1 keeps Fedora 7's gcc from crashing...
    // num_points won't work.  :(
    const unsigned int eighthPoints = (num_points - 1) / 8;

    float* outPtr = outputVector;
    const float* inPtr = inputVector;
    __m256 upperBound = _mm256_set1_ps(bound);
    __m256 lowerBound = _mm256_set1_ps(-bound);
    __m256 next3old1;
    __m256 next4;
    __m256 boundAdjust;
    __m256 posBoundAdjust = _mm256_set1_ps(-2 * bound); // Subtract when we're above.
    __m256 negBoundAdjust = _mm256_set1_ps(2 * bound);  // Add when we're below.
    // Do the first 8 by hand since we're going in from the saveValue:
    *outPtr = *inPtr - *saveValue;
    if (*outPtr > bound)
        *outPtr -= 2 * bound;
    if (*outPtr < -bound)
        *outPtr += 2 * bound;
    inPtr++;
    outPtr++;
    for (j = 1; j < ((8 < num_points) ? 8 : num_points); j++) {
        *outPtr = *(inPtr) - *(inPtr - 1);
        if (*outPtr > bound)
            *outPtr -= 2 * bound;
        if (*outPtr < -bound)
            *outPtr += 2 * bound;
        inPtr++;
        outPtr++;
    }

    for (; number < eighthPoints; number++) {
        // Load data
        next3old1 = _mm256_loadu_ps((float*)(inPtr - 1));
        next4 = _mm256_loadu_ps(inPtr);
        inPtr += 8;
        // Subtract and store:
        next3old1 = _mm256_sub_ps(next4, next3old1);
        // Bound:
        boundAdjust = _mm256_cmp_ps(next3old1, upperBound, _CMP_GT_OS);
        boundAdjust = _mm256_and_ps(boundAdjust, posBoundAdjust);
        next4 = _mm256_cmp_ps(next3old1, lowerBound, _CMP_LT_OS);
        next4 = _mm256_and_ps(next4, negBoundAdjust);
        boundAdjust = _mm256_or_ps(next4, boundAdjust);
        // Make sure we're in the bounding interval:
        next3old1 = _mm256_add_ps(next3old1, boundAdjust);
        _mm256_storeu_ps(outPtr, next3old1); // Store the results back into the output
        outPtr += 8;
    }

    for (number = (8 > (eighthPoints * 8) ? 8 : (8 * eighthPoints)); number < num_points;
         number++) {
        *outPtr = *(inPtr) - *(inPtr - 1);
        if (*outPtr > bound)
            *outPtr -= 2 * bound;
        if (*outPtr < -bound)
            *outPtr += 2 * bound;
        inPtr++;
        outPtr++;
    }

    *saveValue = inputVector[num_points - 1];
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32f_s32f_32f_fm_detect_32f_rvv(float* outputVector,
                                                       const float* inputVector,
                                                       const float bound,
                                                       float* saveValue,
                                                       unsigned int num_points)
{
    if (num_points < 1)
        return;

    *outputVector = *inputVector - *saveValue;
    if (*outputVector > bound)
        *outputVector -= 2 * bound;
    if (*outputVector < -bound)
        *outputVector += 2 * bound;
    ++inputVector;
    ++outputVector;

    vfloat32m8_t v2bound = __riscv_vfmv_v_f_f32m8(bound * 2, __riscv_vsetvlmax_e32m8());

    size_t n = num_points - 1;
    for (size_t vl; n > 0; n -= vl, inputVector += vl, outputVector += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t va = __riscv_vle32_v_f32m8(inputVector, vl);
        vfloat32m8_t vb = __riscv_vle32_v_f32m8(inputVector - 1, vl);
        vfloat32m8_t v = __riscv_vfsub(va, vb, vl);
        v = __riscv_vfsub_mu(__riscv_vmfgt(v, bound, vl), v, v, v2bound, vl);
        v = __riscv_vfadd_mu(__riscv_vmflt(v, -bound, vl), v, v, v2bound, vl);
        __riscv_vse32(outputVector, v, vl);
    }

    *saveValue = inputVector[-1];
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32f_s32f_32f_fm_detect_32f_u_H */
