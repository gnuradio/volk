/* -*- c++ -*- */
/*
 * Copyright 2017 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_s32f_s32f_mod_range_32f
 *
 * \b wraps floating point numbers to stay within a defined [min,max] range
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_s32f_s32f_mod_range_32f(float* outputVector, const float* inputVector,
 * const float lower_bound, const float upper_bound, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li inputVector: The input vector
 * \li lower_bound: The lower output boundary
 * \li upper_bound: The upper output boundary
 * \li num_points The number of data points.
 *
 * \b Outputs
 * \li outputVector: The vector where the results will be stored.
 *
 * \endcode
 */

#ifndef INCLUDED_VOLK_32F_S32F_S32F_MOD_RANGE_32F_A_H
#define INCLUDED_VOLK_32F_S32F_S32F_MOD_RANGE_32F_A_H

#ifdef LV_HAVE_GENERIC

static inline void volk_32f_s32f_s32f_mod_range_32f_generic(float* outputVector,
                                                            const float* inputVector,
                                                            const float lower_bound,
                                                            const float upper_bound,
                                                            unsigned int num_points)
{
    float* outPtr = outputVector;
    const float* inPtr;
    const float distance = upper_bound - lower_bound;

    for (inPtr = inputVector; inPtr < inputVector + num_points; inPtr++) {
        float val = *inPtr;
        if (val < lower_bound) {
            float excess = lower_bound - val;
            signed int count = (int)(excess / distance);
            *outPtr = val + (count + 1) * distance;
        } else if (val > upper_bound) {
            float excess = val - upper_bound;
            signed int count = (int)(excess / distance);
            *outPtr = val - (count + 1) * distance;
        } else
            *outPtr = val;
        outPtr++;
    }
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_AVX
#include <xmmintrin.h>

static inline void volk_32f_s32f_s32f_mod_range_32f_u_avx(float* outputVector,
                                                          const float* inputVector,
                                                          const float lower_bound,
                                                          const float upper_bound,
                                                          unsigned int num_points)
{
    const __m256 lower = _mm256_set1_ps(lower_bound);
    const __m256 upper = _mm256_set1_ps(upper_bound);
    const __m256 distance = _mm256_sub_ps(upper, lower);
    __m256 input, output;
    __m256 is_smaller, is_bigger;
    __m256 excess, adj;

    const float* inPtr = inputVector;
    float* outPtr = outputVector;
    const size_t eight_points = num_points / 8;
    for (size_t counter = 0; counter < eight_points; counter++) {
        input = _mm256_loadu_ps(inPtr);
        // calculate mask: input < lower, input > upper
        is_smaller = _mm256_cmp_ps(
            input, lower, _CMP_LT_OQ); // 0x11: Less than, ordered, non-signalling
        is_bigger = _mm256_cmp_ps(
            input, upper, _CMP_GT_OQ); // 0x1e: greater than, ordered, non-signalling
        // find out how far we are out-of-bound – positive values!
        excess = _mm256_and_ps(_mm256_sub_ps(lower, input), is_smaller);
        excess =
            _mm256_or_ps(_mm256_and_ps(_mm256_sub_ps(input, upper), is_bigger), excess);
        // how many do we have to add? (int(excess/distance+1)*distance)
        excess = _mm256_div_ps(excess, distance);
        // round down
        excess = _mm256_cvtepi32_ps(_mm256_cvttps_epi32(excess));
        // plus 1
        adj = _mm256_set1_ps(1.0f);
        excess = _mm256_add_ps(excess, adj);
        // get the sign right, adj is still {1.0f,1.0f,1.0f,1.0f}
        adj = _mm256_and_ps(adj, is_smaller);
        adj = _mm256_or_ps(_mm256_and_ps(_mm256_set1_ps(-1.0f), is_bigger), adj);
        // scale by distance, sign
        excess = _mm256_mul_ps(_mm256_mul_ps(excess, adj), distance);
        output = _mm256_add_ps(input, excess);
        _mm256_storeu_ps(outPtr, output);
        inPtr += 8;
        outPtr += 8;
    }

    volk_32f_s32f_s32f_mod_range_32f_generic(
        outPtr, inPtr, lower_bound, upper_bound, num_points - eight_points * 8);
}
static inline void volk_32f_s32f_s32f_mod_range_32f_a_avx(float* outputVector,
                                                          const float* inputVector,
                                                          const float lower_bound,
                                                          const float upper_bound,
                                                          unsigned int num_points)
{
    const __m256 lower = _mm256_set1_ps(lower_bound);
    const __m256 upper = _mm256_set1_ps(upper_bound);
    const __m256 distance = _mm256_sub_ps(upper, lower);
    __m256 input, output;
    __m256 is_smaller, is_bigger;
    __m256 excess, adj;

    const float* inPtr = inputVector;
    float* outPtr = outputVector;
    const size_t eight_points = num_points / 8;
    for (size_t counter = 0; counter < eight_points; counter++) {
        input = _mm256_load_ps(inPtr);
        // calculate mask: input < lower, input > upper
        is_smaller = _mm256_cmp_ps(
            input, lower, _CMP_LT_OQ); // 0x11: Less than, ordered, non-signalling
        is_bigger = _mm256_cmp_ps(
            input, upper, _CMP_GT_OQ); // 0x1e: greater than, ordered, non-signalling
        // find out how far we are out-of-bound – positive values!
        excess = _mm256_and_ps(_mm256_sub_ps(lower, input), is_smaller);
        excess =
            _mm256_or_ps(_mm256_and_ps(_mm256_sub_ps(input, upper), is_bigger), excess);
        // how many do we have to add? (int(excess/distance+1)*distance)
        excess = _mm256_div_ps(excess, distance);
        // round down
        excess = _mm256_cvtepi32_ps(_mm256_cvttps_epi32(excess));
        // plus 1
        adj = _mm256_set1_ps(1.0f);
        excess = _mm256_add_ps(excess, adj);
        // get the sign right, adj is still {1.0f,1.0f,1.0f,1.0f}
        adj = _mm256_and_ps(adj, is_smaller);
        adj = _mm256_or_ps(_mm256_and_ps(_mm256_set1_ps(-1.0f), is_bigger), adj);
        // scale by distance, sign
        excess = _mm256_mul_ps(_mm256_mul_ps(excess, adj), distance);
        output = _mm256_add_ps(input, excess);
        _mm256_store_ps(outPtr, output);
        inPtr += 8;
        outPtr += 8;
    }

    volk_32f_s32f_s32f_mod_range_32f_generic(
        outPtr, inPtr, lower_bound, upper_bound, num_points - eight_points * 8);
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_SSE2
#include <xmmintrin.h>

static inline void volk_32f_s32f_s32f_mod_range_32f_u_sse2(float* outputVector,
                                                           const float* inputVector,
                                                           const float lower_bound,
                                                           const float upper_bound,
                                                           unsigned int num_points)
{
    const __m128 lower = _mm_set_ps1(lower_bound);
    const __m128 upper = _mm_set_ps1(upper_bound);
    const __m128 distance = _mm_sub_ps(upper, lower);
    __m128 input, output;
    __m128 is_smaller, is_bigger;
    __m128 excess, adj;

    const float* inPtr = inputVector;
    float* outPtr = outputVector;
    const size_t quarter_points = num_points / 4;
    for (size_t counter = 0; counter < quarter_points; counter++) {
        input = _mm_loadu_ps(inPtr);
        // calculate mask: input < lower, input > upper
        is_smaller = _mm_cmplt_ps(input, lower);
        is_bigger = _mm_cmpgt_ps(input, upper);
        // find out how far we are out-of-bound – positive values!
        excess = _mm_and_ps(_mm_sub_ps(lower, input), is_smaller);
        excess = _mm_or_ps(_mm_and_ps(_mm_sub_ps(input, upper), is_bigger), excess);
        // how many do we have to add? (int(excess/distance+1)*distance)
        excess = _mm_div_ps(excess, distance);
        // round down
        excess = _mm_cvtepi32_ps(_mm_cvttps_epi32(excess));
        // plus 1
        adj = _mm_set_ps1(1.0f);
        excess = _mm_add_ps(excess, adj);
        // get the sign right, adj is still {1.0f,1.0f,1.0f,1.0f}
        adj = _mm_and_ps(adj, is_smaller);
        adj = _mm_or_ps(_mm_and_ps(_mm_set_ps1(-1.0f), is_bigger), adj);
        // scale by distance, sign
        excess = _mm_mul_ps(_mm_mul_ps(excess, adj), distance);
        output = _mm_add_ps(input, excess);
        _mm_storeu_ps(outPtr, output);
        inPtr += 4;
        outPtr += 4;
    }

    volk_32f_s32f_s32f_mod_range_32f_generic(
        outPtr, inPtr, lower_bound, upper_bound, num_points - quarter_points * 4);
}
static inline void volk_32f_s32f_s32f_mod_range_32f_a_sse2(float* outputVector,
                                                           const float* inputVector,
                                                           const float lower_bound,
                                                           const float upper_bound,
                                                           unsigned int num_points)
{
    const __m128 lower = _mm_set_ps1(lower_bound);
    const __m128 upper = _mm_set_ps1(upper_bound);
    const __m128 distance = _mm_sub_ps(upper, lower);
    __m128 input, output;
    __m128 is_smaller, is_bigger;
    __m128 excess, adj;

    const float* inPtr = inputVector;
    float* outPtr = outputVector;
    const size_t quarter_points = num_points / 4;
    for (size_t counter = 0; counter < quarter_points; counter++) {
        input = _mm_load_ps(inPtr);
        // calculate mask: input < lower, input > upper
        is_smaller = _mm_cmplt_ps(input, lower);
        is_bigger = _mm_cmpgt_ps(input, upper);
        // find out how far we are out-of-bound – positive values!
        excess = _mm_and_ps(_mm_sub_ps(lower, input), is_smaller);
        excess = _mm_or_ps(_mm_and_ps(_mm_sub_ps(input, upper), is_bigger), excess);
        // how many do we have to add? (int(excess/distance+1)*distance)
        excess = _mm_div_ps(excess, distance);
        // round down – for some reason, SSE doesn't come with a 4x float -> 4x int32
        // conversion.
        excess = _mm_cvtepi32_ps(_mm_cvttps_epi32(excess));
        // plus 1
        adj = _mm_set_ps1(1.0f);
        excess = _mm_add_ps(excess, adj);
        // get the sign right, adj is still {1.0f,1.0f,1.0f,1.0f}
        adj = _mm_and_ps(adj, is_smaller);
        adj = _mm_or_ps(_mm_and_ps(_mm_set_ps1(-1.0f), is_bigger), adj);
        // scale by distance, sign
        excess = _mm_mul_ps(_mm_mul_ps(excess, adj), distance);
        output = _mm_add_ps(input, excess);
        _mm_store_ps(outPtr, output);
        inPtr += 4;
        outPtr += 4;
    }

    volk_32f_s32f_s32f_mod_range_32f_generic(
        outPtr, inPtr, lower_bound, upper_bound, num_points - quarter_points * 4);
}
#endif /* LV_HAVE_SSE2 */

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_s32f_s32f_mod_range_32f_u_sse(float* outputVector,
                                                          const float* inputVector,
                                                          const float lower_bound,
                                                          const float upper_bound,
                                                          unsigned int num_points)
{
    const __m128 lower = _mm_set_ps1(lower_bound);
    const __m128 upper = _mm_set_ps1(upper_bound);
    const __m128 distance = _mm_sub_ps(upper, lower);
    __m128 input, output;
    __m128 is_smaller, is_bigger;
    __m128 excess, adj;
    __m128i rounddown;

    const float* inPtr = inputVector;
    float* outPtr = outputVector;
    const size_t quarter_points = num_points / 4;
    for (size_t counter = 0; counter < quarter_points; counter++) {
        input = _mm_loadu_ps(inPtr);
        // calculate mask: input < lower, input > upper
        is_smaller = _mm_cmplt_ps(input, lower);
        is_bigger = _mm_cmpgt_ps(input, upper);
        // find out how far we are out-of-bound – positive values!
        excess = _mm_and_ps(_mm_sub_ps(lower, input), is_smaller);
        excess = _mm_or_ps(_mm_and_ps(_mm_sub_ps(input, upper), is_bigger), excess);
        // how many do we have to add? (int(excess/distance+1)*distance)
        excess = _mm_div_ps(excess, distance);
        // round down – for some reason
        rounddown = _mm_cvttps_epi32(excess);
        excess = _mm_cvtepi32_ps(rounddown);
        // plus 1
        adj = _mm_set_ps1(1.0f);
        excess = _mm_add_ps(excess, adj);
        // get the sign right, adj is still {1.0f,1.0f,1.0f,1.0f}
        adj = _mm_and_ps(adj, is_smaller);
        adj = _mm_or_ps(_mm_and_ps(_mm_set_ps1(-1.0f), is_bigger), adj);
        // scale by distance, sign
        excess = _mm_mul_ps(_mm_mul_ps(excess, adj), distance);
        output = _mm_add_ps(input, excess);
        _mm_storeu_ps(outPtr, output);
        inPtr += 4;
        outPtr += 4;
    }

    volk_32f_s32f_s32f_mod_range_32f_generic(
        outPtr, inPtr, lower_bound, upper_bound, num_points - quarter_points * 4);
}
static inline void volk_32f_s32f_s32f_mod_range_32f_a_sse(float* outputVector,
                                                          const float* inputVector,
                                                          const float lower_bound,
                                                          const float upper_bound,
                                                          unsigned int num_points)
{
    const __m128 lower = _mm_set_ps1(lower_bound);
    const __m128 upper = _mm_set_ps1(upper_bound);
    const __m128 distance = _mm_sub_ps(upper, lower);
    __m128 input, output;
    __m128 is_smaller, is_bigger;
    __m128 excess, adj;
    __m128i rounddown;

    const float* inPtr = inputVector;
    float* outPtr = outputVector;
    const size_t quarter_points = num_points / 4;
    for (size_t counter = 0; counter < quarter_points; counter++) {
        input = _mm_load_ps(inPtr);
        // calculate mask: input < lower, input > upper
        is_smaller = _mm_cmplt_ps(input, lower);
        is_bigger = _mm_cmpgt_ps(input, upper);
        // find out how far we are out-of-bound – positive values!
        excess = _mm_and_ps(_mm_sub_ps(lower, input), is_smaller);
        excess = _mm_or_ps(_mm_and_ps(_mm_sub_ps(input, upper), is_bigger), excess);
        // how many do we have to add? (int(excess/distance+1)*distance)
        excess = _mm_div_ps(excess, distance);
        // round down
        rounddown = _mm_cvttps_epi32(excess);
        excess = _mm_cvtepi32_ps(rounddown);
        // plus 1
        adj = _mm_set_ps1(1.0f);
        excess = _mm_add_ps(excess, adj);
        // get the sign right, adj is still {1.0f,1.0f,1.0f,1.0f}
        adj = _mm_and_ps(adj, is_smaller);
        adj = _mm_or_ps(_mm_and_ps(_mm_set_ps1(-1.0f), is_bigger), adj);
        // scale by distance, sign
        excess = _mm_mul_ps(_mm_mul_ps(excess, adj), distance);
        output = _mm_add_ps(input, excess);
        _mm_store_ps(outPtr, output);
        inPtr += 4;
        outPtr += 4;
    }

    volk_32f_s32f_s32f_mod_range_32f_generic(
        outPtr, inPtr, lower_bound, upper_bound, num_points - quarter_points * 4);
}
#endif /* LV_HAVE_SSE */


#endif /* INCLUDED_VOLK_32F_S32F_S32F_MOD_RANGE_32F_A_H */
