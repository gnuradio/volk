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

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32f_s32f_s32f_mod_range_32f_neon(float* outputVector,
                                                         const float* inputVector,
                                                         const float lower_bound,
                                                         const float upper_bound,
                                                         unsigned int num_points)
{
    const float32x4_t lower = vdupq_n_f32(lower_bound);
    const float32x4_t upper = vdupq_n_f32(upper_bound);
    const float32x4_t distance = vsubq_f32(upper, lower);
    const float32x4_t fone = vdupq_n_f32(1.0f);
    const float32x4_t fmone = vdupq_n_f32(-1.0f);

    const float* inPtr = inputVector;
    float* outPtr = outputVector;
    const size_t quarter_points = num_points / 4;

    for (size_t counter = 0; counter < quarter_points; counter++) {
        float32x4_t input = vld1q_f32(inPtr);

        // Calculate masks
        uint32x4_t is_smaller = vcltq_f32(input, lower);
        uint32x4_t is_bigger = vcgtq_f32(input, upper);

        // Find excess (positive values for both cases)
        float32x4_t excess_low = vsubq_f32(lower, input);
        float32x4_t excess_high = vsubq_f32(input, upper);
        float32x4_t excess = vbslq_f32(is_smaller, excess_low, vdupq_n_f32(0.0f));
        excess = vbslq_f32(is_bigger, excess_high, excess);

        // Calculate count: int(excess / distance)
        float32x4_t recip = vrecpeq_f32(distance);
        recip = vmulq_f32(recip, vrecpsq_f32(distance, recip));
        excess = vmulq_f32(excess, recip);

        // Truncate to integer and back to float
        int32x4_t excess_int = vcvtq_s32_f32(excess);
        excess = vcvtq_f32_s32(excess_int);

        // Add 1
        excess = vaddq_f32(excess, fone);

        // Get sign adjustment
        float32x4_t adj = vbslq_f32(is_smaller, fone, vdupq_n_f32(0.0f));
        adj = vbslq_f32(is_bigger, fmone, adj);

        // Scale by distance and sign
        excess = vmulq_f32(vmulq_f32(excess, adj), distance);
        float32x4_t output = vaddq_f32(input, excess);

        vst1q_f32(outPtr, output);
        inPtr += 4;
        outPtr += 4;
    }

    volk_32f_s32f_s32f_mod_range_32f_generic(
        outPtr, inPtr, lower_bound, upper_bound, num_points - quarter_points * 4);
}

#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32f_s32f_s32f_mod_range_32f_neonv8(float* outputVector,
                                                           const float* inputVector,
                                                           const float lower_bound,
                                                           const float upper_bound,
                                                           unsigned int num_points)
{
    const float32x4_t lower = vdupq_n_f32(lower_bound);
    const float32x4_t upper = vdupq_n_f32(upper_bound);
    const float32x4_t distance = vsubq_f32(upper, lower);
    const float32x4_t fone = vdupq_n_f32(1.0f);
    const float32x4_t fmone = vdupq_n_f32(-1.0f);

    const float* inPtr = inputVector;
    float* outPtr = outputVector;
    const size_t eighth_points = num_points / 8;

    for (size_t counter = 0; counter < eighth_points; counter++) {
        __VOLK_PREFETCH(inPtr + 16);

        float32x4_t input0 = vld1q_f32(inPtr);
        float32x4_t input1 = vld1q_f32(inPtr + 4);

        // Calculate masks
        uint32x4_t is_smaller0 = vcltq_f32(input0, lower);
        uint32x4_t is_smaller1 = vcltq_f32(input1, lower);
        uint32x4_t is_bigger0 = vcgtq_f32(input0, upper);
        uint32x4_t is_bigger1 = vcgtq_f32(input1, upper);

        // Find excess
        float32x4_t excess_low0 = vsubq_f32(lower, input0);
        float32x4_t excess_low1 = vsubq_f32(lower, input1);
        float32x4_t excess_high0 = vsubq_f32(input0, upper);
        float32x4_t excess_high1 = vsubq_f32(input1, upper);
        float32x4_t excess0 = vbslq_f32(is_smaller0, excess_low0, vdupq_n_f32(0.0f));
        float32x4_t excess1 = vbslq_f32(is_smaller1, excess_low1, vdupq_n_f32(0.0f));
        excess0 = vbslq_f32(is_bigger0, excess_high0, excess0);
        excess1 = vbslq_f32(is_bigger1, excess_high1, excess1);

        // Calculate count using division
        excess0 = vdivq_f32(excess0, distance);
        excess1 = vdivq_f32(excess1, distance);

        // Truncate to integer and back to float
        int32x4_t excess_int0 = vcvtq_s32_f32(excess0);
        int32x4_t excess_int1 = vcvtq_s32_f32(excess1);
        excess0 = vcvtq_f32_s32(excess_int0);
        excess1 = vcvtq_f32_s32(excess_int1);

        // Add 1
        excess0 = vaddq_f32(excess0, fone);
        excess1 = vaddq_f32(excess1, fone);

        // Get sign adjustment
        float32x4_t adj0 = vbslq_f32(is_smaller0, fone, vdupq_n_f32(0.0f));
        float32x4_t adj1 = vbslq_f32(is_smaller1, fone, vdupq_n_f32(0.0f));
        adj0 = vbslq_f32(is_bigger0, fmone, adj0);
        adj1 = vbslq_f32(is_bigger1, fmone, adj1);

        // Scale by distance and sign
        excess0 = vmulq_f32(vmulq_f32(excess0, adj0), distance);
        excess1 = vmulq_f32(vmulq_f32(excess1, adj1), distance);
        float32x4_t output0 = vaddq_f32(input0, excess0);
        float32x4_t output1 = vaddq_f32(input1, excess1);

        vst1q_f32(outPtr, output0);
        vst1q_f32(outPtr + 4, output1);
        inPtr += 8;
        outPtr += 8;
    }

    volk_32f_s32f_s32f_mod_range_32f_generic(
        outPtr, inPtr, lower_bound, upper_bound, num_points - eighth_points * 8);
}

#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32f_s32f_s32f_mod_range_32f_rvv(float* outputVector,
                                                        const float* inputVector,
                                                        const float lower_bound,
                                                        const float upper_bound,
                                                        unsigned int num_points)
{
    const float dist = upper_bound - lower_bound;
    size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t vdist = __riscv_vfmv_v_f_f32m4(dist, vlmax);
    vfloat32m4_t vmdist = __riscv_vfmv_v_f_f32m4(-dist, vlmax);
    vfloat32m4_t vupper = __riscv_vfmv_v_f_f32m4(upper_bound, vlmax);
    vfloat32m4_t vlower = __riscv_vfmv_v_f_f32m4(lower_bound, vlmax);
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, outputVector += vl, inputVector += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4_t v = __riscv_vle32_v_f32m4(inputVector, vl);
        vfloat32m4_t vlt = __riscv_vfsub(vlower, v, vl);
        vfloat32m4_t vgt = __riscv_vfsub(v, vupper, vl);
        vbool8_t mlt = __riscv_vmflt(v, vlower, vl);
        vfloat32m4_t vmul = __riscv_vmerge(vmdist, vdist, mlt, vl);
        vfloat32m4_t vcnt = __riscv_vfdiv(__riscv_vmerge(vgt, vlt, mlt, vl), vdist, vl);
        vcnt = __riscv_vfcvt_f(__riscv_vadd(__riscv_vfcvt_rtz_x(vcnt, vl), 1, vl), vl);
        vbool8_t mgt = __riscv_vmfgt(v, vupper, vl);
        v = __riscv_vfmacc_mu(__riscv_vmor(mlt, mgt, vl), v, vcnt, vmul, vl);

        __riscv_vse32(outputVector, v, vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_VOLK_32F_S32F_S32F_MOD_RANGE_32F_A_H */
