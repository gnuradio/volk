/* -*- c++ -*- */
/*
 * Copyright 2016 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_16ic_x2_dot_prod_16ic
 *
 * \b Overview
 *
 * Multiplies two input complex vectors (16-bit integer each component) and accumulates
 * them, storing the result. Results are saturated so never go beyond the limits of the
 * data type.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_16ic_x2_dot_prod_16ic(lv_16sc_t* result, const lv_16sc_t* in_a, const
 * lv_16sc_t* in_b, unsigned int num_points); \endcode
 *
 * \b Inputs
 * \li in_a:          One of the vectors to be multiplied and accumulated.
 * \li in_b:          The other vector to be multiplied and accumulated.
 * \li num_points:    Number of complex values to be multiplied together, accumulated and
 * stored into \p result
 *
 * \b Outputs
 * \li result:        Value of the accumulated result.
 *
 */

#ifndef INCLUDED_volk_16ic_x2_dot_prod_16ic_H
#define INCLUDED_volk_16ic_x2_dot_prod_16ic_H

#include <volk/saturation_arithmetic.h>
#include <volk/volk_common.h>
#include <volk/volk_complex.h>


#ifdef LV_HAVE_GENERIC

static inline void volk_16ic_x2_dot_prod_16ic_generic(lv_16sc_t* result,
                                                      const lv_16sc_t* in_a,
                                                      const lv_16sc_t* in_b,
                                                      unsigned int num_points)
{
    result[0] = lv_cmake((int16_t)0, (int16_t)0);
    unsigned int n;
    for (n = 0; n < num_points; n++) {
        lv_16sc_t tmp = in_a[n] * in_b[n];
        result[0] = lv_cmake(sat_adds16i(lv_creal(result[0]), lv_creal(tmp)),
                             sat_adds16i(lv_cimag(result[0]), lv_cimag(tmp)));
    }
}

#endif /*LV_HAVE_GENERIC*/


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_16ic_x2_dot_prod_16ic_a_sse2(lv_16sc_t* out,
                                                     const lv_16sc_t* in_a,
                                                     const lv_16sc_t* in_b,
                                                     unsigned int num_points)
{
    lv_16sc_t dotProduct = lv_cmake((int16_t)0, (int16_t)0);

    const unsigned int sse_iters = num_points / 4;
    unsigned int number;

    const lv_16sc_t* _in_a = in_a;
    const lv_16sc_t* _in_b = in_b;
    lv_16sc_t* _out = out;

    if (sse_iters > 0) {
        __m128i a, b, c, c_sr, mask_imag, mask_real, real, imag, imag1, imag2, b_sl, a_sl,
            realcacc, imagcacc;
        __VOLK_ATTR_ALIGNED(16) lv_16sc_t dotProductVector[4];

        realcacc = _mm_setzero_si128();
        imagcacc = _mm_setzero_si128();

        mask_imag = _mm_set_epi8(
            0xFF, 0xFF, 0, 0, 0xFF, 0xFF, 0, 0, 0xFF, 0xFF, 0, 0, 0xFF, 0xFF, 0, 0);
        mask_real = _mm_set_epi8(
            0, 0, 0xFF, 0xFF, 0, 0, 0xFF, 0xFF, 0, 0, 0xFF, 0xFF, 0, 0, 0xFF, 0xFF);

        for (number = 0; number < sse_iters; number++) {
            // a[127:0]=[a3.i,a3.r,a2.i,a2.r,a1.i,a1.r,a0.i,a0.r]
            a = _mm_load_si128(
                (__m128i*)_in_a); // load (2 byte imag, 2 byte real) x 4 into 128 bits reg
            __VOLK_PREFETCH(_in_a + 8);
            b = _mm_load_si128((__m128i*)_in_b);
            __VOLK_PREFETCH(_in_b + 8);
            c = _mm_mullo_epi16(a, b); // a3.i*b3.i, a3.r*b3.r, ....

            c_sr = _mm_srli_si128(c, 2); // Shift a right by imm8 bytes while shifting in
                                         // zeros, and store the results in dst.
            real = _mm_subs_epi16(c, c_sr);

            b_sl = _mm_slli_si128(b, 2); // b3.r, b2.i ....
            a_sl = _mm_slli_si128(a, 2); // a3.r, a2.i ....

            imag1 = _mm_mullo_epi16(a, b_sl); // a3.i*b3.r, ....
            imag2 = _mm_mullo_epi16(b, a_sl); // b3.i*a3.r, ....

            imag = _mm_adds_epi16(imag1, imag2); // with saturation arithmetic!

            realcacc = _mm_adds_epi16(realcacc, real);
            imagcacc = _mm_adds_epi16(imagcacc, imag);

            _in_a += 4;
            _in_b += 4;
        }

        realcacc = _mm_and_si128(realcacc, mask_real);
        imagcacc = _mm_and_si128(imagcacc, mask_imag);

        a = _mm_or_si128(realcacc, imagcacc);

        _mm_store_si128((__m128i*)dotProductVector,
                        a); // Store the results back into the dot product vector

        for (number = 0; number < 4; ++number) {
            dotProduct = lv_cmake(
                sat_adds16i(lv_creal(dotProduct), lv_creal(dotProductVector[number])),
                sat_adds16i(lv_cimag(dotProduct), lv_cimag(dotProductVector[number])));
        }
    }

    for (number = 0; number < (num_points % 4); ++number) {
        lv_16sc_t tmp = (*_in_a++) * (*_in_b++);
        dotProduct = lv_cmake(sat_adds16i(lv_creal(dotProduct), lv_creal(tmp)),
                              sat_adds16i(lv_cimag(dotProduct), lv_cimag(tmp)));
    }

    *_out = dotProduct;
}

#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_16ic_x2_dot_prod_16ic_u_sse2(lv_16sc_t* out,
                                                     const lv_16sc_t* in_a,
                                                     const lv_16sc_t* in_b,
                                                     unsigned int num_points)
{
    lv_16sc_t dotProduct = lv_cmake((int16_t)0, (int16_t)0);

    const unsigned int sse_iters = num_points / 4;

    const lv_16sc_t* _in_a = in_a;
    const lv_16sc_t* _in_b = in_b;
    lv_16sc_t* _out = out;
    unsigned int number;

    if (sse_iters > 0) {
        __m128i a, b, c, c_sr, mask_imag, mask_real, real, imag, imag1, imag2, b_sl, a_sl,
            realcacc, imagcacc, result;
        __VOLK_ATTR_ALIGNED(16) lv_16sc_t dotProductVector[4];

        realcacc = _mm_setzero_si128();
        imagcacc = _mm_setzero_si128();

        mask_imag = _mm_set_epi8(
            0xFF, 0xFF, 0, 0, 0xFF, 0xFF, 0, 0, 0xFF, 0xFF, 0, 0, 0xFF, 0xFF, 0, 0);
        mask_real = _mm_set_epi8(
            0, 0, 0xFF, 0xFF, 0, 0, 0xFF, 0xFF, 0, 0, 0xFF, 0xFF, 0, 0, 0xFF, 0xFF);

        for (number = 0; number < sse_iters; number++) {
            // a[127:0]=[a3.i,a3.r,a2.i,a2.r,a1.i,a1.r,a0.i,a0.r]
            a = _mm_loadu_si128(
                (__m128i*)_in_a); // load (2 byte imag, 2 byte real) x 4 into 128 bits reg
            __VOLK_PREFETCH(_in_a + 8);
            b = _mm_loadu_si128((__m128i*)_in_b);
            __VOLK_PREFETCH(_in_b + 8);
            c = _mm_mullo_epi16(a, b); // a3.i*b3.i, a3.r*b3.r, ....

            c_sr = _mm_srli_si128(c, 2); // Shift a right by imm8 bytes while shifting in
                                         // zeros, and store the results in dst.
            real = _mm_subs_epi16(c, c_sr);

            b_sl = _mm_slli_si128(b, 2); // b3.r, b2.i ....
            a_sl = _mm_slli_si128(a, 2); // a3.r, a2.i ....

            imag1 = _mm_mullo_epi16(a, b_sl); // a3.i*b3.r, ....
            imag2 = _mm_mullo_epi16(b, a_sl); // b3.i*a3.r, ....

            imag = _mm_adds_epi16(imag1, imag2); // with saturation arithmetic!

            realcacc = _mm_adds_epi16(realcacc, real);
            imagcacc = _mm_adds_epi16(imagcacc, imag);

            _in_a += 4;
            _in_b += 4;
        }

        realcacc = _mm_and_si128(realcacc, mask_real);
        imagcacc = _mm_and_si128(imagcacc, mask_imag);

        result = _mm_or_si128(realcacc, imagcacc);

        _mm_storeu_si128((__m128i*)dotProductVector,
                         result); // Store the results back into the dot product vector

        for (number = 0; number < 4; ++number) {
            dotProduct = lv_cmake(
                sat_adds16i(lv_creal(dotProduct), lv_creal(dotProductVector[number])),
                sat_adds16i(lv_cimag(dotProduct), lv_cimag(dotProductVector[number])));
        }
    }

    for (number = 0; number < (num_points % 4); ++number) {
        lv_16sc_t tmp = (*_in_a++) * (*_in_b++);
        dotProduct = lv_cmake(sat_adds16i(lv_creal(dotProduct), lv_creal(tmp)),
                              sat_adds16i(lv_cimag(dotProduct), lv_cimag(tmp)));
    }

    *_out = dotProduct;
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16ic_x2_dot_prod_16ic_u_avx2(lv_16sc_t* out,
                                                     const lv_16sc_t* in_a,
                                                     const lv_16sc_t* in_b,
                                                     unsigned int num_points)
{
    lv_16sc_t dotProduct = lv_cmake((int16_t)0, (int16_t)0);

    const unsigned int avx_iters = num_points / 8;

    const lv_16sc_t* _in_a = in_a;
    const lv_16sc_t* _in_b = in_b;
    lv_16sc_t* _out = out;
    unsigned int number;

    if (avx_iters > 0) {
        __m256i a, b, c, c_sr, mask_imag, mask_real, real, imag, imag1, imag2, b_sl, a_sl,
            realcacc, imagcacc, result;
        __VOLK_ATTR_ALIGNED(32) lv_16sc_t dotProductVector[8];

        realcacc = _mm256_setzero_si256();
        imagcacc = _mm256_setzero_si256();

        mask_imag = _mm256_set_epi8(0xFF,
                                    0xFF,
                                    0,
                                    0,
                                    0xFF,
                                    0xFF,
                                    0,
                                    0,
                                    0xFF,
                                    0xFF,
                                    0,
                                    0,
                                    0xFF,
                                    0xFF,
                                    0,
                                    0,
                                    0xFF,
                                    0xFF,
                                    0,
                                    0,
                                    0xFF,
                                    0xFF,
                                    0,
                                    0,
                                    0xFF,
                                    0xFF,
                                    0,
                                    0,
                                    0xFF,
                                    0xFF,
                                    0,
                                    0);
        mask_real = _mm256_set_epi8(0,
                                    0,
                                    0xFF,
                                    0xFF,
                                    0,
                                    0,
                                    0xFF,
                                    0xFF,
                                    0,
                                    0,
                                    0xFF,
                                    0xFF,
                                    0,
                                    0,
                                    0xFF,
                                    0xFF,
                                    0,
                                    0,
                                    0xFF,
                                    0xFF,
                                    0,
                                    0,
                                    0xFF,
                                    0xFF,
                                    0,
                                    0,
                                    0xFF,
                                    0xFF,
                                    0,
                                    0,
                                    0xFF,
                                    0xFF);

        for (number = 0; number < avx_iters; number++) {
            a = _mm256_loadu_si256((__m256i*)_in_a);
            __VOLK_PREFETCH(_in_a + 16);
            b = _mm256_loadu_si256((__m256i*)_in_b);
            __VOLK_PREFETCH(_in_b + 16);
            c = _mm256_mullo_epi16(a, b);

            c_sr = _mm256_srli_si256(c, 2); // Shift a right by imm8 bytes while shifting
                                            // in zeros, and store the results in dst.
            real = _mm256_subs_epi16(c, c_sr);

            b_sl = _mm256_slli_si256(b, 2);
            a_sl = _mm256_slli_si256(a, 2);

            imag1 = _mm256_mullo_epi16(a, b_sl);
            imag2 = _mm256_mullo_epi16(b, a_sl);

            imag = _mm256_adds_epi16(imag1, imag2); // with saturation arithmetic!

            realcacc = _mm256_adds_epi16(realcacc, real);
            imagcacc = _mm256_adds_epi16(imagcacc, imag);

            _in_a += 8;
            _in_b += 8;
        }

        realcacc = _mm256_and_si256(realcacc, mask_real);
        imagcacc = _mm256_and_si256(imagcacc, mask_imag);

        result = _mm256_or_si256(realcacc, imagcacc);

        _mm256_storeu_si256((__m256i*)dotProductVector,
                            result); // Store the results back into the dot product vector

        for (number = 0; number < 8; ++number) {
            dotProduct = lv_cmake(
                sat_adds16i(lv_creal(dotProduct), lv_creal(dotProductVector[number])),
                sat_adds16i(lv_cimag(dotProduct), lv_cimag(dotProductVector[number])));
        }
    }

    for (number = 0; number < (num_points % 8); ++number) {
        lv_16sc_t tmp = (*_in_a++) * (*_in_b++);
        dotProduct = lv_cmake(sat_adds16i(lv_creal(dotProduct), lv_creal(tmp)),
                              sat_adds16i(lv_cimag(dotProduct), lv_cimag(tmp)));
    }

    *_out = dotProduct;
}
#endif /* LV_HAVE_AVX2 */


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16ic_x2_dot_prod_16ic_a_avx2(lv_16sc_t* out,
                                                     const lv_16sc_t* in_a,
                                                     const lv_16sc_t* in_b,
                                                     unsigned int num_points)
{
    lv_16sc_t dotProduct = lv_cmake((int16_t)0, (int16_t)0);

    const unsigned int avx_iters = num_points / 8;

    const lv_16sc_t* _in_a = in_a;
    const lv_16sc_t* _in_b = in_b;
    lv_16sc_t* _out = out;
    unsigned int number;

    if (avx_iters > 0) {
        __m256i a, b, c, c_sr, mask_imag, mask_real, real, imag, imag1, imag2, b_sl, a_sl,
            realcacc, imagcacc, result;
        __VOLK_ATTR_ALIGNED(32) lv_16sc_t dotProductVector[8];

        realcacc = _mm256_setzero_si256();
        imagcacc = _mm256_setzero_si256();

        mask_imag = _mm256_set_epi8(0xFF,
                                    0xFF,
                                    0,
                                    0,
                                    0xFF,
                                    0xFF,
                                    0,
                                    0,
                                    0xFF,
                                    0xFF,
                                    0,
                                    0,
                                    0xFF,
                                    0xFF,
                                    0,
                                    0,
                                    0xFF,
                                    0xFF,
                                    0,
                                    0,
                                    0xFF,
                                    0xFF,
                                    0,
                                    0,
                                    0xFF,
                                    0xFF,
                                    0,
                                    0,
                                    0xFF,
                                    0xFF,
                                    0,
                                    0);
        mask_real = _mm256_set_epi8(0,
                                    0,
                                    0xFF,
                                    0xFF,
                                    0,
                                    0,
                                    0xFF,
                                    0xFF,
                                    0,
                                    0,
                                    0xFF,
                                    0xFF,
                                    0,
                                    0,
                                    0xFF,
                                    0xFF,
                                    0,
                                    0,
                                    0xFF,
                                    0xFF,
                                    0,
                                    0,
                                    0xFF,
                                    0xFF,
                                    0,
                                    0,
                                    0xFF,
                                    0xFF,
                                    0,
                                    0,
                                    0xFF,
                                    0xFF);

        for (number = 0; number < avx_iters; number++) {
            a = _mm256_load_si256((__m256i*)_in_a);
            __VOLK_PREFETCH(_in_a + 16);
            b = _mm256_load_si256((__m256i*)_in_b);
            __VOLK_PREFETCH(_in_b + 16);
            c = _mm256_mullo_epi16(a, b);

            c_sr = _mm256_srli_si256(c, 2); // Shift a right by imm8 bytes while shifting
                                            // in zeros, and store the results in dst.
            real = _mm256_subs_epi16(c, c_sr);

            b_sl = _mm256_slli_si256(b, 2);
            a_sl = _mm256_slli_si256(a, 2);

            imag1 = _mm256_mullo_epi16(a, b_sl);
            imag2 = _mm256_mullo_epi16(b, a_sl);

            imag = _mm256_adds_epi16(imag1, imag2); // with saturation arithmetic!

            realcacc = _mm256_adds_epi16(realcacc, real);
            imagcacc = _mm256_adds_epi16(imagcacc, imag);

            _in_a += 8;
            _in_b += 8;
        }

        realcacc = _mm256_and_si256(realcacc, mask_real);
        imagcacc = _mm256_and_si256(imagcacc, mask_imag);

        result = _mm256_or_si256(realcacc, imagcacc);

        _mm256_store_si256((__m256i*)dotProductVector,
                           result); // Store the results back into the dot product vector

        for (number = 0; number < 8; ++number) {
            dotProduct = lv_cmake(
                sat_adds16i(lv_creal(dotProduct), lv_creal(dotProductVector[number])),
                sat_adds16i(lv_cimag(dotProduct), lv_cimag(dotProductVector[number])));
        }
    }

    for (number = 0; number < (num_points % 8); ++number) {
        lv_16sc_t tmp = (*_in_a++) * (*_in_b++);
        dotProduct = lv_cmake(sat_adds16i(lv_creal(dotProduct), lv_creal(tmp)),
                              sat_adds16i(lv_cimag(dotProduct), lv_cimag(tmp)));
    }

    *_out = dotProduct;
}
#endif /* LV_HAVE_AVX2 */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_16ic_x2_dot_prod_16ic_neon(lv_16sc_t* out,
                                                   const lv_16sc_t* in_a,
                                                   const lv_16sc_t* in_b,
                                                   unsigned int num_points)
{
    unsigned int quarter_points = num_points / 4;
    unsigned int number;

    lv_16sc_t* a_ptr = (lv_16sc_t*)in_a;
    lv_16sc_t* b_ptr = (lv_16sc_t*)in_b;
    *out = lv_cmake((int16_t)0, (int16_t)0);

    if (quarter_points > 0) {
        // for 2-lane vectors, 1st lane holds the real part,
        // 2nd lane holds the imaginary part
        int16x4x2_t a_val, b_val, c_val, accumulator;
        int16x4x2_t tmp_real, tmp_imag;
        __VOLK_ATTR_ALIGNED(16) lv_16sc_t accum_result[4];
        accumulator.val[0] = vdup_n_s16(0);
        accumulator.val[1] = vdup_n_s16(0);
        lv_16sc_t dotProduct = lv_cmake((int16_t)0, (int16_t)0);

        for (number = 0; number < quarter_points; ++number) {
            a_val = vld2_s16((int16_t*)a_ptr); // a0r|a1r|a2r|a3r || a0i|a1i|a2i|a3i
            b_val = vld2_s16((int16_t*)b_ptr); // b0r|b1r|b2r|b3r || b0i|b1i|b2i|b3i
            __VOLK_PREFETCH(a_ptr + 8);
            __VOLK_PREFETCH(b_ptr + 8);

            // multiply the real*real and imag*imag to get real result
            // a0r*b0r|a1r*b1r|a2r*b2r|a3r*b3r
            tmp_real.val[0] = vmul_s16(a_val.val[0], b_val.val[0]);
            // a0i*b0i|a1i*b1i|a2i*b2i|a3i*b3i
            tmp_real.val[1] = vmul_s16(a_val.val[1], b_val.val[1]);

            // Multiply cross terms to get the imaginary result
            // a0r*b0i|a1r*b1i|a2r*b2i|a3r*b3i
            tmp_imag.val[0] = vmul_s16(a_val.val[0], b_val.val[1]);
            // a0i*b0r|a1i*b1r|a2i*b2r|a3i*b3r
            tmp_imag.val[1] = vmul_s16(a_val.val[1], b_val.val[0]);

            c_val.val[0] = vqsub_s16(tmp_real.val[0], tmp_real.val[1]);
            c_val.val[1] = vqadd_s16(tmp_imag.val[0], tmp_imag.val[1]);

            accumulator.val[0] = vqadd_s16(accumulator.val[0], c_val.val[0]);
            accumulator.val[1] = vqadd_s16(accumulator.val[1], c_val.val[1]);

            a_ptr += 4;
            b_ptr += 4;
        }

        vst2_s16((int16_t*)accum_result, accumulator);
        for (number = 0; number < 4; ++number) {
            dotProduct = lv_cmake(
                sat_adds16i(lv_creal(dotProduct), lv_creal(accum_result[number])),
                sat_adds16i(lv_cimag(dotProduct), lv_cimag(accum_result[number])));
        }

        *out = dotProduct;
    }

    // tail case
    for (number = quarter_points * 4; number < num_points; ++number) {
        *out += (*a_ptr++) * (*b_ptr++);
    }
}

#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_16ic_x2_dot_prod_16ic_neon_vma(lv_16sc_t* out,
                                                       const lv_16sc_t* in_a,
                                                       const lv_16sc_t* in_b,
                                                       unsigned int num_points)
{
    unsigned int quarter_points = num_points / 4;
    unsigned int number;

    lv_16sc_t* a_ptr = (lv_16sc_t*)in_a;
    lv_16sc_t* b_ptr = (lv_16sc_t*)in_b;
    // for 2-lane vectors, 1st lane holds the real part,
    // 2nd lane holds the imaginary part
    int16x4x2_t a_val, b_val, accumulator;
    int16x4x2_t tmp;
    __VOLK_ATTR_ALIGNED(16) lv_16sc_t accum_result[4];
    accumulator.val[0] = vdup_n_s16(0);
    accumulator.val[1] = vdup_n_s16(0);

    for (number = 0; number < quarter_points; ++number) {
        a_val = vld2_s16((int16_t*)a_ptr); // a0r|a1r|a2r|a3r || a0i|a1i|a2i|a3i
        b_val = vld2_s16((int16_t*)b_ptr); // b0r|b1r|b2r|b3r || b0i|b1i|b2i|b3i
        __VOLK_PREFETCH(a_ptr + 8);
        __VOLK_PREFETCH(b_ptr + 8);

        tmp.val[0] = vmul_s16(a_val.val[0], b_val.val[0]);
        tmp.val[1] = vmul_s16(a_val.val[1], b_val.val[0]);

        // use multiply accumulate/subtract to get result
        tmp.val[0] = vmls_s16(tmp.val[0], a_val.val[1], b_val.val[1]);
        tmp.val[1] = vmla_s16(tmp.val[1], a_val.val[0], b_val.val[1]);

        accumulator.val[0] = vqadd_s16(accumulator.val[0], tmp.val[0]);
        accumulator.val[1] = vqadd_s16(accumulator.val[1], tmp.val[1]);

        a_ptr += 4;
        b_ptr += 4;
    }

    vst2_s16((int16_t*)accum_result, accumulator);
    *out = accum_result[0] + accum_result[1] + accum_result[2] + accum_result[3];

    // tail case
    for (number = quarter_points * 4; number < num_points; ++number) {
        *out += (*a_ptr++) * (*b_ptr++);
    }
}

#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_16ic_x2_dot_prod_16ic_neon_optvma(lv_16sc_t* out,
                                                          const lv_16sc_t* in_a,
                                                          const lv_16sc_t* in_b,
                                                          unsigned int num_points)
{
    unsigned int quarter_points = num_points / 4;
    unsigned int number;

    lv_16sc_t* a_ptr = (lv_16sc_t*)in_a;
    lv_16sc_t* b_ptr = (lv_16sc_t*)in_b;
    // for 2-lane vectors, 1st lane holds the real part,
    // 2nd lane holds the imaginary part
    int16x4x2_t a_val, b_val, accumulator1, accumulator2;

    __VOLK_ATTR_ALIGNED(16) lv_16sc_t accum_result[4];
    accumulator1.val[0] = vdup_n_s16(0);
    accumulator1.val[1] = vdup_n_s16(0);
    accumulator2.val[0] = vdup_n_s16(0);
    accumulator2.val[1] = vdup_n_s16(0);

    for (number = 0; number < quarter_points; ++number) {
        a_val = vld2_s16((int16_t*)a_ptr); // a0r|a1r|a2r|a3r || a0i|a1i|a2i|a3i
        b_val = vld2_s16((int16_t*)b_ptr); // b0r|b1r|b2r|b3r || b0i|b1i|b2i|b3i
        __VOLK_PREFETCH(a_ptr + 8);
        __VOLK_PREFETCH(b_ptr + 8);

        // use 2 accumulators to remove inter-instruction data dependencies
        accumulator1.val[0] = vmla_s16(accumulator1.val[0], a_val.val[0], b_val.val[0]);
        accumulator2.val[0] = vmls_s16(accumulator2.val[0], a_val.val[1], b_val.val[1]);
        accumulator1.val[1] = vmla_s16(accumulator1.val[1], a_val.val[0], b_val.val[1]);
        accumulator2.val[1] = vmla_s16(accumulator2.val[1], a_val.val[1], b_val.val[0]);

        a_ptr += 4;
        b_ptr += 4;
    }

    accumulator1.val[0] = vqadd_s16(accumulator1.val[0], accumulator2.val[0]);
    accumulator1.val[1] = vqadd_s16(accumulator1.val[1], accumulator2.val[1]);

    vst2_s16((int16_t*)accum_result, accumulator1);
    *out = accum_result[0] + accum_result[1] + accum_result[2] + accum_result[3];

    // tail case
    for (number = quarter_points * 4; number < num_points; ++number) {
        *out += (*a_ptr++) * (*b_ptr++);
    }
}

#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_16ic_x2_dot_prod_16ic_neonv8(lv_16sc_t* out,
                                                     const lv_16sc_t* in_a,
                                                     const lv_16sc_t* in_b,
                                                     unsigned int num_points)
{
    unsigned int eighth_points = num_points / 8;
    unsigned int number;

    const lv_16sc_t* a_ptr = in_a;
    const lv_16sc_t* b_ptr = in_b;

    /* Use 128-bit registers with deinterleaved loads for better throughput */
    int16x8x2_t a_val, b_val;
    int16x8_t acc_real1 = vdupq_n_s16(0);
    int16x8_t acc_real2 = vdupq_n_s16(0);
    int16x8_t acc_imag1 = vdupq_n_s16(0);
    int16x8_t acc_imag2 = vdupq_n_s16(0);

    for (number = 0; number < eighth_points; ++number) {
        a_val = vld2q_s16((int16_t*)a_ptr);
        b_val = vld2q_s16((int16_t*)b_ptr);
        __VOLK_PREFETCH(a_ptr + 16);
        __VOLK_PREFETCH(b_ptr + 16);

        /* real = ar*br - ai*bi, use two accumulators to avoid dependency */
        acc_real1 = vmlaq_s16(acc_real1, a_val.val[0], b_val.val[0]);
        acc_real2 = vmlsq_s16(acc_real2, a_val.val[1], b_val.val[1]);

        /* imag = ar*bi + ai*br, use two accumulators */
        acc_imag1 = vmlaq_s16(acc_imag1, a_val.val[0], b_val.val[1]);
        acc_imag2 = vmlaq_s16(acc_imag2, a_val.val[1], b_val.val[0]);

        a_ptr += 8;
        b_ptr += 8;
    }

    /* Combine accumulators with saturation */
    int16x8_t acc_real = vqaddq_s16(acc_real1, acc_real2);
    int16x8_t acc_imag = vqaddq_s16(acc_imag1, acc_imag2);

    /* Horizontal sum using ARMv8 vaddvq */
    int16_t sum_real = vaddvq_s16(acc_real);
    int16_t sum_imag = vaddvq_s16(acc_imag);

    *out = lv_cmake(sum_real, sum_imag);

    /* Tail case */
    for (number = eighth_points * 8; number < num_points; ++number) {
        lv_16sc_t tmp = in_a[number] * in_b[number];
        *out = lv_cmake(sat_adds16i(lv_creal(*out), lv_creal(tmp)),
                        sat_adds16i(lv_cimag(*out), lv_cimag(tmp)));
    }
}

#endif /* LV_HAVE_NEONV8 */


#ifdef LV_HAVE_RVV
#include "volk_32fc_x2_dot_prod_32fc.h"

static inline void volk_16ic_x2_dot_prod_16ic_rvv(lv_16sc_t* result,
                                                  const lv_16sc_t* in_a,
                                                  const lv_16sc_t* in_b,
                                                  unsigned int num_points)
{
    vint16m4_t vsumr = __riscv_vmv_v_x_i16m4(0, __riscv_vsetvlmax_e16m4());
    vint16m4_t vsumi = vsumr;
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, in_a += vl, in_b += vl) {
        vl = __riscv_vsetvl_e16m4(n);
        vint32m8_t va = __riscv_vle32_v_i32m8((const int32_t*)in_a, vl);
        vint32m8_t vb = __riscv_vle32_v_i32m8((const int32_t*)in_b, vl);
        vint16m4_t var = __riscv_vnsra(va, 0, vl), vai = __riscv_vnsra(va, 16, vl);
        vint16m4_t vbr = __riscv_vnsra(vb, 0, vl), vbi = __riscv_vnsra(vb, 16, vl);
        vint16m4_t vr = __riscv_vnmsac(__riscv_vmul(var, vbr, vl), vai, vbi, vl);
        vint16m4_t vi = __riscv_vmacc(__riscv_vmul(var, vbi, vl), vai, vbr, vl);
        vsumr = __riscv_vadd_tu(vsumr, vsumr, vr, vl);
        vsumi = __riscv_vadd_tu(vsumi, vsumi, vi, vl);
    }
    size_t vl = __riscv_vsetvlmax_e16m1();
    vint16m1_t vr = RISCV_SHRINK4(vadd, i, 16, vsumr);
    vint16m1_t vi = RISCV_SHRINK4(vadd, i, 16, vsumi);
    vint16m1_t z = __riscv_vmv_s_x_i16m1(0, vl);
    *result = lv_cmake(__riscv_vmv_x(__riscv_vredsum(vr, z, vl)),
                       __riscv_vmv_x(__riscv_vredsum(vi, z, vl)));
}
#endif /*LV_HAVE_RVV*/

#ifdef LV_HAVE_RVVSEG
#include "volk_32fc_x2_dot_prod_32fc.h"


static inline void volk_16ic_x2_dot_prod_16ic_rvvseg(lv_16sc_t* result,
                                                     const lv_16sc_t* in_a,
                                                     const lv_16sc_t* in_b,
                                                     unsigned int num_points)
{
    vint16m4_t vsumr = __riscv_vmv_v_x_i16m4(0, __riscv_vsetvlmax_e16m4());
    vint16m4_t vsumi = vsumr;
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, in_a += vl, in_b += vl) {
        vl = __riscv_vsetvl_e16m4(n);
        vint16m4x2_t va = __riscv_vlseg2e16_v_i16m4x2((const int16_t*)in_a, vl);
        vint16m4x2_t vb = __riscv_vlseg2e16_v_i16m4x2((const int16_t*)in_b, vl);
        vint16m4_t var = __riscv_vget_i16m4(va, 0), vai = __riscv_vget_i16m4(va, 1);
        vint16m4_t vbr = __riscv_vget_i16m4(vb, 0), vbi = __riscv_vget_i16m4(vb, 1);
        vint16m4_t vr = __riscv_vnmsac(__riscv_vmul(var, vbr, vl), vai, vbi, vl);
        vint16m4_t vi = __riscv_vmacc(__riscv_vmul(var, vbi, vl), vai, vbr, vl);
        vsumr = __riscv_vadd_tu(vsumr, vsumr, vr, vl);
        vsumi = __riscv_vadd_tu(vsumi, vsumi, vi, vl);
    }
    size_t vl = __riscv_vsetvlmax_e16m1();
    vint16m1_t vr = RISCV_SHRINK4(vadd, i, 16, vsumr);
    vint16m1_t vi = RISCV_SHRINK4(vadd, i, 16, vsumi);
    vint16m1_t z = __riscv_vmv_s_x_i16m1(0, vl);
    *result = lv_cmake(__riscv_vmv_x(__riscv_vredsum(vr, z, vl)),
                       __riscv_vmv_x(__riscv_vredsum(vi, z, vl)));
}
#endif /*LV_HAVE_RVVSEG*/

#endif /*INCLUDED_volk_16ic_x2_dot_prod_16ic_H*/
