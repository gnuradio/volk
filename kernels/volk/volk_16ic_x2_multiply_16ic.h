/* -*- c++ -*- */
/*
 * Copyright 2016 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * GNU Radio is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * GNU Radio is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Radio; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

/*!
 * \page volk_16ic_x2_multiply_16ic
 *
 * \b Overview
 *
 * Multiplies two input complex vectors, point-by-point, storing the result in the third vector.
 * WARNING: Saturation is not checked.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_16ic_x2_multiply_16ic(lv_16sc_t* result, const lv_16sc_t* in_a, const lv_16sc_t* in_b, unsigned int num_points);
 * \endcode
 *
 * \b Inputs
 * \li in_a: One of the vectors to be multiplied.
 * \li in_b: The other vector to be multiplied.
 * \li num_points: The number of complex data points to be multiplied from both input vectors.
 *
 * \b Outputs
 * \li result: The vector where the results will be stored.
 *
 */

#ifndef INCLUDED_volk_16ic_x2_multiply_16ic_H
#define INCLUDED_volk_16ic_x2_multiply_16ic_H

#include <volk/volk_common.h>
#include <volk/volk_complex.h>

#ifdef LV_HAVE_GENERIC

static inline void volk_16ic_x2_multiply_16ic_generic(lv_16sc_t* result, const lv_16sc_t* in_a, const lv_16sc_t* in_b, unsigned int num_points)
{
    unsigned int n;
    for (n = 0; n < num_points; n++)
        {
            result[n] = in_a[n] * in_b[n];
        }
}

#endif /*LV_HAVE_GENERIC*/


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_16ic_x2_multiply_16ic_a_sse2(lv_16sc_t* out, const lv_16sc_t* in_a, const lv_16sc_t* in_b, unsigned int num_points)
{
    const unsigned int sse_iters = num_points / 4;
    __m128i a, b, c, c_sr, mask_imag, mask_real, real, imag, imag1, imag2, b_sl, a_sl, result;

    mask_imag = _mm_set_epi8(255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0);
    mask_real = _mm_set_epi8(0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255);

    const lv_16sc_t* _in_a = in_a;
    const lv_16sc_t* _in_b = in_b;
    lv_16sc_t* _out = out;
    unsigned int number;

    for(number = 0; number < sse_iters; number++)
        {
            a = _mm_load_si128((__m128i*)_in_a); //load (2 byte imag, 2 byte real) x 4 into 128 bits reg
            b = _mm_load_si128((__m128i*)_in_b);
            c = _mm_mullo_epi16 (a, b); // a3.i*b3.i, a3.r*b3.r, ....

            c_sr = _mm_srli_si128 (c, 2); // Shift a right by imm8 bytes while shifting in zeros, and store the results in dst.
            real = _mm_subs_epi16 (c, c_sr);
            real = _mm_and_si128 (real, mask_real); // a3.r*b3.r-a3.i*b3.i , 0,  a3.r*b3.r- a3.i*b3.i

            b_sl = _mm_slli_si128(b, 2); // b3.r, b2.i ....
            a_sl = _mm_slli_si128(a, 2); // a3.r, a2.i ....

            imag1 = _mm_mullo_epi16(a, b_sl); // a3.i*b3.r, ....
            imag2 = _mm_mullo_epi16(b, a_sl); // b3.i*a3.r, ....

            imag = _mm_adds_epi16(imag1, imag2);
            imag = _mm_and_si128 (imag, mask_imag); // a3.i*b3.r+b3.i*a3.r, 0, ...

            result = _mm_or_si128 (real, imag);

            _mm_store_si128((__m128i*)_out, result);

            _in_a += 4;
            _in_b += 4;
            _out += 4;
        }

    for (number = sse_iters * 4; number < num_points; ++number)
        {
            *_out++ = (*_in_a++) * (*_in_b++);
        }
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_16ic_x2_multiply_16ic_u_sse2(lv_16sc_t* out, const lv_16sc_t* in_a, const lv_16sc_t* in_b, unsigned int num_points)
{
    const unsigned int sse_iters = num_points / 4;
    __m128i a, b, c, c_sr, mask_imag, mask_real, real, imag, imag1,imag2, b_sl, a_sl, result;

    mask_imag = _mm_set_epi8(255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0);
    mask_real = _mm_set_epi8(0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255);

    const lv_16sc_t* _in_a = in_a;
    const lv_16sc_t* _in_b = in_b;
    lv_16sc_t* _out = out;
    unsigned int number;

    for(number = 0; number < sse_iters; number++)
        {
            a = _mm_loadu_si128((__m128i*)_in_a); //load (2 byte imag, 2 byte real) x 4 into 128 bits reg
            b = _mm_loadu_si128((__m128i*)_in_b);
            c = _mm_mullo_epi16 (a, b); // a3.i*b3.i, a3.r*b3.r, ....

            c_sr = _mm_srli_si128 (c, 2); // Shift a right by imm8 bytes while shifting in zeros, and store the results in dst.
            real = _mm_subs_epi16 (c, c_sr);
            real = _mm_and_si128 (real, mask_real); // a3.r*b3.r-a3.i*b3.i , 0,  a3.r*b3.r- a3.i*b3.i

            b_sl = _mm_slli_si128(b, 2); // b3.r, b2.i ....
            a_sl = _mm_slli_si128(a, 2); // a3.r, a2.i ....

            imag1 = _mm_mullo_epi16(a, b_sl); // a3.i*b3.r, ....
            imag2 = _mm_mullo_epi16(b, a_sl); // b3.i*a3.r, ....

            imag = _mm_adds_epi16(imag1, imag2);
            imag = _mm_and_si128 (imag, mask_imag); // a3.i*b3.r+b3.i*a3.r, 0, ...

            result = _mm_or_si128 (real, imag);

            _mm_storeu_si128((__m128i*)_out, result);

            _in_a += 4;
            _in_b += 4;
            _out += 4;
        }

    for (number = sse_iters * 4; number < num_points; ++number)
        {
            *_out++ = (*_in_a++) * (*_in_b++);
        }
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16ic_x2_multiply_16ic_u_avx2(lv_16sc_t* out, const lv_16sc_t* in_a, const lv_16sc_t* in_b, unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int avx2_points = num_points / 8;

    const lv_16sc_t* _in_a = in_a;
    const lv_16sc_t* _in_b = in_b;
    lv_16sc_t* _out = out;

    __m256i a, b, c, c_sr,  real, imag, imag1, imag2, b_sl, a_sl, result;

    const __m256i mask_imag = _mm256_set_epi8(255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0);
    const __m256i mask_real = _mm256_set_epi8(0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255);

    for(;number < avx2_points; number++)
        {
            a = _mm256_loadu_si256((__m256i*)_in_a); // Load the ar + ai, br + bi as ar,ai,br,bi
            b = _mm256_loadu_si256((__m256i*)_in_b); // Load the cr + ci, dr + di as cr,ci,dr,di
            c = _mm256_mullo_epi16(a, b);

            c_sr = _mm256_srli_si256(c, 2); // Shift a right by imm8 bytes while shifting in zeros, and store the results in dst.
            real = _mm256_subs_epi16(c, c_sr);
            real = _mm256_and_si256(real, mask_real); // a3.r*b3.r-a3.i*b3.i , 0,  a3.r*b3.r- a3.i*b3.i

            b_sl = _mm256_slli_si256(b, 2); // b3.r, b2.i ....
            a_sl = _mm256_slli_si256(a, 2); // a3.r, a2.i ....

            imag1 = _mm256_mullo_epi16(a, b_sl); // a3.i*b3.r, ....
            imag2 = _mm256_mullo_epi16(b, a_sl); // b3.i*a3.r, ....

            imag = _mm256_adds_epi16(imag1, imag2);
            imag = _mm256_and_si256(imag, mask_imag); // a3.i*b3.r+b3.i*a3.r, 0, ...

            result = _mm256_or_si256(real, imag);

            _mm256_storeu_si256((__m256i*)_out, result);

            _in_a += 8;
            _in_b += 8;
            _out += 8;
        }
    _mm256_zeroupper();
    number = avx2_points * 8;
    for(;number < num_points; number++)
        {
            *_out++ = (*_in_a++) * (*_in_b++);
        }
}
#endif /* LV_HAVE_AVX2  */


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_16ic_x2_multiply_16ic_a_avx2(lv_16sc_t* out, const lv_16sc_t* in_a, const lv_16sc_t* in_b, unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int avx2_points = num_points / 8;

    const lv_16sc_t* _in_a = in_a;
    const lv_16sc_t* _in_b = in_b;
    lv_16sc_t* _out = out;

    __m256i a, b, c, c_sr,  real, imag, imag1, imag2, b_sl, a_sl, result;

    const __m256i mask_imag = _mm256_set_epi8(255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0);
    const __m256i mask_real = _mm256_set_epi8(0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255);

    for(;number < avx2_points; number++)
        {
            a = _mm256_load_si256((__m256i*)_in_a); // Load the ar + ai, br + bi as ar,ai,br,bi
            b = _mm256_load_si256((__m256i*)_in_b); // Load the cr + ci, dr + di as cr,ci,dr,di
            c = _mm256_mullo_epi16(a, b);

            c_sr = _mm256_srli_si256(c, 2); // Shift a right by imm8 bytes while shifting in zeros, and store the results in dst.
            real = _mm256_subs_epi16(c, c_sr);
            real = _mm256_and_si256(real, mask_real); // a3.r*b3.r-a3.i*b3.i , 0,  a3.r*b3.r- a3.i*b3.i

            b_sl = _mm256_slli_si256(b, 2); // b3.r, b2.i ....
            a_sl = _mm256_slli_si256(a, 2); // a3.r, a2.i ....

            imag1 = _mm256_mullo_epi16(a, b_sl); // a3.i*b3.r, ....
            imag2 = _mm256_mullo_epi16(b, a_sl); // b3.i*a3.r, ....

            imag = _mm256_adds_epi16(imag1, imag2);
            imag = _mm256_and_si256(imag, mask_imag); // a3.i*b3.r+b3.i*a3.r, 0, ...

            result = _mm256_or_si256(real, imag);

            _mm256_store_si256((__m256i*)_out, result);

            _in_a += 8;
            _in_b += 8;
            _out += 8;
        }
    _mm256_zeroupper();
    number = avx2_points * 8;
    for(;number < num_points; number++)
        {
            *_out++ = (*_in_a++) * (*_in_b++);
        }
}
#endif /* LV_HAVE_AVX2  */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_16ic_x2_multiply_16ic_neon(lv_16sc_t* out, const lv_16sc_t* in_a, const lv_16sc_t* in_b, unsigned int num_points)
{
    lv_16sc_t *a_ptr = (lv_16sc_t*) in_a;
    lv_16sc_t *b_ptr = (lv_16sc_t*) in_b;
    unsigned int quarter_points = num_points / 4;
    int16x4x2_t a_val, b_val, c_val;
    int16x4x2_t tmp_real, tmp_imag;
    unsigned int number = 0;

    for(number = 0; number < quarter_points; ++number)
        {
            a_val = vld2_s16((int16_t*)a_ptr); // a0r|a1r|a2r|a3r || a0i|a1i|a2i|a3i
            b_val = vld2_s16((int16_t*)b_ptr); // b0r|b1r|b2r|b3r || b0i|b1i|b2i|b3i
            __builtin_prefetch(a_ptr + 4);
            __builtin_prefetch(b_ptr + 4);

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

            // store the results
            c_val.val[0] = vsub_s16(tmp_real.val[0], tmp_real.val[1]);
            c_val.val[1] = vadd_s16(tmp_imag.val[0], tmp_imag.val[1]);
            vst2_s16((int16_t*)out, c_val);

            a_ptr += 4;
            b_ptr += 4;
            out += 4;
        }

    for(number = quarter_points * 4; number < num_points; number++)
        {
            *out++ = (*a_ptr++) * (*b_ptr++);
        }
}
#endif /* LV_HAVE_NEON */

#endif /*INCLUDED_volk_16ic_x2_multiply_16ic_H*/
