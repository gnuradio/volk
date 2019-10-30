/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
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
 * \page volk_32fc_x2_conjugate_dot_prod_32fc
 *
 * \b Overview
 *
 * This block computes the conjugate dot product (or inner product)
 * between two vectors, the \p input and \p taps vectors. Given a set
 * of \p num_points taps, the result is the sum of products between
 * the input vector and the conjugate of the taps. The result is a
 * single value stored in the \p result address and is returned as a
 * complex float.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_x2_conjugate_dot_prod_32fc(lv_32fc_t* result, const lv_32fc_t* input, const lv_32fc_t* taps, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li input: vector of complex floats.
 * \li taps:  complex float taps.
 * \li num_points: number of samples in both \p input and \p taps.
 *
 * \b Outputs
 * \li result: pointer to a complex float value to hold the dot product result.
 *
 * \b Example
 * \code
 * int N = 10000;
 *
 * <FIXME>
 *
 * volk_32fc_x2_conjugate_dot_prod_32fc();
 *
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_x2_conjugate_dot_prod_32fc_u_H
#define INCLUDED_volk_32fc_x2_conjugate_dot_prod_32fc_u_H


#include<volk/volk_complex.h>


#ifdef LV_HAVE_GENERIC

static inline void volk_32fc_x2_conjugate_dot_prod_32fc_generic(lv_32fc_t* result, const lv_32fc_t* input, const lv_32fc_t* taps, unsigned int num_points) {

  const unsigned int num_bytes = num_points*8;

  float * res = (float*) result;
  float * in = (float*) input;
  float * tp = (float*) taps;
  unsigned int n_2_ccomplex_blocks = num_bytes >> 4;
  unsigned int isodd = (num_bytes >> 3) &1;

  float sum0[2] = {0,0};
  float sum1[2] = {0,0};
  unsigned int i = 0;

  for(i = 0; i < n_2_ccomplex_blocks; ++i) {
    sum0[0] += in[0] * tp[0] + in[1] * tp[1];
    sum0[1] += (-in[0] * tp[1]) + in[1] * tp[0];
    sum1[0] += in[2] * tp[2] + in[3] * tp[3];
    sum1[1] += (-in[2] * tp[3]) + in[3] * tp[2];

    in += 4;
    tp += 4;
  }

  res[0] = sum0[0] + sum1[0];
  res[1] = sum0[1] + sum1[1];

  for(i = 0; i < isodd; ++i) {
    *result += input[(num_bytes >> 3) - 1] * lv_conj(taps[(num_bytes >> 3) - 1]);
  }
}

#endif /*LV_HAVE_GENERIC*/

#ifdef LV_HAVE_AVX

#include <immintrin.h>

static inline void volk_32fc_x2_conjugate_dot_prod_32fc_u_avx(lv_32fc_t* result,
    const lv_32fc_t* input, const lv_32fc_t* taps, unsigned int num_points)
{
  // Partial sums for indices i, i+1, i+2 and i+3.
  __m256 sum_a_mult_b_real = _mm256_setzero_ps();
  __m256 sum_a_mult_b_imag = _mm256_setzero_ps();

  for (long unsigned i = 0; i < (num_points & ~3u); i += 4) {
    /* Four complex elements a time are processed.
     * (ar + j⋅ai)*conj(br + j⋅bi) =
     * ar⋅br + ai⋅bi + j⋅(ai⋅br − ar⋅bi)
     */

    /* Load input and taps, split and duplicate real und imaginary parts of taps.
     * a: | ai,i+3 | ar,i+3 | … | ai,i+1 | ar,i+1 | ai,i+0 | ar,i+0 |
     * b: | bi,i+3 | br,i+3 | … | bi,i+1 | br,i+1 | bi,i+0 | br,i+0 |
     * b_real: | br,i+3 | br,i+3 | … | br,i+1 | br,i+1 | br,i+0 | br,i+0 |
     * b_imag: | bi,i+3 | bi,i+3 | … | bi,i+1 | bi,i+1 | bi,i+0 | bi,i+0 |
     */
    __m256 a = _mm256_loadu_ps((const float *) &input[i]);
    __m256 b = _mm256_loadu_ps((const float *) &taps[i]);
    __m256 b_real = _mm256_moveldup_ps(b);
    __m256 b_imag = _mm256_movehdup_ps(b);

    // Add | ai⋅br,i+3 | ar⋅br,i+3 | … | ai⋅br,i+0 | ar⋅br,i+0 | to partial sum.
    sum_a_mult_b_real = _mm256_add_ps(sum_a_mult_b_real, _mm256_mul_ps(a, b_real));
    // Add | ai⋅bi,i+3 | −ar⋅bi,i+3 | … | ai⋅bi,i+0 | −ar⋅bi,i+0 | to partial sum.
    sum_a_mult_b_imag = _mm256_addsub_ps(sum_a_mult_b_imag, _mm256_mul_ps(a, b_imag));
  }

  // Swap position of −ar⋅bi and ai⋅bi.
  sum_a_mult_b_imag = _mm256_permute_ps(sum_a_mult_b_imag, _MM_SHUFFLE(2, 3, 0, 1));
  // | ai⋅br + ai⋅bi | ai⋅br − ar⋅bi |, sum contains four such partial sums.
  __m256 sum = _mm256_add_ps(sum_a_mult_b_real, sum_a_mult_b_imag);
  /* Sum the four partial sums: Add high half of vector sum to the low one, i.e.
   * s1 + s3 and s0 + s2 …
   */
  sum = _mm256_add_ps(sum, _mm256_permute2f128_ps(sum, sum, 0x01));
  // … and now (s0 + s2) + (s1 + s3)
  sum = _mm256_add_ps(sum, _mm256_permute_ps(sum, _MM_SHUFFLE(1, 0, 3, 2)));
  // Store result.
  __m128 lower = _mm256_extractf128_ps(sum, 0);
  _mm_storel_pi((__m64 *) result, lower);

  // Handle the last elements if num_points mod 4 is bigger than 0.
  for (long unsigned i = num_points & ~3u; i < num_points; ++i) {
    *result += lv_cmake(
        lv_creal(input[i]) * lv_creal(taps[i]) + lv_cimag(input[i]) * lv_cimag(taps[i]),
        lv_cimag(input[i]) * lv_creal(taps[i]) - lv_creal(input[i]) * lv_cimag(taps[i]));
  }
}

#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_SSE3

#include <xmmintrin.h>
#include <pmmintrin.h>

static inline void volk_32fc_x2_conjugate_dot_prod_32fc_u_sse3(lv_32fc_t* result,
    const lv_32fc_t* input, const lv_32fc_t* taps, unsigned int num_points)
{
  // Partial sums for indices i and i+1.
  __m128 sum_a_mult_b_real = _mm_setzero_ps();
  __m128 sum_a_mult_b_imag = _mm_setzero_ps();

  for (long unsigned i = 0; i < (num_points & ~1u); i += 2) {
    /* Two complex elements a time are processed.
     * (ar + j⋅ai)*conj(br + j⋅bi) =
     * ar⋅br + ai⋅bi + j⋅(ai⋅br − ar⋅bi)
     */

    /* Load input and taps, split and duplicate real und imaginary parts of taps.
     * a: | ai,i+1 | ar,i+1 | ai,i+0 | ar,i+0 |
     * b: | bi,i+1 | br,i+1 | bi,i+0 | br,i+0 |
     * b_real: | br,i+1 | br,i+1 | br,i+0 | br,i+0 |
     * b_imag: | bi,i+1 | bi,i+1 | bi,i+0 | bi,i+0 |
     */
    __m128 a = _mm_loadu_ps((const float *) &input[i]);
    __m128 b = _mm_loadu_ps((const float *) &taps[i]);
    __m128 b_real = _mm_moveldup_ps(b);
    __m128 b_imag = _mm_movehdup_ps(b);

    // Add | ai⋅br,i+1 | ar⋅br,i+1 | ai⋅br,i+0 | ar⋅br,i+0 | to partial sum.
    sum_a_mult_b_real = _mm_add_ps(sum_a_mult_b_real, _mm_mul_ps(a, b_real));
    // Add | ai⋅bi,i+1 | −ar⋅bi,i+1 | ai⋅bi,i+0 | −ar⋅bi,i+0 | to partial sum.
    sum_a_mult_b_imag = _mm_addsub_ps(sum_a_mult_b_imag, _mm_mul_ps(a, b_imag));
  }

  // Swap position of −ar⋅bi and ai⋅bi.
  sum_a_mult_b_imag = _mm_shuffle_ps(sum_a_mult_b_imag, sum_a_mult_b_imag,
      _MM_SHUFFLE(2, 3, 0, 1));
  // | ai⋅br + ai⋅bi | ai⋅br − ar⋅bi |, sum contains two such partial sums.
  __m128 sum = _mm_add_ps(sum_a_mult_b_real, sum_a_mult_b_imag);
  // Sum the two partial sums.
  sum = _mm_add_ps(sum, _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(1, 0, 3, 2)));
  // Store result.
  _mm_storel_pi((__m64 *) result, sum);

  // Handle the last element if num_points mod 2 is 1.
  if (num_points & 1u) {
    *result += lv_cmake(
        lv_creal(input[num_points - 1]) * lv_creal(taps[num_points - 1]) +
        lv_cimag(input[num_points - 1]) * lv_cimag(taps[num_points - 1]),
        lv_cimag(input[num_points - 1]) * lv_creal(taps[num_points - 1]) -
        lv_creal(input[num_points - 1]) * lv_cimag(taps[num_points - 1]));
  }
}

#endif /*LV_HAVE_SSE3*/

#ifdef LV_HAVE_NEON
#include <arm_neon.h>
static inline void volk_32fc_x2_conjugate_dot_prod_32fc_neon(lv_32fc_t* result, const lv_32fc_t* input, const lv_32fc_t* taps, unsigned int num_points) {

    unsigned int quarter_points = num_points / 4;
    unsigned int number;

    lv_32fc_t* a_ptr = (lv_32fc_t*) taps;
    lv_32fc_t* b_ptr = (lv_32fc_t*) input;
    // for 2-lane vectors, 1st lane holds the real part,
    // 2nd lane holds the imaginary part
    float32x4x2_t a_val, b_val, accumulator;
    float32x4x2_t tmp_imag;
    accumulator.val[0] = vdupq_n_f32(0);
    accumulator.val[1] = vdupq_n_f32(0);

    for(number = 0; number < quarter_points; ++number) {
        a_val = vld2q_f32((float*)a_ptr); // a0r|a1r|a2r|a3r || a0i|a1i|a2i|a3i
        b_val = vld2q_f32((float*)b_ptr); // b0r|b1r|b2r|b3r || b0i|b1i|b2i|b3i
        __VOLK_PREFETCH(a_ptr+8);
        __VOLK_PREFETCH(b_ptr+8);

        // do the first multiply
        tmp_imag.val[1] = vmulq_f32(a_val.val[1], b_val.val[0]);
        tmp_imag.val[0] = vmulq_f32(a_val.val[0], b_val.val[0]);

        // use multiply accumulate/subtract to get result
        tmp_imag.val[1] = vmlsq_f32(tmp_imag.val[1], a_val.val[0], b_val.val[1]);
        tmp_imag.val[0] = vmlaq_f32(tmp_imag.val[0], a_val.val[1], b_val.val[1]);

        accumulator.val[0] = vaddq_f32(accumulator.val[0], tmp_imag.val[0]);
        accumulator.val[1] = vaddq_f32(accumulator.val[1], tmp_imag.val[1]);

        // increment pointers
        a_ptr += 4;
        b_ptr += 4;
    }
    lv_32fc_t accum_result[4];
    vst2q_f32((float*)accum_result, accumulator);
    *result = accum_result[0] + accum_result[1] + accum_result[2] + accum_result[3];

    // tail case
    for(number = quarter_points*4; number < num_points; ++number) {
      *result += (*a_ptr++) * lv_conj(*b_ptr++);
    }
    *result = lv_conj(*result);

}
#endif /*LV_HAVE_NEON*/

#endif /*INCLUDED_volk_32fc_x2_conjugate_dot_prod_32fc_u_H*/

#ifndef INCLUDED_volk_32fc_x2_conjugate_dot_prod_32fc_a_H
#define INCLUDED_volk_32fc_x2_conjugate_dot_prod_32fc_a_H

#include <volk/volk_common.h>
#include<volk/volk_complex.h>
#include<stdio.h>


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32fc_x2_conjugate_dot_prod_32fc_a_avx(lv_32fc_t* result,
    const lv_32fc_t* input, const lv_32fc_t* taps, unsigned int num_points)
{
  // Partial sums for indices i, i+1, i+2 and i+3.
  __m256 sum_a_mult_b_real = _mm256_setzero_ps();
  __m256 sum_a_mult_b_imag = _mm256_setzero_ps();

  for (long unsigned i = 0; i < (num_points & ~3u); i += 4) {
    /* Four complex elements a time are processed.
     * (ar + j⋅ai)*conj(br + j⋅bi) =
     * ar⋅br + ai⋅bi + j⋅(ai⋅br − ar⋅bi)
     */

    /* Load input and taps, split and duplicate real und imaginary parts of taps.
     * a: | ai,i+3 | ar,i+3 | … | ai,i+1 | ar,i+1 | ai,i+0 | ar,i+0 |
     * b: | bi,i+3 | br,i+3 | … | bi,i+1 | br,i+1 | bi,i+0 | br,i+0 |
     * b_real: | br,i+3 | br,i+3 | … | br,i+1 | br,i+1 | br,i+0 | br,i+0 |
     * b_imag: | bi,i+3 | bi,i+3 | … | bi,i+1 | bi,i+1 | bi,i+0 | bi,i+0 |
     */
    __m256 a = _mm256_load_ps((const float *) &input[i]);
    __m256 b = _mm256_load_ps((const float *) &taps[i]);
    __m256 b_real = _mm256_moveldup_ps(b);
    __m256 b_imag = _mm256_movehdup_ps(b);

    // Add | ai⋅br,i+3 | ar⋅br,i+3 | … | ai⋅br,i+0 | ar⋅br,i+0 | to partial sum.
    sum_a_mult_b_real = _mm256_add_ps(sum_a_mult_b_real, _mm256_mul_ps(a, b_real));
    // Add | ai⋅bi,i+3 | −ar⋅bi,i+3 | … | ai⋅bi,i+0 | −ar⋅bi,i+0 | to partial sum.
    sum_a_mult_b_imag = _mm256_addsub_ps(sum_a_mult_b_imag, _mm256_mul_ps(a, b_imag));
  }

  // Swap position of −ar⋅bi and ai⋅bi.
  sum_a_mult_b_imag = _mm256_permute_ps(sum_a_mult_b_imag, _MM_SHUFFLE(2, 3, 0, 1));
  // | ai⋅br + ai⋅bi | ai⋅br − ar⋅bi |, sum contains four such partial sums.
  __m256 sum = _mm256_add_ps(sum_a_mult_b_real, sum_a_mult_b_imag);
  /* Sum the four partial sums: Add high half of vector sum to the low one, i.e.
   * s1 + s3 and s0 + s2 …
   */
  sum = _mm256_add_ps(sum, _mm256_permute2f128_ps(sum, sum, 0x01));
  // … and now (s0 + s2) + (s1 + s3)
  sum = _mm256_add_ps(sum, _mm256_permute_ps(sum, _MM_SHUFFLE(1, 0, 3, 2)));
  // Store result.
  __m128 lower = _mm256_extractf128_ps(sum, 0);
  _mm_storel_pi((__m64 *) result, lower);

  // Handle the last elements if num_points mod 4 is bigger than 0.
  for (long unsigned i = num_points & ~3u; i < num_points; ++i) {
    *result += lv_cmake(
        lv_creal(input[i]) * lv_creal(taps[i]) + lv_cimag(input[i]) * lv_cimag(taps[i]),
        lv_cimag(input[i]) * lv_creal(taps[i]) - lv_creal(input[i]) * lv_cimag(taps[i]));
  }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_SSE3

#include <xmmintrin.h>
#include <pmmintrin.h>

static inline void volk_32fc_x2_conjugate_dot_prod_32fc_a_sse3(lv_32fc_t* result,
    const lv_32fc_t* input, const lv_32fc_t* taps, unsigned int num_points)
{
  // Partial sums for indices i and i+1.
  __m128 sum_a_mult_b_real = _mm_setzero_ps();
  __m128 sum_a_mult_b_imag = _mm_setzero_ps();

  for (long unsigned i = 0; i < (num_points & ~1u); i += 2) {
    /* Two complex elements a time are processed.
     * (ar + j⋅ai)*conj(br + j⋅bi) =
     * ar⋅br + ai⋅bi + j⋅(ai⋅br − ar⋅bi)
     */

    /* Load input and taps, split and duplicate real und imaginary parts of taps.
     * a: | ai,i+1 | ar,i+1 | ai,i+0 | ar,i+0 |
     * b: | bi,i+1 | br,i+1 | bi,i+0 | br,i+0 |
     * b_real: | br,i+1 | br,i+1 | br,i+0 | br,i+0 |
     * b_imag: | bi,i+1 | bi,i+1 | bi,i+0 | bi,i+0 |
     */
    __m128 a = _mm_load_ps((const float *) &input[i]);
    __m128 b = _mm_load_ps((const float *) &taps[i]);
    __m128 b_real = _mm_moveldup_ps(b);
    __m128 b_imag = _mm_movehdup_ps(b);

    // Add | ai⋅br,i+1 | ar⋅br,i+1 | ai⋅br,i+0 | ar⋅br,i+0 | to partial sum.
    sum_a_mult_b_real = _mm_add_ps(sum_a_mult_b_real, _mm_mul_ps(a, b_real));
    // Add | ai⋅bi,i+1 | −ar⋅bi,i+1 | ai⋅bi,i+0 | −ar⋅bi,i+0 | to partial sum.
    sum_a_mult_b_imag = _mm_addsub_ps(sum_a_mult_b_imag, _mm_mul_ps(a, b_imag));
  }

  // Swap position of −ar⋅bi and ai⋅bi.
  sum_a_mult_b_imag = _mm_shuffle_ps(sum_a_mult_b_imag, sum_a_mult_b_imag,
      _MM_SHUFFLE(2, 3, 0, 1));
  // | ai⋅br + ai⋅bi | ai⋅br − ar⋅bi |, sum contains two such partial sums.
  __m128 sum = _mm_add_ps(sum_a_mult_b_real, sum_a_mult_b_imag);
  // Sum the two partial sums.
  sum = _mm_add_ps(sum, _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(1, 0, 3, 2)));
  // Store result.
  _mm_storel_pi((__m64 *) result, sum);

  // Handle the last element if num_points mod 2 is 1.
  if (num_points & 1u) {
    *result += lv_cmake(
        lv_creal(input[num_points - 1]) * lv_creal(taps[num_points - 1]) +
        lv_cimag(input[num_points - 1]) * lv_cimag(taps[num_points - 1]),
        lv_cimag(input[num_points - 1]) * lv_creal(taps[num_points - 1]) -
        lv_creal(input[num_points - 1]) * lv_cimag(taps[num_points - 1]));
  }
}

#endif /*LV_HAVE_SSE3*/


#ifdef LV_HAVE_GENERIC


static inline void volk_32fc_x2_conjugate_dot_prod_32fc_a_generic(lv_32fc_t* result, const lv_32fc_t* input, const lv_32fc_t* taps, unsigned int num_points) {

  const unsigned int num_bytes = num_points*8;

  float * res = (float*) result;
  float * in = (float*) input;
  float * tp = (float*) taps;
  unsigned int n_2_ccomplex_blocks = num_bytes >> 4;
  unsigned int isodd = (num_bytes >> 3) &1;

  float sum0[2] = {0,0};
  float sum1[2] = {0,0};
  unsigned int i = 0;

  for(i = 0; i < n_2_ccomplex_blocks; ++i) {
    sum0[0] += in[0] * tp[0] + in[1] * tp[1];
    sum0[1] += (-in[0] * tp[1]) + in[1] * tp[0];
    sum1[0] += in[2] * tp[2] + in[3] * tp[3];
    sum1[1] += (-in[2] * tp[3]) + in[3] * tp[2];

    in += 4;
    tp += 4;
  }

  res[0] = sum0[0] + sum1[0];
  res[1] = sum0[1] + sum1[1];

  for(i = 0; i < isodd; ++i) {
    *result += input[(num_bytes >> 3) - 1] * lv_conj(taps[(num_bytes >> 3) - 1]);
  }
}

#endif /*LV_HAVE_GENERIC*/


#if LV_HAVE_SSE && LV_HAVE_64

static inline void volk_32fc_x2_conjugate_dot_prod_32fc_a_sse(lv_32fc_t* result, const lv_32fc_t* input, const lv_32fc_t* taps, unsigned int num_points) {

  const unsigned int num_bytes = num_points*8;

  __VOLK_ATTR_ALIGNED(16) static const uint32_t conjugator[4]= {0x00000000, 0x80000000, 0x00000000, 0x80000000};

  __VOLK_ASM __VOLK_VOLATILE
    (
     "#  ccomplex_conjugate_dotprod_generic (float* result, const float *input,\n\t"
     "#                         const float *taps, unsigned num_bytes)\n\t"
     "#    float sum0 = 0;\n\t"
     "#    float sum1 = 0;\n\t"
     "#    float sum2 = 0;\n\t"
     "#    float sum3 = 0;\n\t"
     "#    do {\n\t"
     "#      sum0 += input[0] * taps[0] - input[1] * taps[1];\n\t"
     "#      sum1 += input[0] * taps[1] + input[1] * taps[0];\n\t"
     "#      sum2 += input[2] * taps[2] - input[3] * taps[3];\n\t"
     "#      sum3 += input[2] * taps[3] + input[3] * taps[2];\n\t"
     "#      input += 4;\n\t"
     "#      taps += 4;  \n\t"
     "#    } while (--n_2_ccomplex_blocks != 0);\n\t"
     "#    result[0] = sum0 + sum2;\n\t"
     "#    result[1] = sum1 + sum3;\n\t"
     "# TODO: prefetch and better scheduling\n\t"
     "  xor    %%r9,  %%r9\n\t"
     "  xor    %%r10, %%r10\n\t"
     "  movq   %[conjugator], %%r9\n\t"
     "  movq   %%rcx, %%rax\n\t"
     "  movaps 0(%%r9), %%xmm8\n\t"
     "  movq   %%rcx, %%r8\n\t"
     "  movq   %[rsi],  %%r9\n\t"
     "  movq   %[rdx], %%r10\n\t"
     "	xorps	%%xmm6, %%xmm6		# zero accumulators\n\t"
     "	movaps	0(%%r9), %%xmm0\n\t"
     "	xorps	%%xmm7, %%xmm7		# zero accumulators\n\t"
     "	movups	0(%%r10), %%xmm2\n\t"
     "	shr	$5, %%rax		# rax = n_2_ccomplex_blocks / 2\n\t"
     "  shr     $4, %%r8\n\t"
     "  xorps  %%xmm8, %%xmm2\n\t"
     "	jmp	.%=L1_test\n\t"
     "	# 4 taps / loop\n\t"
     "	# something like ?? cycles / loop\n\t"
     ".%=Loop1:	\n\t"
     "# complex prod: C += A * B,  w/ temp Z & Y (or B), xmmPN=$0x8000000080000000\n\t"
     "#	movaps	(%%r9), %%xmmA\n\t"
     "#	movaps	(%%r10), %%xmmB\n\t"
     "#	movaps	%%xmmA, %%xmmZ\n\t"
     "#	shufps	$0xb1, %%xmmZ, %%xmmZ	# swap internals\n\t"
     "#	mulps	%%xmmB, %%xmmA\n\t"
     "#	mulps	%%xmmZ, %%xmmB\n\t"
     "#	# SSE replacement for: pfpnacc %%xmmB, %%xmmA\n\t"
     "#	xorps	%%xmmPN, %%xmmA\n\t"
     "#	movaps	%%xmmA, %%xmmZ\n\t"
     "#	unpcklps %%xmmB, %%xmmA\n\t"
     "#	unpckhps %%xmmB, %%xmmZ\n\t"
     "#	movaps	%%xmmZ, %%xmmY\n\t"
     "#	shufps	$0x44, %%xmmA, %%xmmZ	# b01000100\n\t"
     "#	shufps	$0xee, %%xmmY, %%xmmA	# b11101110\n\t"
     "#	addps	%%xmmZ, %%xmmA\n\t"
     "#	addps	%%xmmA, %%xmmC\n\t"
     "# A=xmm0, B=xmm2, Z=xmm4\n\t"
     "# A'=xmm1, B'=xmm3, Z'=xmm5\n\t"
     "	movaps	16(%%r9), %%xmm1\n\t"
     "	movaps	%%xmm0, %%xmm4\n\t"
     "	mulps	%%xmm2, %%xmm0\n\t"
     "	shufps	$0xb1, %%xmm4, %%xmm4	# swap internals\n\t"
     "	movaps	16(%%r10), %%xmm3\n\t"
     "	movaps	%%xmm1, %%xmm5\n\t"
     "  xorps   %%xmm8, %%xmm3\n\t"
     "	addps	%%xmm0, %%xmm6\n\t"
     "	mulps	%%xmm3, %%xmm1\n\t"
     "	shufps	$0xb1, %%xmm5, %%xmm5	# swap internals\n\t"
     "	addps	%%xmm1, %%xmm6\n\t"
     "	mulps	%%xmm4, %%xmm2\n\t"
     "	movaps	32(%%r9), %%xmm0\n\t"
     "	addps	%%xmm2, %%xmm7\n\t"
     "	mulps	%%xmm5, %%xmm3\n\t"
     "	add	$32, %%r9\n\t"
     "	movaps	32(%%r10), %%xmm2\n\t"
     "	addps	%%xmm3, %%xmm7\n\t"
     "	add	$32, %%r10\n\t"
     "  xorps   %%xmm8, %%xmm2\n\t"
     ".%=L1_test:\n\t"
     "	dec	%%rax\n\t"
     "	jge	.%=Loop1\n\t"
     "	# We've handled the bulk of multiplies up to here.\n\t"
     "	# Let's sse if original n_2_ccomplex_blocks was odd.\n\t"
     "	# If so, we've got 2 more taps to do.\n\t"
     "	and	$1, %%r8\n\t"
     "	je	.%=Leven\n\t"
     "	# The count was odd, do 2 more taps.\n\t"
     "	# Note that we've already got mm0/mm2 preloaded\n\t"
     "	# from the main loop.\n\t"
     "	movaps	%%xmm0, %%xmm4\n\t"
     "	mulps	%%xmm2, %%xmm0\n\t"
     "	shufps	$0xb1, %%xmm4, %%xmm4	# swap internals\n\t"
     "	addps	%%xmm0, %%xmm6\n\t"
     "	mulps	%%xmm4, %%xmm2\n\t"
     "	addps	%%xmm2, %%xmm7\n\t"
     ".%=Leven:\n\t"
     "	# neg inversor\n\t"
     "	xorps	%%xmm1, %%xmm1\n\t"
     "	mov	$0x80000000, %%r9\n\t"
     "	movd	%%r9, %%xmm1\n\t"
     "	shufps	$0x11, %%xmm1, %%xmm1	# b00010001 # 0 -0 0 -0\n\t"
     "	# pfpnacc\n\t"
     "	xorps	%%xmm1, %%xmm6\n\t"
     "	movaps	%%xmm6, %%xmm2\n\t"
     "	unpcklps %%xmm7, %%xmm6\n\t"
     "	unpckhps %%xmm7, %%xmm2\n\t"
     "	movaps	%%xmm2, %%xmm3\n\t"
     "	shufps	$0x44, %%xmm6, %%xmm2	# b01000100\n\t"
     "	shufps	$0xee, %%xmm3, %%xmm6	# b11101110\n\t"
     "	addps	%%xmm2, %%xmm6\n\t"
     "					# xmm6 = r1 i2 r3 i4\n\t"
     "	movhlps	%%xmm6, %%xmm4		# xmm4 = r3 i4 ?? ??\n\t"
     "	addps	%%xmm4, %%xmm6		# xmm6 = r1+r3 i2+i4 ?? ??\n\t"
     "	movlps	%%xmm6, (%[rdi])		# store low 2x32 bits (complex) to memory\n\t"
     :
     :[rsi] "r" (input), [rdx] "r" (taps), "c" (num_bytes), [rdi] "r" (result), [conjugator] "r" (conjugator)
     :"rax", "r8", "r9", "r10"
     );

  int getem = num_bytes % 16;

  for(; getem > 0; getem -= 8) {
    *result += (input[(num_bytes >> 3) - 1] * lv_conj(taps[(num_bytes >> 3) - 1]));
  }
}
#endif

#if LV_HAVE_SSE && LV_HAVE_32
static inline void volk_32fc_x2_conjugate_dot_prod_32fc_a_sse_32(lv_32fc_t* result, const lv_32fc_t* input, const lv_32fc_t* taps, unsigned int num_points) {

  const unsigned int num_bytes = num_points*8;

  __VOLK_ATTR_ALIGNED(16) static const uint32_t conjugator[4]= {0x00000000, 0x80000000, 0x00000000, 0x80000000};

  int bound = num_bytes >> 4;
  int leftovers = num_bytes % 16;

  __VOLK_ASM __VOLK_VOLATILE
    (
     "	#pushl	%%ebp\n\t"
     "	#movl	%%esp, %%ebp\n\t"
     "	#movl	12(%%ebp), %%eax		# input\n\t"
     "	#movl	16(%%ebp), %%edx		# taps\n\t"
     "	#movl	20(%%ebp), %%ecx                # n_bytes\n\t"
     "  movaps  0(%[conjugator]), %%xmm1\n\t"
     "	xorps	%%xmm6, %%xmm6		# zero accumulators\n\t"
     "	movaps	0(%[eax]), %%xmm0\n\t"
     "	xorps	%%xmm7, %%xmm7		# zero accumulators\n\t"
     "	movaps	0(%[edx]), %%xmm2\n\t"
     "  movl    %[ecx], (%[out])\n\t"
     "	shrl	$5, %[ecx]		# ecx = n_2_ccomplex_blocks / 2\n\t"

     "  xorps   %%xmm1, %%xmm2\n\t"
     "	jmp	.%=L1_test\n\t"
     "	# 4 taps / loop\n\t"
     "	# something like ?? cycles / loop\n\t"
     ".%=Loop1:	\n\t"
     "# complex prod: C += A * B,  w/ temp Z & Y (or B), xmmPN=$0x8000000080000000\n\t"
     "#	movaps	(%[eax]), %%xmmA\n\t"
     "#	movaps	(%[edx]), %%xmmB\n\t"
     "#	movaps	%%xmmA, %%xmmZ\n\t"
     "#	shufps	$0xb1, %%xmmZ, %%xmmZ	# swap internals\n\t"
     "#	mulps	%%xmmB, %%xmmA\n\t"
     "#	mulps	%%xmmZ, %%xmmB\n\t"
     "#	# SSE replacement for: pfpnacc %%xmmB, %%xmmA\n\t"
     "#	xorps	%%xmmPN, %%xmmA\n\t"
     "#	movaps	%%xmmA, %%xmmZ\n\t"
     "#	unpcklps %%xmmB, %%xmmA\n\t"
     "#	unpckhps %%xmmB, %%xmmZ\n\t"
     "#	movaps	%%xmmZ, %%xmmY\n\t"
     "#	shufps	$0x44, %%xmmA, %%xmmZ	# b01000100\n\t"
     "#	shufps	$0xee, %%xmmY, %%xmmA	# b11101110\n\t"
     "#	addps	%%xmmZ, %%xmmA\n\t"
     "#	addps	%%xmmA, %%xmmC\n\t"
     "# A=xmm0, B=xmm2, Z=xmm4\n\t"
     "# A'=xmm1, B'=xmm3, Z'=xmm5\n\t"
     "	movaps	16(%[edx]), %%xmm3\n\t"
     "	movaps	%%xmm0, %%xmm4\n\t"
     "  xorps   %%xmm1, %%xmm3\n\t"
     "	mulps	%%xmm2, %%xmm0\n\t"
     "	movaps	16(%[eax]), %%xmm1\n\t"
     "	shufps	$0xb1, %%xmm4, %%xmm4	# swap internals\n\t"
     "	movaps	%%xmm1, %%xmm5\n\t"
     "	addps	%%xmm0, %%xmm6\n\t"
     "	mulps	%%xmm3, %%xmm1\n\t"
     "	shufps	$0xb1, %%xmm5, %%xmm5	# swap internals\n\t"
     "	addps	%%xmm1, %%xmm6\n\t"
     "  movaps  0(%[conjugator]), %%xmm1\n\t"
     "	mulps	%%xmm4, %%xmm2\n\t"
     "	movaps	32(%[eax]), %%xmm0\n\t"
     "	addps	%%xmm2, %%xmm7\n\t"
     "	mulps	%%xmm5, %%xmm3\n\t"
     "	addl	$32, %[eax]\n\t"
     "	movaps	32(%[edx]), %%xmm2\n\t"
     "	addps	%%xmm3, %%xmm7\n\t"
     "  xorps   %%xmm1, %%xmm2\n\t"
     "	addl	$32, %[edx]\n\t"
     ".%=L1_test:\n\t"
     "	decl	%[ecx]\n\t"
     "	jge	.%=Loop1\n\t"
     "	# We've handled the bulk of multiplies up to here.\n\t"
     "	# Let's sse if original n_2_ccomplex_blocks was odd.\n\t"
     "	# If so, we've got 2 more taps to do.\n\t"
     "	movl	0(%[out]), %[ecx]		# n_2_ccomplex_blocks\n\t"
     "  shrl    $4, %[ecx]\n\t"
     "	andl	$1, %[ecx]\n\t"
     "	je	.%=Leven\n\t"
     "	# The count was odd, do 2 more taps.\n\t"
     "	# Note that we've already got mm0/mm2 preloaded\n\t"
     "	# from the main loop.\n\t"
     "	movaps	%%xmm0, %%xmm4\n\t"
     "	mulps	%%xmm2, %%xmm0\n\t"
     "	shufps	$0xb1, %%xmm4, %%xmm4	# swap internals\n\t"
     "	addps	%%xmm0, %%xmm6\n\t"
     "	mulps	%%xmm4, %%xmm2\n\t"
     "	addps	%%xmm2, %%xmm7\n\t"
     ".%=Leven:\n\t"
     "	# neg inversor\n\t"
     "  #movl 8(%%ebp), %[eax] \n\t"
     "	xorps	%%xmm1, %%xmm1\n\t"
     "  movl	$0x80000000, (%[out])\n\t"
     "	movss	(%[out]), %%xmm1\n\t"
     "	shufps	$0x11, %%xmm1, %%xmm1	# b00010001 # 0 -0 0 -0\n\t"
     "	# pfpnacc\n\t"
     "	xorps	%%xmm1, %%xmm6\n\t"
     "	movaps	%%xmm6, %%xmm2\n\t"
     "	unpcklps %%xmm7, %%xmm6\n\t"
     "	unpckhps %%xmm7, %%xmm2\n\t"
     "	movaps	%%xmm2, %%xmm3\n\t"
     "	shufps	$0x44, %%xmm6, %%xmm2	# b01000100\n\t"
     "	shufps	$0xee, %%xmm3, %%xmm6	# b11101110\n\t"
     "	addps	%%xmm2, %%xmm6\n\t"
     "					# xmm6 = r1 i2 r3 i4\n\t"
     "	#movl	8(%%ebp), %[eax]		# @result\n\t"
     "	movhlps	%%xmm6, %%xmm4		# xmm4 = r3 i4 ?? ??\n\t"
     "	addps	%%xmm4, %%xmm6		# xmm6 = r1+r3 i2+i4 ?? ??\n\t"
     "	movlps	%%xmm6, (%[out])		# store low 2x32 bits (complex) to memory\n\t"
     "	#popl	%%ebp\n\t"
     :
     : [eax] "r" (input), [edx] "r" (taps), [ecx] "r" (num_bytes), [out] "r" (result), [conjugator] "r" (conjugator)
     );

  for(; leftovers > 0; leftovers -= 8) {
    *result += (input[(bound << 1)] * lv_conj(taps[(bound << 1)]));
  }
}
#endif /*LV_HAVE_SSE*/


#endif /*INCLUDED_volk_32fc_x2_conjugate_dot_prod_32fc_a_H*/
