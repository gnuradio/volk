/* -*- c++ -*- */
/*
 * Copyright 2014 Free Software Foundation, Inc.
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
 * \page volk_32f_binary_slicer_32i
 *
 * \b Overview
 *
 * Slices input floats and and returns 1 when the input >= 0 and 0
 * when < 0. Results are returned as 32-bit ints.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_binary_slicer_32i(int* cVector, const float* aVector, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li aVector: The input vector of floats.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li cVector: The output vector of 32-bit ints.
 *
 * \b Example
 * Generate ints of a 7-bit barker code from floats.
 * \code
 *   int N = 7;
 *   unsigned int alignment = volk_get_alignment();
 *   float* in = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   int32_t* out = (int32_t*)volk_malloc(sizeof(int32_t)*N, alignment);
 *
 *   in[0] = 0.9f;
 *   in[1] = 1.1f;
 *   in[2] = 0.4f;
 *   in[3] = -0.7f;
 *   in[5] = -1.2f;
 *   in[6] = 0.2f;
 *   in[7] = -0.8f;
 *
 *   volk_32f_binary_slicer_32i(out, in, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       VOLK_LOG("out(%i) = %i\n", ii, out[ii]);
 *   }
 *
 *   volk_free(in);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_binary_slicer_32i_H
#define INCLUDED_volk_32f_binary_slicer_32i_H


#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_binary_slicer_32i_generic(int* cVector, const float* aVector, unsigned int num_points)
{
  int* cPtr = cVector;
  const float* aPtr = aVector;
  unsigned int number = 0;

  for(number = 0; number < num_points; number++){
    if( *aPtr++ >= 0) {
      *cPtr++ = 1;
    }
    else {
      *cPtr++ = 0;
    }
  }
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_binary_slicer_32i_generic_branchless(int* cVector, const float* aVector, unsigned int num_points)
{
  int* cPtr = cVector;
  const float* aPtr = aVector;
  unsigned int number = 0;

  for(number = 0; number < num_points; number++){
    *cPtr++ = (*aPtr++ >= 0);
  }
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void
volk_32f_binary_slicer_32i_a_sse2(int* cVector, const float* aVector, unsigned int num_points)
{
  int* cPtr = cVector;
  const float* aPtr = aVector;
  unsigned int number = 0;

  unsigned int quarter_points = num_points / 4;
  __m128 a_val, res_f;
  __m128i res_i, binary_i;
  __m128 zero_val;
  zero_val = _mm_set1_ps (0.0f);

  for(number = 0; number < quarter_points; number++){
    a_val = _mm_load_ps(aPtr);

    res_f = _mm_cmpge_ps (a_val, zero_val);
    res_i = _mm_cvtps_epi32 (res_f);
    binary_i = _mm_srli_epi32 (res_i, 31);

    _mm_store_si128((__m128i*)cPtr, binary_i);

    cPtr += 4;
    aPtr += 4;
  }

  for(number = quarter_points * 4; number < num_points; number++){
    if( *aPtr++ >= 0) {
      *cPtr++ = 1;
    }
    else {
      *cPtr++ = 0;
    }
  }
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32f_binary_slicer_32i_a_avx(int* cVector, const float* aVector, unsigned int num_points)
{
  int* cPtr = cVector;
  const float* aPtr = aVector;
  unsigned int number = 0;

  unsigned int quarter_points = num_points / 8;
  __m256 a_val, res_f, binary_f;
  __m256i binary_i;
  __m256 zero_val, one_val;
  zero_val = _mm256_set1_ps (0.0f);
  one_val = _mm256_set1_ps (1.0f);

  for(number = 0; number < quarter_points; number++){
    a_val = _mm256_load_ps(aPtr);

    res_f = _mm256_cmp_ps (a_val, zero_val, 13);
    binary_f = _mm256_and_ps (res_f, one_val);
    binary_i = _mm256_cvtps_epi32(binary_f);

    _mm256_store_si256((__m256i *)cPtr, binary_i);

    cPtr += 8;
    aPtr += 8;
  }

  for(number = quarter_points * 8; number < num_points; number++){
    if( *aPtr++ >= 0) {
      *cPtr++ = 1;
    }
    else {
      *cPtr++ = 0;
    }
  }
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void
volk_32f_binary_slicer_32i_u_sse2(int* cVector, const float* aVector, unsigned int num_points)
{
  int* cPtr = cVector;
  const float* aPtr = aVector;
  unsigned int number = 0;

  unsigned int quarter_points = num_points / 4;
  __m128 a_val, res_f;
  __m128i res_i, binary_i;
  __m128 zero_val;
  zero_val = _mm_set1_ps (0.0f);

  for(number = 0; number < quarter_points; number++){
    a_val = _mm_loadu_ps(aPtr);

    res_f = _mm_cmpge_ps (a_val, zero_val);
    res_i = _mm_cvtps_epi32 (res_f);
    binary_i = _mm_srli_epi32 (res_i, 31);

    _mm_storeu_si128((__m128i*)cPtr, binary_i);

    cPtr += 4;
    aPtr += 4;
  }

  for(number = quarter_points * 4; number < num_points; number++){
    if( *aPtr++ >= 0) {
      *cPtr++ = 1;
    }
    else {
      *cPtr++ = 0;
    }
  }
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32f_binary_slicer_32i_u_avx(int* cVector, const float* aVector, unsigned int num_points)
{
  int* cPtr = cVector;
  const float* aPtr = aVector;
  unsigned int number = 0;

  unsigned int quarter_points = num_points / 8;
  __m256 a_val, res_f, binary_f;
  __m256i binary_i;
  __m256 zero_val, one_val;
  zero_val = _mm256_set1_ps (0.0f);
  one_val = _mm256_set1_ps (1.0f);

  for(number = 0; number < quarter_points; number++){
    a_val = _mm256_loadu_ps(aPtr);

    res_f = _mm256_cmp_ps (a_val, zero_val, 13);
    binary_f = _mm256_and_ps (res_f, one_val);
    binary_i = _mm256_cvtps_epi32(binary_f);

    _mm256_storeu_si256((__m256i*)cPtr, binary_i);

    cPtr += 8;
    aPtr += 8;
  }

  for(number = quarter_points * 8; number < num_points; number++){
    if( *aPtr++ >= 0) {
      *cPtr++ = 1;
    }
    else {
      *cPtr++ = 0;
    }
  }
}
#endif /* LV_HAVE_AVX */


#endif /* INCLUDED_volk_32f_binary_slicer_32i_H */
