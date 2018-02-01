/* -*- c++ -*- */
/*
 * Copyright 2015 Free Software Foundation, Inc.
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
 * \page volk_8i_x2_saturated_sum_8i
 *
 * \b Overview
 *
 * Carry out a saturated addition on a vectors of signed 8-bit integers.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_8i_x2_saturated_sum_8i(int8_t* outVector, const int8_t* inVectorA, const int8_t* inVectorB, const unsigned int num_vals)
 * \endcode
 *
 * \b Inputs
 * \li inVectorA: The first input vector of 8-bit integers.
 * \li inVectorB: The second input vector of 8-bit integers.
 * \li num_vals: The number of data points.
 *
 * \b Outputs
 * \li outVector: The resulting output of 8-bit integers.
 *
 * \b Example
 *
 * The follow example adds some vectors such that the result will eventually saturate
 *
 * \code
 *   int N = 100;
 *   unsigned int alignment = volk_get_alignment();
 *   int8_t* vecA = (int8_t*)volk_malloc(sizeof(int8_t)*N, alignment);
 *   int8_t* vecB = (int8_t*)volk_malloc(sizeof(int8_t)*N, alignment);
 *   int8_t* out = (int8_t*)volk_malloc(sizeof(int8_t)*N, alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       vecA[ii] = ii;
 *       vecB[ii] = 100+ii;
 *   }
 *
 *   volk_8i_x2_saturated_sum_8i(out, vecA, vecB, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("out[%u] = %d\n", ii, out[ii]);
 *   }
 *
 *   volk_free(vecA);
 *   volk_free(vecB);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_8i_x2_saturated_sum_8i_u_H
#define INCLUDED_volk_8i_x2_saturated_sum_8i_u_H

#include <stdint.h>
#include <stdio.h>

#ifdef LV_HAVE_GENERIC

static inline void
volk_8i_x2_saturated_sum_8i_generic(int8_t* outVector,
                                    const int8_t* inVectorA, const int8_t* inVectorB,
                                    const unsigned int num_vals)
{
  unsigned int i = 0;
  int16_t tmpPlus; // Need to move to more bits to workaround possible overflow

  for(; i < num_vals; i++){
    tmpPlus = inVectorB[i] + inVectorA[i];
    // Saturate to avoid overflow
    tmpPlus = (tmpPlus > 127) ? 127 : tmpPlus;
    tmpPlus = (tmpPlus < -128) ? -128 : tmpPlus;

    outVector[i] = ((int8_t)tmpPlus);
  }
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_SSE2

#include <x86intrin.h>

static inline void
volk_8i_x2_saturated_sum_8i_u_sse2(int8_t* outVector,
                                   const int8_t* inVectorA, const int8_t* inVectorB,
                                   const unsigned int num_vals)
{
  const unsigned int VEC_SIZE = 16;
  const unsigned int vecNum = num_vals / VEC_SIZE;
  unsigned int i = 0;
  int16_t tmpPlus; // Need to move to more bits to workaround possible overflow

  for(; i < vecNum; i++) {
    __m128i a = _mm_loadu_si128((const __m128i*)(inVectorA + VEC_SIZE*i));
    __m128i b = _mm_loadu_si128((const __m128i*)(inVectorB + VEC_SIZE*i));
    __m128i result = _mm_adds_epi8(b, a);
    _mm_storeu_si128((__m128i*)(outVector + VEC_SIZE*i), result);
  }

  i = VEC_SIZE*vecNum;
  for(; i < num_vals; i++) {
    tmpPlus = inVectorB[i] + inVectorA[i];
    // Saturate to avoid overflow
    tmpPlus = (tmpPlus > 127) ? 127 : tmpPlus;
    tmpPlus = (tmpPlus < -128) ? -128 : tmpPlus;

    outVector[i] = ((int8_t)tmpPlus);
  }
}

#endif /* LV_HAVE_SSE2 for unaligned */


#ifdef LV_HAVE_AVX2

#include <x86intrin.h>

static inline void
volk_8i_x2_saturated_sum_8i_u_avx2(int8_t* outVector,
                                   const int8_t* inVectorA, const int8_t* inVectorB,
                                   const unsigned int num_vals)
{
  const unsigned int VEC_SIZE = 32;
  const unsigned int vecNum = num_vals / VEC_SIZE;
  unsigned int i = 0;
  int16_t tmpPlus; // Need to move to more bits to workaround possible overflow

  for(; i < vecNum; i++) {
    __m256i a = _mm256_loadu_si256((const __m256i*)(inVectorA + VEC_SIZE*i));
    __m256i b = _mm256_loadu_si256((const __m256i*)(inVectorB + VEC_SIZE*i));
    __m256i result = _mm256_adds_epi8(b, a);
    _mm256_storeu_si256((__m256i*)(outVector + VEC_SIZE*i), result);
  }

  i = VEC_SIZE*vecNum;
  for(; i < num_vals; i++) {
    tmpPlus = inVectorB[i] + inVectorA[i];
    // Saturate to avoid overflow
    tmpPlus = (tmpPlus > 127) ? 127 : tmpPlus;
    tmpPlus = (tmpPlus < -128) ? -128 : tmpPlus;

    outVector[i] = ((int8_t)tmpPlus);
  }
}

#endif /* LV_HAVE_AVX2 for unaligned */

#endif /* INCLUDED_VOLK_8s_SATURATED_SUM_8s_UNALIGNED8_H */


#ifndef INCLUDED_volk_8i_x2_saturated_sum_8i_a_H
#define INCLUDED_volk_8i_x2_saturated_sum_8i_a_H

#ifdef LV_HAVE_SSE2

#include <x86intrin.h>

static inline void
volk_8i_x2_saturated_sum_8i_a_sse2(int8_t* outVector,
                                   const int8_t* inVectorA, const int8_t* inVectorB,
                                   const unsigned int num_vals)
{
  const unsigned int VEC_SIZE = 16;
  const unsigned int vecNum = num_vals / VEC_SIZE;
  unsigned int i = 0;
  int16_t tmpPlus; // Need to move to more bits to workaround possible overflow

  for(; i < vecNum; i++) {
    __m128i a = _mm_load_si128((const __m128i*)(inVectorA + VEC_SIZE*i));
    __m128i b = _mm_load_si128((const __m128i*)(inVectorB + VEC_SIZE*i));
    __m128i result = _mm_adds_epi8(b, a);
    _mm_store_si128((__m128i*)(outVector + VEC_SIZE*i), result);
  }

  i = VEC_SIZE*vecNum;
  for(; i < num_vals; i++) {
    tmpPlus = inVectorB[i] + inVectorA[i];
    // Saturate to avoid overflow
    tmpPlus = (tmpPlus > 127) ? 127 : tmpPlus;
    tmpPlus = (tmpPlus < -128) ? -128 : tmpPlus;

    outVector[i] = ((int8_t)tmpPlus);
  }
}

#endif /* LV_HAVE_SSE2 for aligned */


#ifdef LV_HAVE_AVX2

#include <x86intrin.h>

static inline void
volk_8i_x2_saturated_sum_8i_a_avx2(int8_t* outVector,
                                   const int8_t* inVectorA, const int8_t* inVectorB,
                                   const unsigned int num_vals)
{
  const unsigned int VEC_SIZE = 32;
  const unsigned int vecNum = num_vals / VEC_SIZE;
  unsigned int i = 0;
  int16_t tmpPlus; // Need to move to more bits to workaround possible overflow

  for(; i < vecNum; i++) {
    __m256i a = _mm256_load_si256((const __m256i*)(inVectorA + VEC_SIZE*i));
    __m256i b = _mm256_load_si256((const __m256i*)(inVectorB + VEC_SIZE*i));
    __m256i result = _mm256_adds_epi8(b, a);
    _mm256_store_si256((__m256i*)(outVector + VEC_SIZE*i), result);
  }

  i = VEC_SIZE*vecNum;
  for(; i < num_vals; i++) {
    tmpPlus = inVectorB[i] + inVectorA[i];
    // Saturate to avoid overflow
    tmpPlus = (tmpPlus > 127) ? 127 : tmpPlus;
    tmpPlus = (tmpPlus < -128) ? -128 : tmpPlus;

    outVector[i] = ((int8_t)tmpPlus);
  }
}

#endif /* LV_HAVE_AVX2 for aligned */


#ifdef LV_HAVE_NEON

#include <arm_neon.h>

static inline void
volk_8i_x2_saturated_sum_8i_a_neon(int8_t* outVector,
                                   const int8_t* inVectorA, const int8_t* inVectorB,
                                   const unsigned int num_vals)
{
  const unsigned int VEC_SIZE = 16;
  const unsigned int vecNum = num_vals / VEC_SIZE;
  unsigned int i = 0;
  int16_t tmpPlus; // Need to move to more bits to workaround possible overflow

  for(; i < vecNum; i++) {
		int8x16_t a = vld1q_s8((const int8x16_t*)(inVectorA + VEC_SIZE*i));
		int8x16_t b = vld1q_s8((const int8x16_t*)(inVectorB + VEC_SIZE*i));
		vst1q_s8((const int8x16_t*)(outVector + VEC_SIZE*i), vqaddq_s8(b, a));
  }

  i = VEC_SIZE*vecNum;
  for(; i < num_vals; i++) {
    tmpPlus = inVectorB[i] + inVectorA[i];
    // Saturate to avoid overflow
    tmpPlus = (tmpPlus > 127) ? 127 : tmpPlus;
    tmpPlus = (tmpPlus < -128) ? -128 : tmpPlus;

    outVector[i] = ((int8_t)tmpPlus);
  }
}

#endif /* LV_HAVE_NEON for aligned */

#endif /* INCLUDED_VOLK_8s_SATURATED_SUM_8s_ALIGNED8_H */
