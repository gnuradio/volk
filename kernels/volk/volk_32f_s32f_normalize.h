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
 * \page volk_32f_s32f_normalize
 *
 * \b Overview
 *
 * Normalizes all points in the buffer by the scalar value (divides
 * each data point by the scalar value).
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_s32f_normalize(float* vecBuffer, const float scalar, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li vecBuffer: The buffer of values to be vectorized.
 * \li scalar: The scale value to be applied to each buffer value.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li vecBuffer: returns as an in-place calculation.
 *
 * \b Example
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   float* increasing = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* out = (float*)volk_malloc(sizeof(float)*N, alignment);
 *
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       increasing[ii] = 2.f * ((float)ii / (float)N) - 1.f;
 *   }
 *
 *   // Normalize by the smallest delta (0.2 in this example)
 *   float scale = 5.0f;
 *
 *   volk_32f_s32f_normalize(increasing, scale, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("increasing[%u] = %f\n", ii, increasing[ii]);
 *   }
 *
 *   volk_free(increasing);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_s32f_normalize_a_H
#define INCLUDED_volk_32f_s32f_normalize_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_s32f_normalize_a_sse(float* vecBuffer, const float scalar, unsigned int num_points){
  unsigned int number = 0;
  float* inputPtr = vecBuffer;

  const float invScalar = 1.0 / scalar;
  __m128 vecScalar = _mm_set_ps1(invScalar);

  __m128 input1;

  const uint64_t quarterPoints = num_points / 4;
  for(;number < quarterPoints; number++){

    input1 = _mm_load_ps(inputPtr);

    input1 = _mm_mul_ps(input1, vecScalar);

    _mm_store_ps(inputPtr, input1);

    inputPtr += 4;
  }

  number = quarterPoints*4;
  for(; number < num_points; number++){
    *inputPtr *= invScalar;
    inputPtr++;
  }
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_GENERIC

static inline void volk_32f_s32f_normalize_generic(float* vecBuffer, const float scalar, unsigned int num_points){
  unsigned int number = 0;
  float* inputPtr = vecBuffer;
  const float invScalar = 1.0 / scalar;
  for(number = 0; number < num_points; number++){
    *inputPtr *= invScalar;
    inputPtr++;
  }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_ORC

extern void volk_32f_s32f_normalize_a_orc_impl(float* dst, float* src, const float scalar, unsigned int num_points);
static inline void volk_32f_s32f_normalize_u_orc(float* vecBuffer, const float scalar, unsigned int num_points){
    float invscalar = 1.0 / scalar;
    volk_32f_s32f_normalize_a_orc_impl(vecBuffer, vecBuffer, invscalar, num_points);
}
#endif /* LV_HAVE_GENERIC */




#endif /* INCLUDED_volk_32f_s32f_normalize_a_H */
