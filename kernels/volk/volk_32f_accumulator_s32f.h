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
 * \page volk_32f_accumulator_s32f
 *
 * \b Overview
 *
 * Accumulates the values in the input buffer.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_accumulator_s32f(float* result, const float* inputBuffer, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li inputBuffer The buffer of data to be accumulated
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li result The accumulated result.
 *
 * \b Example
 * Calculate the sum of numbers  0 through 99
 * \code
 *   int N = 100;
 *   unsigned int alignment = volk_get_alignment();
 *   float* increasing = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* out = (float*)volk_malloc(sizeof(float), alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       increasing[ii] = (float)ii;
 *   }
 *
 *   volk_32f_accumulator_s32f(out, increasing, N);
 *
 *   printf("sum(1..100) = %1.2f\n", out[0]);
 *
 *   volk_free(increasing);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_accumulator_s32f_a_H
#define INCLUDED_volk_32f_accumulator_s32f_a_H

#include <volk/volk_common.h>
#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_SSE

#include <xmmintrin.h>

static inline void
volk_32f_accumulator_s32f_a_sse(float* result, const float* inputBuffer, unsigned int num_points)
{
  float returnValue = 0;
  unsigned int number = 0;
  const unsigned int quarterPoints = num_points / 4;

  const float* aPtr = inputBuffer;
  __VOLK_ATTR_ALIGNED(16) float tempBuffer[4];

  __m128 accumulator = _mm_setzero_ps();
  __m128 aVal = _mm_setzero_ps();

  for(;number < quarterPoints; number++){
    aVal = _mm_load_ps(aPtr);
    accumulator = _mm_add_ps(accumulator, aVal);
    aPtr += 4;
  }

  _mm_store_ps(tempBuffer,accumulator); // Store the results back into the C container

  returnValue = tempBuffer[0];
  returnValue += tempBuffer[1];
  returnValue += tempBuffer[2];
  returnValue += tempBuffer[3];

  number = quarterPoints * 4;
  for(;number < num_points; number++){
    returnValue += (*aPtr++);
  }
  *result = returnValue;
}

#endif /* LV_HAVE_SSE */



#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_accumulator_s32f_generic(float* result, const float* inputBuffer, unsigned int num_points)
{
  const float* aPtr = inputBuffer;
  unsigned int number = 0;
  float returnValue = 0;

  for(;number < num_points; number++){
    returnValue += (*aPtr++);
  }
  *result = returnValue;
}

#endif /* LV_HAVE_GENERIC */

#endif /* INCLUDED_volk_32f_accumulator_s32f_a_H */
