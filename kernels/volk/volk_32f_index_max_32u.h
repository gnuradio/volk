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
 * \page volk_32f_index_max_32u
 *
 * \b Overview
 *
 * Returns Argmax_i x[i]. Finds and returns the index which contains the maximum value in the given vector.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_index_max_32u(uint32_t* target, const float* src0, uint32_t num_points)
 * \endcode
 *
 * \b Inputs
 * \li src0: The input vector of floats.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li target: The index of the maximum value in the input buffer.
 *
 * \b Example
 * \code
 *   int N = 10;
 *   uint32_t alignment = volk_get_alignment();
 *   float* in = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   uint32_t* out = (uint32_t*)volk_malloc(sizeof(uint32_t), alignment);
 *
 *   for(uint32_t ii = 0; ii < N; ++ii){
 *       float x = (float)ii;
 *       // a parabola with a maximum at x=4
 *       in[ii] = -(x-4) * (x-4) + 5;
 *   }
 *
 *   volk_32f_index_max_32u(out, in, N);
 *
 *   printf("maximum is %1.2f at index %u\n", in[*out], *out);
 *
 *   volk_free(in);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_index_max_32u_a_H
#define INCLUDED_volk_32f_index_max_32u_a_H

#include <volk/volk_common.h>
#include <volk/volk_common.h>
#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_SSE4_1
#include<smmintrin.h>

static inline void
volk_32f_index_max_32u_a_sse4_1(uint32_t* target, const float* src0, uint32_t num_points)
{
  if(num_points > 0){
    uint32_t number = 0;
    const uint32_t quarterPoints = num_points / 4;

    float* inputPtr = (float*)src0;

    __m128 indexIncrementValues = _mm_set1_ps(4);
    __m128 currentIndexes = _mm_set_ps(-1,-2,-3,-4);

    float max = src0[0];
    float index = 0;
    __m128 maxValues = _mm_set1_ps(max);
    __m128 maxValuesIndex = _mm_setzero_ps();
    __m128 compareResults;
    __m128 currentValues;

    __VOLK_ATTR_ALIGNED(16) float maxValuesBuffer[4];
    __VOLK_ATTR_ALIGNED(16) float maxIndexesBuffer[4];

    for(;number < quarterPoints; number++){

      currentValues  = _mm_load_ps(inputPtr); inputPtr += 4;
      currentIndexes = _mm_add_ps(currentIndexes, indexIncrementValues);

      compareResults = _mm_cmpgt_ps(maxValues, currentValues);

      maxValuesIndex = _mm_blendv_ps(currentIndexes, maxValuesIndex, compareResults);
      maxValues      = _mm_blendv_ps(currentValues, maxValues, compareResults);
    }

    // Calculate the largest value from the remaining 4 points
    _mm_store_ps(maxValuesBuffer, maxValues);
    _mm_store_ps(maxIndexesBuffer, maxValuesIndex);

    for(number = 0; number < 4; number++){
      if(maxValuesBuffer[number] > max){
	index = maxIndexesBuffer[number];
	max = maxValuesBuffer[number];
      }
    }

    number = quarterPoints * 4;
    for(;number < num_points; number++){
      if(src0[number] > max){
	index = number;
	max = src0[number];
      }
    }
    target[0] = (uint32_t)index;
  }
}

#endif /*LV_HAVE_SSE4_1*/


#ifdef LV_HAVE_SSE

#include<xmmintrin.h>

static inline void
volk_32f_index_max_32u_a_sse(uint32_t* target, const float* src0, uint32_t num_points)
{
  if(num_points > 0){
    uint32_t number = 0;
    const uint32_t quarterPoints = num_points / 4;

    float* inputPtr = (float*)src0;

    __m128 indexIncrementValues = _mm_set1_ps(4);
    __m128 currentIndexes = _mm_set_ps(-1,-2,-3,-4);

    float max = src0[0];
    float index = 0;
    __m128 maxValues = _mm_set1_ps(max);
    __m128 maxValuesIndex = _mm_setzero_ps();
    __m128 compareResults;
    __m128 currentValues;

    __VOLK_ATTR_ALIGNED(16) float maxValuesBuffer[4];
    __VOLK_ATTR_ALIGNED(16) float maxIndexesBuffer[4];

    for(;number < quarterPoints; number++){

      currentValues  = _mm_load_ps(inputPtr); inputPtr += 4;
      currentIndexes = _mm_add_ps(currentIndexes, indexIncrementValues);

      compareResults = _mm_cmpgt_ps(maxValues, currentValues);

      maxValuesIndex = _mm_or_ps(_mm_and_ps(compareResults, maxValuesIndex) , _mm_andnot_ps(compareResults, currentIndexes));

      maxValues      = _mm_or_ps(_mm_and_ps(compareResults, maxValues) , _mm_andnot_ps(compareResults, currentValues));
    }

    // Calculate the largest value from the remaining 4 points
    _mm_store_ps(maxValuesBuffer, maxValues);
    _mm_store_ps(maxIndexesBuffer, maxValuesIndex);

    for(number = 0; number < 4; number++){
      if(maxValuesBuffer[number] > max){
	index = maxIndexesBuffer[number];
	max = maxValuesBuffer[number];
      }
    }

    number = quarterPoints * 4;
    for(;number < num_points; number++){
      if(src0[number] > max){
	index = number;
	max = src0[number];
      }
    }
    target[0] = (uint32_t)index;
  }
}

#endif /*LV_HAVE_SSE*/


#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_index_max_32u_generic(uint32_t* target, const float* src0, uint32_t num_points)
{
  if(num_points > 0){
    float max = src0[0];
    uint32_t index = 0;

    uint32_t i = 1;

    for(; i < num_points; ++i) {
      if(src0[i] > max){
        index = i;
        max = src0[i];
      }
    }
    target[0] = index;
  }
}

#endif /*LV_HAVE_GENERIC*/


#endif /*INCLUDED_volk_32f_index_max_32u_a_H*/
