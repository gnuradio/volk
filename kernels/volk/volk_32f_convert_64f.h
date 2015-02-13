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
 * \page volk_32f_convert_64f
 *
 * \b Overview
 *
 * Converts float values into doubles.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_convert_64f(double* outputVector, const float* inputVector, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li inputVector: The vector of floats to convert to doubles.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li outputVector: returns the converted doubles.
 *
 * \b Example
 * Generate floats and convert them to doubles.
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   float* in = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   double* out = (double*)volk_malloc(sizeof(double)*N, alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       in[ii] = (float)ii;
 *   }
 *
 *   volk_32f_convert_64f(out, in, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("out(%i) = %g\n", ii, out[ii]);
 *   }
 *
 *   volk_free(in);
 *   volk_free(out);
 * \endcode
 */


#ifndef INCLUDED_volk_32f_convert_64f_u_H
#define INCLUDED_volk_32f_convert_64f_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_convert_64f_u_avx(double* outputVector, const float* inputVector, unsigned int num_points){
  unsigned int number = 0;

  const unsigned int quarterPoints = num_points / 4;

  const float* inputVectorPtr = (const float*)inputVector;
  double* outputVectorPtr = outputVector;
  __m256d ret;
  __m128 inputVal;

  for(;number < quarterPoints; number++){
    inputVal = _mm_loadu_ps(inputVectorPtr); inputVectorPtr += 4;

    ret = _mm256_cvtps_pd(inputVal);
    _mm256_storeu_pd(outputVectorPtr, ret);

    outputVectorPtr += 4;
  }

  number = quarterPoints * 4;
  for(; number < num_points; number++){
    outputVector[number] = (double)(inputVector[number]);
  }
}

#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_32f_convert_64f_u_sse2(double* outputVector, const float* inputVector, unsigned int num_points){
  unsigned int number = 0;

  const unsigned int quarterPoints = num_points / 4;

  const float* inputVectorPtr = (const float*)inputVector;
  double* outputVectorPtr = outputVector;
  __m128d ret;
  __m128 inputVal;

  for(;number < quarterPoints; number++){
    inputVal = _mm_loadu_ps(inputVectorPtr); inputVectorPtr += 4;

    ret = _mm_cvtps_pd(inputVal);

    _mm_storeu_pd(outputVectorPtr, ret);
    outputVectorPtr += 2;

    inputVal = _mm_movehl_ps(inputVal, inputVal);

    ret = _mm_cvtps_pd(inputVal);

    _mm_storeu_pd(outputVectorPtr, ret);
    outputVectorPtr += 2;
  }

  number = quarterPoints * 4;
  for(; number < num_points; number++){
    outputVector[number] = (double)(inputVector[number]);
  }
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_GENERIC

static inline void volk_32f_convert_64f_generic(double* outputVector, const float* inputVector, unsigned int num_points){
  double* outputVectorPtr = outputVector;
  const float* inputVectorPtr = inputVector;
  unsigned int number = 0;

  for(number = 0; number < num_points; number++){
    *outputVectorPtr++ = ((double)(*inputVectorPtr++));
  }
}
#endif /* LV_HAVE_GENERIC */




#endif /* INCLUDED_volk_32f_convert_64f_u_H */


#ifndef INCLUDED_volk_32f_convert_64f_a_H
#define INCLUDED_volk_32f_convert_64f_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32f_convert_64f_a_avx(double* outputVector, const float* inputVector, unsigned int num_points){
  unsigned int number = 0;

  const unsigned int quarterPoints = num_points / 4;

  const float* inputVectorPtr = (const float*)inputVector;
  double* outputVectorPtr = outputVector;
  __m256d ret;
  __m128 inputVal;

  for(;number < quarterPoints; number++){
    inputVal = _mm_load_ps(inputVectorPtr); inputVectorPtr += 4;

    ret = _mm256_cvtps_pd(inputVal);
    _mm256_store_pd(outputVectorPtr, ret);

    outputVectorPtr += 4;
  }

  number = quarterPoints * 4;
  for(; number < num_points; number++){
    outputVector[number] = (double)(inputVector[number]);
  }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_32f_convert_64f_a_sse2(double* outputVector, const float* inputVector, unsigned int num_points){
  unsigned int number = 0;

  const unsigned int quarterPoints = num_points / 4;

  const float* inputVectorPtr = (const float*)inputVector;
  double* outputVectorPtr = outputVector;
  __m128d ret;
  __m128 inputVal;

  for(;number < quarterPoints; number++){
    inputVal = _mm_load_ps(inputVectorPtr); inputVectorPtr += 4;

    ret = _mm_cvtps_pd(inputVal);

    _mm_store_pd(outputVectorPtr, ret);
    outputVectorPtr += 2;

    inputVal = _mm_movehl_ps(inputVal, inputVal);

    ret = _mm_cvtps_pd(inputVal);

    _mm_store_pd(outputVectorPtr, ret);
    outputVectorPtr += 2;
  }

  number = quarterPoints * 4;
  for(; number < num_points; number++){
    outputVector[number] = (double)(inputVector[number]);
  }
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_GENERIC

static inline void volk_32f_convert_64f_a_generic(double* outputVector, const float* inputVector, unsigned int num_points){
  double* outputVectorPtr = outputVector;
  const float* inputVectorPtr = inputVector;
  unsigned int number = 0;

  for(number = 0; number < num_points; number++){
    *outputVectorPtr++ = ((double)(*inputVectorPtr++));
  }
}
#endif /* LV_HAVE_GENERIC */




#endif /* INCLUDED_volk_32f_convert_64f_a_H */
