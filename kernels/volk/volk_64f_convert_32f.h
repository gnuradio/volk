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
 * \page volk_64f_convert_32f
 *
 * \b Overview
 *
 * Converts doubles into floats.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_64f_convert_32f(float* outputVector, const double* inputVector, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li inputVector: The vector of doubles to convert to floats.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li outputVector: returns the converted floats.
 *
 * \b Example
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   double* increasing = (double*)volk_malloc(sizeof(double)*N, alignment);
 *   float* out = (float*)volk_malloc(sizeof(float)*N, alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       increasing[ii] = (double)ii;
 *   }
 *
 *   volk_64f_convert_32f(out, increasing, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("out[%u] = %1.2f\n", ii, out[ii]);
 *   }
 *
 *   volk_free(increasing);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_64f_convert_32f_u_H
#define INCLUDED_volk_64f_convert_32f_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_64f_convert_32f_u_avx(float* outputVector, const double* inputVector, unsigned int num_points){
  unsigned int number = 0;

  const unsigned int oneEightPoints = num_points / 8;

  const double* inputVectorPtr = (const double*)inputVector;
  float* outputVectorPtr = outputVector;
  __m128 ret1, ret2;
  __m256d inputVal1, inputVal2;

  for(;number < oneEightPoints; number++){
    inputVal1 = _mm256_loadu_pd(inputVectorPtr); inputVectorPtr += 4;
    inputVal2 = _mm256_loadu_pd(inputVectorPtr); inputVectorPtr += 4;

    ret1 = _mm256_cvtpd_ps(inputVal1);
    ret2 = _mm256_cvtpd_ps(inputVal2);

    _mm_storeu_ps(outputVectorPtr, ret1);
    outputVectorPtr += 4;

    _mm_storeu_ps(outputVectorPtr, ret2);
    outputVectorPtr += 4;
  }

  number = oneEightPoints * 8;
  for(; number < num_points; number++){
    outputVector[number] = (float)(inputVector[number]);
  }
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_64f_convert_32f_u_sse2(float* outputVector, const double* inputVector, unsigned int num_points){
  unsigned int number = 0;

  const unsigned int quarterPoints = num_points / 4;

  const double* inputVectorPtr = (const double*)inputVector;
  float* outputVectorPtr = outputVector;
  __m128 ret, ret2;
  __m128d inputVal1, inputVal2;

  for(;number < quarterPoints; number++){
    inputVal1 = _mm_loadu_pd(inputVectorPtr); inputVectorPtr += 2;
    inputVal2 = _mm_loadu_pd(inputVectorPtr); inputVectorPtr += 2;

    ret = _mm_cvtpd_ps(inputVal1);
    ret2 = _mm_cvtpd_ps(inputVal2);

    ret = _mm_movelh_ps(ret, ret2);

    _mm_storeu_ps(outputVectorPtr, ret);
    outputVectorPtr += 4;
  }

  number = quarterPoints * 4;
  for(; number < num_points; number++){
    outputVector[number] = (float)(inputVector[number]);
  }
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_GENERIC

static inline void volk_64f_convert_32f_generic(float* outputVector, const double* inputVector, unsigned int num_points){
  float* outputVectorPtr = outputVector;
  const double* inputVectorPtr = inputVector;
  unsigned int number = 0;

  for(number = 0; number < num_points; number++){
    *outputVectorPtr++ = ((float)(*inputVectorPtr++));
  }
}
#endif /* LV_HAVE_GENERIC */




#endif /* INCLUDED_volk_64f_convert_32f_u_H */
#ifndef INCLUDED_volk_64f_convert_32f_a_H
#define INCLUDED_volk_64f_convert_32f_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_64f_convert_32f_a_avx(float* outputVector, const double* inputVector, unsigned int num_points){
  unsigned int number = 0;

  const unsigned int oneEightPoints = num_points / 8;

  const double* inputVectorPtr = (const double*)inputVector;
  float* outputVectorPtr = outputVector;
  __m128 ret1, ret2;
  __m256d inputVal1, inputVal2;

  for(;number < oneEightPoints; number++){
    inputVal1 = _mm256_load_pd(inputVectorPtr); inputVectorPtr += 4;
    inputVal2 = _mm256_load_pd(inputVectorPtr); inputVectorPtr += 4;

    ret1 = _mm256_cvtpd_ps(inputVal1);
    ret2 = _mm256_cvtpd_ps(inputVal2);

    _mm_store_ps(outputVectorPtr, ret1);
    outputVectorPtr += 4;

    _mm_store_ps(outputVectorPtr, ret2);
    outputVectorPtr += 4;
  }

  number = oneEightPoints * 8;
  for(; number < num_points; number++){
    outputVector[number] = (float)(inputVector[number]);
  }
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void volk_64f_convert_32f_a_sse2(float* outputVector, const double* inputVector, unsigned int num_points){
  unsigned int number = 0;

  const unsigned int quarterPoints = num_points / 4;

  const double* inputVectorPtr = (const double*)inputVector;
  float* outputVectorPtr = outputVector;
  __m128 ret, ret2;
  __m128d inputVal1, inputVal2;

  for(;number < quarterPoints; number++){
    inputVal1 = _mm_load_pd(inputVectorPtr); inputVectorPtr += 2;
    inputVal2 = _mm_load_pd(inputVectorPtr); inputVectorPtr += 2;

    ret = _mm_cvtpd_ps(inputVal1);
    ret2 = _mm_cvtpd_ps(inputVal2);

    ret = _mm_movelh_ps(ret, ret2);

    _mm_store_ps(outputVectorPtr, ret);
    outputVectorPtr += 4;
  }

  number = quarterPoints * 4;
  for(; number < num_points; number++){
    outputVector[number] = (float)(inputVector[number]);
  }
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_GENERIC

static inline void volk_64f_convert_32f_a_generic(float* outputVector, const double* inputVector, unsigned int num_points){
  float* outputVectorPtr = outputVector;
  const double* inputVectorPtr = inputVector;
  unsigned int number = 0;

  for(number = 0; number < num_points; number++){
    *outputVectorPtr++ = ((float)(*inputVectorPtr++));
  }
}
#endif /* LV_HAVE_GENERIC */




#endif /* INCLUDED_volk_64f_convert_32f_a_H */
