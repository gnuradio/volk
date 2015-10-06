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
 * \page volk_32fc_magnitude_squared_32f
 *
 * \b Overview
 *
 * Calculates the magnitude squared of the complexVector and stores
 * the results in the magnitudeVector.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_magnitude_squared_32f(float* magnitudeVector, const lv_32fc_t* complexVector, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li complexVector: The complex input vector.
 * \li num_points: The number of samples.
 *
 * \b Outputs
 * \li magnitudeVector: The output value.
 *
 * \b Example
 * Calculate the magnitude squared of \f$x^2 + x\f$ for points around the unit circle.
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   lv_32fc_t* in  = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *   float* magnitude = (float*)volk_malloc(sizeof(float)*N, alignment);
 *
 *   for(unsigned int ii = 0; ii < N/2; ++ii){
 *       float real = 2.f * ((float)ii / (float)N) - 1.f;
 *       float imag = std::sqrt(1.f - real * real);
 *       in[ii] = lv_cmake(real, imag);
 *       in[ii] = in[ii] * in[ii] + in[ii];
 *       in[N-ii] = lv_cmake(real, imag);
 *       in[N-ii] = in[N-ii] * in[N-ii] + in[N-ii];
 *   }
 *
 *   volk_32fc_magnitude_32f(magnitude, in, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("out(%i) = %+.1f\n", ii, magnitude[ii]);
 *   }
 *
 *   volk_free(in);
 *   volk_free(magnitude);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_magnitude_squared_32f_u_H
#define INCLUDED_volk_32fc_magnitude_squared_32f_u_H

#include <inttypes.h>
#include <stdio.h>
#include <math.h>

#ifdef LV_HAVE_AVX
#include <immintrin.h>
#include <volk/volk_avx_intrinsics.h>

static inline void
volk_32fc_magnitude_squared_32f_u_avx(float* magnitudeVector, const lv_32fc_t* complexVector,
                                      unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int eighthPoints = num_points / 8;

  const float* complexVectorPtr = (float*) complexVector;
  float* magnitudeVectorPtr = magnitudeVector;

  __m256 cplxValue1, cplxValue2, result;

  for(; number < eighthPoints; number++){
    cplxValue1 = _mm256_loadu_ps(complexVectorPtr);
    cplxValue2 = _mm256_loadu_ps(complexVectorPtr + 8);
    result = _mm256_magnitudesquared_ps(cplxValue1, cplxValue2);
    _mm256_storeu_ps(magnitudeVectorPtr, result);

    complexVectorPtr += 16;
    magnitudeVectorPtr += 8;
  }

  number = eighthPoints * 8;
  for(; number < num_points; number++){
    float val1Real = *complexVectorPtr++;
    float val1Imag = *complexVectorPtr++;
    *magnitudeVectorPtr++ = (val1Real * val1Real) + (val1Imag * val1Imag);
  }
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>
#include <volk/volk_sse3_intrinsics.h>

static inline void
volk_32fc_magnitude_squared_32f_u_sse3(float* magnitudeVector, const lv_32fc_t* complexVector,
                                       unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int quarterPoints = num_points / 4;

  const float* complexVectorPtr = (float*) complexVector;
  float* magnitudeVectorPtr = magnitudeVector;

  __m128 cplxValue1, cplxValue2, result;
  for(; number < quarterPoints; number++){
    cplxValue1 = _mm_loadu_ps(complexVectorPtr);
    complexVectorPtr += 4;

    cplxValue2 = _mm_loadu_ps(complexVectorPtr);
    complexVectorPtr += 4;

    result = _mm_magnitudesquared_ps_sse3(cplxValue1, cplxValue2);
    _mm_storeu_ps(magnitudeVectorPtr, result);
    magnitudeVectorPtr += 4;
  }

  number = quarterPoints * 4;
  for(; number < num_points; number++){
    float val1Real = *complexVectorPtr++;
    float val1Imag = *complexVectorPtr++;
    *magnitudeVectorPtr++ = (val1Real * val1Real) + (val1Imag * val1Imag);
  }
}
#endif /* LV_HAVE_SSE3 */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>
#include <volk/volk_sse_intrinsics.h>

static inline void
volk_32fc_magnitude_squared_32f_u_sse(float* magnitudeVector, const lv_32fc_t* complexVector,
                                      unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int quarterPoints = num_points / 4;

  const float* complexVectorPtr = (float*) complexVector;
  float* magnitudeVectorPtr = magnitudeVector;

  __m128 cplxValue1, cplxValue2, result;

  for(; number < quarterPoints; number++){
    cplxValue1 = _mm_loadu_ps(complexVectorPtr);
    complexVectorPtr += 4;

    cplxValue2 = _mm_loadu_ps(complexVectorPtr);
    complexVectorPtr += 4;

    result = _mm_magnitudesquared_ps(cplxValue1, cplxValue2);
    _mm_storeu_ps(magnitudeVectorPtr, result);
    magnitudeVectorPtr += 4;
  }

  number = quarterPoints * 4;
  for(; number < num_points; number++){
    float val1Real = *complexVectorPtr++;
    float val1Imag = *complexVectorPtr++;
    *magnitudeVectorPtr++ = (val1Real * val1Real) + (val1Imag * val1Imag);
  }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_GENERIC

static inline void
volk_32fc_magnitude_squared_32f_generic(float* magnitudeVector, const lv_32fc_t* complexVector,
                                        unsigned int num_points)
{
  const float* complexVectorPtr = (float*)complexVector;
  float* magnitudeVectorPtr = magnitudeVector;
  unsigned int number = 0;
  for(number = 0; number < num_points; number++){
    const float real = *complexVectorPtr++;
    const float imag = *complexVectorPtr++;
    *magnitudeVectorPtr++ = (real*real) + (imag*imag);
  }
}
#endif /* LV_HAVE_GENERIC */



#endif /* INCLUDED_volk_32fc_magnitude_32f_u_H */
#ifndef INCLUDED_volk_32fc_magnitude_squared_32f_a_H
#define INCLUDED_volk_32fc_magnitude_squared_32f_a_H

#include <inttypes.h>
#include <stdio.h>
#include <math.h>

#ifdef LV_HAVE_AVX
#include <immintrin.h>
#include <volk/volk_avx_intrinsics.h>

static inline void
volk_32fc_magnitude_squared_32f_a_avx(float* magnitudeVector, const lv_32fc_t* complexVector,
                                      unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int eighthPoints = num_points / 8;

  const float* complexVectorPtr = (float*) complexVector;
  float* magnitudeVectorPtr = magnitudeVector;

  __m256 cplxValue1, cplxValue2, result;
  for(; number < eighthPoints; number++){
    cplxValue1 = _mm256_load_ps(complexVectorPtr);
    complexVectorPtr += 8;

    cplxValue2 = _mm256_load_ps(complexVectorPtr);
    complexVectorPtr += 8;

    result = _mm256_magnitudesquared_ps(cplxValue1, cplxValue2);
    _mm256_store_ps(magnitudeVectorPtr, result);
    magnitudeVectorPtr += 8;
  }

  number = eighthPoints * 8;
  for(; number < num_points; number++){
    float val1Real = *complexVectorPtr++;
    float val1Imag = *complexVectorPtr++;
    *magnitudeVectorPtr++ = (val1Real * val1Real) + (val1Imag * val1Imag);
  }
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>
#include <volk/volk_sse3_intrinsics.h>

static inline void
volk_32fc_magnitude_squared_32f_a_sse3(float* magnitudeVector, const lv_32fc_t* complexVector,
                                       unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int quarterPoints = num_points / 4;

  const float* complexVectorPtr = (float*) complexVector;
  float* magnitudeVectorPtr = magnitudeVector;

  __m128 cplxValue1, cplxValue2, result;
  for(; number < quarterPoints; number++){
    cplxValue1 = _mm_load_ps(complexVectorPtr);
    complexVectorPtr += 4;

    cplxValue2 = _mm_load_ps(complexVectorPtr);
    complexVectorPtr += 4;

    result = _mm_magnitudesquared_ps_sse3(cplxValue1, cplxValue2);
    _mm_store_ps(magnitudeVectorPtr, result);
    magnitudeVectorPtr += 4;
  }

  number = quarterPoints * 4;
  for(; number < num_points; number++){
    float val1Real = *complexVectorPtr++;
    float val1Imag = *complexVectorPtr++;
    *magnitudeVectorPtr++ = (val1Real * val1Real) + (val1Imag * val1Imag);
  }
}
#endif /* LV_HAVE_SSE3 */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>
#include <volk/volk_sse_intrinsics.h>

static inline void
volk_32fc_magnitude_squared_32f_a_sse(float* magnitudeVector, const lv_32fc_t* complexVector,
                                      unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int quarterPoints = num_points / 4;

  const float* complexVectorPtr = (float*)complexVector;
  float* magnitudeVectorPtr = magnitudeVector;

  __m128 cplxValue1, cplxValue2, result;
  for(;number < quarterPoints; number++){
    cplxValue1 = _mm_load_ps(complexVectorPtr);
    complexVectorPtr += 4;

    cplxValue2 = _mm_load_ps(complexVectorPtr);
    complexVectorPtr += 4;

    result = _mm_magnitudesquared_ps(cplxValue1, cplxValue2);
    _mm_store_ps(magnitudeVectorPtr, result);
    magnitudeVectorPtr += 4;
  }

  number = quarterPoints * 4;
  for(; number < num_points; number++){
    float val1Real = *complexVectorPtr++;
    float val1Imag = *complexVectorPtr++;
    *magnitudeVectorPtr++ = (val1Real * val1Real) + (val1Imag * val1Imag);
  }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void
volk_32fc_magnitude_squared_32f_neon(float* magnitudeVector, const lv_32fc_t* complexVector,
                                     unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int quarterPoints = num_points / 4;

  const float* complexVectorPtr = (float*)complexVector;
  float* magnitudeVectorPtr = magnitudeVector;

  float32x4x2_t cmplx_val;
  float32x4_t result;
  for(;number < quarterPoints; number++){
    cmplx_val = vld2q_f32(complexVectorPtr);
    complexVectorPtr += 8;

    cmplx_val.val[0] = vmulq_f32(cmplx_val.val[0], cmplx_val.val[0]); // Square the values
    cmplx_val.val[1] = vmulq_f32(cmplx_val.val[1], cmplx_val.val[1]); // Square the values

    result = vaddq_f32(cmplx_val.val[0], cmplx_val.val[1]); // Add the I2 and Q2 values

    vst1q_f32(magnitudeVectorPtr, result);
    magnitudeVectorPtr += 4;
  }

  number = quarterPoints * 4;
  for(; number < num_points; number++){
    float val1Real = *complexVectorPtr++;
    float val1Imag = *complexVectorPtr++;
    *magnitudeVectorPtr++ = (val1Real * val1Real) + (val1Imag * val1Imag);
  }
}
#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_GENERIC

static inline void
volk_32fc_magnitude_squared_32f_a_generic(float* magnitudeVector, const lv_32fc_t* complexVector,
                                          unsigned int num_points)
{
  const float* complexVectorPtr = (float*)complexVector;
  float* magnitudeVectorPtr = magnitudeVector;
  unsigned int number = 0;
  for(number = 0; number < num_points; number++){
    const float real = *complexVectorPtr++;
    const float imag = *complexVectorPtr++;
    *magnitudeVectorPtr++ = (real*real) + (imag*imag);
  }
}
#endif /* LV_HAVE_GENERIC */

#endif /* INCLUDED_volk_32fc_magnitude_32f_a_H */
