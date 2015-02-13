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
 * \page volk_32fc_deinterleave_64f_x2
 *
 * \b Overview
 *
 * Deinterleaves the complex floating point vector into I & Q vector
 * data. The output vectors are converted to doubles.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_deinterleave_64f_x2(double* iBuffer, double* qBuffer, const lv_32fc_t* complexVector, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li complexVector: The complex input vector.
 * \li num_points: The number of complex data values to be deinterleaved.
 *
 * \b Outputs
 * \li iBuffer: The I buffer output data.
 * \li qBuffer: The Q buffer output data.
 *
 * \b Example
 * Generate complex numbers around the top half of the unit circle and
 * deinterleave in to real and imaginary double buffers.
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   lv_32fc_t* in  = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *   double* re = (double*)volk_malloc(sizeof(double)*N, alignment);
 *   double* im = (double*)volk_malloc(sizeof(double)*N, alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       float real = 2.f * ((float)ii / (float)N) - 1.f;
 *       float imag = std::sqrt(1.f - real * real);
 *       in[ii] = lv_cmake(real, imag);
 *   }
 *
 *   volk_32fc_deinterleave_64f_x2(re, im, in, N);
 *
 *   printf("          re  | im\n");
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("out(%i) = %+.1g | %+.1g\n", ii, re[ii], im[ii]);
 *   }
 *
 *   volk_free(in);
 *   volk_free(re);
 *   volk_free(im);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_deinterleave_64f_x2_u_H
#define INCLUDED_volk_32fc_deinterleave_64f_x2_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32fc_deinterleave_64f_x2_u_avx(double* iBuffer, double* qBuffer, const lv_32fc_t* complexVector,
                                    unsigned int num_points)
{
  unsigned int number = 0;

  const float* complexVectorPtr = (float*)complexVector;
  double* iBufferPtr = iBuffer;
  double* qBufferPtr = qBuffer;

  const unsigned int quarterPoints = num_points / 4;
  __m256 cplxValue;
  __m128 complexH, complexL, fVal;
  __m256d dVal;

  for(;number < quarterPoints; number++){

    cplxValue = _mm256_loadu_ps(complexVectorPtr);
    complexVectorPtr += 8;

    complexH = _mm256_extractf128_ps(cplxValue, 1);
    complexL = _mm256_extractf128_ps(cplxValue, 0);

    // Arrange in i1i2i1i2 format
    fVal = _mm_shuffle_ps(complexL, complexH, _MM_SHUFFLE(2,0,2,0));
    dVal = _mm256_cvtps_pd(fVal);
    _mm256_storeu_pd(iBufferPtr, dVal);

    // Arrange in q1q2q1q2 format
    fVal = _mm_shuffle_ps(complexL, complexH, _MM_SHUFFLE(3,1,3,1));
    dVal = _mm256_cvtps_pd(fVal);
    _mm256_storeu_pd(qBufferPtr, dVal);

    iBufferPtr += 4;
    qBufferPtr += 4;
  }

  number = quarterPoints * 4;
  for(; number < num_points; number++){
    *iBufferPtr++ = *complexVectorPtr++;
    *qBufferPtr++ = *complexVectorPtr++;
  }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void
volk_32fc_deinterleave_64f_x2_u_sse2(double* iBuffer, double* qBuffer, const lv_32fc_t* complexVector,
                                     unsigned int num_points)
{
  unsigned int number = 0;

  const float* complexVectorPtr = (float*)complexVector;
  double* iBufferPtr = iBuffer;
  double* qBufferPtr = qBuffer;

  const unsigned int halfPoints = num_points / 2;
  __m128 cplxValue, fVal;
  __m128d dVal;

  for(;number < halfPoints; number++){

    cplxValue = _mm_loadu_ps(complexVectorPtr);
    complexVectorPtr += 4;

    // Arrange in i1i2i1i2 format
    fVal = _mm_shuffle_ps(cplxValue, cplxValue, _MM_SHUFFLE(2,0,2,0));
    dVal = _mm_cvtps_pd(fVal);
    _mm_storeu_pd(iBufferPtr, dVal);

    // Arrange in q1q2q1q2 format
    fVal = _mm_shuffle_ps(cplxValue, cplxValue, _MM_SHUFFLE(3,1,3,1));
    dVal = _mm_cvtps_pd(fVal);
    _mm_storeu_pd(qBufferPtr, dVal);

    iBufferPtr += 2;
    qBufferPtr += 2;
  }

  number = halfPoints * 2;
  for(; number < num_points; number++){
    *iBufferPtr++ = *complexVectorPtr++;
    *qBufferPtr++ = *complexVectorPtr++;
  }
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_GENERIC

static inline void
volk_32fc_deinterleave_64f_x2_generic(double* iBuffer, double* qBuffer, const lv_32fc_t* complexVector,
                                      unsigned int num_points)
{
  unsigned int number = 0;
  const float* complexVectorPtr = (float*)complexVector;
  double* iBufferPtr = iBuffer;
  double* qBufferPtr = qBuffer;

  for(number = 0; number < num_points; number++){
    *iBufferPtr++ = (double)*complexVectorPtr++;
    *qBufferPtr++ = (double)*complexVectorPtr++;
  }
}
#endif /* LV_HAVE_GENERIC */




#endif /* INCLUDED_volk_32fc_deinterleave_64f_x2_u_H */
#ifndef INCLUDED_volk_32fc_deinterleave_64f_x2_a_H
#define INCLUDED_volk_32fc_deinterleave_64f_x2_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32fc_deinterleave_64f_x2_a_avx(double* iBuffer, double* qBuffer, const lv_32fc_t* complexVector,
                                    unsigned int num_points)
{
  unsigned int number = 0;

  const float* complexVectorPtr = (float*)complexVector;
  double* iBufferPtr = iBuffer;
  double* qBufferPtr = qBuffer;

  const unsigned int quarterPoints = num_points / 4;
  __m256 cplxValue;
  __m128 complexH, complexL, fVal;
  __m256d dVal;

  for(;number < quarterPoints; number++){

    cplxValue = _mm256_load_ps(complexVectorPtr);
    complexVectorPtr += 8;

    complexH = _mm256_extractf128_ps(cplxValue, 1);
    complexL = _mm256_extractf128_ps(cplxValue, 0);

    // Arrange in i1i2i1i2 format
    fVal = _mm_shuffle_ps(complexL, complexH, _MM_SHUFFLE(2,0,2,0));
    dVal = _mm256_cvtps_pd(fVal);
    _mm256_store_pd(iBufferPtr, dVal);

    // Arrange in q1q2q1q2 format
    fVal = _mm_shuffle_ps(complexL, complexH, _MM_SHUFFLE(3,1,3,1));
    dVal = _mm256_cvtps_pd(fVal);
    _mm256_store_pd(qBufferPtr, dVal);

    iBufferPtr += 4;
    qBufferPtr += 4;
  }

  number = quarterPoints * 4;
  for(; number < num_points; number++){
    *iBufferPtr++ = *complexVectorPtr++;
    *qBufferPtr++ = *complexVectorPtr++;
  }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void
volk_32fc_deinterleave_64f_x2_a_sse2(double* iBuffer, double* qBuffer, const lv_32fc_t* complexVector,
                                     unsigned int num_points)
{
  unsigned int number = 0;

  const float* complexVectorPtr = (float*)complexVector;
  double* iBufferPtr = iBuffer;
  double* qBufferPtr = qBuffer;

  const unsigned int halfPoints = num_points / 2;
  __m128 cplxValue, fVal;
  __m128d dVal;

  for(;number < halfPoints; number++){

    cplxValue = _mm_load_ps(complexVectorPtr);
    complexVectorPtr += 4;

    // Arrange in i1i2i1i2 format
    fVal = _mm_shuffle_ps(cplxValue, cplxValue, _MM_SHUFFLE(2,0,2,0));
    dVal = _mm_cvtps_pd(fVal);
    _mm_store_pd(iBufferPtr, dVal);

    // Arrange in q1q2q1q2 format
    fVal = _mm_shuffle_ps(cplxValue, cplxValue, _MM_SHUFFLE(3,1,3,1));
    dVal = _mm_cvtps_pd(fVal);
    _mm_store_pd(qBufferPtr, dVal);

    iBufferPtr += 2;
    qBufferPtr += 2;
  }

  number = halfPoints * 2;
  for(; number < num_points; number++){
    *iBufferPtr++ = *complexVectorPtr++;
    *qBufferPtr++ = *complexVectorPtr++;
  }
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_GENERIC

static inline void
volk_32fc_deinterleave_64f_x2_a_generic(double* iBuffer, double* qBuffer, const lv_32fc_t* complexVector,
                                        unsigned int num_points)
{
  unsigned int number = 0;
  const float* complexVectorPtr = (float*)complexVector;
  double* iBufferPtr = iBuffer;
  double* qBufferPtr = qBuffer;

  for(number = 0; number < num_points; number++){
    *iBufferPtr++ = (double)*complexVectorPtr++;
    *qBufferPtr++ = (double)*complexVectorPtr++;
  }
}
#endif /* LV_HAVE_GENERIC */


#endif /* INCLUDED_volk_32fc_deinterleave_64f_x2_a_H */
