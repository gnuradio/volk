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
 * \page volk_32fc_deinterleave_real_32f
 *
 * \b Overview
 *
 * Deinterleaves the complex floating point vector and return the real
 * part (inphase) of the samples.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_deinterleave_real_32f(float* iBuffer, const lv_32fc_t* complexVector, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li complexVector: The complex input vector.
 * \li num_points: The number of complex data values to be deinterleaved.
 *
 * \b Outputs
 * \li iBuffer: The I buffer output data.
 *
 * \b Example
 * Generate complex numbers around the top half of the unit circle and
 * extract all of the real parts to a float buffer.
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   lv_32fc_t* in  = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *   float* re = (float*)volk_malloc(sizeof(float)*N, alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       float real = 2.f * ((float)ii / (float)N) - 1.f;
 *       float imag = std::sqrt(1.f - real * real);
 *       in[ii] = lv_cmake(real, imag);
 *   }
 *
 *   volk_32fc_deinterleave_real_32f(re, in, N);
 *
 *   printf("          real part\n");
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("out(%i) = %+.1f\n", ii, re[ii]);
 *   }
 *
 *   volk_free(in);
 *   volk_free(re);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_deinterleave_real_32f_a_H
#define INCLUDED_volk_32fc_deinterleave_real_32f_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void
volk_32fc_deinterleave_real_32f_a_avx2(float* iBuffer, const lv_32fc_t* complexVector,
                                      unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int eighthPoints = num_points / 8;

  const float* complexVectorPtr = (const float*)complexVector;
  float* iBufferPtr = iBuffer;

  __m256 cplxValue1, cplxValue2;
  __m256 iValue;
  __m256i idx = _mm256_set_epi32(7,6,3,2,5,4,1,0);
  for(;number < eighthPoints; number++){

    cplxValue1 = _mm256_load_ps(complexVectorPtr);
    complexVectorPtr += 8;

    cplxValue2 = _mm256_load_ps(complexVectorPtr);
    complexVectorPtr += 8;

    // Arrange in i1i2i3i4 format
    iValue = _mm256_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(2,0,2,0));
    iValue = _mm256_permutevar8x32_ps(iValue,idx);

    _mm256_store_ps(iBufferPtr, iValue);

    iBufferPtr += 8;
  }

  number = eighthPoints * 8;
  for(; number < num_points; number++){
    *iBufferPtr++ = *complexVectorPtr++;
    complexVectorPtr++;
  }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void
volk_32fc_deinterleave_real_32f_a_sse(float* iBuffer, const lv_32fc_t* complexVector,
                                      unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int quarterPoints = num_points / 4;

  const float* complexVectorPtr = (const float*)complexVector;
  float* iBufferPtr = iBuffer;

  __m128 cplxValue1, cplxValue2, iValue;
  for(;number < quarterPoints; number++){

    cplxValue1 = _mm_load_ps(complexVectorPtr);
    complexVectorPtr += 4;

    cplxValue2 = _mm_load_ps(complexVectorPtr);
    complexVectorPtr += 4;

    // Arrange in i1i2i3i4 format
    iValue = _mm_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(2,0,2,0));

    _mm_store_ps(iBufferPtr, iValue);

    iBufferPtr += 4;
  }

  number = quarterPoints * 4;
  for(; number < num_points; number++){
    *iBufferPtr++ = *complexVectorPtr++;
    complexVectorPtr++;
  }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_GENERIC

static inline void
volk_32fc_deinterleave_real_32f_generic(float* iBuffer, const lv_32fc_t* complexVector,
                                        unsigned int num_points)
{
  unsigned int number = 0;
  const float* complexVectorPtr = (float*)complexVector;
  float* iBufferPtr = iBuffer;
  for(number = 0; number < num_points; number++){
    *iBufferPtr++ = *complexVectorPtr++;
    complexVectorPtr++;
  }
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void
volk_32fc_deinterleave_real_32f_neon(float* iBuffer, const lv_32fc_t* complexVector,
                                     unsigned int num_points)
{
  unsigned int number = 0;
  unsigned int quarter_points = num_points / 4;
  const float* complexVectorPtr = (float*)complexVector;
  float* iBufferPtr = iBuffer;
  float32x4x2_t complexInput;

  for(number = 0; number < quarter_points; number++){
    complexInput = vld2q_f32(complexVectorPtr);
    vst1q_f32( iBufferPtr, complexInput.val[0] );
    complexVectorPtr += 8;
    iBufferPtr += 4;
  }

  for(number = quarter_points*4; number < num_points; number++){
    *iBufferPtr++ = *complexVectorPtr++;
    complexVectorPtr++;
  }
}
#endif /* LV_HAVE_NEON */

#endif /* INCLUDED_volk_32fc_deinterleave_real_32f_a_H */


#ifndef INCLUDED_volk_32fc_deinterleave_real_32f_u_H
#define INCLUDED_volk_32fc_deinterleave_real_32f_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void
volk_32fc_deinterleave_real_32f_u_avx2(float* iBuffer, const lv_32fc_t* complexVector,
                                      unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int eighthPoints = num_points / 8;

  const float* complexVectorPtr = (const float*)complexVector;
  float* iBufferPtr = iBuffer;

  __m256 cplxValue1, cplxValue2;
  __m256 iValue;
  __m256i idx = _mm256_set_epi32(7,6,3,2,5,4,1,0);
  for(;number < eighthPoints; number++){

    cplxValue1 = _mm256_loadu_ps(complexVectorPtr);
    complexVectorPtr += 8;

    cplxValue2 = _mm256_loadu_ps(complexVectorPtr);
    complexVectorPtr += 8;

    // Arrange in i1i2i3i4 format
    iValue = _mm256_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(2,0,2,0));
    iValue = _mm256_permutevar8x32_ps(iValue,idx);

    _mm256_storeu_ps(iBufferPtr, iValue);

    iBufferPtr += 8;
  }

  number = eighthPoints * 8;
  for(; number < num_points; number++){
    *iBufferPtr++ = *complexVectorPtr++;
    complexVectorPtr++;
  }
}
#endif /* LV_HAVE_AVX2 */

#endif /* INCLUDED_volk_32fc_deinterleave_real_32f_u_H */
