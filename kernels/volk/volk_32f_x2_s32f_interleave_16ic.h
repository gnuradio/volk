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
 * \page volk_32f_x2_s32f_interleave_16ic
 *
 * \b Overview
 *
 * Takes input vector iBuffer as the real (inphase) part and input
 * vector qBuffer as the imag (quadrature) part and combines them into
 * a complex output vector. The output is scaled by the input scalar
 * value and convert to a 16-bit short comlex number.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_x2_s32f_interleave_16ic(lv_16sc_t* complexVector, const float* iBuffer, const float* qBuffer, const float scalar, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li iBuffer: Input vector of samples for the real part.
 * \li qBuffer: Input vector of samples for the imaginary part.
 * \;i scalar:  The scalar value used to scale the values before converting to shorts.
 * \li num_points: The number of values in both input vectors.
 *
 * \b Outputs
 * \li complexVector: The output vector of complex numbers.
 *
 * \b Example
 * Generate points around the unit circle and convert to complex integers.
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   float* imag = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* real = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   lv_16sc_t* out = (lv_16sc_t*)volk_malloc(sizeof(lv_16sc_t)*N, alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       real[ii] = 2.f * ((float)ii / (float)N) - 1.f;
 *       imag[ii] = std::sqrt(1.f - real[ii] * real[ii]);
 *   }
 *   // Normalize by smallest delta (0.02 in this example)
 *   float scale = 50.f;
 *
 *   volk_32f_x2_s32f_interleave_16ic(out, imag, real, scale, N);
 *
 *  for(unsigned int ii = 0; ii < N; ++ii){
 *      printf("out[%u] = %i + %ij\n", ii, std::real(out[ii]), std::imag(out[ii]));
 *  }
 *
 *   volk_free(imag);
 *   volk_free(real);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_x2_s32f_interleave_16ic_a_H
#define INCLUDED_volk_32f_x2_s32f_interleave_16ic_a_H

#include <volk/volk_common.h>
#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void
volk_32f_x2_s32f_interleave_16ic_a_sse2(lv_16sc_t* complexVector, const float* iBuffer,
                                        const float* qBuffer, const float scalar, unsigned int num_points)
{
  unsigned int number = 0;
  const float* iBufferPtr = iBuffer;
  const float* qBufferPtr = qBuffer;

  __m128 vScalar = _mm_set_ps1(scalar);

  const unsigned int quarterPoints = num_points / 4;

  __m128 iValue, qValue, cplxValue1, cplxValue2;
  __m128i intValue1, intValue2;

  int16_t* complexVectorPtr = (int16_t*)complexVector;

  for(;number < quarterPoints; number++){
    iValue = _mm_load_ps(iBufferPtr);
    qValue = _mm_load_ps(qBufferPtr);

    // Interleaves the lower two values in the i and q variables into one buffer
    cplxValue1 = _mm_unpacklo_ps(iValue, qValue);
    cplxValue1 = _mm_mul_ps(cplxValue1, vScalar);

    // Interleaves the upper two values in the i and q variables into one buffer
    cplxValue2 = _mm_unpackhi_ps(iValue, qValue);
    cplxValue2 = _mm_mul_ps(cplxValue2, vScalar);

    intValue1 = _mm_cvtps_epi32(cplxValue1);
    intValue2 = _mm_cvtps_epi32(cplxValue2);

    intValue1 = _mm_packs_epi32(intValue1, intValue2);

    _mm_store_si128((__m128i*)complexVectorPtr, intValue1);
    complexVectorPtr += 8;

    iBufferPtr += 4;
    qBufferPtr += 4;
  }

  number = quarterPoints * 4;
  complexVectorPtr = (int16_t*)(&complexVector[number]);
  for(; number < num_points; number++){
    *complexVectorPtr++ = (int16_t)(*iBufferPtr++ * scalar);
    *complexVectorPtr++ = (int16_t)(*qBufferPtr++ * scalar);
  }
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void
volk_32f_x2_s32f_interleave_16ic_a_sse(lv_16sc_t* complexVector, const float* iBuffer,
                                       const float* qBuffer, const float scalar, unsigned int num_points)
{
  unsigned int number = 0;
  const float* iBufferPtr = iBuffer;
  const float* qBufferPtr = qBuffer;

  __m128 vScalar = _mm_set_ps1(scalar);

  const unsigned int quarterPoints = num_points / 4;

  __m128 iValue, qValue, cplxValue;

  int16_t* complexVectorPtr = (int16_t*)complexVector;

  __VOLK_ATTR_ALIGNED(16) float floatBuffer[4];

  for(;number < quarterPoints; number++){
    iValue = _mm_load_ps(iBufferPtr);
    qValue = _mm_load_ps(qBufferPtr);

    // Interleaves the lower two values in the i and q variables into one buffer
    cplxValue = _mm_unpacklo_ps(iValue, qValue);
    cplxValue = _mm_mul_ps(cplxValue, vScalar);

    _mm_store_ps(floatBuffer, cplxValue);

    *complexVectorPtr++ = (int16_t)(floatBuffer[0]);
    *complexVectorPtr++ = (int16_t)(floatBuffer[1]);
    *complexVectorPtr++ = (int16_t)(floatBuffer[2]);
    *complexVectorPtr++ = (int16_t)(floatBuffer[3]);

    // Interleaves the upper two values in the i and q variables into one buffer
    cplxValue = _mm_unpackhi_ps(iValue, qValue);
    cplxValue = _mm_mul_ps(cplxValue, vScalar);

    _mm_store_ps(floatBuffer, cplxValue);

    *complexVectorPtr++ = (int16_t)(floatBuffer[0]);
    *complexVectorPtr++ = (int16_t)(floatBuffer[1]);
    *complexVectorPtr++ = (int16_t)(floatBuffer[2]);
    *complexVectorPtr++ = (int16_t)(floatBuffer[3]);

    iBufferPtr += 4;
    qBufferPtr += 4;
  }

  number = quarterPoints * 4;
  complexVectorPtr = (int16_t*)(&complexVector[number]);
  for(; number < num_points; number++){
    *complexVectorPtr++ = (int16_t)(*iBufferPtr++ * scalar);
    *complexVectorPtr++ = (int16_t)(*qBufferPtr++ * scalar);
  }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_x2_s32f_interleave_16ic_generic(lv_16sc_t* complexVector, const float* iBuffer,
                                         const float* qBuffer, const float scalar, unsigned int num_points)
{
  int16_t* complexVectorPtr = (int16_t*)complexVector;
  const float* iBufferPtr = iBuffer;
  const float* qBufferPtr = qBuffer;
  unsigned int number = 0;

  for(number = 0; number < num_points; number++){
    *complexVectorPtr++ = (int16_t)(*iBufferPtr++ * scalar);
    *complexVectorPtr++ = (int16_t)(*qBufferPtr++ * scalar);
  }
}
#endif /* LV_HAVE_GENERIC */


#endif /* INCLUDED_volk_32f_x2_s32f_interleave_16ic_a_H */
