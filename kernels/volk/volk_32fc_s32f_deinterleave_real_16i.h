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
 * \page volk_32fc_s32f_deinterleave_real_16i
 *
 * \b Overview
 *
 * Deinterleaves the complex floating point vector and return the real
 * part (inphase) of the samples scaled to 16-bit shorts.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_s32f_deinterleave_real_16i(int16_t* iBuffer, const lv_32fc_t* complexVector, const float scalar, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li complexVector: The complex input vector.
 * \li scalar: The value to be multiplied against each of the input vectors..
 * \li num_points: The number of complex data values to be deinterleaved.
 *
 * \b Outputs
 * \li iBuffer: The I buffer output data.
 *
 * \b Example
 * Generate points around the unit circle and map them to integers with
 * magnitude 50 to preserve smallest deltas.
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   lv_32fc_t* in  = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *   int16_t* out = (int16_t*)volk_malloc(sizeof(int16_t)*N, alignment);
 *   float scale = 50.f;
 *
 *   for(unsigned int ii = 0; ii < N/2; ++ii){
 *       // Generate points around the unit circle
 *       float real = -4.f * ((float)ii / (float)N) + 1.f;
 *       float imag = std::sqrt(1.f - real * real);
 *       in[ii] = lv_cmake(real, imag);
 *       in[ii+N/2] = lv_cmake(-real, -imag);
 *   }
 *
 *   volk_32fc_s32f_deinterleave_real_16i(out, in, scale, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("out[%u] = %i\n", ii, out[ii]);
 *   }
 *
 *   volk_free(in);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_s32f_deinterleave_real_16i_a_H
#define INCLUDED_volk_32fc_s32f_deinterleave_real_16i_a_H

#include <volk/volk_common.h>
#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void
volk_32fc_s32f_deinterleave_real_16i_a_sse(int16_t* iBuffer, const lv_32fc_t* complexVector,
                                           const float scalar, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int quarterPoints = num_points / 4;

  const float* complexVectorPtr = (float*)complexVector;
  int16_t* iBufferPtr = iBuffer;

  __m128 vScalar = _mm_set_ps1(scalar);

  __m128 cplxValue1, cplxValue2, iValue;

  __VOLK_ATTR_ALIGNED(16) float floatBuffer[4];

  for(;number < quarterPoints; number++){
    cplxValue1 = _mm_load_ps(complexVectorPtr);
    complexVectorPtr += 4;

    cplxValue2 = _mm_load_ps(complexVectorPtr);
    complexVectorPtr += 4;

    // Arrange in i1i2i3i4 format
    iValue = _mm_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(2,0,2,0));

    iValue = _mm_mul_ps(iValue, vScalar);

    _mm_store_ps(floatBuffer, iValue);
    *iBufferPtr++ = (int16_t)(floatBuffer[0]);
    *iBufferPtr++ = (int16_t)(floatBuffer[1]);
    *iBufferPtr++ = (int16_t)(floatBuffer[2]);
    *iBufferPtr++ = (int16_t)(floatBuffer[3]);
  }

  number = quarterPoints * 4;
  iBufferPtr = &iBuffer[number];
  for(; number < num_points; number++){
    *iBufferPtr++ = (int16_t)(*complexVectorPtr++ * scalar);
    complexVectorPtr++;
  }
}

#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_GENERIC

static inline void
volk_32fc_s32f_deinterleave_real_16i_generic(int16_t* iBuffer, const lv_32fc_t* complexVector,
                                             const float scalar, unsigned int num_points)
{
  const float* complexVectorPtr = (float*)complexVector;
  int16_t* iBufferPtr = iBuffer;
  unsigned int number = 0;
  for(number = 0; number < num_points; number++){
    *iBufferPtr++ = (int16_t)(*complexVectorPtr++ * scalar);
    complexVectorPtr++;
  }
}

#endif /* LV_HAVE_GENERIC */

#endif /* INCLUDED_volk_32fc_s32f_deinterleave_real_16i_a_H */
