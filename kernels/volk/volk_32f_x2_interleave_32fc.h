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
 * \page volk_32f_x2_interleave_32fc
 *
 * \b Overview
 *
 * Takes input vector iBuffer as the real (inphase) part and input
 * vector qBuffer as the imag (quadrature) part and combines them into
 * a complex output vector.
 *
 * c[i] = complex(a[i], b[i])
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_x2_interleave_32fc(lv_32fc_t* complexVector, const float* iBuffer, const float* qBuffer, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li iBuffer: Input vector of samples for the real part.
 * \li qBuffer: Input vector of samples for the imaginary part.
 * \li num_points: The number of values in both input vectors.
 *
 * \b Outputs
 * \li complexVector: The output vector of complex numbers.
 *
 * \b Example
 * Generate the top half of the unit circle with real points equally spaced.
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   float* imag = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* real = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   lv_32fc_t* out = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       real[ii] = 2.f * ((float)ii / (float)N) - 1.f;
 *       imag[ii] = std::sqrt(1.f - real[ii] * real[ii]);
 *   }
 *
 *   volk_32f_x2_interleave_32fc(out, imag, real, N);
 *
 *  for(unsigned int ii = 0; ii < N; ++ii){
 *      printf("out[%u] = %1.2f + %1.2fj\n", ii, std::real(out[ii]), std::imag(out[ii]));
 *  }
 *
 *   volk_free(imag);
 *   volk_free(real);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_x2_interleave_32fc_a_H
#define INCLUDED_volk_32f_x2_interleave_32fc_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void
volk_32f_x2_interleave_32fc_a_sse(lv_32fc_t* complexVector, const float* iBuffer,
                                  const float* qBuffer, unsigned int num_points)
{
  unsigned int number = 0;
  float* complexVectorPtr = (float*)complexVector;
  const float* iBufferPtr = iBuffer;
  const float* qBufferPtr = qBuffer;

  const uint64_t quarterPoints = num_points / 4;

  __m128 iValue, qValue, cplxValue;
  for(;number < quarterPoints; number++){
    iValue = _mm_load_ps(iBufferPtr);
    qValue = _mm_load_ps(qBufferPtr);

    // Interleaves the lower two values in the i and q variables into one buffer
    cplxValue = _mm_unpacklo_ps(iValue, qValue);
    _mm_store_ps(complexVectorPtr, cplxValue);
    complexVectorPtr += 4;

    // Interleaves the upper two values in the i and q variables into one buffer
    cplxValue = _mm_unpackhi_ps(iValue, qValue);
    _mm_store_ps(complexVectorPtr, cplxValue);
    complexVectorPtr += 4;

    iBufferPtr += 4;
    qBufferPtr += 4;
  }

  number = quarterPoints * 4;
  for(; number < num_points; number++){
    *complexVectorPtr++ = *iBufferPtr++;
    *complexVectorPtr++ = *qBufferPtr++;
  }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void
volk_32f_x2_interleave_32fc_neon(lv_32fc_t* complexVector, const float* iBuffer,
                                 const float* qBuffer, unsigned int num_points)
{
  unsigned int quarter_points = num_points / 4;
  unsigned int number;
  float* complexVectorPtr = (float*) complexVector;

  float32x4x2_t complex_vec;
  for(number=0; number < quarter_points; ++number) {
    complex_vec.val[0] = vld1q_f32(iBuffer);
    complex_vec.val[1] = vld1q_f32(qBuffer);
    vst2q_f32(complexVectorPtr, complex_vec);
    iBuffer += 4;
    qBuffer += 4;
    complexVectorPtr += 8;
  }

  for(number=quarter_points * 4; number < num_points; ++number) {
    *complexVectorPtr++ = *iBuffer++;
    *complexVectorPtr++ = *qBuffer++;
  }
}
#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_x2_interleave_32fc_generic(lv_32fc_t* complexVector, const float* iBuffer,
                                    const float* qBuffer, unsigned int num_points)
{
  float* complexVectorPtr = (float*)complexVector;
  const float* iBufferPtr = iBuffer;
  const float* qBufferPtr = qBuffer;
  unsigned int number;

  for(number = 0; number < num_points; number++){
    *complexVectorPtr++ = *iBufferPtr++;
    *complexVectorPtr++ = *qBufferPtr++;
  }
}
#endif /* LV_HAVE_GENERIC */



#endif /* INCLUDED_volk_32f_x2_interleave_32fc_a_H */
