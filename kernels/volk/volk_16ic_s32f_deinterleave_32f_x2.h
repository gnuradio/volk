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
 * \page volk_16ic_s32f_deinterleave_32f_x2
 *
 * \b Overview
 *
 * Deinterleaves the complex 16 bit vector into I & Q vector data and
 * returns the result as two vectors of floats that have been scaled.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 *  void volk_16ic_s32f_deinterleave_32f_x2(float* iBuffer, float* qBuffer, const lv_16sc_t* complexVector, const float scalar, unsigned int num_points){
 * \endcode
 *
 * \b Inputs
 * \li complexVector: The complex input vector of 16-bit shorts.
 * \li scalar: The value to be divided against each sample of the input complex vector.
 * \li num_points: The number of complex data values to be deinterleaved.
 *
 * \b Outputs
 * \li iBuffer: The floating point I buffer output data.
 * \li qBuffer: The floating point Q buffer output data.
 *
 * \b Example
 * \code
 * int N = 10000;
 *
 * volk_16ic_s32f_deinterleave_32f_x2();
 *
 * volk_free(x);
 * volk_free(t);
 * \endcode
 */

#ifndef INCLUDED_volk_16ic_s32f_deinterleave_32f_x2_a_H
#define INCLUDED_volk_16ic_s32f_deinterleave_32f_x2_a_H

#include <volk/volk_common.h>
#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline
void volk_16ic_s32f_deinterleave_32f_x2_a_sse(float* iBuffer, float* qBuffer, const lv_16sc_t* complexVector,
                                              const float scalar, unsigned int num_points)
{
  float* iBufferPtr = iBuffer;
  float* qBufferPtr = qBuffer;

  uint64_t number = 0;
  const uint64_t quarterPoints = num_points / 4;
  __m128 cplxValue1, cplxValue2, iValue, qValue;

  __m128 invScalar = _mm_set_ps1(1.0/scalar);
  int16_t* complexVectorPtr = (int16_t*)complexVector;

  __VOLK_ATTR_ALIGNED(16) float floatBuffer[8];

  for(;number < quarterPoints; number++){

    floatBuffer[0] = (float)(complexVectorPtr[0]);
    floatBuffer[1] = (float)(complexVectorPtr[1]);
    floatBuffer[2] = (float)(complexVectorPtr[2]);
    floatBuffer[3] = (float)(complexVectorPtr[3]);

    floatBuffer[4] = (float)(complexVectorPtr[4]);
    floatBuffer[5] = (float)(complexVectorPtr[5]);
    floatBuffer[6] = (float)(complexVectorPtr[6]);
    floatBuffer[7] = (float)(complexVectorPtr[7]);

    cplxValue1 = _mm_load_ps(&floatBuffer[0]);
    cplxValue2 = _mm_load_ps(&floatBuffer[4]);

    complexVectorPtr += 8;

    cplxValue1 = _mm_mul_ps(cplxValue1, invScalar);
    cplxValue2 = _mm_mul_ps(cplxValue2, invScalar);

    // Arrange in i1i2i3i4 format
    iValue = _mm_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(2,0,2,0));
    // Arrange in q1q2q3q4 format
    qValue = _mm_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(3,1,3,1));

    _mm_store_ps(iBufferPtr, iValue);
    _mm_store_ps(qBufferPtr, qValue);

    iBufferPtr += 4;
    qBufferPtr += 4;
  }

  number = quarterPoints * 4;
  complexVectorPtr = (int16_t*)&complexVector[number];
  for(; number < num_points; number++){
    *iBufferPtr++ = (float)(*complexVectorPtr++) / scalar;
    *qBufferPtr++ = (float)(*complexVectorPtr++) / scalar;
  }
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_GENERIC

static inline void
volk_16ic_s32f_deinterleave_32f_x2_generic(float* iBuffer, float* qBuffer, const lv_16sc_t* complexVector,
                                           const float scalar, unsigned int num_points)
{
  const int16_t* complexVectorPtr = (const int16_t*)complexVector;
  float* iBufferPtr = iBuffer;
  float* qBufferPtr = qBuffer;
  unsigned int number;
  for(number = 0; number < num_points; number++){
    *iBufferPtr++ = (float)(*complexVectorPtr++) / scalar;
    *qBufferPtr++ = (float)(*complexVectorPtr++) / scalar;
  }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>
static inline void
volk_16ic_s32f_deinterleave_32f_x2_neon(float* iBuffer, float* qBuffer, const lv_16sc_t* complexVector,
                                        const float scalar, unsigned int num_points)
{
  const int16_t* complexVectorPtr = (const int16_t*)complexVector;
  float* iBufferPtr = iBuffer;
  float* qBufferPtr = qBuffer;
  unsigned int eighth_points = num_points / 4;
  unsigned int number;
  float iScalar = 1.f/scalar;
  float32x4_t invScalar;
  invScalar = vld1q_dup_f32(&iScalar);

  int16x4x2_t complexInput_s16;
  int32x4x2_t complexInput_s32;
  float32x4x2_t complexFloat;

  for(number = 0; number < eighth_points; number++){
    complexInput_s16 = vld2_s16(complexVectorPtr);
    complexInput_s32.val[0] = vmovl_s16(complexInput_s16.val[0]);
    complexInput_s32.val[1] = vmovl_s16(complexInput_s16.val[1]);
    complexFloat.val[0] = vcvtq_f32_s32(complexInput_s32.val[0]);
    complexFloat.val[1] = vcvtq_f32_s32(complexInput_s32.val[1]);
    complexFloat.val[0] = vmulq_f32(complexFloat.val[0], invScalar);
    complexFloat.val[1] = vmulq_f32(complexFloat.val[1], invScalar);
    vst1q_f32(iBufferPtr, complexFloat.val[0]);
    vst1q_f32(qBufferPtr, complexFloat.val[1]);
    complexVectorPtr += 8;
    iBufferPtr += 4;
    qBufferPtr += 4;
  }

  for(number = eighth_points*4; number < num_points; number++){
    *iBufferPtr++ = (float)(*complexVectorPtr++) / scalar;
    *qBufferPtr++ = (float)(*complexVectorPtr++) / scalar;
  }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_ORC
extern void
volk_16ic_s32f_deinterleave_32f_x2_a_orc_impl(float* iBuffer, float* qBuffer, const lv_16sc_t* complexVector,
                                              const float scalar, unsigned int num_points);

static inline void
volk_16ic_s32f_deinterleave_32f_x2_u_orc(float* iBuffer, float* qBuffer, const lv_16sc_t* complexVector,
                                         const float scalar, unsigned int num_points)
{
  volk_16ic_s32f_deinterleave_32f_x2_a_orc_impl(iBuffer, qBuffer, complexVector, scalar, num_points);
}
#endif /* LV_HAVE_ORC */


#endif /* INCLUDED_volk_16ic_s32f_deinterleave_32f_x2_a_H */
