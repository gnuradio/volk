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
 * \page volk_16ic_s32f_magnitude_32f
 *
 * \b Overview
 *
 * Computes the magnitude of the complexVector and stores the results
 * in the magnitudeVector as a scaled floating point number.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_16ic_s32f_magnitude_32f(float* magnitudeVector, const lv_16sc_t* complexVector, const float scalar, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li complexVector: The complex input vector of complex 16-bit shorts.
 * \li scalar: The value to be divided against each sample of the input complex vector.
 * \li num_points: The number of samples.
 *
 * \b Outputs
 * \li magnitudeVector: The magnitude of the complex values.
 *
 * \b Example
 * \code
 * int N = 10000;
 *
 * volk_16ic_s32f_magnitude_32f();
 *
 * volk_free(x);
 * volk_free(t);
 * \endcode
 */

#ifndef INCLUDED_volk_16ic_s32f_magnitude_32f_a_H
#define INCLUDED_volk_16ic_s32f_magnitude_32f_a_H

#include <volk/volk_common.h>
#include <inttypes.h>
#include <stdio.h>
#include <math.h>

#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>

static inline void
volk_16ic_s32f_magnitude_32f_a_sse3(float* magnitudeVector, const lv_16sc_t* complexVector,
                                    const float scalar, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int quarterPoints = num_points / 4;

  const int16_t* complexVectorPtr = (const int16_t*)complexVector;
  float* magnitudeVectorPtr = magnitudeVector;

  __m128 invScalar = _mm_set_ps1(1.0/scalar);

  __m128 cplxValue1, cplxValue2, result;

  __VOLK_ATTR_ALIGNED(16) float inputFloatBuffer[8];

  for(;number < quarterPoints; number++){

    inputFloatBuffer[0] = (float)(complexVectorPtr[0]);
    inputFloatBuffer[1] = (float)(complexVectorPtr[1]);
    inputFloatBuffer[2] = (float)(complexVectorPtr[2]);
    inputFloatBuffer[3] = (float)(complexVectorPtr[3]);

    inputFloatBuffer[4] = (float)(complexVectorPtr[4]);
    inputFloatBuffer[5] = (float)(complexVectorPtr[5]);
    inputFloatBuffer[6] = (float)(complexVectorPtr[6]);
    inputFloatBuffer[7] = (float)(complexVectorPtr[7]);

    cplxValue1 = _mm_load_ps(&inputFloatBuffer[0]);
    cplxValue2 = _mm_load_ps(&inputFloatBuffer[4]);

    complexVectorPtr += 8;

    cplxValue1 = _mm_mul_ps(cplxValue1, invScalar);
    cplxValue2 = _mm_mul_ps(cplxValue2, invScalar);

    cplxValue1 = _mm_mul_ps(cplxValue1, cplxValue1); // Square the values
    cplxValue2 = _mm_mul_ps(cplxValue2, cplxValue2); // Square the Values

    result = _mm_hadd_ps(cplxValue1, cplxValue2); // Add the I2 and Q2 values

    result = _mm_sqrt_ps(result); // Square root the values

    _mm_store_ps(magnitudeVectorPtr, result);

    magnitudeVectorPtr += 4;
  }

  number = quarterPoints * 4;
  magnitudeVectorPtr = &magnitudeVector[number];
  complexVectorPtr = (const int16_t*)&complexVector[number];
  for(; number < num_points; number++){
    float val1Real = (float)(*complexVectorPtr++) / scalar;
    float val1Imag = (float)(*complexVectorPtr++) / scalar;
    *magnitudeVectorPtr++ = sqrtf((val1Real * val1Real) + (val1Imag * val1Imag));
  }
}
#endif /* LV_HAVE_SSE3 */

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void
volk_16ic_s32f_magnitude_32f_a_sse(float* magnitudeVector, const lv_16sc_t* complexVector,
                                   const float scalar, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int quarterPoints = num_points / 4;

  const int16_t* complexVectorPtr = (const int16_t*)complexVector;
  float* magnitudeVectorPtr = magnitudeVector;

  const float iScalar = 1.0 / scalar;
  __m128 invScalar = _mm_set_ps1(iScalar);

  __m128 cplxValue1, cplxValue2, result, re, im;

  __VOLK_ATTR_ALIGNED(16) float inputFloatBuffer[8];

  for(;number < quarterPoints; number++){
    inputFloatBuffer[0] = (float)(complexVectorPtr[0]);
    inputFloatBuffer[1] = (float)(complexVectorPtr[1]);
    inputFloatBuffer[2] = (float)(complexVectorPtr[2]);
    inputFloatBuffer[3] = (float)(complexVectorPtr[3]);

    inputFloatBuffer[4] = (float)(complexVectorPtr[4]);
    inputFloatBuffer[5] = (float)(complexVectorPtr[5]);
    inputFloatBuffer[6] = (float)(complexVectorPtr[6]);
    inputFloatBuffer[7] = (float)(complexVectorPtr[7]);

    cplxValue1 = _mm_load_ps(&inputFloatBuffer[0]);
    cplxValue2 = _mm_load_ps(&inputFloatBuffer[4]);

    re = _mm_shuffle_ps(cplxValue1, cplxValue2, 0x88);
    im = _mm_shuffle_ps(cplxValue1, cplxValue2, 0xdd);

    complexVectorPtr += 8;

    cplxValue1 = _mm_mul_ps(re, invScalar);
    cplxValue2 = _mm_mul_ps(im, invScalar);

    cplxValue1 = _mm_mul_ps(cplxValue1, cplxValue1); // Square the values
    cplxValue2 = _mm_mul_ps(cplxValue2, cplxValue2); // Square the Values

    result = _mm_add_ps(cplxValue1, cplxValue2); // Add the I2 and Q2 values

    result = _mm_sqrt_ps(result); // Square root the values

    _mm_store_ps(magnitudeVectorPtr, result);

    magnitudeVectorPtr += 4;
  }

  number = quarterPoints * 4;
  magnitudeVectorPtr = &magnitudeVector[number];
  complexVectorPtr = (const int16_t*)&complexVector[number];
  for(; number < num_points; number++){
    float val1Real = (float)(*complexVectorPtr++) * iScalar;
    float val1Imag = (float)(*complexVectorPtr++) * iScalar;
    *magnitudeVectorPtr++ = sqrtf((val1Real * val1Real) + (val1Imag * val1Imag));
  }
}


#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_GENERIC

static inline void
volk_16ic_s32f_magnitude_32f_generic(float* magnitudeVector, const lv_16sc_t* complexVector,
                                     const float scalar, unsigned int num_points)
{
  const int16_t* complexVectorPtr = (const int16_t*)complexVector;
  float* magnitudeVectorPtr = magnitudeVector;
  unsigned int number = 0;
  const float invScalar = 1.0 / scalar;
  for(number = 0; number < num_points; number++){
    float real = ( (float) (*complexVectorPtr++)) * invScalar;
    float imag = ( (float) (*complexVectorPtr++)) * invScalar;
    *magnitudeVectorPtr++ = sqrtf((real*real) + (imag*imag));
  }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_ORC_DISABLED

extern void
volk_16ic_s32f_magnitude_32f_a_orc_impl(float* magnitudeVector, const lv_16sc_t* complexVector,
                                        const float scalar, unsigned int num_points);

static inline void
volk_16ic_s32f_magnitude_32f_u_orc(float* magnitudeVector, const lv_16sc_t* complexVector,
                                   const float scalar, unsigned int num_points)
{
  volk_16ic_s32f_magnitude_32f_a_orc_impl(magnitudeVector, complexVector, scalar, num_points);
}
#endif /* LV_HAVE_ORC */


#endif /* INCLUDED_volk_16ic_s32f_magnitude_32f_a_H */
