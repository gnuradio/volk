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
 * \page volk_32fc_s32f_magnitude_16i
 *
 * \b Overview
 *
 * Calculates the magnitude of the complexVector and stores the
 * results in the magnitudeVector. The results are scaled and
 * converted into 16-bit shorts.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_s32f_magnitude_16i(int16_t* magnitudeVector, const lv_32fc_t* complexVector, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li complexVector: The complex input vector.
 * \li num_points: The number of samples.
 *
 * \b Outputs
 * \li magnitudeVector: The output value as 16-bit shorts.
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
 *   volk_32fc_s32f_magnitude_16i(out, in, scale, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("out[%u] = %i\n", ii, out[ii]);
 *   }
 *
 *   volk_free(in);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_s32f_magnitude_16i_a_H
#define INCLUDED_volk_32fc_s32f_magnitude_16i_a_H

#include <volk/volk_common.h>
#include <inttypes.h>
#include <stdio.h>
#include <math.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void
volk_32fc_s32f_magnitude_16i_a_avx2(int16_t* magnitudeVector, const lv_32fc_t* complexVector,
                                    const float scalar, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int eighthPoints = num_points / 8;

  const float* complexVectorPtr = (const float*)complexVector;
  int16_t* magnitudeVectorPtr = magnitudeVector;

  __m256 vScalar = _mm256_set1_ps(scalar);
  __m256i idx = _mm256_set_epi32(0,0,0,0,5,1,4,0);
  __m256 cplxValue1, cplxValue2, result;
  __m256i resultInt;
  __m128i resultShort;

  for(;number < eighthPoints; number++){
    cplxValue1 = _mm256_load_ps(complexVectorPtr);
    complexVectorPtr += 8;

    cplxValue2 = _mm256_load_ps(complexVectorPtr);
    complexVectorPtr += 8;

    cplxValue1 = _mm256_mul_ps(cplxValue1, cplxValue1); // Square the values
    cplxValue2 = _mm256_mul_ps(cplxValue2, cplxValue2); // Square the Values

    result = _mm256_hadd_ps(cplxValue1, cplxValue2); // Add the I2 and Q2 values

    result = _mm256_sqrt_ps(result);

    result = _mm256_mul_ps(result, vScalar);

    resultInt = _mm256_cvtps_epi32(result);
    resultInt = _mm256_packs_epi32(resultInt, resultInt);
    resultInt = _mm256_permutevar8x32_epi32(resultInt, idx); //permute to compensate for shuffling in hadd and packs
    resultShort = _mm256_extracti128_si256(resultInt,0);
    _mm_store_si128((__m128i*)magnitudeVectorPtr,resultShort);
    magnitudeVectorPtr += 8;
  }

  number = eighthPoints * 8;
  magnitudeVectorPtr = &magnitudeVector[number];
  for(; number < num_points; number++){
    float val1Real = *complexVectorPtr++;
    float val1Imag = *complexVectorPtr++;
    *magnitudeVectorPtr++ = (int16_t)rintf(sqrtf((val1Real * val1Real) + (val1Imag * val1Imag)) * scalar);
  }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>

static inline void
volk_32fc_s32f_magnitude_16i_a_sse3(int16_t* magnitudeVector, const lv_32fc_t* complexVector,
                                    const float scalar, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int quarterPoints = num_points / 4;

  const float* complexVectorPtr = (const float*)complexVector;
  int16_t* magnitudeVectorPtr = magnitudeVector;

  __m128 vScalar = _mm_set_ps1(scalar);

  __m128 cplxValue1, cplxValue2, result;

  __VOLK_ATTR_ALIGNED(16) float floatBuffer[4];

  for(;number < quarterPoints; number++){
    cplxValue1 = _mm_load_ps(complexVectorPtr);
    complexVectorPtr += 4;

    cplxValue2 = _mm_load_ps(complexVectorPtr);
    complexVectorPtr += 4;

    cplxValue1 = _mm_mul_ps(cplxValue1, cplxValue1); // Square the values
    cplxValue2 = _mm_mul_ps(cplxValue2, cplxValue2); // Square the Values

    result = _mm_hadd_ps(cplxValue1, cplxValue2); // Add the I2 and Q2 values

    result = _mm_sqrt_ps(result);

    result = _mm_mul_ps(result, vScalar);

    _mm_store_ps(floatBuffer, result);
    *magnitudeVectorPtr++ = (int16_t)rintf(floatBuffer[0]);
    *magnitudeVectorPtr++ = (int16_t)rintf(floatBuffer[1]);
    *magnitudeVectorPtr++ = (int16_t)rintf(floatBuffer[2]);
    *magnitudeVectorPtr++ = (int16_t)rintf(floatBuffer[3]);
  }

  number = quarterPoints * 4;
  magnitudeVectorPtr = &magnitudeVector[number];
  for(; number < num_points; number++){
    float val1Real = *complexVectorPtr++;
    float val1Imag = *complexVectorPtr++;
    *magnitudeVectorPtr++ = (int16_t)rintf(sqrtf((val1Real * val1Real) + (val1Imag * val1Imag)) * scalar);
  }
}
#endif /* LV_HAVE_SSE3 */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void
volk_32fc_s32f_magnitude_16i_a_sse(int16_t* magnitudeVector, const lv_32fc_t* complexVector,
                                   const float scalar, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int quarterPoints = num_points / 4;

  const float* complexVectorPtr = (const float*)complexVector;
  int16_t* magnitudeVectorPtr = magnitudeVector;

  __m128 vScalar = _mm_set_ps1(scalar);

  __m128 cplxValue1, cplxValue2, iValue, qValue, result;

  __VOLK_ATTR_ALIGNED(16) float floatBuffer[4];

  for(;number < quarterPoints; number++){
    cplxValue1 = _mm_load_ps(complexVectorPtr);
    complexVectorPtr += 4;

    cplxValue2 = _mm_load_ps(complexVectorPtr);
    complexVectorPtr += 4;

    // Arrange in i1i2i3i4 format
    iValue = _mm_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(2,0,2,0));
    // Arrange in q1q2q3q4 format
    qValue = _mm_shuffle_ps(cplxValue1, cplxValue2, _MM_SHUFFLE(3,1,3,1));

    iValue = _mm_mul_ps(iValue, iValue); // Square the I values
    qValue = _mm_mul_ps(qValue, qValue); // Square the Q Values

    result = _mm_add_ps(iValue, qValue); // Add the I2 and Q2 values

    result = _mm_sqrt_ps(result);

    result = _mm_mul_ps(result, vScalar);

    _mm_store_ps(floatBuffer, result);
    *magnitudeVectorPtr++ = (int16_t)rintf(floatBuffer[0]);
    *magnitudeVectorPtr++ = (int16_t)rintf(floatBuffer[1]);
    *magnitudeVectorPtr++ = (int16_t)rintf(floatBuffer[2]);
    *magnitudeVectorPtr++ = (int16_t)rintf(floatBuffer[3]);
  }

  number = quarterPoints * 4;
  magnitudeVectorPtr = &magnitudeVector[number];
  for(; number < num_points; number++){
    float val1Real = *complexVectorPtr++;
    float val1Imag = *complexVectorPtr++;
    *magnitudeVectorPtr++ = (int16_t)rintf(sqrtf((val1Real * val1Real) + (val1Imag * val1Imag)) * scalar);
  }
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_GENERIC

static inline void
volk_32fc_s32f_magnitude_16i_generic(int16_t* magnitudeVector, const lv_32fc_t* complexVector,
                                     const float scalar, unsigned int num_points)
{
  const float* complexVectorPtr = (float*)complexVector;
  int16_t* magnitudeVectorPtr = magnitudeVector;
  unsigned int number = 0;
  for(number = 0; number < num_points; number++){
    const float real = *complexVectorPtr++;
    const float imag = *complexVectorPtr++;
    *magnitudeVectorPtr++ = (int16_t)rintf(sqrtf((real*real) + (imag*imag)) * scalar);
  }
}
#endif /* LV_HAVE_GENERIC */


#endif /* INCLUDED_volk_32fc_s32f_magnitude_16i_a_H */

#ifndef INCLUDED_volk_32fc_s32f_magnitude_16i_u_H
#define INCLUDED_volk_32fc_s32f_magnitude_16i_u_H

#include <volk/volk_common.h>
#include <inttypes.h>
#include <stdio.h>
#include <math.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void
volk_32fc_s32f_magnitude_16i_u_avx2(int16_t* magnitudeVector, const lv_32fc_t* complexVector,
                                    const float scalar, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int eighthPoints = num_points / 8;

  const float* complexVectorPtr = (const float*)complexVector;
  int16_t* magnitudeVectorPtr = magnitudeVector;

  __m256 vScalar = _mm256_set1_ps(scalar);
  __m256i idx = _mm256_set_epi32(0,0,0,0,5,1,4,0);
  __m256 cplxValue1, cplxValue2, result;
  __m256i resultInt;
  __m128i resultShort;

  for(;number < eighthPoints; number++){
    cplxValue1 = _mm256_loadu_ps(complexVectorPtr);
    complexVectorPtr += 8;

    cplxValue2 = _mm256_loadu_ps(complexVectorPtr);
    complexVectorPtr += 8;

    cplxValue1 = _mm256_mul_ps(cplxValue1, cplxValue1); // Square the values
    cplxValue2 = _mm256_mul_ps(cplxValue2, cplxValue2); // Square the Values

    result = _mm256_hadd_ps(cplxValue1, cplxValue2); // Add the I2 and Q2 values

    result = _mm256_sqrt_ps(result);

    result = _mm256_mul_ps(result, vScalar);

    resultInt = _mm256_cvtps_epi32(result);
    resultInt = _mm256_packs_epi32(resultInt, resultInt);
    resultInt = _mm256_permutevar8x32_epi32(resultInt, idx); //permute to compensate for shuffling in hadd and packs
    resultShort = _mm256_extracti128_si256(resultInt,0);
    _mm_storeu_si128((__m128i*)magnitudeVectorPtr,resultShort);
    magnitudeVectorPtr += 8;
  }

  number = eighthPoints * 8;
  magnitudeVectorPtr = &magnitudeVector[number];
  for(; number < num_points; number++){
    float val1Real = *complexVectorPtr++;
    float val1Imag = *complexVectorPtr++;
    *magnitudeVectorPtr++ = (int16_t)rintf(sqrtf((val1Real * val1Real) + (val1Imag * val1Imag)) * scalar);
  }
}
#endif /* LV_HAVE_AVX2 */

#endif /* INCLUDED_volk_32fc_s32f_magnitude_16i_u_H */
