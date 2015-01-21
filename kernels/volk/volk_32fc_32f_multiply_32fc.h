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
 * \page volk_32fc_32f_multiply_32fc
 *
 * \b Overview
 *
 * Multiplies a complex vector by a floating point vector and returns
 * the complex result.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_32f_multiply_32fc(lv_32fc_t* cVector, const lv_32fc_t* aVector, const float* bVector, unsigned int num_points);
 * \endcode
 *
 * \b Inputs
 * \li aVector: The input vector of complex floats.
 * \li bVector: The input vector of floats.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li outputVector: The output vector complex floats.
 *
 * \b Example
 * \code
 * int N = 10000;
 *
 * volk_32fc_32f_multiply_32fc();
 *
 * volk_free(x);
 * volk_free(t);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_32f_multiply_32fc_a_H
#define INCLUDED_volk_32fc_32f_multiply_32fc_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32fc_32f_multiply_32fc_a_avx(lv_32fc_t* cVector, const lv_32fc_t* aVector,
                                  const float* bVector, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int eighthPoints = num_points / 8;

  lv_32fc_t* cPtr = cVector;
  const lv_32fc_t* aPtr = aVector;
  const float* bPtr=  bVector;

  __m256 aVal1, aVal2, bVal, bVal1, bVal2, cVal1, cVal2;

  __m256i permute_mask = _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0);

  for(;number < eighthPoints; number++){

    aVal1 = _mm256_load_ps((float *)aPtr);
    aPtr += 4;

    aVal2 = _mm256_load_ps((float *)aPtr);
    aPtr += 4;

    bVal = _mm256_load_ps(bPtr); // b0|b1|b2|b3|b4|b5|b6|b7
    bPtr += 8;

    bVal1 = _mm256_permute2f128_ps(bVal, bVal, 0x00); // b0|b1|b2|b3|b0|b1|b2|b3
    bVal2 = _mm256_permute2f128_ps(bVal, bVal, 0x11); // b4|b5|b6|b7|b4|b5|b6|b7

    bVal1 = _mm256_permutevar_ps(bVal1, permute_mask); // b0|b0|b1|b1|b2|b2|b3|b3
    bVal2 = _mm256_permutevar_ps(bVal2, permute_mask); // b4|b4|b5|b5|b6|b6|b7|b7

    cVal1 = _mm256_mul_ps(aVal1, bVal1);
    cVal2 = _mm256_mul_ps(aVal2, bVal2);

    _mm256_store_ps((float*)cPtr,cVal1); // Store the results back into the C container
    cPtr += 4;

    _mm256_store_ps((float*)cPtr,cVal2); // Store the results back into the C container
    cPtr += 4;
  }

  number = eighthPoints * 8;
  for(;number < num_points; ++number){
    *cPtr++ = (*aPtr++) * (*bPtr++);
  }
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void
volk_32fc_32f_multiply_32fc_a_sse(lv_32fc_t* cVector, const lv_32fc_t* aVector,
                                  const float* bVector, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int quarterPoints = num_points / 4;

  lv_32fc_t* cPtr = cVector;
  const lv_32fc_t* aPtr = aVector;
  const float* bPtr=  bVector;

  __m128 aVal1, aVal2, bVal, bVal1, bVal2, cVal;
  for(;number < quarterPoints; number++){

    aVal1 = _mm_load_ps((const float*)aPtr);
    aPtr += 2;

    aVal2 = _mm_load_ps((const float*)aPtr);
    aPtr += 2;

    bVal = _mm_load_ps(bPtr);
    bPtr += 4;

    bVal1 = _mm_shuffle_ps(bVal, bVal, _MM_SHUFFLE(1,1,0,0));
    bVal2 = _mm_shuffle_ps(bVal, bVal, _MM_SHUFFLE(3,3,2,2));

    cVal = _mm_mul_ps(aVal1, bVal1);

    _mm_store_ps((float*)cPtr,cVal); // Store the results back into the C container
    cPtr += 2;

    cVal = _mm_mul_ps(aVal2, bVal2);

    _mm_store_ps((float*)cPtr,cVal); // Store the results back into the C container

    cPtr += 2;
  }

  number = quarterPoints * 4;
  for(;number < num_points; number++){
    *cPtr++ = (*aPtr++) * (*bPtr);
    bPtr++;
  }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_GENERIC

static inline void
volk_32fc_32f_multiply_32fc_generic(lv_32fc_t* cVector, const lv_32fc_t* aVector,
                                    const float* bVector, unsigned int num_points)
{
  lv_32fc_t* cPtr = cVector;
  const lv_32fc_t* aPtr = aVector;
  const float* bPtr=  bVector;
  unsigned int number = 0;

  for(number = 0; number < num_points; number++){
    *cPtr++ = (*aPtr++) * (*bPtr++);
  }
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void
volk_32fc_32f_multiply_32fc_neon(lv_32fc_t* cVector, const lv_32fc_t* aVector,
                                 const float* bVector, unsigned int num_points)
{
  lv_32fc_t* cPtr = cVector;
  const lv_32fc_t* aPtr = aVector;
  const float* bPtr=  bVector;
  unsigned int number = 0;
  unsigned int quarter_points = num_points / 4;

  float32x4x2_t inputVector, outputVector;
  float32x4_t tapsVector;
  for(number = 0; number < quarter_points; number++){
    inputVector = vld2q_f32((float*)aPtr);
    tapsVector = vld1q_f32(bPtr);

    outputVector.val[0] = vmulq_f32(inputVector.val[0], tapsVector);
    outputVector.val[1] = vmulq_f32(inputVector.val[1], tapsVector);

    vst2q_f32((float*)cPtr, outputVector);
    aPtr += 4;
    bPtr += 4;
    cPtr += 4;
  }

  for(number = quarter_points * 4; number < num_points; number++){
    *cPtr++ = (*aPtr++) * (*bPtr++);
  }
}
#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_ORC

extern void
volk_32fc_32f_multiply_32fc_a_orc_impl(lv_32fc_t* cVector, const lv_32fc_t* aVector,
                                       const float* bVector, unsigned int num_points);

static inline void
volk_32fc_32f_multiply_32fc_u_orc(lv_32fc_t* cVector, const lv_32fc_t* aVector,
                                  const float* bVector, unsigned int num_points)
{
  volk_32fc_32f_multiply_32fc_a_orc_impl(cVector, aVector, bVector, num_points);
}

#endif /* LV_HAVE_GENERIC */


#endif /* INCLUDED_volk_32fc_32f_multiply_32fc_a_H */
