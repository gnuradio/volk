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
 * \page volk_32i_x2_and_32i
 *
 * \b Overview
 *
 * Computes the Boolean AND operation between two input 32-bit integer vectors.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32i_x2_and_32i(int32_t* cVector, const int32_t* aVector, const int32_t* bVector, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li aVector: Input vector of samples.
 * \li bVector: Input vector of samples.
 * \li num_points: The number of values.
 *
 * \b Outputs
 * \li cVector: The output vector.
 *
 * \b Example
 * \code
 * int N = 10000;
 *
 * volk_32i_x2_and_32i();
 *
 * volk_free(x);
 * \endcode
 */

#ifndef INCLUDED_volk_32i_x2_and_32i_a_H
#define INCLUDED_volk_32i_x2_and_32i_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void
volk_32i_x2_and_32i_a_sse(int32_t* cVector, const int32_t* aVector,
                          const int32_t* bVector, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int quarterPoints = num_points / 4;

  float* cPtr = (float*)cVector;
  const float* aPtr = (float*)aVector;
  const float* bPtr = (float*)bVector;

  __m128 aVal, bVal, cVal;
  for(;number < quarterPoints; number++){

    aVal = _mm_load_ps(aPtr);
    bVal = _mm_load_ps(bPtr);

    cVal = _mm_and_ps(aVal, bVal);

    _mm_store_ps(cPtr,cVal); // Store the results back into the C container

    aPtr += 4;
    bPtr += 4;
    cPtr += 4;
  }

  number = quarterPoints * 4;
  for(;number < num_points; number++){
    cVector[number] = aVector[number] & bVector[number];
  }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void
volk_32i_x2_and_32i_neon(int32_t* cVector, const int32_t* aVector, const int32_t* bVector, unsigned int num_points)
{
  int32_t* cPtr = cVector;
  const int32_t* aPtr = aVector;
  const int32_t* bPtr=  bVector;
  unsigned int number = 0;
  unsigned int quarter_points = num_points / 4;

  int32x4_t a_val, b_val, c_val;

  for(number = 0; number < quarter_points; number++){
    a_val = vld1q_s32(aPtr);
    b_val = vld1q_s32(bPtr);
    c_val = vandq_s32(a_val, b_val);
    vst1q_s32(cPtr, c_val);
    aPtr += 4;
    bPtr += 4;
    cPtr += 4;
  }

  for(number = quarter_points * 4; number < num_points; number++){
    *cPtr++ = (*aPtr++) & (*bPtr++);
  }
}
#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_GENERIC

static inline void
volk_32i_x2_and_32i_generic(int32_t* cVector, const int32_t* aVector,
                            const int32_t* bVector, unsigned int num_points)
{
  int32_t* cPtr = cVector;
  const int32_t* aPtr = aVector;
  const int32_t* bPtr=  bVector;
  unsigned int number = 0;

  for(number = 0; number < num_points; number++){
    *cPtr++ = (*aPtr++) & (*bPtr++);
  }
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_ORC
extern void
volk_32i_x2_and_32i_a_orc_impl(int32_t* cVector, const int32_t* aVector,
                               const int32_t* bVector, unsigned int num_points);

static inline void
volk_32i_x2_and_32i_u_orc(int32_t* cVector, const int32_t* aVector,
                          const int32_t* bVector, unsigned int num_points)
{
  volk_32i_x2_and_32i_a_orc_impl(cVector, aVector, bVector, num_points);
}
#endif /* LV_HAVE_ORC */


#endif /* INCLUDED_volk_32i_x2_and_32i_a_H */
