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
 * \page volk_32f_x2_dot_prod_16i
 *
 * \b Overview
 *
 * This block computes the dot product (or inner product) between two
 * vectors, the \p input and \p taps vectors. Given a set of \p
 * num_points taps, the result is the sum of products between the two
 * vectors. The result is a single value stored in the \p result
 * address and is conerted to a fixed-point short.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_x2_dot_prod_16i(int16_t* result, const float* input, const float* taps, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li input: vector of floats.
 * \li taps:  float taps.
 * \li num_points: number of samples in both \p input and \p taps.
 *
 * \b Outputs
 * \li result: pointer to a short value to hold the dot product result.
 *
 * \b Example
 * \code
 * int N = 10000;
 *
 * <FIXME>
 *
 * volk_32f_x2_dot_prod_16i();
 *
 * \endcode
 */

#ifndef INCLUDED_volk_32f_x2_dot_prod_16i_H
#define INCLUDED_volk_32f_x2_dot_prod_16i_H

#include <volk/volk_common.h>
#include <stdio.h>


#ifdef LV_HAVE_GENERIC


static inline void volk_32f_x2_dot_prod_16i_generic(int16_t* result, const float* input, const float* taps, unsigned int num_points) {

  float dotProduct = 0;
  const float* aPtr = input;
  const float* bPtr=  taps;
  unsigned int number = 0;

  for(number = 0; number < num_points; number++){
    dotProduct += ((*aPtr++) * (*bPtr++));
  }

  *result = (int16_t)dotProduct;
}

#endif /*LV_HAVE_GENERIC*/


#ifdef LV_HAVE_AVX

static inline void volk_32f_x2_dot_prod_16i_a_avx(int16_t* result, const  float* input, const  float* taps, unsigned int num_points) {

  unsigned int number = 0;
  const unsigned int thirtySecondPoints = num_points / 32;

  float dotProduct = 0;
  const float* aPtr = input;
  const float* bPtr = taps;

  __m256 a0Val, a1Val, a2Val, a3Val;
  __m256 b0Val, b1Val, b2Val, b3Val;
  __m256 c0Val, c1Val, c2Val, c3Val;

  __m256 dotProdVal0 = _mm256_setzero_ps();
  __m256 dotProdVal1 = _mm256_setzero_ps();
  __m256 dotProdVal2 = _mm256_setzero_ps();
  __m256 dotProdVal3 = _mm256_setzero_ps();

  for(;number < thirtySecondPoints; number++){

    a0Val = _mm256_load_ps(aPtr);
    a1Val = _mm256_load_ps(aPtr+8);
    a2Val = _mm256_load_ps(aPtr+16);
    a3Val = _mm256_load_ps(aPtr+24);

    b0Val = _mm256_load_ps(bPtr);
    b1Val = _mm256_load_ps(bPtr+8);
    b2Val = _mm256_load_ps(bPtr+16);
    b3Val = _mm256_load_ps(bPtr+24);

    c0Val = _mm256_mul_ps(a0Val, b0Val);
    c1Val = _mm256_mul_ps(a1Val, b1Val);
    c2Val = _mm256_mul_ps(a2Val, b2Val);
    c3Val = _mm256_mul_ps(a3Val, b3Val);

    dotProdVal0 = _mm256_add_ps(c0Val, dotProdVal0);
    dotProdVal1 = _mm256_add_ps(c1Val, dotProdVal1);
    dotProdVal2 = _mm256_add_ps(c2Val, dotProdVal2);
    dotProdVal3 = _mm256_add_ps(c3Val, dotProdVal3);

    aPtr += 32;
    bPtr += 32;
  }

  dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal1);
  dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal2);
  dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal3);

  __VOLK_ATTR_ALIGNED(32) float dotProductVector[8];

  _mm256_store_ps(dotProductVector,dotProdVal0); // Store the results back into the dot product vector

  dotProduct = dotProductVector[0];
  dotProduct += dotProductVector[1];
  dotProduct += dotProductVector[2];
  dotProduct += dotProductVector[3];
  dotProduct += dotProductVector[4];
  dotProduct += dotProductVector[5];
  dotProduct += dotProductVector[6];
  dotProduct += dotProductVector[7];

  number = thirtySecondPoints*32;
  for(;number < num_points; number++){
    dotProduct += ((*aPtr++) * (*bPtr++));
  }

  *result = (short)dotProduct;
}

#endif /*LV_HAVE_AVX*/


#ifdef LV_HAVE_AVX

static inline void volk_32f_x2_dot_prod_16i_u_avx(int16_t* result, const  float* input, const  float* taps, unsigned int num_points) {

  unsigned int number = 0;
  const unsigned int thirtySecondPoints = num_points / 32;

  float dotProduct = 0;
  const float* aPtr = input;
  const float* bPtr = taps;

  __m256 a0Val, a1Val, a2Val, a3Val;
  __m256 b0Val, b1Val, b2Val, b3Val;
  __m256 c0Val, c1Val, c2Val, c3Val;

  __m256 dotProdVal0 = _mm256_setzero_ps();
  __m256 dotProdVal1 = _mm256_setzero_ps();
  __m256 dotProdVal2 = _mm256_setzero_ps();
  __m256 dotProdVal3 = _mm256_setzero_ps();

  for(;number < thirtySecondPoints; number++){

    a0Val = _mm256_loadu_ps(aPtr);
    a1Val = _mm256_loadu_ps(aPtr+8);
    a2Val = _mm256_loadu_ps(aPtr+16);
    a3Val = _mm256_loadu_ps(aPtr+24);

    b0Val = _mm256_loadu_ps(bPtr);
    b1Val = _mm256_loadu_ps(bPtr+8);
    b2Val = _mm256_loadu_ps(bPtr+16);
    b3Val = _mm256_loadu_ps(bPtr+24);

    c0Val = _mm256_mul_ps(a0Val, b0Val);
    c1Val = _mm256_mul_ps(a1Val, b1Val);
    c2Val = _mm256_mul_ps(a2Val, b2Val);
    c3Val = _mm256_mul_ps(a3Val, b3Val);

    dotProdVal0 = _mm256_add_ps(c0Val, dotProdVal0);
    dotProdVal1 = _mm256_add_ps(c1Val, dotProdVal1);
    dotProdVal2 = _mm256_add_ps(c2Val, dotProdVal2);
    dotProdVal3 = _mm256_add_ps(c3Val, dotProdVal3);

    aPtr += 32;
    bPtr += 32;
  }

  dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal1);
  dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal2);
  dotProdVal0 = _mm256_add_ps(dotProdVal0, dotProdVal3);

  __VOLK_ATTR_ALIGNED(32) float dotProductVector[8];

  _mm256_storeu_ps(dotProductVector,dotProdVal0); // Store the results back into the dot product vector

  dotProduct = dotProductVector[0];
  dotProduct += dotProductVector[1];
  dotProduct += dotProductVector[2];
  dotProduct += dotProductVector[3];
  dotProduct += dotProductVector[4];
  dotProduct += dotProductVector[5];
  dotProduct += dotProductVector[6];
  dotProduct += dotProductVector[7];

  number = thirtySecondPoints*32;
  for(;number < num_points; number++){
    dotProduct += ((*aPtr++) * (*bPtr++));
  }

  *result = (short)dotProduct;
}

#endif /*LV_HAVE_AVX*/


#ifdef LV_HAVE_SSE

static inline void volk_32f_x2_dot_prod_16i_a_sse(int16_t* result, const  float* input, const  float* taps, unsigned int num_points) {

  unsigned int number = 0;
  const unsigned int sixteenthPoints = num_points / 16;

  float dotProduct = 0;
  const float* aPtr = input;
  const float* bPtr = taps;

  __m128 a0Val, a1Val, a2Val, a3Val;
  __m128 b0Val, b1Val, b2Val, b3Val;
  __m128 c0Val, c1Val, c2Val, c3Val;

  __m128 dotProdVal0 = _mm_setzero_ps();
  __m128 dotProdVal1 = _mm_setzero_ps();
  __m128 dotProdVal2 = _mm_setzero_ps();
  __m128 dotProdVal3 = _mm_setzero_ps();

  for(;number < sixteenthPoints; number++){

    a0Val = _mm_load_ps(aPtr);
    a1Val = _mm_load_ps(aPtr+4);
    a2Val = _mm_load_ps(aPtr+8);
    a3Val = _mm_load_ps(aPtr+12);
    b0Val = _mm_load_ps(bPtr);
    b1Val = _mm_load_ps(bPtr+4);
    b2Val = _mm_load_ps(bPtr+8);
    b3Val = _mm_load_ps(bPtr+12);

    c0Val = _mm_mul_ps(a0Val, b0Val);
    c1Val = _mm_mul_ps(a1Val, b1Val);
    c2Val = _mm_mul_ps(a2Val, b2Val);
    c3Val = _mm_mul_ps(a3Val, b3Val);

    dotProdVal0 = _mm_add_ps(c0Val, dotProdVal0);
    dotProdVal1 = _mm_add_ps(c1Val, dotProdVal1);
    dotProdVal2 = _mm_add_ps(c2Val, dotProdVal2);
    dotProdVal3 = _mm_add_ps(c3Val, dotProdVal3);

    aPtr += 16;
    bPtr += 16;
  }

  dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal1);
  dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal2);
  dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal3);

  __VOLK_ATTR_ALIGNED(16) float dotProductVector[4];

  _mm_store_ps(dotProductVector,dotProdVal0); // Store the results back into the dot product vector

  dotProduct = dotProductVector[0];
  dotProduct += dotProductVector[1];
  dotProduct += dotProductVector[2];
  dotProduct += dotProductVector[3];

  number = sixteenthPoints*16;
  for(;number < num_points; number++){
    dotProduct += ((*aPtr++) * (*bPtr++));
  }

  *result = (short)dotProduct;
}

#endif /*LV_HAVE_SSE*/


#ifdef LV_HAVE_SSE

static inline void volk_32f_x2_dot_prod_16i_u_sse(int16_t* result, const  float* input, const  float* taps, unsigned int num_points) {

  unsigned int number = 0;
  const unsigned int sixteenthPoints = num_points / 16;

  float dotProduct = 0;
  const float* aPtr = input;
  const float* bPtr = taps;

  __m128 a0Val, a1Val, a2Val, a3Val;
  __m128 b0Val, b1Val, b2Val, b3Val;
  __m128 c0Val, c1Val, c2Val, c3Val;

  __m128 dotProdVal0 = _mm_setzero_ps();
  __m128 dotProdVal1 = _mm_setzero_ps();
  __m128 dotProdVal2 = _mm_setzero_ps();
  __m128 dotProdVal3 = _mm_setzero_ps();

  for(;number < sixteenthPoints; number++){

    a0Val = _mm_loadu_ps(aPtr);
    a1Val = _mm_loadu_ps(aPtr+4);
    a2Val = _mm_loadu_ps(aPtr+8);
    a3Val = _mm_loadu_ps(aPtr+12);
    b0Val = _mm_loadu_ps(bPtr);
    b1Val = _mm_loadu_ps(bPtr+4);
    b2Val = _mm_loadu_ps(bPtr+8);
    b3Val = _mm_loadu_ps(bPtr+12);

    c0Val = _mm_mul_ps(a0Val, b0Val);
    c1Val = _mm_mul_ps(a1Val, b1Val);
    c2Val = _mm_mul_ps(a2Val, b2Val);
    c3Val = _mm_mul_ps(a3Val, b3Val);

    dotProdVal0 = _mm_add_ps(c0Val, dotProdVal0);
    dotProdVal1 = _mm_add_ps(c1Val, dotProdVal1);
    dotProdVal2 = _mm_add_ps(c2Val, dotProdVal2);
    dotProdVal3 = _mm_add_ps(c3Val, dotProdVal3);

    aPtr += 16;
    bPtr += 16;
  }

  dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal1);
  dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal2);
  dotProdVal0 = _mm_add_ps(dotProdVal0, dotProdVal3);

  __VOLK_ATTR_ALIGNED(16) float dotProductVector[4];

  _mm_store_ps(dotProductVector,dotProdVal0); // Store the results back into the dot product vector

  dotProduct = dotProductVector[0];
  dotProduct += dotProductVector[1];
  dotProduct += dotProductVector[2];
  dotProduct += dotProductVector[3];

  number = sixteenthPoints*16;
  for(;number < num_points; number++){
    dotProduct += ((*aPtr++) * (*bPtr++));
  }

  *result = (short)dotProduct;
}

#endif /*LV_HAVE_SSE*/

#endif /*INCLUDED_volk_32f_x2_dot_prod_16i_H*/
