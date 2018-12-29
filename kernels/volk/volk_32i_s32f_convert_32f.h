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
 * \page volk_32i_s32f_convert_32f
 *
 * \b Overview
 *
 * Converts the samples in the inputVector from 32-bit integers into
 * floating point values and then divides them by the input scalar.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32i_s32f_convert_32f(float* outputVector, const int32_t* inputVector, const float scalar, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li inputVector: The vector of 32-bit integers.
 * \li scalar: The value that the output is divided by after being converted to a float.
 * \li num_points: The number of values.
 *
 * \b Outputs
 * \li complexVector: The output vector of floats.
 *
 * \b Example
 * Convert full-range integers to floats in range [0,1].
 * \code
 *   int N = 1<<8;
 *   unsigned int alignment = volk_get_alignment();
 *
 *   int32_t* x = (int32_t*)volk_malloc(N*sizeof(int32_t), alignment);
 *   float* z = (float*)volk_malloc(N*sizeof(float), alignment);
 *   float scale = (float)N;
 *   for(unsigned int ii=0; ii<N; ++ii){
 *       x[ii] = ii;
 *   }
 *
 *   volk_32i_s32f_convert_32f(z, x, scale, N);
 *
 *   volk_free(x);
 *   volk_free(z);
 * \endcode
 */

#ifndef INCLUDED_volk_32i_s32f_convert_32f_u_H
#define INCLUDED_volk_32i_s32f_convert_32f_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void
volk_32i_s32f_convert_32f_neon(float* outputVector,
                               const int32_t* inputVector,
                               const float scalar,
                               unsigned int num_points) {
    float* outputVectorPtr = outputVector;
    const int32_t* inputVectorPtr = inputVector;
    const float iScalar = 1.0 / scalar;
    unsigned int number;
    unsigned int quarter_points = num_points / 4;
    int32x4_t input_vec;
    float32x4_t ouput_vec, iscalar_vec;
    
    iscalar_vec = vdupq_n_f32(iScalar);
    
    for(number = 0; number < quarter_points; number++) {
        // load s32
        input_vec = vld1q_s32(inputVectorPtr);
        // convert s32 to f32
        ouput_vec = vcvtq_f32_s32(input_vec);
        // scale
        ouput_vec = vmulq_f32(ouput_vec, iscalar_vec);
        // store
        vst1q_f32(outputVectorPtr, ouput_vec);
        // move pointers ahead
        outputVectorPtr+=4;
        inputVectorPtr+=4;
    }
    
    // deal with the rest
    for(number = quarter_points * 4; number < num_points; number++) {
        *outputVectorPtr++ = ((float)(*inputVectorPtr++)) * iScalar;
    }
}

#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void
volk_32i_s32f_convert_32f_u_avx512f(float* outputVector, const int32_t* inputVector,
                                 const float scalar, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int onesixteenthPoints = num_points / 16;

  float* outputVectorPtr = outputVector;
  const float iScalar = 1.0 / scalar;
  __m512 invScalar = _mm512_set1_ps(iScalar);
  int32_t* inputPtr = (int32_t*)inputVector;
  __m512i inputVal;
  __m512 ret;

  for(;number < onesixteenthPoints; number++){
    // Load the values
    inputVal = _mm512_loadu_si512((__m512i*)inputPtr);

    ret = _mm512_cvtepi32_ps(inputVal);
    ret = _mm512_mul_ps(ret, invScalar);

    _mm512_storeu_ps(outputVectorPtr, ret);

    outputVectorPtr += 16;
    inputPtr += 16;
  }

  number = onesixteenthPoints * 16;
  for(; number < num_points; number++){
    outputVector[number] =((float)(inputVector[number])) * iScalar;
  }
}
#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void
volk_32i_s32f_convert_32f_u_avx2(float* outputVector, const int32_t* inputVector,
                                 const float scalar, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int oneEightPoints = num_points / 8;

  float* outputVectorPtr = outputVector;
  const float iScalar = 1.0 / scalar;
  __m256 invScalar = _mm256_set1_ps(iScalar);
  int32_t* inputPtr = (int32_t*)inputVector;
  __m256i inputVal;
  __m256 ret;

  for(;number < oneEightPoints; number++){
    // Load the 4 values
    inputVal = _mm256_loadu_si256((__m256i*)inputPtr);

    ret = _mm256_cvtepi32_ps(inputVal);
    ret = _mm256_mul_ps(ret, invScalar);

    _mm256_storeu_ps(outputVectorPtr, ret);

    outputVectorPtr += 8;
    inputPtr += 8;
  }

  number = oneEightPoints * 8;
  for(; number < num_points; number++){
    outputVector[number] =((float)(inputVector[number])) * iScalar;
  }
}
#endif /* LV_HAVE_AVX2 */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void
volk_32i_s32f_convert_32f_u_sse2(float* outputVector, const int32_t* inputVector,
                                 const float scalar, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int quarterPoints = num_points / 4;

  float* outputVectorPtr = outputVector;
  const float iScalar = 1.0 / scalar;
  __m128 invScalar = _mm_set_ps1(iScalar);
  int32_t* inputPtr = (int32_t*)inputVector;
  __m128i inputVal;
  __m128 ret;

  for(;number < quarterPoints; number++){
    // Load the 4 values
    inputVal = _mm_loadu_si128((__m128i*)inputPtr);

    ret = _mm_cvtepi32_ps(inputVal);
    ret = _mm_mul_ps(ret, invScalar);

    _mm_storeu_ps(outputVectorPtr, ret);

    outputVectorPtr += 4;
    inputPtr += 4;
  }

  number = quarterPoints * 4;
  for(; number < num_points; number++){
    outputVector[number] =((float)(inputVector[number])) * iScalar;
  }
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_GENERIC

static inline void
volk_32i_s32f_convert_32f_generic(float* outputVector, const int32_t* inputVector,
                                  const float scalar, unsigned int num_points)
{
  float* outputVectorPtr = outputVector;
  const int32_t* inputVectorPtr = inputVector;
  unsigned int number = 0;
  const float iScalar = 1.0 / scalar;

  for(number = 0; number < num_points; number++){
    *outputVectorPtr++ = ((float)(*inputVectorPtr++)) * iScalar;
  }
}
#endif /* LV_HAVE_GENERIC */

#endif /* INCLUDED_volk_32i_s32f_convert_32f_u_H */



#ifndef INCLUDED_volk_32i_s32f_convert_32f_a_H
#define INCLUDED_volk_32i_s32f_convert_32f_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void
volk_32i_s32f_convert_32f_a_avx512f(float* outputVector, const int32_t* inputVector,
                                 const float scalar, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int onesixteenthPoints = num_points / 16;

  float* outputVectorPtr = outputVector;
  const float iScalar = 1.0 / scalar;
  __m512 invScalar = _mm512_set1_ps(iScalar);
  int32_t* inputPtr = (int32_t*)inputVector;
  __m512i inputVal;
  __m512 ret;

  for(;number < onesixteenthPoints; number++){
    // Load the values
    inputVal = _mm512_load_si512((__m512i*)inputPtr);

    ret = _mm512_cvtepi32_ps(inputVal);
    ret = _mm512_mul_ps(ret, invScalar);

    _mm512_store_ps(outputVectorPtr, ret);

    outputVectorPtr += 16;
    inputPtr += 16;
  }

  number = onesixteenthPoints * 16;
  for(; number < num_points; number++){
    outputVector[number] =((float)(inputVector[number])) * iScalar;
  }
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void
volk_32i_s32f_convert_32f_a_avx2(float* outputVector, const int32_t* inputVector,
                                 const float scalar, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int oneEightPoints = num_points / 8;

  float* outputVectorPtr = outputVector;
  const float iScalar = 1.0 / scalar;
  __m256 invScalar = _mm256_set1_ps(iScalar);
  int32_t* inputPtr = (int32_t*)inputVector;
  __m256i inputVal;
  __m256 ret;

  for(;number < oneEightPoints; number++){
    // Load the 4 values
    inputVal = _mm256_load_si256((__m256i*)inputPtr);

    ret = _mm256_cvtepi32_ps(inputVal);
    ret = _mm256_mul_ps(ret, invScalar);

    _mm256_store_ps(outputVectorPtr, ret);

    outputVectorPtr += 8;
    inputPtr += 8;
  }

  number = oneEightPoints * 8;
  for(; number < num_points; number++){
    outputVector[number] =((float)(inputVector[number])) * iScalar;
  }
}
#endif /* LV_HAVE_AVX2 */


#ifdef LV_HAVE_SSE2
#include <emmintrin.h>

static inline void
volk_32i_s32f_convert_32f_a_sse2(float* outputVector, const int32_t* inputVector,
                                 const float scalar, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int quarterPoints = num_points / 4;

  float* outputVectorPtr = outputVector;
  const float iScalar = 1.0 / scalar;
  __m128 invScalar = _mm_set_ps1(iScalar);
  int32_t* inputPtr = (int32_t*)inputVector;
  __m128i inputVal;
  __m128 ret;

  for(;number < quarterPoints; number++){
    // Load the 4 values
    inputVal = _mm_load_si128((__m128i*)inputPtr);

    ret = _mm_cvtepi32_ps(inputVal);
    ret = _mm_mul_ps(ret, invScalar);

    _mm_store_ps(outputVectorPtr, ret);

    outputVectorPtr += 4;
    inputPtr += 4;
  }

  number = quarterPoints * 4;
  for(; number < num_points; number++){
    outputVector[number] =((float)(inputVector[number])) * iScalar;
  }
}
#endif /* LV_HAVE_SSE2 */


#ifdef LV_HAVE_GENERIC

static inline void
volk_32i_s32f_convert_32f_a_generic(float* outputVector, const int32_t* inputVector,
                                    const float scalar, unsigned int num_points)
{
  float* outputVectorPtr = outputVector;
  const int32_t* inputVectorPtr = inputVector;
  unsigned int number = 0;
  const float iScalar = 1.0 / scalar;

  for(number = 0; number < num_points; number++){
    *outputVectorPtr++ = ((float)(*inputVectorPtr++)) * iScalar;
  }
}
#endif /* LV_HAVE_GENERIC */




#endif /* INCLUDED_volk_32i_s32f_convert_32f_a_H */
