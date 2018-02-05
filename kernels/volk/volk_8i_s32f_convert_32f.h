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
 * \page volk_8i_s32f_convert_32f
 *
 * \b Overview
 *
 * Convert the input vector of 8-bit chars to a vector of floats. The
 * floats are then divided by the scalar factor.  shorts.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_8i_s32f_convert_32f(float* outputVector, const int8_t* inputVector, const float scalar, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li inputVector: The input vector of 8-bit chars.
 * \li scalar: the scaling factor used to divide the results of the conversion.
 * \li num_points: The number of values.
 *
 * \b Outputs
 * \li outputVector: The output 16-bit shorts.
 *
 * \b Example
 * \code
 * int N = 10000;
 *
 * volk_8i_s32f_convert_32f();
 *
 * volk_free(x);
 * \endcode
 */

#ifndef INCLUDED_volk_8i_s32f_convert_32f_u_H
#define INCLUDED_volk_8i_s32f_convert_32f_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void
volk_8i_s32f_convert_32f_u_sse4_1(float* outputVector, const int8_t* inputVector,
                                  const float scalar, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int sixteenthPoints = num_points / 16;

  float* outputVectorPtr = outputVector;
  const float iScalar = 1.0 / scalar;
  __m128 invScalar = _mm_set_ps1( iScalar );
  const int8_t* inputVectorPtr = inputVector;
  __m128 ret;
  __m128i inputVal;
  __m128i interimVal;

  for(;number < sixteenthPoints; number++){
    inputVal = _mm_loadu_si128((__m128i*)inputVectorPtr);

    interimVal = _mm_cvtepi8_epi32(inputVal);
    ret = _mm_cvtepi32_ps(interimVal);
    ret = _mm_mul_ps(ret, invScalar);
    _mm_storeu_ps(outputVectorPtr, ret);
    outputVectorPtr += 4;

    inputVal = _mm_srli_si128(inputVal, 4);
    interimVal = _mm_cvtepi8_epi32(inputVal);
    ret = _mm_cvtepi32_ps(interimVal);
    ret = _mm_mul_ps(ret, invScalar);
    _mm_storeu_ps(outputVectorPtr, ret);
    outputVectorPtr += 4;

    inputVal = _mm_srli_si128(inputVal, 4);
    interimVal = _mm_cvtepi8_epi32(inputVal);
    ret = _mm_cvtepi32_ps(interimVal);
    ret = _mm_mul_ps(ret, invScalar);
    _mm_storeu_ps(outputVectorPtr, ret);
    outputVectorPtr += 4;

    inputVal = _mm_srli_si128(inputVal, 4);
    interimVal = _mm_cvtepi8_epi32(inputVal);
    ret = _mm_cvtepi32_ps(interimVal);
    ret = _mm_mul_ps(ret, invScalar);
    _mm_storeu_ps(outputVectorPtr, ret);
    outputVectorPtr += 4;

    inputVectorPtr += 16;
  }

  number = sixteenthPoints * 16;
  for(; number < num_points; number++){
    outputVector[number] = (float)(inputVector[number]) * iScalar;
  }
}
#endif /* LV_HAVE_SSE4_1 */

#ifdef LV_HAVE_GENERIC

static inline void
volk_8i_s32f_convert_32f_generic(float* outputVector, const int8_t* inputVector,
                                 const float scalar, unsigned int num_points)
{
  float* outputVectorPtr = outputVector;
  const int8_t* inputVectorPtr = inputVector;
  unsigned int number = 0;
  const float iScalar = 1.0 / scalar;

  for(number = 0; number < num_points; number++){
    *outputVectorPtr++ = ((float)(*inputVectorPtr++)) * iScalar;
  }
}
#endif /* LV_HAVE_GENERIC */



#endif /* INCLUDED_VOLK_8s_CONVERT_32f_UNALIGNED8_H */
#ifndef INCLUDED_volk_8i_s32f_convert_32f_a_H
#define INCLUDED_volk_8i_s32f_convert_32f_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void
volk_8i_s32f_convert_32f_a_sse4_1(float* outputVector, const int8_t* inputVector,
                                  const float scalar, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int sixteenthPoints = num_points / 16;

  float* outputVectorPtr = outputVector;
  const float iScalar = 1.0 / scalar;
  __m128 invScalar = _mm_set_ps1(iScalar);
  const int8_t* inputVectorPtr = inputVector;
  __m128 ret;
  __m128i inputVal;
  __m128i interimVal;

  for(;number < sixteenthPoints; number++){
    inputVal = _mm_load_si128((__m128i*)inputVectorPtr);

    interimVal = _mm_cvtepi8_epi32(inputVal);
    ret = _mm_cvtepi32_ps(interimVal);
    ret = _mm_mul_ps(ret, invScalar);
    _mm_store_ps(outputVectorPtr, ret);
    outputVectorPtr += 4;

    inputVal = _mm_srli_si128(inputVal, 4);
    interimVal = _mm_cvtepi8_epi32(inputVal);
    ret = _mm_cvtepi32_ps(interimVal);
    ret = _mm_mul_ps(ret, invScalar);
    _mm_store_ps(outputVectorPtr, ret);
    outputVectorPtr += 4;

    inputVal = _mm_srli_si128(inputVal, 4);
    interimVal = _mm_cvtepi8_epi32(inputVal);
    ret = _mm_cvtepi32_ps(interimVal);
    ret = _mm_mul_ps(ret, invScalar);
    _mm_store_ps(outputVectorPtr, ret);
    outputVectorPtr += 4;

    inputVal = _mm_srli_si128(inputVal, 4);
    interimVal = _mm_cvtepi8_epi32(inputVal);
    ret = _mm_cvtepi32_ps(interimVal);
    ret = _mm_mul_ps(ret, invScalar);
    _mm_store_ps(outputVectorPtr, ret);
    outputVectorPtr += 4;

    inputVectorPtr += 16;
  }

  number = sixteenthPoints * 16;
  for(; number < num_points; number++){
    outputVector[number] = (float)(inputVector[number]) * iScalar;
  }
}
#endif /* LV_HAVE_SSE4_1 */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void
volk_8i_s32f_convert_32f_neon(float* outputVector, const int8_t* inputVector,
                              const float scalar, unsigned int num_points)
{
  float* outputVectorPtr = outputVector;
  const int8_t* inputVectorPtr = inputVector;

  const float iScalar = 1.0 / scalar;
  const float32x4_t qiScalar = vdupq_n_f32(iScalar);

  int8x8x2_t inputVal;
  float32x4x2_t outputFloat;
  int16x8_t tmp;
  
  unsigned int number = 0;
  const unsigned int sixteenthPoints = num_points / 16;
  for(;number < sixteenthPoints; number++){
      __VOLK_PREFETCH(inputVectorPtr+16);

	  inputVal = vld2_s8(inputVectorPtr);
	  inputVal = vzip_s8(inputVal.val[0], inputVal.val[1]);
	  inputVectorPtr += 16;

      tmp = vmovl_s8(inputVal.val[0]);

      outputFloat.val[0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(tmp)));
      outputFloat.val[0] = vmulq_f32(outputFloat.val[0], qiScalar);
      vst1q_f32(outputVectorPtr, outputFloat.val[0]);
      outputVectorPtr += 4;

      outputFloat.val[1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(tmp)));
      outputFloat.val[1] = vmulq_f32(outputFloat.val[1], qiScalar);
      vst1q_f32(outputVectorPtr, outputFloat.val[1]);
      outputVectorPtr += 4;

      tmp = vmovl_s8(inputVal.val[1]);

      outputFloat.val[0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(tmp)));
      outputFloat.val[0] = vmulq_f32(outputFloat.val[0], qiScalar);
      vst1q_f32(outputVectorPtr, outputFloat.val[0]);
      outputVectorPtr += 4;

      outputFloat.val[1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(tmp)));
      outputFloat.val[1] = vmulq_f32(outputFloat.val[1], qiScalar);
      vst1q_f32(outputVectorPtr, outputFloat.val[1]);
      outputVectorPtr += 4;
  }
  for(number = sixteenthPoints * 16; number < num_points; number++){
      *outputVectorPtr++ = ((float)(*inputVectorPtr++)) * iScalar;
  }
}

#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_GENERIC

static inline void
volk_8i_s32f_convert_32f_a_generic(float* outputVector, const int8_t* inputVector,
                                   const float scalar, unsigned int num_points)
{
  float* outputVectorPtr = outputVector;
  const int8_t* inputVectorPtr = inputVector;
  unsigned int number = 0;
  const float iScalar = 1.0 / scalar;

  for(number = 0; number < num_points; number++){
    *outputVectorPtr++ = ((float)(*inputVectorPtr++)) * iScalar;
  }
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_ORC
extern void
volk_8i_s32f_convert_32f_a_orc_impl(float* outputVector, const int8_t* inputVector,
                                    const float scalar, unsigned int num_points);

static inline void
volk_8i_s32f_convert_32f_u_orc(float* outputVector, const int8_t* inputVector,
                               const float scalar, unsigned int num_points)
{
  float invscalar = 1.0 / scalar;
  volk_8i_s32f_convert_32f_a_orc_impl(outputVector, inputVector, invscalar, num_points);
}
#endif /* LV_HAVE_ORC */



#endif /* INCLUDED_VOLK_8s_CONVERT_32f_ALIGNED8_H */
