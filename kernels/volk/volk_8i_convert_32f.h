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
 * \page volk_8i_convert_32f
 *
 * \b Overview
 *
 * Convert the input vector of 8-bit signed ints ranging [-128,127] to
 * a vector of floats ranging [-1,0.9921875].
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_8i_convert_32f(float* outputVector, const int8_t* inputVector, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li inputVector: The input vector of 8-bit signed ints.
 * \li num_points: The number of values.
 *
 * \b Outputs
 * \li outputVector: The output 32-bit floats.
 *
 * \b Example
 * \code
 * int N = 10000;
 *
 * <FIXME>
 *
 * volk_8i_convert_32f();
 *
 * volk_free(x);
 * \endcode
 */

#ifndef INCLUDED_volk_8i_convert_32f_u_H
#define INCLUDED_volk_8i_convert_32f_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void
volk_8i_convert_32f_u_sse4_1(float* outputVector, const int8_t* inputVector,
                             unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int sixteenthPoints = num_points / 16;

  float* outputVectorPtr = outputVector;
  const float multiplier = 1.0 / 128;
  __m128 mulScalar = _mm_set_ps1( multiplier );
  const int8_t* inputVectorPtr = inputVector;
  __m128 ret;
  __m128i inputVal;
  __m128i interimVal;

  for(;number < sixteenthPoints; number++){
    inputVal = _mm_loadu_si128((__m128i*)inputVectorPtr);

    interimVal = _mm_cvtepi8_epi32(inputVal);
    ret = _mm_cvtepi32_ps(interimVal);
    ret = _mm_mul_ps(ret, mulScalar);
    _mm_storeu_ps(outputVectorPtr, ret);
    outputVectorPtr += 4;

    inputVal = _mm_srli_si128(inputVal, 4);
    interimVal = _mm_cvtepi8_epi32(inputVal);
    ret = _mm_cvtepi32_ps(interimVal);
    ret = _mm_mul_ps(ret, mulScalar);
    _mm_storeu_ps(outputVectorPtr, ret);
    outputVectorPtr += 4;

    inputVal = _mm_srli_si128(inputVal, 4);
    interimVal = _mm_cvtepi8_epi32(inputVal);
    ret = _mm_cvtepi32_ps(interimVal);
    ret = _mm_mul_ps(ret, mulScalar);
    _mm_storeu_ps(outputVectorPtr, ret);
    outputVectorPtr += 4;

    inputVal = _mm_srli_si128(inputVal, 4);
    interimVal = _mm_cvtepi8_epi32(inputVal);
    ret = _mm_cvtepi32_ps(interimVal);
    ret = _mm_mul_ps(ret, mulScalar);
    _mm_storeu_ps(outputVectorPtr, ret);
    outputVectorPtr += 4;

    inputVectorPtr += 16;
  }

  number = sixteenthPoints * 16;
  for(; number < num_points; number++){
    outputVector[number] = (float)(inputVector[number]) * multiplier;
  }
}
#endif /* LV_HAVE_SSE4_1 */

#ifdef LV_HAVE_GENERIC

static inline void
volk_8i_convert_32f_u_generic(float* outputVector, const int8_t* inputVector,
                              unsigned int num_points)
{
  float* outputVectorPtr = outputVector;
  const int8_t* inputVectorPtr = inputVector;
  unsigned int number = 0;
  const float multiplier = 1.0 / 128;

  for(number = 0; number < num_points; number++){
    *outputVectorPtr++ = (float)(*inputVectorPtr++) * multiplier;
  }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_GENERIC

static inline void
volk_8i_convert_32f_u_generic_lut8(float* outputVector, const int8_t* inputVector,
                                   unsigned int num_points)
{
  /* generate lookup table */
  #define gen(x) (((int8_t)(x)) / 128.0f)
  #define gen16(x) \
    gen(x|0x0), gen(x|0x1), gen(x|0x2), gen(x|0x3), gen(x|0x4), gen(x|0x5), gen(x|0x6), gen(x|0x7),\
    gen(x|0x8), gen(x|0x9), gen(x|0xa), gen(x|0xb), gen(x|0xc), gen(x|0xd), gen(x|0xe), gen(x|0xf)
  static const float lut[256] = {
    gen16(0x00), gen16(0x10), gen16(0x20), gen16(0x30),
    gen16(0x40), gen16(0x50), gen16(0x60), gen16(0x70),
    gen16(0x80), gen16(0x90), gen16(0xa0), gen16(0xb0),
    gen16(0xc0), gen16(0xd0), gen16(0xe0), gen16(0xf0)
  };
  #undef gen
  #undef gen16

  /* index lut to convert */
  float* outputVectorPtr = outputVector;
  const int8_t* inputVectorPtr = inputVector;
  unsigned int number = 0;

  for(number = 0; number < num_points; number++){
    *outputVectorPtr++ = lut[(uint8_t)(*inputVectorPtr++)];
  }
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_ORC
extern void
volk_8i_convert_32f_a_orc_impl(float* outputVector, const int8_t* inputVector,
                               unsigned int num_points);

static inline void
volk_8i_convert_32f_u_orc(float* outputVector, const int8_t* inputVector,
                          unsigned int num_points)
{
  volk_8i_convert_32f_a_orc_impl(outputVector, inputVector, num_points);
}
#endif /* LV_HAVE_ORC */


#endif /* INCLUDED_VOLK_8i_CONVERT_32f_u_H */
#ifndef INCLUDED_volk_8i_convert_32f_a_H
#define INCLUDED_volk_8i_convert_32f_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void
volk_8i_convert_32f_a_sse4_1(float* outputVector, const int8_t* inputVector,
                             unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int sixteenthPoints = num_points / 16;

  float* outputVectorPtr = outputVector;
  const float multiplier = 1.0 / 128;
  __m128 mulScalar = _mm_set_ps1(multiplier);
  const int8_t* inputVectorPtr = inputVector;
  __m128 ret;
  __m128i inputVal;
  __m128i interimVal;

  for(;number < sixteenthPoints; number++){
    inputVal = _mm_load_si128((__m128i*)inputVectorPtr);

    interimVal = _mm_cvtepi8_epi32(inputVal);
    ret = _mm_cvtepi32_ps(interimVal);
    ret = _mm_mul_ps(ret, mulScalar);
    _mm_store_ps(outputVectorPtr, ret);
    outputVectorPtr += 4;

    inputVal = _mm_srli_si128(inputVal, 4);
    interimVal = _mm_cvtepi8_epi32(inputVal);
    ret = _mm_cvtepi32_ps(interimVal);
    ret = _mm_mul_ps(ret, mulScalar);
    _mm_store_ps(outputVectorPtr, ret);
    outputVectorPtr += 4;

    inputVal = _mm_srli_si128(inputVal, 4);
    interimVal = _mm_cvtepi8_epi32(inputVal);
    ret = _mm_cvtepi32_ps(interimVal);
    ret = _mm_mul_ps(ret, mulScalar);
    _mm_store_ps(outputVectorPtr, ret);
    outputVectorPtr += 4;

    inputVal = _mm_srli_si128(inputVal, 4);
    interimVal = _mm_cvtepi8_epi32(inputVal);
    ret = _mm_cvtepi32_ps(interimVal);
    ret = _mm_mul_ps(ret, mulScalar);
    _mm_store_ps(outputVectorPtr, ret);
    outputVectorPtr += 4;

    inputVectorPtr += 16;
  }

  number = sixteenthPoints * 16;
  for(; number < num_points; number++){
    outputVector[number] = (float)(inputVector[number]) * multiplier;
  }
}
#endif /* LV_HAVE_SSE4_1 */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void
volk_8i_convert_32f_neon(float* outputVector, const int8_t* inputVector,
                         unsigned int num_points)
{
  float* outputVectorPtr = outputVector;
  const int8_t* inputVectorPtr = inputVector;

  const float multiplier = 1.0 / 128;

  const float32x4_t qmulScalar = vdupq_n_f32(multiplier);

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
      outputFloat.val[0] = vmulq_f32(outputFloat.val[0], qmulScalar);
      vst1q_f32(outputVectorPtr, outputFloat.val[0]);
      outputVectorPtr += 4;

      outputFloat.val[1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(tmp)));
      outputFloat.val[1] = vmulq_f32(outputFloat.val[1], qmulScalar);
      vst1q_f32(outputVectorPtr, outputFloat.val[1]);
      outputVectorPtr += 4;

      tmp = vmovl_s8(inputVal.val[1]);

      outputFloat.val[0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(tmp)));
      outputFloat.val[0] = vmulq_f32(outputFloat.val[0], qmulScalar);
      vst1q_f32(outputVectorPtr, outputFloat.val[0]);
      outputVectorPtr += 4;

      outputFloat.val[1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(tmp)));
      outputFloat.val[1] = vmulq_f32(outputFloat.val[1], qmulScalar);
      vst1q_f32(outputVectorPtr, outputFloat.val[1]);
      outputVectorPtr += 4;
  }
  for(number = sixteenthPoints * 16; number < num_points; number++){
      *outputVectorPtr++ = ((float)(*inputVectorPtr++) + offset) * multiplier;
  }
}

#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_GENERIC

static inline void
volk_8i_convert_32f_a_generic(float* outputVector, const int8_t* inputVector,
                              unsigned int num_points)
{
  volk_8i_convert_32f_u_generic(outputVector, inputVector, num_points);
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_GENERIC

static inline void
volk_8i_convert_32f_a_generic_lut8(float* outputVector, const int8_t* inputVector,
                                   unsigned int num_points)
{
  volk_8i_convert_32f_u_generic_lut8(outputVector, inputVector, num_points);
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_ORC
extern void
volk_8i_convert_32f_a_orc_impl(float* outputVector, const int8_t* inputVector,
                               unsigned int num_points);

static inline void
volk_8i_convert_32f_a_orc(float* outputVector, const int8_t* inputVector,
                          unsigned int num_points)
{
  volk_8i_convert_32f_a_orc_impl(outputVector, inputVector, num_points);
}
#endif /* LV_HAVE_ORC */



#endif /* INCLUDED_VOLK_8i_CONVERT_32f_a_H */
