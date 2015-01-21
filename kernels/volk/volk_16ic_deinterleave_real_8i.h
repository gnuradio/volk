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
 * \page volk_16ic_deinterleave_real_8i
 *
 * \b Overview
 *
 * Deinterleaves the complex 16 bit vector and returns the real
 * (inphase) part of the signal as an 8-bit value.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_16ic_deinterleave_real_8i(int8_t* iBuffer, const lv_16sc_t* complexVector, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li complexVector: The complex input vector.
 * \li num_points: The number of complex data values to be deinterleaved.
 *
 * \b Outputs
 * \li iBuffer: The I buffer output data with 8-bit precision.
 *
 * \b Example
 * \code
 * int N = 10000;
 *
 * volk_16ic_deinterleave_real_8i();
 *
 * volk_free(x);
 * volk_free(t);
 * \endcode
 */

#ifndef INCLUDED_volk_16ic_deinterleave_real_8i_a_H
#define INCLUDED_volk_16ic_deinterleave_real_8i_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_SSSE3
#include <tmmintrin.h>

static inline void
volk_16ic_deinterleave_real_8i_a_ssse3(int8_t* iBuffer, const lv_16sc_t* complexVector, unsigned int num_points)
{
  unsigned int number = 0;
  const int8_t* complexVectorPtr = (int8_t*)complexVector;
  int8_t* iBufferPtr = iBuffer;
  __m128i iMoveMask1 = _mm_set_epi8(0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 13, 12, 9, 8, 5, 4, 1, 0);
  __m128i iMoveMask2 = _mm_set_epi8(13, 12, 9, 8, 5, 4, 1, 0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
  __m128i complexVal1, complexVal2, complexVal3, complexVal4, iOutputVal;

  unsigned int sixteenthPoints = num_points / 16;

  for(number = 0; number < sixteenthPoints; number++){
    complexVal1 = _mm_load_si128((__m128i*)complexVectorPtr);  complexVectorPtr += 16;
    complexVal2 = _mm_load_si128((__m128i*)complexVectorPtr);  complexVectorPtr += 16;

    complexVal3 = _mm_load_si128((__m128i*)complexVectorPtr);  complexVectorPtr += 16;
    complexVal4 = _mm_load_si128((__m128i*)complexVectorPtr);  complexVectorPtr += 16;

    complexVal1 = _mm_shuffle_epi8(complexVal1, iMoveMask1);
    complexVal2 = _mm_shuffle_epi8(complexVal2, iMoveMask2);

    complexVal1 = _mm_or_si128(complexVal1, complexVal2);

    complexVal3 = _mm_shuffle_epi8(complexVal3, iMoveMask1);
    complexVal4 = _mm_shuffle_epi8(complexVal4, iMoveMask2);

    complexVal3 = _mm_or_si128(complexVal3, complexVal4);


    complexVal1 = _mm_srai_epi16(complexVal1, 8);
    complexVal3 = _mm_srai_epi16(complexVal3, 8);

    iOutputVal = _mm_packs_epi16(complexVal1, complexVal3);

    _mm_store_si128((__m128i*)iBufferPtr, iOutputVal);

    iBufferPtr += 16;
  }

  number = sixteenthPoints * 16;
  int16_t* int16ComplexVectorPtr = (int16_t*)complexVectorPtr;
  for(; number < num_points; number++){
    *iBufferPtr++ = ((int8_t)(*int16ComplexVectorPtr++ >> 8));
    int16ComplexVectorPtr++;
  }
}
#endif /* LV_HAVE_SSSE3 */

#ifdef LV_HAVE_GENERIC

static inline void
volk_16ic_deinterleave_real_8i_generic(int8_t* iBuffer, const lv_16sc_t* complexVector, unsigned int num_points)
{
  unsigned int number = 0;
  int16_t* complexVectorPtr = (int16_t*)complexVector;
  int8_t* iBufferPtr = iBuffer;
  for(number = 0; number < num_points; number++){
    *iBufferPtr++ = ((int8_t)(*complexVectorPtr++ >> 8));
    complexVectorPtr++;
  }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void
volk_16ic_deinterleave_real_8i_neon(int8_t* iBuffer, const lv_16sc_t* complexVector, unsigned int num_points)
{
  const int16_t* complexVectorPtr = (const int16_t*)complexVector;
  int8_t* iBufferPtr = iBuffer;
  unsigned int eighth_points = num_points / 8;
  unsigned int number;

  int16x8x2_t complexInput;
  int8x8_t realOutput;
  for(number = 0; number < eighth_points; number++){
    complexInput = vld2q_s16(complexVectorPtr);
    realOutput = vshrn_n_s16(complexInput.val[0], 8);
    vst1_s8(iBufferPtr, realOutput);
    complexVectorPtr += 16;
    iBufferPtr += 8;
  }

  for(number = eighth_points*8; number < num_points; number++){
    *iBufferPtr++ = ((int8_t)(*complexVectorPtr++ >> 8));
    complexVectorPtr++;
  }
}
#endif

#ifdef LV_HAVE_ORC

extern void
volk_16ic_deinterleave_real_8i_a_orc_impl(int8_t* iBuffer, const lv_16sc_t* complexVector, unsigned int num_points);

static inline void
volk_16ic_deinterleave_real_8i_u_orc(int8_t* iBuffer, const lv_16sc_t* complexVector, unsigned int num_points)
{
    volk_16ic_deinterleave_real_8i_a_orc_impl(iBuffer, complexVector, num_points);
}
#endif /* LV_HAVE_ORC */


#endif /* INCLUDED_volk_16ic_deinterleave_real_8i_a_H */
