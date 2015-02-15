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
 * \page volk_32fc_s32f_atan2_32f
 *
 * \b Overview
 *
 * Computes the arctan for each value in a complex vector and applies
 * a normalization factor.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_s32f_atan2_32f(float* outputVector, const lv_32fc_t* complexVector, const float normalizeFactor, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li inputVector: The byte-aligned input vector containing interleaved IQ data (I = cos, Q = sin).
 * \li normalizeFactor: The atan results are divided by this normalization factor.
 * \li num_points: The number of complex values in \p inputVector.
 *
 * \b Outputs
 * \li outputVector: The vector where the results will be stored.
 *
 * \b Example
 * Calculate the arctangent of points around the unit circle.
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   lv_32fc_t* in  = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *   float* out = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float scale = 1.f; // we want unit circle
 *
 *   for(unsigned int ii = 0; ii < N/2; ++ii){
 *       // Generate points around the unit circle
 *       float real = -4.f * ((float)ii / (float)N) + 1.f;
 *       float imag = std::sqrt(1.f - real * real);
 *       in[ii] = lv_cmake(real, imag);
 *       in[ii+N/2] = lv_cmake(-real, -imag);
 *   }
 *
 *   volk_32fc_s32f_atan2_32f(out, in, scale, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("atan2(%1.2f, %1.2f) = %1.2f\n",
 *           lv_cimag(in[ii]), lv_creal(in[ii]), out[ii]);
 *   }
 *
 *   volk_free(in);
 *   volk_free(out);
 * \endcode
 */


#ifndef INCLUDED_volk_32fc_s32f_atan2_32f_a_H
#define INCLUDED_volk_32fc_s32f_atan2_32f_a_H

#include <inttypes.h>
#include <stdio.h>
#include <math.h>

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

#ifdef LV_HAVE_LIB_SIMDMATH
#include <simdmath.h>
#endif /* LV_HAVE_LIB_SIMDMATH */

static inline void volk_32fc_s32f_atan2_32f_a_sse4_1(float* outputVector,  const lv_32fc_t* complexVector, const float normalizeFactor, unsigned int num_points){
  const float* complexVectorPtr = (float*)complexVector;
  float* outPtr = outputVector;

  unsigned int number = 0;
  const float invNormalizeFactor = 1.0 / normalizeFactor;

#ifdef LV_HAVE_LIB_SIMDMATH
  const unsigned int quarterPoints = num_points / 4;
  __m128 testVector = _mm_set_ps1(2*M_PI);
  __m128 correctVector = _mm_set_ps1(M_PI);
  __m128 vNormalizeFactor = _mm_set_ps1(invNormalizeFactor);
  __m128 phase;
  __m128 complex1, complex2, iValue, qValue;
  __m128 keepMask;

  for (; number < quarterPoints; number++) {
    // Load IQ data:
    complex1 = _mm_load_ps(complexVectorPtr);
    complexVectorPtr += 4;
    complex2 = _mm_load_ps(complexVectorPtr);
    complexVectorPtr += 4;
    // Deinterleave IQ data:
    iValue = _mm_shuffle_ps(complex1, complex2, _MM_SHUFFLE(2,0,2,0));
    qValue = _mm_shuffle_ps(complex1, complex2, _MM_SHUFFLE(3,1,3,1));
    // Arctan to get phase:
    phase = atan2f4(qValue, iValue);
    // When Q = 0 and I < 0, atan2f4 sucks and returns 2pi vice pi.
    // Compare to 2pi:
    keepMask = _mm_cmpneq_ps(phase,testVector);
    phase = _mm_blendv_ps(correctVector, phase, keepMask);
    // done with above correction.
    phase = _mm_mul_ps(phase, vNormalizeFactor);
    _mm_store_ps((float*)outPtr, phase);
    outPtr += 4;
  }
  number = quarterPoints * 4;
#endif /* LV_HAVE_SIMDMATH_H */

  for (; number < num_points; number++) {
    const float real = *complexVectorPtr++;
    const float imag = *complexVectorPtr++;
    *outPtr++ = atan2f(imag, real) * invNormalizeFactor;
  }
}
#endif /* LV_HAVE_SSE4_1 */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

#ifdef LV_HAVE_LIB_SIMDMATH
#include <simdmath.h>
#endif /* LV_HAVE_LIB_SIMDMATH */

static inline void volk_32fc_s32f_atan2_32f_a_sse(float* outputVector,  const lv_32fc_t* complexVector, const float normalizeFactor, unsigned int num_points){
  const float* complexVectorPtr = (float*)complexVector;
  float* outPtr = outputVector;

  unsigned int number = 0;
  const float invNormalizeFactor = 1.0 / normalizeFactor;

#ifdef LV_HAVE_LIB_SIMDMATH
  const unsigned int quarterPoints = num_points / 4;
  __m128 testVector = _mm_set_ps1(2*M_PI);
  __m128 correctVector = _mm_set_ps1(M_PI);
  __m128 vNormalizeFactor = _mm_set_ps1(invNormalizeFactor);
  __m128 phase;
  __m128 complex1, complex2, iValue, qValue;
  __m128 mask;
  __m128 keepMask;

  for (; number < quarterPoints; number++) {
    // Load IQ data:
    complex1 = _mm_load_ps(complexVectorPtr);
    complexVectorPtr += 4;
    complex2 = _mm_load_ps(complexVectorPtr);
    complexVectorPtr += 4;
    // Deinterleave IQ data:
    iValue = _mm_shuffle_ps(complex1, complex2, _MM_SHUFFLE(2,0,2,0));
    qValue = _mm_shuffle_ps(complex1, complex2, _MM_SHUFFLE(3,1,3,1));
    // Arctan to get phase:
    phase = atan2f4(qValue, iValue);
    // When Q = 0 and I < 0, atan2f4 sucks and returns 2pi vice pi.
    // Compare to 2pi:
    keepMask = _mm_cmpneq_ps(phase,testVector);
    phase = _mm_and_ps(phase, keepMask);
    mask = _mm_andnot_ps(keepMask, correctVector);
    phase = _mm_or_ps(phase, mask);
    // done with above correction.
    phase = _mm_mul_ps(phase, vNormalizeFactor);
    _mm_store_ps((float*)outPtr, phase);
    outPtr += 4;
  }
  number = quarterPoints * 4;
#endif /* LV_HAVE_SIMDMATH_H */

  for (; number < num_points; number++) {
    const float real = *complexVectorPtr++;
    const float imag = *complexVectorPtr++;
    *outPtr++ = atan2f(imag, real) * invNormalizeFactor;
  }
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_GENERIC

static inline void volk_32fc_s32f_atan2_32f_generic(float* outputVector, const lv_32fc_t* inputVector, const float normalizeFactor, unsigned int num_points){
  float* outPtr = outputVector;
  const float* inPtr = (float*)inputVector;
  const float invNormalizeFactor = 1.0 / normalizeFactor;
  unsigned int number;
  for ( number = 0; number < num_points; number++) {
    const float real = *inPtr++;
    const float imag = *inPtr++;
    *outPtr++ = atan2f(imag, real) * invNormalizeFactor;
  }
}
#endif /* LV_HAVE_GENERIC */




#endif /* INCLUDED_volk_32fc_s32f_atan2_32f_a_H */
