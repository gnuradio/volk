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
 * \page volk_32f_x2_add_32f
 *
 * \b Overview
 *
 * Adds two vectors together element by element:
 *
 * c[i] = a[i] + b[i]
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_x2_add_32f(float* cVector, const float* aVector, const float* bVector, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li aVector: First vector of input points.
 * \li bVector: Second vector of input points.
 * \li num_points: The number of values in both input vector.
 *
 * \b Outputs
 * \li cVector: The output vector.
 *
 * \b Example
 *
 * The follow example adds the increasing and decreasing vectors such that the result of every summation pair is 10
 *
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   float* increasing = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* decreasing = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* out = (float*)volk_malloc(sizeof(float)*N, alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       increasing[ii] = (float)ii;
 *       decreasing[ii] = 10.f - (float)ii;
 *   }
 *
 *   volk_32f_x2_add_32f(out, increasing, decreasing, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("out[%u] = %1.2f\n", ii, out[ii]);
 *   }
 *
 *   volk_free(increasing);
 *   volk_free(decreasing);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_x2_add_32f_u_H
#define INCLUDED_volk_32f_x2_add_32f_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void
volk_32f_x2_add_32f_u_sse(float* cVector, const float* aVector,
                          const float* bVector, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int quarterPoints = num_points / 4;

  float* cPtr = cVector;
  const float* aPtr = aVector;
  const float* bPtr=  bVector;

  __m128 aVal, bVal, cVal;
  for(;number < quarterPoints; number++){

    aVal = _mm_loadu_ps(aPtr);
    bVal = _mm_loadu_ps(bPtr);

    cVal = _mm_add_ps(aVal, bVal);

    _mm_storeu_ps(cPtr,cVal); // Store the results back into the C container

    aPtr += 4;
    bPtr += 4;
    cPtr += 4;
  }

  number = quarterPoints * 4;
  for(;number < num_points; number++){
    *cPtr++ = (*aPtr++) + (*bPtr++);
  }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_x2_add_32f_generic(float* cVector, const float* aVector,
                            const float* bVector, unsigned int num_points)
{
  float* cPtr = cVector;
  const float* aPtr = aVector;
  const float* bPtr=  bVector;
  unsigned int number = 0;

  for(number = 0; number < num_points; number++){
    *cPtr++ = (*aPtr++) + (*bPtr++);
  }
}
#endif /* LV_HAVE_GENERIC */


#endif /* INCLUDED_volk_32f_x2_add_32f_u_H */
#ifndef INCLUDED_volk_32f_x2_add_32f_a_H
#define INCLUDED_volk_32f_x2_add_32f_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void
volk_32f_x2_add_32f_a_sse(float* cVector, const float* aVector, const float* bVector, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int quarterPoints = num_points / 4;

  float* cPtr = cVector;
  const float* aPtr = aVector;
  const float* bPtr=  bVector;

  __m128 aVal, bVal, cVal;
  for(;number < quarterPoints; number++){
    aVal = _mm_load_ps(aPtr);
    bVal = _mm_load_ps(bPtr);

    cVal = _mm_add_ps(aVal, bVal);

    _mm_store_ps(cPtr,cVal); // Store the results back into the C container

    aPtr += 4;
    bPtr += 4;
    cPtr += 4;
  }

  number = quarterPoints * 4;
  for(;number < num_points; number++){
    *cPtr++ = (*aPtr++) + (*bPtr++);
  }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void
volk_32f_x2_add_32f_u_neon(float* cVector, const float* aVector,
                           const float* bVector, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int quarterPoints = num_points / 4;

  float* cPtr = cVector;
  const float* aPtr = aVector;
  const float* bPtr=  bVector;
  float32x4_t aVal, bVal, cVal;
  for(number=0; number < quarterPoints; number++){
    // Load in to NEON registers
    aVal = vld1q_f32(aPtr);
    bVal = vld1q_f32(bPtr);
    __builtin_prefetch(aPtr+4);
    __builtin_prefetch(bPtr+4);

    // vector add
    cVal = vaddq_f32(aVal, bVal);
    // Store the results back into the C container
    vst1q_f32(cPtr,cVal);

    aPtr += 4; // q uses quadwords, 4 floats per vadd
    bPtr += 4;
    cPtr += 4;
  }

  number = quarterPoints * 4; // should be = num_points
  for(;number < num_points; number++){
    *cPtr++ = (*aPtr++) + (*bPtr++);
  }
}

#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEON
extern void volk_32f_x2_add_32f_neonasm(float* cVector, const float* aVector, const float* bVector, unsigned int num_points);
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEON
extern void volk_32f_x2_add_32f_neonpipeline(float* cVector, const float* aVector, const float* bVector, unsigned int num_points);
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_x2_add_32f_a_generic(float* cVector, const float* aVector,
                              const float* bVector, unsigned int num_points)
{
  float* cPtr = cVector;
  const float* aPtr = aVector;
  const float* bPtr=  bVector;
  unsigned int number = 0;

  for(number = 0; number < num_points; number++){
    *cPtr++ = (*aPtr++) + (*bPtr++);
  }
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_ORC

extern void
volk_32f_x2_add_32f_a_orc_impl(float* cVector, const float* aVector,
                               const float* bVector, unsigned int num_points);

static inline void
volk_32f_x2_add_32f_u_orc(float* cVector, const float* aVector,
                          const float* bVector, unsigned int num_points){
  volk_32f_x2_add_32f_a_orc_impl(cVector, aVector, bVector, num_points);
}

#endif /* LV_HAVE_ORC */


#endif /* INCLUDED_volk_32f_x2_add_32f_a_H */
