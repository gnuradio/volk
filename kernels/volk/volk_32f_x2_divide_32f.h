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
 * \page volk_32f_x2_divide_32f
 *
 * \b Overview
 *
 * Divides aVector by bVector to produce cVector:
 *
 * c[i] = a[i] / b[i]
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_x2_divide_32f(float* cVector, const float* aVector, const float* bVector, unsigned int num_points)
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
 * Divide an increasing vector by a decreasing vector
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
 *   volk_32f_x2_divide_32f(out, increasing, decreasing, N);
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

#ifndef INCLUDED_volk_32f_x2_divide_32f_a_H
#define INCLUDED_volk_32f_x2_divide_32f_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void
volk_32f_x2_divide_32f_a_sse(float* cVector, const float* aVector,
                             const float* bVector, unsigned int num_points)
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

    cVal = _mm_div_ps(aVal, bVal);

    _mm_store_ps(cPtr,cVal); // Store the results back into the C container

    aPtr += 4;
    bPtr += 4;
    cPtr += 4;
  }

  number = quarterPoints * 4;
  for(;number < num_points; number++){
    *cPtr++ = (*aPtr++) / (*bPtr++);
  }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32f_x2_divide_32f_a_avx(float* cVector, const float* aVector,
                             const float* bVector, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int eigthPoints = num_points / 8;

  float* cPtr = cVector;
  const float* aPtr = aVector;
  const float* bPtr=  bVector;

  __m256 aVal, bVal, cVal;
  for(;number < eigthPoints; number++){
    aVal = _mm256_load_ps(aPtr);
    bVal = _mm256_load_ps(bPtr);

    cVal = _mm256_div_ps(aVal, bVal);

    _mm256_store_ps(cPtr,cVal); // Store the results back into the C container

    aPtr += 8;
    bPtr += 8;
    cPtr += 8;
  }

  number = eigthPoints * 8;
  for(;number < num_points; number++){
    *cPtr++ = (*aPtr++) / (*bPtr++);
  }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void
volk_32f_x2_divide_32f_neon(float* cVector, const float* aVector,
			    const float* bVector, unsigned int num_points)
{
  float* cPtr = cVector;
  const float* aPtr = aVector;
  const float* bPtr = bVector;

  float32x4x4_t aVal, bVal, bInv, cVal;

  const unsigned int eighthPoints = num_points / 16;
  unsigned int number = 0;
  for(; number < eighthPoints; number++){
    aVal = vld4q_f32(aPtr);
    aPtr += 16;
    bVal = vld4q_f32(bPtr);
    bPtr += 16;

    __VOLK_PREFETCH(aPtr+16);
    __VOLK_PREFETCH(bPtr+16);

    bInv.val[0] = vrecpeq_f32(bVal.val[0]);
    bInv.val[0] = vmulq_f32(bInv.val[0], vrecpsq_f32(bInv.val[0], bVal.val[0]));
    bInv.val[0] = vmulq_f32(bInv.val[0], vrecpsq_f32(bInv.val[0], bVal.val[0]));
    cVal.val[0] = vmulq_f32(aVal.val[0], bInv.val[0]);

    bInv.val[1] = vrecpeq_f32(bVal.val[1]);
    bInv.val[1] = vmulq_f32(bInv.val[1], vrecpsq_f32(bInv.val[1], bVal.val[1]));
    bInv.val[1] = vmulq_f32(bInv.val[1], vrecpsq_f32(bInv.val[1], bVal.val[1]));
    cVal.val[1] = vmulq_f32(aVal.val[1], bInv.val[1]);

    bInv.val[2] = vrecpeq_f32(bVal.val[2]);
    bInv.val[2] = vmulq_f32(bInv.val[2], vrecpsq_f32(bInv.val[2], bVal.val[2]));
    bInv.val[2] = vmulq_f32(bInv.val[2], vrecpsq_f32(bInv.val[2], bVal.val[2]));
    cVal.val[2] = vmulq_f32(aVal.val[2], bInv.val[2]);

    bInv.val[3] = vrecpeq_f32(bVal.val[3]);
    bInv.val[3] = vmulq_f32(bInv.val[3], vrecpsq_f32(bInv.val[3], bVal.val[3]));
    bInv.val[3] = vmulq_f32(bInv.val[3], vrecpsq_f32(bInv.val[3], bVal.val[3]));
    cVal.val[3] = vmulq_f32(aVal.val[3], bInv.val[3]);

    vst4q_f32(cPtr, cVal);
    cPtr += 16;
  }

  for(number = eighthPoints * 16; number < num_points; number++){
    *cPtr++ = (*aPtr++) / (*bPtr++);
  }
}

#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_x2_divide_32f_generic(float* cVector, const float* aVector,
                               const float* bVector, unsigned int num_points)
{
  float* cPtr = cVector;
  const float* aPtr = aVector;
  const float* bPtr=  bVector;
  unsigned int number = 0;

  for(number = 0; number < num_points; number++){
    *cPtr++ = (*aPtr++) / (*bPtr++);
  }
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_ORC

extern void
volk_32f_x2_divide_32f_a_orc_impl(float* cVector, const float* aVector,
                                  const float* bVector, unsigned int num_points);

static inline void
volk_32f_x2_divide_32f_u_orc(float* cVector, const float* aVector,
                             const float* bVector, unsigned int num_points)
{
  volk_32f_x2_divide_32f_a_orc_impl(cVector, aVector, bVector, num_points);
}
#endif /* LV_HAVE_ORC */



#endif /* INCLUDED_volk_32f_x2_divide_32f_a_H */


#ifndef INCLUDED_volk_32f_x2_divide_32f_u_H
#define INCLUDED_volk_32f_x2_divide_32f_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32f_x2_divide_32f_u_avx(float* cVector, const float* aVector,
                             const float* bVector, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int eigthPoints = num_points / 8;

  float* cPtr = cVector;
  const float* aPtr = aVector;
  const float* bPtr=  bVector;

  __m256 aVal, bVal, cVal;
  for(;number < eigthPoints; number++){
    aVal = _mm256_loadu_ps(aPtr);
    bVal = _mm256_loadu_ps(bPtr);

    cVal = _mm256_div_ps(aVal, bVal);

    _mm256_storeu_ps(cPtr,cVal); // Store the results back into the C container

    aPtr += 8;
    bPtr += 8;
    cPtr += 8;
  }

  number = eigthPoints * 8;
  for(;number < num_points; number++){
    *cPtr++ = (*aPtr++) / (*bPtr++);
  }
}
#endif /* LV_HAVE_AVX */

#endif /* INCLUDED_volk_32f_x2_divide_32f_u_H */
