/* -*- c++ -*- */
/*
 * Copyright 2018 Free Software Foundation, Inc.
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
 * \page volk_32fc_x2_add_32fcc
 *
 * \b Overview
 *
 * Adds two vectors together element by element:
 *
 * c[i] = a[i] + b[i]
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_x2_add_32fc(lv_32fc_t* cVector, const lv_32fc_t* aVector, const lv_32fc_t* bVector, unsigned int num_points)
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
 *   lv_32fc_t* increasing = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *   lv_32fc_t* decreasing = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *   lv_32fc_t* out = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       increasing[ii] = (lv_32fc_t)ii;
 *       decreasing[ii] = 10.f - (lv_32fc_t)ii;
 *   }
 *
 *   volk_32fc_x2_add_32fc(out, increasing, decreasing, N);
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

#ifndef INCLUDED_volk_32fc_x2_add_32fc_u_H
#define INCLUDED_volk_32fc_x2_add_32fc_u_H

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32fc_x2_add_32fc_u_avx(lv_32fc_t* cVector, const lv_32fc_t* aVector,
                          const lv_32fc_t* bVector, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int quarterPoints = num_points / 4;

  lv_32fc_t* cPtr = cVector;
  const lv_32fc_t* aPtr = aVector;
  const lv_32fc_t* bPtr=  bVector;

  __m256 aVal, bVal, cVal;
  for(;number < quarterPoints; number++){

    aVal = _mm256_loadu_ps((float *) aPtr);
    bVal = _mm256_loadu_ps((float *) bPtr);

    cVal = _mm256_add_ps(aVal, bVal);

    _mm256_storeu_ps((float *) cPtr,cVal); // Store the results back into the C container

    aPtr += 4;
    bPtr += 4;
    cPtr += 4;
  }

  number = quarterPoints * 4;
  for(;number < num_points; number++){
    *cPtr++ = (*aPtr++) + (*bPtr++);
  }
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32fc_x2_add_32fc_a_avx(lv_32fc_t* cVector, const lv_32fc_t* aVector,
                          const lv_32fc_t* bVector, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int quarterPoints = num_points / 4;

  lv_32fc_t* cPtr = cVector;
  const lv_32fc_t* aPtr = aVector;
  const lv_32fc_t* bPtr=  bVector;

  __m256 aVal, bVal, cVal;
  for(;number < quarterPoints; number++){

    aVal = _mm256_load_ps((float*) aPtr);
    bVal = _mm256_load_ps((float*) bPtr);

    cVal = _mm256_add_ps(aVal, bVal);

    _mm256_store_ps((float*) cPtr,cVal); // Store the results back into the C container

    aPtr += 4;
    bPtr += 4;
    cPtr += 4;
  }

  number = quarterPoints * 4;
  for(;number < num_points; number++){
    *cPtr++ = (*aPtr++) + (*bPtr++);
  }
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void
volk_32fc_x2_add_32fc_u_sse(lv_32fc_t* cVector, const lv_32fc_t* aVector,
                          const lv_32fc_t* bVector, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int halfPoints = num_points / 2;

  lv_32fc_t* cPtr = cVector;
  const lv_32fc_t* aPtr = aVector;
  const lv_32fc_t* bPtr=  bVector;

  __m128 aVal, bVal, cVal;
  for(;number < halfPoints; number++){

    aVal = _mm_loadu_ps((float *) aPtr);
    bVal = _mm_loadu_ps((float *) bPtr);

    cVal = _mm_add_ps(aVal, bVal);

    _mm_storeu_ps((float*) cPtr, cVal); // Store the results back into the C container

    aPtr += 2;
    bPtr += 2;
    cPtr += 2;
  }

  number = halfPoints * 2;
  for(;number < num_points; number++){
    *cPtr++ = (*aPtr++) + (*bPtr++);
  }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_GENERIC

static inline void
volk_32fc_x2_add_32fc_generic(lv_32fc_t* cVector, const lv_32fc_t* aVector,
                            const lv_32fc_t* bVector, unsigned int num_points)
{
  lv_32fc_t* cPtr = cVector;
  const lv_32fc_t* aPtr = aVector;
  const lv_32fc_t* bPtr=  bVector;
  unsigned int number = 0;

  for(number = 0; number < num_points; number++){
    *cPtr++ = (*aPtr++) + (*bPtr++);
  }
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void
volk_32fc_x2_add_32fc_a_sse(lv_32fc_t* cVector, const lv_32fc_t* aVector, const lv_32fc_t* bVector, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int halfPoints = num_points / 2;

  lv_32fc_t* cPtr = cVector;
  const lv_32fc_t* aPtr = aVector;
  const lv_32fc_t* bPtr=  bVector;

  __m128 aVal, bVal, cVal;
  for(;number < halfPoints; number++){
    aVal = _mm_load_ps((float *) aPtr);
    bVal = _mm_load_ps((float *) bPtr);

    cVal = _mm_add_ps(aVal, bVal);

    _mm_store_ps((float *) cPtr,cVal); // Store the results back into the C container

    aPtr += 2;
    bPtr += 2;
    cPtr += 2;
  }

  number = halfPoints * 2;
  for(;number < num_points; number++){
    *cPtr++ = (*aPtr++) + (*bPtr++);
  }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void
volk_32fc_x2_add_32fc_u_neon(lv_32fc_t* cVector, const lv_32fc_t* aVector,
                           const lv_32fc_t* bVector, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int halfPoints = num_points / 2;

  lv_32fc_t* cPtr = cVector;
  const lv_32fc_t* aPtr = aVector;
  const lv_32fc_t* bPtr=  bVector;
  float32x4_t aVal, bVal, cVal;
  for(number=0; number < halfPoints; number++){
    // Load in to NEON registers
    aVal = vld1q_f32((const float32_t*)(aPtr));
    bVal = vld1q_f32((const float32_t*)(bPtr));
    __VOLK_PREFETCH(aPtr+2);
    __VOLK_PREFETCH(bPtr+2);

    // vector add
    cVal = vaddq_f32(aVal, bVal);
    // Store the results back into the C container
    vst1q_f32((float*)(cPtr),cVal);

    aPtr += 2; // q uses quadwords, 4 lv_32fc_ts per vadd
    bPtr += 2;
    cPtr += 2;
  }

  number = halfPoints * 2; // should be = num_points
  for(;number < num_points; number++){
    *cPtr++ = (*aPtr++) + (*bPtr++);
  }
}

#endif /* LV_HAVE_NEON */


#endif /* INCLUDED_volk_32fc_x2_add_32fc_a_H */
