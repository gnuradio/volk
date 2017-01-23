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
 * \page volk_32fc_conjugate_32fc
 *
 * \b Overview
 *
 * Takes the conjugate of a complex vector.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_conjugate_32fc(lv_32fc_t* cVector, const lv_32fc_t* aVector, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li aVector: The input vector of complex floats.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li bVector: The output vector of complex floats.
 *
 * \b Example
 * Generate points around the top half of the unit circle and conjugate them
 * to give bottom half of the unit circle.
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   lv_32fc_t* in  = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *   lv_32fc_t* out = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       float real = 2.f * ((float)ii / (float)N) - 1.f;
 *       float imag = std::sqrt(1.f - real * real);
 *       in[ii] = lv_cmake(real, imag);
 *   }
 *
 *   volk_32fc_conjugate_32fc(out, in, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("out(%i) = %.1f + %.1fi\n", ii, lv_creal(out[ii]), lv_cimag(out[ii]));
 *   }
 *
 *   volk_free(in);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_conjugate_32fc_u_H
#define INCLUDED_volk_32fc_conjugate_32fc_u_H

#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_complex.h>
#include <float.h>

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32fc_conjugate_32fc_u_avx(lv_32fc_t* cVector, const lv_32fc_t* aVector, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int quarterPoints = num_points / 4;

  __m256 x;
  lv_32fc_t* c = cVector;
  const lv_32fc_t* a = aVector;

  __m256 conjugator = _mm256_setr_ps(0, -0.f, 0, -0.f, 0, -0.f, 0, -0.f);

  for(;number < quarterPoints; number++){

    x = _mm256_loadu_ps((float*)a); // Load the complex data as ar,ai,br,bi

    x = _mm256_xor_ps(x, conjugator); // conjugate register

    _mm256_storeu_ps((float*)c,x); // Store the results back into the C container

    a += 4;
    c += 4;
  }

  number = quarterPoints * 4;

  for(;number < num_points; number++) {
    *c++ = lv_conj(*a++);
  }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>

static inline void
volk_32fc_conjugate_32fc_u_sse3(lv_32fc_t* cVector, const lv_32fc_t* aVector, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int halfPoints = num_points / 2;

  __m128 x;
  lv_32fc_t* c = cVector;
  const lv_32fc_t* a = aVector;

  __m128 conjugator = _mm_setr_ps(0, -0.f, 0, -0.f);

  for(;number < halfPoints; number++){

    x = _mm_loadu_ps((float*)a); // Load the complex data as ar,ai,br,bi

    x = _mm_xor_ps(x, conjugator); // conjugate register

    _mm_storeu_ps((float*)c,x); // Store the results back into the C container

    a += 2;
    c += 2;
  }

  if((num_points % 2) != 0) {
    *c = lv_conj(*a);
  }
}
#endif /* LV_HAVE_SSE3 */

#ifdef LV_HAVE_GENERIC

static inline void
volk_32fc_conjugate_32fc_generic(lv_32fc_t* cVector, const lv_32fc_t* aVector, unsigned int num_points)
{
  lv_32fc_t* cPtr = cVector;
  const lv_32fc_t* aPtr = aVector;
  unsigned int number = 0;

  for(number = 0; number < num_points; number++){
    *cPtr++ = lv_conj(*aPtr++);
  }
}
#endif /* LV_HAVE_GENERIC */


#endif /* INCLUDED_volk_32fc_conjugate_32fc_u_H */
#ifndef INCLUDED_volk_32fc_conjugate_32fc_a_H
#define INCLUDED_volk_32fc_conjugate_32fc_a_H

#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_complex.h>
#include <float.h>

#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void
volk_32fc_conjugate_32fc_a_avx(lv_32fc_t* cVector, const lv_32fc_t* aVector, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int quarterPoints = num_points / 4;

  __m256 x;
  lv_32fc_t* c = cVector;
  const lv_32fc_t* a = aVector;

  __m256 conjugator = _mm256_setr_ps(0, -0.f, 0, -0.f, 0, -0.f, 0, -0.f);

  for(;number < quarterPoints; number++){

    x = _mm256_load_ps((float*)a); // Load the complex data as ar,ai,br,bi

    x = _mm256_xor_ps(x, conjugator); // conjugate register

    _mm256_store_ps((float*)c,x); // Store the results back into the C container

    a += 4;
    c += 4;
  }

  number = quarterPoints * 4;

  for(;number < num_points; number++) {
    *c++ = lv_conj(*a++);
  }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>

static inline void
volk_32fc_conjugate_32fc_a_sse3(lv_32fc_t* cVector, const lv_32fc_t* aVector, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int halfPoints = num_points / 2;

  __m128 x;
  lv_32fc_t* c = cVector;
  const lv_32fc_t* a = aVector;

  __m128 conjugator = _mm_setr_ps(0, -0.f, 0, -0.f);

  for(;number < halfPoints; number++){

    x = _mm_load_ps((float*)a); // Load the complex data as ar,ai,br,bi

    x = _mm_xor_ps(x, conjugator); // conjugate register

    _mm_store_ps((float*)c,x); // Store the results back into the C container

    a += 2;
    c += 2;
  }

  if((num_points % 2) != 0) {
    *c = lv_conj(*a);
  }
}
#endif /* LV_HAVE_SSE3 */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void
volk_32fc_conjugate_32fc_a_neon(lv_32fc_t* cVector, const lv_32fc_t* aVector, unsigned int num_points)
{
  unsigned int number;
  const unsigned int quarterPoints = num_points / 4;

  float32x4x2_t x;
  lv_32fc_t* c = cVector;
  const lv_32fc_t* a = aVector;

  for(number=0; number < quarterPoints; number++){
    __VOLK_PREFETCH(a+4);
    x = vld2q_f32((float*)a); // Load the complex data as ar,br,cr,dr; ai,bi,ci,di

    // xor the imaginary lane
    x.val[1] = vnegq_f32( x.val[1]);

    vst2q_f32((float*)c,x); // Store the results back into the C container

    a += 4;
    c += 4;
  }

  for(number=quarterPoints*4; number < num_points; number++){
    *c++ = lv_conj(*a++);
  }
}
#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_GENERIC

static inline void
volk_32fc_conjugate_32fc_a_generic(lv_32fc_t* cVector, const lv_32fc_t* aVector, unsigned int num_points)
{
  lv_32fc_t* cPtr = cVector;
  const lv_32fc_t* aPtr = aVector;
  unsigned int number = 0;

  for(number = 0; number < num_points; number++){
    *cPtr++ = lv_conj(*aPtr++);
  }
}
#endif /* LV_HAVE_GENERIC */


#endif /* INCLUDED_volk_32fc_conjugate_32fc_a_H */
