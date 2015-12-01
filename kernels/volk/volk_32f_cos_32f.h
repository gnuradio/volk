/* -*- c++ -*- */
/*
 * Copyright 2014 Free Software Foundation, Inc.
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
 * \page volk_32f_cos_32f
 *
 * \b Overview
 *
 * Computes cosine of the input vector and stores results in the output vector.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_cos_32f(float* bVector, const float* aVector, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li aVector: The input vector of floats.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li bVector: The vector where results will be stored.
 *
 * \b Example
 * Calculate cos(theta) for common angles.
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   float* in = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* out = (float*)volk_malloc(sizeof(float)*N, alignment);
 *
 *   in[0] = 0.000;
 *   in[1] = 0.524;
 *   in[2] = 0.786;
 *   in[3] = 1.047;
 *   in[4] = 1.571;
 *   in[5] = 1.571;
 *   in[6] = 2.094;
 *   in[7] = 2.356;
 *   in[8] = 2.618;
 *   in[9] = 3.142;
 *
 *   volk_32f_cos_32f(out, in, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("cos(%1.3f) = %1.3f\n", in[ii], out[ii]);
 *   }
 *
 *   volk_free(in);
 *   volk_free(out);
 * \endcode
 */

#include <stdio.h>
#include <math.h>
#include <inttypes.h>

#ifndef INCLUDED_volk_32f_cos_32f_a_H
#define INCLUDED_volk_32f_cos_32f_a_H

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void
 volk_32f_cos_32f_a_sse4_1(float* bVector, const float* aVector, unsigned int num_points)
{
  float* bPtr = bVector;
  const float* aPtr = aVector;

  unsigned int number = 0;
  unsigned int quarterPoints = num_points / 4;
  unsigned int i = 0;

  __m128 aVal, s, r, m4pi, pio4A, pio4B, pio4C, cp1, cp2, cp3, cp4, cp5, ffours, ftwos, fones, fzeroes;
  __m128 sine, cosine;
  __m128i q, ones, twos, fours;

  m4pi = _mm_set1_ps(1.273239544735162542821171882678754627704620361328125);
  pio4A = _mm_set1_ps(0.7853981554508209228515625);
  pio4B = _mm_set1_ps(0.794662735614792836713604629039764404296875e-8);
  pio4C = _mm_set1_ps(0.306161699786838294306516483068750264552437361480769e-16);
  ffours = _mm_set1_ps(4.0);
  ftwos = _mm_set1_ps(2.0);
  fones = _mm_set1_ps(1.0);
  fzeroes = _mm_setzero_ps();
  __m128i zeroes = _mm_set1_epi32(0);
  ones = _mm_set1_epi32(1);
  __m128i allones = _mm_set1_epi32(0xffffffff);
  twos = _mm_set1_epi32(2);
  fours = _mm_set1_epi32(4);

  cp1 = _mm_set1_ps(1.0);
  cp2 = _mm_set1_ps(0.08333333333333333);
  cp3 = _mm_set1_ps(0.002777777777777778);
  cp4 = _mm_set1_ps(4.96031746031746e-05);
  cp5 = _mm_set1_ps(5.511463844797178e-07);
  union bit128 condition1;
  union bit128 condition3;

  for(;number < quarterPoints; number++){

    aVal = _mm_load_ps(aPtr);
    // s = fabs(aVal)
    s = _mm_sub_ps(aVal, _mm_and_ps(_mm_mul_ps(aVal, ftwos), _mm_cmplt_ps(aVal, fzeroes)));
    // q = (int) (s * (4/pi)), floor(aVal / (pi/4))
    q = _mm_cvtps_epi32(_mm_floor_ps(_mm_mul_ps(s, m4pi)));
    // r = q + q&1, q indicates quadrant, r gives
    r = _mm_cvtepi32_ps(_mm_add_epi32(q, _mm_and_si128(q, ones)));

    s = _mm_sub_ps(s, _mm_mul_ps(r, pio4A));
    s = _mm_sub_ps(s, _mm_mul_ps(r, pio4B));
    s = _mm_sub_ps(s, _mm_mul_ps(r, pio4C));

    s = _mm_div_ps(s, _mm_set1_ps(8.0));    // The constant is 2^N, for 3 times argument reduction
    s = _mm_mul_ps(s, s);
    // Evaluate Taylor series
    s = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_sub_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_sub_ps(_mm_mul_ps(s, cp5), cp4), s), cp3), s), cp2), s), cp1), s);

    for(i = 0; i < 3; i++)
      s = _mm_mul_ps(s, _mm_sub_ps(ffours, s));
    s = _mm_div_ps(s, ftwos);

    sine = _mm_sqrt_ps(_mm_mul_ps(_mm_sub_ps(ftwos, s), s));
    cosine = _mm_sub_ps(fones, s);

    // if(((q+1)&2) != 0) { cosine=sine;}
    condition1.int_vec = _mm_cmpeq_epi32(_mm_and_si128(_mm_add_epi32(q, ones), twos), zeroes);
    condition1.int_vec = _mm_xor_si128(allones, condition1.int_vec);

    // if(((q+2)&4) != 0) { cosine = -cosine;}
    condition3.int_vec = _mm_cmpeq_epi32(_mm_and_si128(_mm_add_epi32(q, twos), fours), zeroes);
    condition3.int_vec = _mm_xor_si128(allones, condition3.int_vec);

    cosine = _mm_add_ps(cosine, _mm_and_ps(_mm_sub_ps(sine, cosine), condition1.float_vec));
    cosine = _mm_sub_ps(cosine, _mm_and_ps(_mm_mul_ps(cosine, _mm_set1_ps(2.0f)), condition3.float_vec));
    _mm_store_ps(bPtr, cosine);
    aPtr += 4;
    bPtr += 4;
  }

  number = quarterPoints * 4;
  for(;number < num_points; number++){
    *bPtr++ = cos(*aPtr++);
  }
}

#endif /* LV_HAVE_SSE4_1 for aligned */

#endif /* INCLUDED_volk_32f_cos_32f_a_H */



#ifndef INCLUDED_volk_32f_cos_32f_u_H
#define INCLUDED_volk_32f_cos_32f_u_H

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

static inline void
volk_32f_cos_32f_u_sse4_1(float* bVector, const float* aVector, unsigned int num_points)
{
  float* bPtr = bVector;
  const float* aPtr = aVector;

  unsigned int number = 0;
  unsigned int quarterPoints = num_points / 4;
  unsigned int i = 0;

  __m128 aVal, s, m4pi, pio4A, pio4B, cp1, cp2, cp3, cp4, cp5, ffours, ftwos, fones, fzeroes;
  __m128 sine, cosine, condition1, condition3;
  __m128i q, r, ones, twos, fours;

  m4pi = _mm_set1_ps(1.273239545);
  pio4A = _mm_set1_ps(0.78515625);
  pio4B = _mm_set1_ps(0.241876e-3);
  ffours = _mm_set1_ps(4.0);
  ftwos = _mm_set1_ps(2.0);
  fones = _mm_set1_ps(1.0);
  fzeroes = _mm_setzero_ps();
  ones = _mm_set1_epi32(1);
  twos = _mm_set1_epi32(2);
  fours = _mm_set1_epi32(4);

  cp1 = _mm_set1_ps(1.0);
  cp2 = _mm_set1_ps(0.83333333e-1);
  cp3 = _mm_set1_ps(0.2777778e-2);
  cp4 = _mm_set1_ps(0.49603e-4);
  cp5 = _mm_set1_ps(0.551e-6);

  for(;number < quarterPoints; number++){
    aVal = _mm_loadu_ps(aPtr);
    s = _mm_sub_ps(aVal, _mm_and_ps(_mm_mul_ps(aVal, ftwos), _mm_cmplt_ps(aVal, fzeroes)));
    q = _mm_cvtps_epi32(_mm_floor_ps(_mm_mul_ps(s, m4pi)));
    r = _mm_add_epi32(q, _mm_and_si128(q, ones));

    s = _mm_sub_ps(s, _mm_mul_ps(_mm_cvtepi32_ps(r), pio4A));
    s = _mm_sub_ps(s, _mm_mul_ps(_mm_cvtepi32_ps(r), pio4B));

    s = _mm_div_ps(s, _mm_set1_ps(8.0));    // The constant is 2^N, for 3 times argument reduction
    s = _mm_mul_ps(s, s);
    // Evaluate Taylor series
    s = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_sub_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_sub_ps(_mm_mul_ps(s, cp5), cp4), s), cp3), s), cp2), s), cp1), s);

    for(i = 0; i < 3; i++){
      s = _mm_mul_ps(s, _mm_sub_ps(ffours, s));
    }
    s = _mm_div_ps(s, ftwos);

    sine = _mm_sqrt_ps(_mm_mul_ps(_mm_sub_ps(ftwos, s), s));
    cosine = _mm_sub_ps(fones, s);

    condition1 = _mm_cmpneq_ps(_mm_cvtepi32_ps(_mm_and_si128(_mm_add_epi32(q, ones), twos)), fzeroes);

    condition3 = _mm_cmpneq_ps(_mm_cvtepi32_ps(_mm_and_si128(_mm_add_epi32(q, twos), fours)), fzeroes);

    cosine = _mm_add_ps(cosine, _mm_and_ps(_mm_sub_ps(sine, cosine), condition1));
    cosine = _mm_sub_ps(cosine, _mm_and_ps(_mm_mul_ps(cosine, _mm_set1_ps(2.0f)), condition3));
    _mm_storeu_ps(bPtr, cosine);
    aPtr += 4;
    bPtr += 4;
  }

  number = quarterPoints * 4;
  for(;number < num_points; number++){
    *bPtr++ = cos(*aPtr++);
  }
}

#endif /* LV_HAVE_SSE4_1 for unaligned */


#ifdef LV_HAVE_GENERIC

/*
 * For derivation see
 * Shibata, Naoki, "Efficient evaluation methods of elementary functions
 * suitable for SIMD computation," in Springer-Verlag 2010
 */
static inline void
volk_32f_cos_32f_generic_fast(float* bVector, const float* aVector, unsigned int num_points)
{
  float* bPtr = bVector;
  const float* aPtr = aVector;

  float m4pi = 1.273239544735162542821171882678754627704620361328125;
  float pio4A = 0.7853981554508209228515625;
  float pio4B = 0.794662735614792836713604629039764404296875e-8;
  float pio4C = 0.306161699786838294306516483068750264552437361480769e-16;
  int N = 3; // order of argument reduction

  unsigned int number;
  for(number = 0; number < num_points; number++){
      float s = fabs(*aPtr);
      int q = (int)(s * m4pi);
      int r = q + (q&1);
      s -= r * pio4A;
      s -= r * pio4B;
      s -= r * pio4C;

      s = s * 0.125; // 2^-N (<--3)
      s = s*s;
      s = ((((s/1814400. - 1.0/20160.0)*s + 1.0/360.0)*s - 1.0/12.0)*s + 1.0)*s;

      int i;
      for(i=0; i < N; ++i) {
          s = (4.0-s)*s;
      }
      s = s/2.0;

      float sine = sqrt((2.0-s)*s);
      float cosine = 1-s;

      if (((q+1) & 2) != 0) {
          s = cosine;
          cosine = sine;
          sine = s;
      }
      if (((q+2) & 4) != 0) {
          cosine = -cosine;
      }
      *bPtr = cosine;
      bPtr++;
      aPtr++;
  }
}

#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_cos_32f_generic(float* bVector, const float* aVector, unsigned int num_points)
{
  float* bPtr = bVector;
  const float* aPtr = aVector;
  unsigned int number = 0;

  for(; number < num_points; number++){
    *bPtr++ = cos(*aPtr++);
  }
}

#endif /* LV_HAVE_GENERIC */

#endif /* INCLUDED_volk_32f_cos_32f_u_H */
