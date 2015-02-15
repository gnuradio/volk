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
 * \page volk_32fc_x2_s32f_square_dist_scalar_mult_32f
 *
 * \b Overview
 *
 * Calculates the square distance between a single complex input for each
 * point in a complex vector scaled by a scalar value.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_x2_s32f_square_dist_scalar_mult_32f(float* target, lv_32fc_t* src0, lv_32fc_t* points, float scalar, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li src0: The complex input. Only the first point is used.
 * \li points: A complex vector of reference points.
 * \li scalar: A float to scale the distances by
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li target: A vector of distances between src0 and the vector of points.
 *
 * \b Example
 * Calculate the distance between an input and reference points in a square
 * 16-qam constellation. Normalize distances by the area of the constellation.
 * \code
 *   int N = 16;
 *   unsigned int alignment = volk_get_alignment();
 *   lv_32fc_t* constellation  = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *   lv_32fc_t* rx  = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *   float* out = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float const_vals[] = {-3, -1, 1, 3};
 *
 *   unsigned int jj = 0;
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       constellation[ii] = lv_cmake(const_vals[ii%4], const_vals[jj]);
 *       if((ii+1)%4 == 0) ++jj;
 *   }
 *
 *   *rx = lv_cmake(0.5f, 2.f);
 *   float scale = 1.f/64.f; // 1 / constellation area
 *
 *   volk_32fc_x2_s32f_square_dist_scalar_mult_32f(out, rx, constellation, scale, N);
 *
 *   printf("Distance from each constellation point:\n");
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("%.4f  ", out[ii]);
 *       if((ii+1)%4 == 0) printf("\n");
 *   }
 *
 *   volk_free(rx);
 *   volk_free(constellation);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_x2_s32f_square_dist_scalar_mult_32f_a_H
#define INCLUDED_volk_32fc_x2_s32f_square_dist_scalar_mult_32f_a_H

#include<inttypes.h>
#include<stdio.h>
#include<volk/volk_complex.h>
#include <string.h>

#ifdef LV_HAVE_SSE3
#include<xmmintrin.h>
#include<pmmintrin.h>

static inline void
volk_32fc_x2_s32f_square_dist_scalar_mult_32f_a_sse3(float* target, lv_32fc_t* src0, lv_32fc_t* points,
                                                     float scalar, unsigned int num_points)
{
  const unsigned int num_bytes = num_points*8;

  __m128 xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;

  lv_32fc_t diff;
  memset(&diff, 0x0, 2*sizeof(float));

  float sq_dist = 0.0;
  int bound = num_bytes >> 5;
  int leftovers0 = (num_bytes >> 4) & 1;
  int leftovers1 = (num_bytes >> 3) & 1;
  int i = 0;

  xmm1 = _mm_setzero_ps();
  xmm1 = _mm_loadl_pi(xmm1, (__m64*)src0);
  xmm2 = _mm_load_ps((float*)&points[0]);
  xmm8 = _mm_load1_ps(&scalar);
  xmm1 = _mm_movelh_ps(xmm1, xmm1);
  xmm3 = _mm_load_ps((float*)&points[2]);

  for(; i < bound - 1; ++i) {
    xmm4 = _mm_sub_ps(xmm1, xmm2);
    xmm5 = _mm_sub_ps(xmm1, xmm3);
    points += 4;
    xmm6 = _mm_mul_ps(xmm4, xmm4);
    xmm7 = _mm_mul_ps(xmm5, xmm5);

    xmm2 = _mm_load_ps((float*)&points[0]);

    xmm4 = _mm_hadd_ps(xmm6, xmm7);

    xmm3 = _mm_load_ps((float*)&points[2]);

    xmm4 = _mm_mul_ps(xmm4, xmm8);

    _mm_store_ps(target, xmm4);

    target += 4;
  }

  xmm4 = _mm_sub_ps(xmm1, xmm2);
  xmm5 = _mm_sub_ps(xmm1, xmm3);

  points += 4;
  xmm6 = _mm_mul_ps(xmm4, xmm4);
  xmm7 = _mm_mul_ps(xmm5, xmm5);

  xmm4 = _mm_hadd_ps(xmm6, xmm7);

  xmm4 = _mm_mul_ps(xmm4, xmm8);

  _mm_store_ps(target, xmm4);

  target += 4;

  for(i = 0; i < leftovers0; ++i) {
    xmm2 = _mm_load_ps((float*)&points[0]);

    xmm4 = _mm_sub_ps(xmm1, xmm2);

    points += 2;

    xmm6 = _mm_mul_ps(xmm4, xmm4);

    xmm4 = _mm_hadd_ps(xmm6, xmm6);

    xmm4 = _mm_mul_ps(xmm4, xmm8);

    _mm_storeh_pi((__m64*)target, xmm4);

    target += 2;
  }

  for(i = 0; i < leftovers1; ++i) {

    diff = src0[0] - points[0];

    sq_dist = scalar * (lv_creal(diff) * lv_creal(diff) + lv_cimag(diff) * lv_cimag(diff));

    target[0] = sq_dist;
  }
}

#endif /*LV_HAVE_SSE3*/


#ifdef LV_HAVE_GENERIC
static inline void
volk_32fc_x2_s32f_square_dist_scalar_mult_32f_generic(float* target, lv_32fc_t* src0, lv_32fc_t* points,
                                                      float scalar, unsigned int num_points)
{
  const unsigned int num_bytes = num_points*8;

  lv_32fc_t diff;
  float sq_dist;
  unsigned int i = 0;

  for(; i < num_bytes >> 3; ++i) {
    diff = src0[0] - points[i];

    sq_dist = scalar * (lv_creal(diff) * lv_creal(diff) + lv_cimag(diff) * lv_cimag(diff));

    target[i] = sq_dist;
  }
}

#endif /*LV_HAVE_GENERIC*/


#endif /*INCLUDED_volk_32fc_x2_s32f_square_dist_scalar_mult_32f_a_H*/
