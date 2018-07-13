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
 * \page volk_32fc_index_max_16u
 *
 * \b Overview
 *
 * Returns Argmax_i mag(x[i]). Finds and returns the index which contains the
 * maximum magnitude for complex points in the given vector.
 *
 * Note that num_points is a uint32_t, but the return value is
 * uint16_t. Providing a vector larger than the max of a uint16_t
 * (65536) would miss anything outside of this boundary. The kernel
 * will check the length of num_points and cap it to this max value,
 * anyways.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_index_max_16u(uint16_t* target, lv_32fc_t* src0, uint32_t num_points)
 * \endcode
 *
 * \b Inputs
 * \li src0: The complex input vector.
 * \li num_points: The number of samples.
 *
 * \b Outputs
 * \li target: The index of the point with maximum magnitude.
 *
 * \b Example
 * Calculate the index of the maximum value of \f$x^2 + x\f$ for points around
 * the unit circle.
 * \code
 *   int N = 10;
 *   uint32_t alignment = volk_get_alignment();
 *   lv_32fc_t* in  = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *   uint16_t* max = (uint16_t*)volk_malloc(sizeof(uint16_t), alignment);
 *
 *   for(uint32_t ii = 0; ii < N/2; ++ii){
 *       float real = 2.f * ((float)ii / (float)N) - 1.f;
 *       float imag = std::sqrt(1.f - real * real);
 *       in[ii] = lv_cmake(real, imag);
 *       in[ii] = in[ii] * in[ii] + in[ii];
 *       in[N-ii] = lv_cmake(real, imag);
 *       in[N-ii] = in[N-ii] * in[N-ii] + in[N-ii];
 *   }
 *
 *   volk_32fc_index_max_16u(max, in, N);
 *
 *   printf("index of max value = %u\n",  *max);
 *
 *   volk_free(in);
 *   volk_free(max);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_index_max_16u_a_H
#define INCLUDED_volk_32fc_index_max_16u_a_H

#include <volk/volk_common.h>
#include <inttypes.h>
#include <stdio.h>
#include <limits.h>
#include <volk/volk_complex.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void
volk_32fc_index_max_16u_a_avx2(uint16_t* target, lv_32fc_t* src0,
                               uint32_t num_points)
{
  num_points = (num_points > USHRT_MAX) ? USHRT_MAX : num_points;
  // Branchless version, if we think it'll make a difference
  //num_points = USHRT_MAX ^ ((num_points ^ USHRT_MAX) & -(num_points < USHRT_MAX));

  const uint32_t num_bytes = num_points*8;

  union bit256 holderf;
  union bit256 holderi;
  float sq_dist = 0.0;

  union bit256 xmm5, xmm4;
  __m256 xmm1, xmm2, xmm3;
  __m256i xmm8, xmm11, xmm12, xmmfive, xmmfour, xmm9, holder0, holder1, xmm10;

  xmm5.int_vec = xmmfive = _mm256_setzero_si256();
  xmm4.int_vec = xmmfour = _mm256_setzero_si256();
  holderf.int_vec = holder0 = _mm256_setzero_si256();
  holderi.int_vec = holder1 = _mm256_setzero_si256();

  int bound = num_bytes >> 6;
  int leftovers0 = (num_bytes >> 5) & 1;
  int leftovers1 = (num_bytes >> 4) & 1;
  int i = 0;

  xmm8 = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
  xmm9 =  _mm256_setzero_si256(); //=xmm8
  xmm10 = _mm256_set1_epi32(8);
  xmm3 = _mm256_setzero_ps();

  __m256i idx = _mm256_set_epi32(7,6,3,2,5,4,1,0);
  for(; i < bound; ++i) {
    xmm1 = _mm256_load_ps((float*)src0);
    xmm2 = _mm256_load_ps((float*)&src0[4]);

    src0 += 8;

    xmm1 = _mm256_mul_ps(xmm1, xmm1);
    xmm2 = _mm256_mul_ps(xmm2, xmm2);

    xmm1 = _mm256_hadd_ps(xmm1, xmm2);
    xmm1 = _mm256_permutevar8x32_ps(xmm1, idx);

    xmm3 = _mm256_max_ps(xmm1, xmm3);

    xmm4.float_vec = _mm256_cmp_ps(xmm1, xmm3, 1);
    xmm5.float_vec = _mm256_cmp_ps(xmm1, xmm3, 0);

    xmm11 = _mm256_and_si256(xmm8, xmm5.int_vec);
    xmm12 = _mm256_and_si256(xmm9, xmm4.int_vec);

    xmm9 = _mm256_add_epi32(xmm11,  xmm12);

    xmm8 = _mm256_add_epi32(xmm8, xmm10);
  }
  xmm10 = _mm256_set1_epi32(4);
  for(; i < leftovers0; ++i) {
    xmm1 = _mm256_load_ps((float*)src0);

    src0 += 4;

    xmm1 = _mm256_mul_ps(xmm1, xmm1);

    xmm1 = _mm256_hadd_ps(xmm1, xmm1);
    xmm1 = _mm256_permutevar8x32_ps(xmm1, idx);

    xmm3 = _mm256_max_ps(xmm1, xmm3);

    xmm4.float_vec = _mm256_cmp_ps(xmm1, xmm3, 1);
    xmm5.float_vec = _mm256_cmp_ps(xmm1, xmm3, 0);

    xmm11 = _mm256_and_si256(xmm8, xmm5.int_vec);
    xmm12 = _mm256_and_si256(xmm9, xmm4.int_vec);

    xmm9 = _mm256_add_epi32(xmm11,  xmm12);

    xmm8 = _mm256_add_epi32(xmm8, xmm10);
  }

  idx = _mm256_set_epi32(1,0,1,0,1,0,1,0);
  xmm10 = _mm256_set1_epi32(2);
  for(i = 0; i < leftovers1; ++i) {
      xmm2 = _mm256_load_ps((float*)src0);

      xmm1 = _mm256_permutevar8x32_ps(bit256_p(&xmm8)->float_vec, idx);
      xmm8 = bit256_p(&xmm1)->int_vec;

      xmm2 = _mm256_mul_ps(xmm2, xmm2);

      src0 += 2;

      xmm1 = _mm256_hadd_ps(xmm2, xmm2);

      xmm3 = _mm256_max_ps(xmm1, xmm3);

      xmm4.float_vec = _mm256_cmp_ps(xmm1, xmm3,1);
      xmm5.float_vec = _mm256_cmp_ps(xmm1, xmm3,0);

      xmm11 = _mm256_and_si256(xmm8, xmm5.int_vec);
      xmm12 = _mm256_and_si256(xmm9, xmm4.int_vec);

      xmm9 = _mm256_add_epi32(xmm11, xmm12);

      xmm8 = _mm256_add_epi32(xmm8, xmm10);
  }

  /*
  idx = _mm256_setzero_si256();
  for(i = 0; i < leftovers2; ++i) {
    //printf("%u, %u, %u, %u\n", ((uint32_t*)&xmm9)[0], ((uint32_t*)&xmm9)[1], ((uint32_t*)&xmm9)[2], ((uint32_t*)&xmm9)[3]);

    sq_dist = lv_creal(src0[0]) * lv_creal(src0[0]) + lv_cimag(src0[0]) * lv_cimag(src0[0]);

    //xmm = _mm_load1_ps(&sq_dist);//insert?
    xmm2 = _mm256_set1_ps(sq_dist);
    //xmm2 = _mm256_insertf128_ps(xmm2, xmm, 0);

    xmm1 = xmm3;

    xmm3 = _mm256_max_ps(xmm3, xmm2);//only lowest 32bit value
    xmm3 = _mm256_permutevar8x32_ps(xmm3, idx);

    xmm4.float_vec = _mm256_cmp_ps(xmm1, xmm3, 1);
    xmm5.float_vec = _mm256_cmp_ps(xmm1, xmm3, 0);

    xmm8 = _mm256_permutevar8x32_epi32(xmm8, idx);

    xmm11 = _mm256_and_si256(xmm8, xmm4.int_vec);
    xmm12 = _mm256_and_si256(xmm9, xmm5.int_vec);

    xmm9 = _mm256_add_epi32(xmm11, xmm12);
}*/

  _mm256_store_ps((float*)&(holderf.f), xmm3);
  _mm256_store_si256(&(holderi.int_vec), xmm9);

  target[0] = holderi.i[0];
  sq_dist = holderf.f[0];
  target[0] = (holderf.f[1] > sq_dist) ? holderi.i[1] : target[0];
  sq_dist = (holderf.f[1] > sq_dist) ? holderf.f[1] : sq_dist;
  target[0] = (holderf.f[2] > sq_dist) ? holderi.i[2] : target[0];
  sq_dist = (holderf.f[2] > sq_dist) ? holderf.f[2] : sq_dist;
  target[0] = (holderf.f[3] > sq_dist) ? holderi.i[3] : target[0];
  sq_dist = (holderf.f[3] > sq_dist) ? holderf.f[3] : sq_dist;
  target[0] = (holderf.f[4] > sq_dist) ? holderi.i[4] : target[0];
  sq_dist = (holderf.f[4] > sq_dist) ? holderf.f[4] : sq_dist;
  target[0] = (holderf.f[5] > sq_dist) ? holderi.i[5] : target[0];
  sq_dist = (holderf.f[5] > sq_dist) ? holderf.f[5] : sq_dist;
  target[0] = (holderf.f[6] > sq_dist) ? holderi.i[6] : target[0];
  sq_dist = (holderf.f[6] > sq_dist) ? holderf.f[6] : sq_dist;
  target[0] = (holderf.f[7] > sq_dist) ? holderi.i[7] : target[0];
  sq_dist = (holderf.f[7] > sq_dist) ? holderf.f[7] : sq_dist;

}

#endif /*LV_HAVE_AVX2*/

#ifdef LV_HAVE_SSE3
#include <xmmintrin.h>
#include <pmmintrin.h>

static inline void
volk_32fc_index_max_16u_a_sse3(uint16_t* target, lv_32fc_t* src0,
                               uint32_t num_points)
{
  num_points = (num_points > USHRT_MAX) ? USHRT_MAX : num_points;
  // Branchless version, if we think it'll make a difference
  //num_points = USHRT_MAX ^ ((num_points ^ USHRT_MAX) & -(num_points < USHRT_MAX));

  const uint32_t num_bytes = num_points*8;

  union bit128 holderf;
  union bit128 holderi;
  float sq_dist = 0.0;

  union bit128 xmm5, xmm4;
  __m128 xmm1, xmm2, xmm3;
  __m128i xmm8, xmm11, xmm12, xmmfive, xmmfour, xmm9, holder0, holder1, xmm10;

  xmm5.int_vec = xmmfive = _mm_setzero_si128();
  xmm4.int_vec = xmmfour = _mm_setzero_si128();
  holderf.int_vec = holder0 = _mm_setzero_si128();
  holderi.int_vec = holder1 = _mm_setzero_si128();

  int bound = num_bytes >> 5;
  int leftovers0 = (num_bytes >> 4) & 1;
  int leftovers1 = (num_bytes >> 3) & 1;
  int i = 0;

  xmm8 = _mm_set_epi32(3, 2, 1, 0);//remember the crazy reverse order!
  xmm9 = _mm_setzero_si128();
  xmm10 = _mm_set_epi32(4, 4, 4, 4);
  xmm3 = _mm_setzero_ps();
  //printf("%f, %f, %f, %f\n", ((float*)&xmm10)[0], ((float*)&xmm10)[1], ((float*)&xmm10)[2], ((float*)&xmm10)[3]);

  for(; i < bound; ++i) {
    xmm1 = _mm_load_ps((float*)src0);
    xmm2 = _mm_load_ps((float*)&src0[2]);

    src0 += 4;

    xmm1 = _mm_mul_ps(xmm1, xmm1);
    xmm2 = _mm_mul_ps(xmm2, xmm2);

    xmm1 = _mm_hadd_ps(xmm1, xmm2);

    xmm3 = _mm_max_ps(xmm1, xmm3);

    xmm4.float_vec = _mm_cmplt_ps(xmm1, xmm3);
    xmm5.float_vec = _mm_cmpeq_ps(xmm1, xmm3);

    xmm11 = _mm_and_si128(xmm8, xmm5.int_vec);
    xmm12 = _mm_and_si128(xmm9, xmm4.int_vec);

    xmm9 = _mm_add_epi32(xmm11,  xmm12);

    xmm8 = _mm_add_epi32(xmm8, xmm10);

    //printf("%f, %f, %f, %f\n", ((float*)&xmm3)[0], ((float*)&xmm3)[1], ((float*)&xmm3)[2], ((float*)&xmm3)[3]);
    //printf("%u, %u, %u, %u\n", ((uint32_t*)&xmm10)[0], ((uint32_t*)&xmm10)[1], ((uint32_t*)&xmm10)[2], ((uint32_t*)&xmm10)[3]);
  }


  for(i = 0; i < leftovers0; ++i) {
    xmm2 = _mm_load_ps((float*)src0);

    xmm1 = _mm_movelh_ps(bit128_p(&xmm8)->float_vec, bit128_p(&xmm8)->float_vec);
    xmm8 = bit128_p(&xmm1)->int_vec;

    xmm2 = _mm_mul_ps(xmm2, xmm2);

    src0 += 2;

    xmm1 = _mm_hadd_ps(xmm2, xmm2);

    xmm3 = _mm_max_ps(xmm1, xmm3);

    xmm10 = _mm_set_epi32(2, 2, 2, 2);//load1_ps((float*)&init[2]);

    xmm4.float_vec = _mm_cmplt_ps(xmm1, xmm3);
    xmm5.float_vec = _mm_cmpeq_ps(xmm1, xmm3);

    xmm11 = _mm_and_si128(xmm8, xmm5.int_vec);
    xmm12 = _mm_and_si128(xmm9, xmm4.int_vec);

    xmm9 = _mm_add_epi32(xmm11, xmm12);

    xmm8 = _mm_add_epi32(xmm8, xmm10);
    //printf("egads%u, %u, %u, %u\n", ((uint32_t*)&xmm9)[0], ((uint32_t*)&xmm9)[1], ((uint32_t*)&xmm9)[2], ((uint32_t*)&xmm9)[3]);
  }

  for(i = 0; i < leftovers1; ++i) {
    //printf("%u, %u, %u, %u\n", ((uint32_t*)&xmm9)[0], ((uint32_t*)&xmm9)[1], ((uint32_t*)&xmm9)[2], ((uint32_t*)&xmm9)[3]);

    sq_dist = lv_creal(src0[0]) * lv_creal(src0[0]) + lv_cimag(src0[0]) * lv_cimag(src0[0]);

    xmm2 = _mm_load1_ps(&sq_dist);

    xmm1 = xmm3;

    xmm3 = _mm_max_ss(xmm3, xmm2);

    xmm4.float_vec = _mm_cmplt_ps(xmm1, xmm3);
    xmm5.float_vec = _mm_cmpeq_ps(xmm1, xmm3);

    xmm8 = _mm_shuffle_epi32(xmm8, 0x00);

    xmm11 = _mm_and_si128(xmm8, xmm4.int_vec);
    xmm12 = _mm_and_si128(xmm9, xmm5.int_vec);

    xmm9 = _mm_add_epi32(xmm11, xmm12);
  }

  //printf("%f, %f, %f, %f\n", ((float*)&xmm3)[0], ((float*)&xmm3)[1], ((float*)&xmm3)[2], ((float*)&xmm3)[3]);
  //printf("%u, %u, %u, %u\n", ((uint32_t*)&xmm9)[0], ((uint32_t*)&xmm9)[1], ((uint32_t*)&xmm9)[2], ((uint32_t*)&xmm9)[3]);

  _mm_store_ps((float*)&(holderf.f), xmm3);
  _mm_store_si128(&(holderi.int_vec), xmm9);

  target[0] = holderi.i[0];
  sq_dist = holderf.f[0];
  target[0] = (holderf.f[1] > sq_dist) ? holderi.i[1] : target[0];
  sq_dist = (holderf.f[1] > sq_dist) ? holderf.f[1] : sq_dist;
  target[0] = (holderf.f[2] > sq_dist) ? holderi.i[2] : target[0];
  sq_dist = (holderf.f[2] > sq_dist) ? holderf.f[2] : sq_dist;
  target[0] = (holderf.f[3] > sq_dist) ? holderi.i[3] : target[0];
  sq_dist = (holderf.f[3] > sq_dist) ? holderf.f[3] : sq_dist;

  /*
  float placeholder = 0.0;
  uint32_t temp0, temp1;
  uint32_t g0 = (((float*)&xmm3)[0] > ((float*)&xmm3)[1]);
  uint32_t l0 = g0 ^ 1;

  uint32_t g1 = (((float*)&xmm3)[1] > ((float*)&xmm3)[2]);
  uint32_t l1 = g1 ^ 1;

  temp0 = g0 * ((uint32_t*)&xmm9)[0] + l0 * ((uint32_t*)&xmm9)[1];
  temp1 = g0 * ((uint32_t*)&xmm9)[2] + l0 * ((uint32_t*)&xmm9)[3];
  sq_dist = g0 * ((float*)&xmm3)[0] + l0 * ((float*)&xmm3)[1];
  placeholder = g0 * ((float*)&xmm3)[2] + l0 * ((float*)&xmm3)[3];

  g0 = (sq_dist > placeholder);
  l0 = g0 ^ 1;
  target[0] = g0 * temp0 + l0 * temp1;
  */
}

#endif /*LV_HAVE_SSE3*/

#ifdef LV_HAVE_GENERIC
static inline void
 volk_32fc_index_max_16u_generic(uint16_t* target, lv_32fc_t* src0,
                                 uint32_t num_points)
{
  num_points = (num_points > USHRT_MAX) ? USHRT_MAX : num_points;

  const uint32_t num_bytes = num_points*8;

  float sq_dist = 0.0;
  float max = 0.0;
  uint16_t index = 0;

  uint32_t i = 0;

  for(; i < num_bytes >> 3; ++i) {
    sq_dist = lv_creal(src0[i]) * lv_creal(src0[i]) + lv_cimag(src0[i]) * lv_cimag(src0[i]);

    index = sq_dist > max ? i : index;
    max = sq_dist > max ? sq_dist : max;
  }
  target[0] = index;
}

#endif /*LV_HAVE_GENERIC*/


#endif /*INCLUDED_volk_32fc_index_max_16u_a_H*/


#ifndef INCLUDED_volk_32fc_index_max_16u_u_H
#define INCLUDED_volk_32fc_index_max_16u_u_H

#include <volk/volk_common.h>
#include <inttypes.h>
#include <stdio.h>
#include <limits.h>
#include <volk/volk_complex.h>

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void
volk_32fc_index_max_16u_u_avx2(uint16_t* target, lv_32fc_t* src0,
                               uint32_t num_points)
{
  num_points = (num_points > USHRT_MAX) ? USHRT_MAX : num_points;
  // Branchless version, if we think it'll make a difference
  //num_points = USHRT_MAX ^ ((num_points ^ USHRT_MAX) & -(num_points < USHRT_MAX));

  const uint32_t num_bytes = num_points*8;

  union bit256 holderf;
  union bit256 holderi;
  float sq_dist = 0.0;

  union bit256 xmm5, xmm4;
  __m256 xmm1, xmm2, xmm3;
  __m256i xmm8, xmm11, xmm12, xmmfive, xmmfour, xmm9, holder0, holder1, xmm10;

  xmm5.int_vec = xmmfive = _mm256_setzero_si256();
  xmm4.int_vec = xmmfour = _mm256_setzero_si256();
  holderf.int_vec = holder0 = _mm256_setzero_si256();
  holderi.int_vec = holder1 = _mm256_setzero_si256();

  int bound = num_bytes >> 6;
  int leftovers0 = (num_bytes >> 5) & 1;
  int leftovers1 = (num_bytes >> 4) & 1;
  int i = 0;

  xmm8 = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
  xmm9 =  _mm256_setzero_si256(); //=xmm8
  xmm10 = _mm256_set1_epi32(8);
  xmm3 = _mm256_setzero_ps();

  __m256i idx = _mm256_set_epi32(7,6,3,2,5,4,1,0);
  for(; i < bound; ++i) {
    xmm1 = _mm256_loadu_ps((float*)src0);
    xmm2 = _mm256_loadu_ps((float*)&src0[4]);

    src0 += 8;

    xmm1 = _mm256_mul_ps(xmm1, xmm1);
    xmm2 = _mm256_mul_ps(xmm2, xmm2);

    xmm1 = _mm256_hadd_ps(xmm1, xmm2);
    xmm1 = _mm256_permutevar8x32_ps(xmm1, idx);

    xmm3 = _mm256_max_ps(xmm1, xmm3);

    xmm4.float_vec = _mm256_cmp_ps(xmm1, xmm3, 1);
    xmm5.float_vec = _mm256_cmp_ps(xmm1, xmm3, 0);

    xmm11 = _mm256_and_si256(xmm8, xmm5.int_vec);
    xmm12 = _mm256_and_si256(xmm9, xmm4.int_vec);

    xmm9 = _mm256_add_epi32(xmm11,  xmm12);

    xmm8 = _mm256_add_epi32(xmm8, xmm10);
  }
  xmm10 = _mm256_set1_epi32(4);
  for(; i < leftovers0; ++i) {
    xmm1 = _mm256_loadu_ps((float*)src0);

    src0 += 4;

    xmm1 = _mm256_mul_ps(xmm1, xmm1);

    xmm1 = _mm256_hadd_ps(xmm1, xmm1);
    xmm1 = _mm256_permutevar8x32_ps(xmm1, idx);

    xmm3 = _mm256_max_ps(xmm1, xmm3);

    xmm4.float_vec = _mm256_cmp_ps(xmm1, xmm3, 1);
    xmm5.float_vec = _mm256_cmp_ps(xmm1, xmm3, 0);

    xmm11 = _mm256_and_si256(xmm8, xmm5.int_vec);
    xmm12 = _mm256_and_si256(xmm9, xmm4.int_vec);

    xmm9 = _mm256_add_epi32(xmm11,  xmm12);

    xmm8 = _mm256_add_epi32(xmm8, xmm10);
  }

  idx = _mm256_set_epi32(1,0,1,0,1,0,1,0);
  xmm10 = _mm256_set1_epi32(2);
  for(i = 0; i < leftovers1; ++i) {
      xmm2 = _mm256_loadu_ps((float*)src0);

      xmm1 = _mm256_permutevar8x32_ps(bit256_p(&xmm8)->float_vec, idx);
      xmm8 = bit256_p(&xmm1)->int_vec;

      xmm2 = _mm256_mul_ps(xmm2, xmm2);

      src0 += 2;

      xmm1 = _mm256_hadd_ps(xmm2, xmm2);

      xmm3 = _mm256_max_ps(xmm1, xmm3);

      xmm4.float_vec = _mm256_cmp_ps(xmm1, xmm3,1);
      xmm5.float_vec = _mm256_cmp_ps(xmm1, xmm3,0);

      xmm11 = _mm256_and_si256(xmm8, xmm5.int_vec);
      xmm12 = _mm256_and_si256(xmm9, xmm4.int_vec);

      xmm9 = _mm256_add_epi32(xmm11, xmm12);

      xmm8 = _mm256_add_epi32(xmm8, xmm10);
  }

  _mm256_storeu_ps((float*)&(holderf.f), xmm3);
  _mm256_storeu_si256(&(holderi.int_vec), xmm9);

  target[0] = holderi.i[0];
  sq_dist = holderf.f[0];
  target[0] = (holderf.f[1] > sq_dist) ? holderi.i[1] : target[0];
  sq_dist = (holderf.f[1] > sq_dist) ? holderf.f[1] : sq_dist;
  target[0] = (holderf.f[2] > sq_dist) ? holderi.i[2] : target[0];
  sq_dist = (holderf.f[2] > sq_dist) ? holderf.f[2] : sq_dist;
  target[0] = (holderf.f[3] > sq_dist) ? holderi.i[3] : target[0];
  sq_dist = (holderf.f[3] > sq_dist) ? holderf.f[3] : sq_dist;
  target[0] = (holderf.f[4] > sq_dist) ? holderi.i[4] : target[0];
  sq_dist = (holderf.f[4] > sq_dist) ? holderf.f[4] : sq_dist;
  target[0] = (holderf.f[5] > sq_dist) ? holderi.i[5] : target[0];
  sq_dist = (holderf.f[5] > sq_dist) ? holderf.f[5] : sq_dist;
  target[0] = (holderf.f[6] > sq_dist) ? holderi.i[6] : target[0];
  sq_dist = (holderf.f[6] > sq_dist) ? holderf.f[6] : sq_dist;
  target[0] = (holderf.f[7] > sq_dist) ? holderi.i[7] : target[0];
  sq_dist = (holderf.f[7] > sq_dist) ? holderf.f[7] : sq_dist;

}

#endif /*LV_HAVE_AVX2*/

#endif /*INCLUDED_volk_32fc_index_max_16u_u_H*/
