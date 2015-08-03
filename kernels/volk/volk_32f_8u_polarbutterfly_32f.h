/* -*- c++ -*- */
/* 
 * Copyright 2015 Free Software Foundation, Inc.
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
 * \page volk_32f_8u_polarbutterfly_32f
 *
 * \b Overview
 *
 * decode butterfly for one bit in polar decoder graph.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * volk_32f_8u_polarbutterfly_32f(float* llrs, unsigned char* u,
 *    const int frame_size, const int frame_exp,
 *    const int stage, const int u_num, const int row)
 * \endcode
 *
 * \b Inputs
 * \li llrs: buffer with LLRs. contains received LLRs and already decoded LLRs.
 * \li u: previously decoded bits
 * \li frame_size: = 2 ^ frame_exp.
 * \li frame_exp: power of 2 value for frame size.
 * \li stage: value in range [0, frame_exp). start stage algorithm goes deeper.
 * \li u_num: bit number currently to be decoded
 * \li row: row in graph to start decoding.
 *
 * \b Outputs
 * \li frame: necessary LLRs for bit [u_num] to be decoded
 *
 * \b Example
 * \code
 * int frame_exp = 10;
 * int frame_size = 0x01 << frame_exp;
 *
 * float* llrs = (float*) volk_malloc(sizeof(float) * frame_size * (frame_exp + 1), volk_get_alignment());
 * unsigned char* u = (unsigned char) volk_malloc(sizeof(unsigned char) * frame_size * (frame_exp + 1), volk_get_alignment());
 *
 *  {some_function_to_write_encoded_bits_to_float_llrs(llrs + frame_size * frame_exp, data)};
 *
 * unsigned int u_num;
 * for(u_num = 0; u_num < frame_size; u_num++){
 *     volk_32f_8u_polarbutterfly_32f_u_avx(llrs, u, frame_size, frame_exp, 0, u_num, u_num);
 *     // next line could first search for frozen bit value and then do bit decision.
 *     u[u_num] = llrs[u_num] > 0 ? 0 : 1;
 * }
 *
 * volk_free(llrs);
 * volk_free(u);
 * \endcode
 */

#ifndef VOLK_KERNELS_VOLK_VOLK_32F_8U_POLARBUTTERFLY_32F_H_
#define VOLK_KERNELS_VOLK_VOLK_32F_8U_POLARBUTTERFLY_32F_H_
#include <math.h>
#include <volk/volk_8u_x2_encodeframepolar_8u.h>

static inline float
llr_odd(const float la, const float lb)
{
  const float ala = fabs(la);
  const float alb = fabs(lb);
  return copysignf(1.0f, la) * copysignf(1.0f, lb) * (ala > alb ? alb : ala);
}

static inline void
llr_odd_stages(float* llrs, int min_stage, const int depth, const int frame_size, const int row)
{
  int loop_stage = depth - 1;
  float* dst_llr_ptr;
  float* src_llr_ptr;
  int stage_size = 0x01 << loop_stage;

  int el;
  while(min_stage <= loop_stage){
    dst_llr_ptr = llrs + loop_stage * frame_size + row;
    src_llr_ptr = dst_llr_ptr + frame_size;
    for(el = 0; el < stage_size; el++){
      *dst_llr_ptr++ = llr_odd(*src_llr_ptr, *(src_llr_ptr + 1));
      src_llr_ptr += 2;
    }

    --loop_stage;
    stage_size >>= 1;
  }
}

static inline float
llr_even(const float la, const float lb, const unsigned char f)
{
  switch(f){
    case 0:
      return lb + la;
    default:
      return lb - la;
  }
}

static inline void
even_u_values(unsigned char* u_even, const unsigned char* u, const int u_num)
{
  u++;
  int i;
  for(i = 1; i < u_num; i += 2){
    *u_even++ = *u;
    u += 2;
  }
}

static inline void
odd_xor_even_values(unsigned char* u_xor, const unsigned char* u, const int u_num)
{
  int i;
  for(i = 1; i < u_num; i += 2){
    *u_xor++ = *u ^ *(u + 1);
    u += 2;
  }
}

static inline int
calculate_max_stage_depth_for_row(const int frame_exp, const int row)
{
  int max_stage_depth = 0;
  int half_stage_size = 0x01;
  int stage_size = half_stage_size << 1;
  while(max_stage_depth < (frame_exp - 1)){ // last stage holds received values.
    if(!(row % stage_size < half_stage_size)){
      break;
    }
    half_stage_size <<= 1;
    stage_size <<= 1;
    max_stage_depth++;
  }
  return max_stage_depth;
}

#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_8u_polarbutterfly_32f_generic(float* llrs, unsigned char* u,
    const int frame_size, const int frame_exp,
    const int stage, const int u_num, const int row)
{
  const int next_stage = stage + 1;

  const int half_stage_size = 0x01 << stage;
  const int stage_size = half_stage_size << 1;

  const bool is_upper_stage_half = row % stage_size < half_stage_size;

//      // this is a natural bit order impl
  float* next_llrs = llrs + frame_size;// LLRs are stored in a consecutive array.
  float* call_row_llr = llrs + row;

  const int section = row - (row % stage_size);
  const int jump_size = ((row % half_stage_size) << 1) % stage_size;

  const int next_upper_row = section + jump_size;
  const int next_lower_row = next_upper_row + 1;

  const float* upper_right_llr_ptr = next_llrs + next_upper_row;
  const float* lower_right_llr_ptr = next_llrs + next_lower_row;

  if(!is_upper_stage_half){
    const int u_pos = u_num >> stage;
    const unsigned char f = u[u_pos - 1];
    *call_row_llr = llr_even(*upper_right_llr_ptr, *lower_right_llr_ptr, f);
    return;
  }

  if(frame_exp > next_stage){
    unsigned char* u_half = u + frame_size;
    odd_xor_even_values(u_half, u, u_num);
    volk_32f_8u_polarbutterfly_32f_generic(next_llrs, u_half, frame_size, frame_exp, next_stage, u_num, next_upper_row);

    even_u_values(u_half, u, u_num);
    volk_32f_8u_polarbutterfly_32f_generic(next_llrs, u_half, frame_size, frame_exp, next_stage, u_num, next_lower_row);
  }

  *call_row_llr = llr_odd(*upper_right_llr_ptr, *lower_right_llr_ptr);
}

#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_AVX
#include <immintrin.h>

/*
 * https://software.intel.com/sites/landingpage/IntrinsicsGuide/#
 * lists '__m256 _mm256_loadu2_m128 (float const* hiaddr, float const* loaddr)'.
 * But GCC 4.8.4 doesn't know about it. Or headers are missing or something. Anyway, it doesn't compile :(
 * This is what I want: llr0 = _mm256_loadu2_m128(src_llr_ptr, src_llr_ptr + 8);
 * also useful but missing: _mm256_set_m128(hi, lo)
 */

static inline void
volk_32f_8u_polarbutterfly_32f_u_avx(float* llrs, unsigned char* u,
    const int frame_size, const int frame_exp,
    const int stage, const int u_num, const int row)
{
  if(row % 2){ // for odd rows just do the only necessary calculation and return.
    const float* next_llrs = llrs + frame_size + row;
    *(llrs + row) = llr_even(*(next_llrs - 1), *next_llrs, u[u_num - 1]);
    return;
  }

  const int max_stage_depth = calculate_max_stage_depth_for_row(frame_exp, row);
  if(max_stage_depth < 3){ // vectorized version needs larger vectors.
    volk_32f_8u_polarbutterfly_32f_generic(llrs, u, frame_size, frame_exp, stage, u_num, row);
    return;
  }

  int loop_stage = max_stage_depth;
  int stage_size = 0x01 << loop_stage;

  float* src_llr_ptr;
  float* dst_llr_ptr;

  __m256 src0, src1, dst;
  __m256 part0, part1;
  __m256 llr0, llr1;

  if(row){ // not necessary for ZERO row. == first bit to be decoded.
    // first do bit combination for all stages
    // effectively encode some decoded bits again.
    unsigned char* u_target = u + frame_size;
    unsigned char* u_temp = u + 2* frame_size;
    memcpy(u_temp, u + u_num - stage_size, sizeof(unsigned char) * stage_size);

    if(stage_size > 15){
      _mm256_zeroupper();
      volk_8u_x2_encodeframepolar_8u_u_ssse3(u_target, u_temp, stage_size);
    }
    else{
      volk_8u_x2_encodeframepolar_8u_generic(u_target, u_temp, stage_size);
    }

    src_llr_ptr = llrs + (max_stage_depth + 1) * frame_size + row - stage_size;
    dst_llr_ptr = llrs + max_stage_depth * frame_size + row;

    const __m128i zeros = _mm_set1_epi8(0x00);
    const __m128i sign_extract = _mm_set1_epi8(0x80);
    const __m128i shuffle_mask0 = _mm_setr_epi8(0xff, 0xff, 0xff, 0x00, 0xff, 0xff, 0xff, 0x01, 0xff, 0xff, 0xff, 0x02, 0xff, 0xff, 0xff, 0x03);
    const __m128i shuffle_mask1 = _mm_setr_epi8(0xff, 0xff, 0xff, 0x04, 0xff, 0xff, 0xff, 0x05, 0xff, 0xff, 0xff, 0x06, 0xff, 0xff, 0xff, 0x07);
    __m128i fbits, sign_bits0, sign_bits1;

    __m256 sign_mask;

    int p;
    for(p = 0; p < stage_size; p += 8){
      _mm256_zeroupper();
      fbits = _mm_loadu_si128((__m128i*) u_target);
      u_target += 8;

      // prepare sign mask for correct +-
      fbits = _mm_cmpgt_epi8(fbits, zeros);
      fbits = _mm_and_si128(fbits, sign_extract);
      sign_bits0 = _mm_shuffle_epi8(fbits, shuffle_mask0);
      sign_bits1 = _mm_shuffle_epi8(fbits, shuffle_mask1);


      src0 = _mm256_loadu_ps(src_llr_ptr);
      src1 = _mm256_loadu_ps(src_llr_ptr + 8);
      src_llr_ptr += 16;

      sign_mask = _mm256_insertf128_ps(sign_mask, _mm_castsi128_ps(sign_bits0), 0x0);
      sign_mask = _mm256_insertf128_ps(sign_mask, _mm_castsi128_ps(sign_bits1), 0x1);

      // deinterleave values
      part0 = _mm256_permute2f128_ps(src0, src1, 0x20);
      part1 = _mm256_permute2f128_ps(src0, src1, 0x31);
      llr0 = _mm256_shuffle_ps(part0, part1, 0x88);
      llr1 = _mm256_shuffle_ps(part0, part1, 0xdd);

      // calculate result
      llr0 = _mm256_xor_ps(llr0, sign_mask);
      dst = _mm256_add_ps(llr0, llr1);

      _mm256_storeu_ps(dst_llr_ptr, dst);
      dst_llr_ptr += 8;
    }

    --loop_stage;
    stage_size >>= 1;
  }

  const int min_stage = stage > 2 ? stage : 2;
  const __m256 sign_mask = _mm256_set1_ps(-0.0);
  const __m256 abs_mask = _mm256_andnot_ps(sign_mask, _mm256_castsi256_ps(_mm256_set1_epi8(0xff)));
  __m256 sign;

  int el;
  while(min_stage < loop_stage){
    dst_llr_ptr = llrs + loop_stage * frame_size + row;
    src_llr_ptr = dst_llr_ptr + frame_size;
    for(el = 0; el < stage_size; el += 8){
      src0 = _mm256_loadu_ps(src_llr_ptr);
      src_llr_ptr += 8;
      src1 = _mm256_loadu_ps(src_llr_ptr);
      src_llr_ptr += 8;

      // deinterleave values
      part0 = _mm256_permute2f128_ps(src0, src1, 0x20);
      part1 = _mm256_permute2f128_ps(src0, src1, 0x31);
      llr0 = _mm256_shuffle_ps(part0, part1, 0x88);
      llr1 = _mm256_shuffle_ps(part0, part1, 0xdd);

      // calculate result
      sign = _mm256_xor_ps(_mm256_and_ps(llr0, sign_mask), _mm256_and_ps(llr1, sign_mask));
      dst = _mm256_min_ps(_mm256_and_ps(llr0, abs_mask), _mm256_and_ps(llr1, abs_mask));
      dst = _mm256_or_ps(dst, sign);

      _mm256_storeu_ps(dst_llr_ptr, dst);
      dst_llr_ptr += 8;
    }

    --loop_stage;
    stage_size >>= 1;

  }

  // for stages < 3 vectors are too small!.
  llr_odd_stages(llrs, stage, loop_stage + 1,frame_size, row);
}

#endif /* LV_HAVE_AVX */

#endif /* VOLK_KERNELS_VOLK_VOLK_32F_8U_POLARBUTTERFLY_32F_H_ */
