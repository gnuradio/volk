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

/*
 * This file is intended to hold AVX2 intrinsics of intrinsics.
 * They should be used in VOLK kernels to avoid copy-paste.
 */

#ifndef INCLUDE_VOLK_VOLK_AVX2_INTRINSICS_H_
#define INCLUDE_VOLK_VOLK_AVX2_INTRINSICS_H_
#include <immintrin.h>
#include "volk/volk_avx_intrinsics.h"

static inline __m256
_mm256_polar_sign_mask_avx2(__m128i fbits){
  const __m128i zeros = _mm_set1_epi8(0x00);
  const __m128i sign_extract = _mm_set1_epi8(0x80);
  const __m256i shuffle_mask = _mm256_setr_epi8(0xff, 0xff, 0xff, 0x00, 0xff, 0xff, 0xff, 0x01, 0xff, 0xff, 0xff, 0x02, 0xff, 0xff, 0xff, 0x03,
                                                 0xff, 0xff, 0xff, 0x04, 0xff, 0xff, 0xff, 0x05, 0xff, 0xff, 0xff, 0x06, 0xff, 0xff, 0xff, 0x07);
  __m256i sign_bits = _mm256_setzero_si256();
  
  fbits = _mm_cmpgt_epi8(fbits, zeros);
  fbits = _mm_and_si128(fbits, sign_extract);
  sign_bits = _mm256_insertf128_si256(sign_bits,fbits,0);
  sign_bits = _mm256_insertf128_si256(sign_bits,fbits,1);
  sign_bits = _mm256_shuffle_epi8(sign_bits, shuffle_mask);

  return _mm256_castsi256_ps(sign_bits);
}

static inline __m256
_mm256_polar_fsign_add_llrs_avx2(__m256 src0, __m256 src1, __m128i fbits){
    // prepare sign mask for correct +-
    __m256 sign_mask = _mm256_polar_sign_mask_avx2(fbits);

    __m256 llr0, llr1;
    _mm256_polar_deinterleave(&llr0, &llr1, src0, src1);

    // calculate result
    llr0 = _mm256_xor_ps(llr0, sign_mask);
    __m256 dst = _mm256_add_ps(llr0, llr1);
    return dst;
}

static inline __m256
_mm256_magnitudesquared_ps_avx2(const __m256 cplxValue0, const __m256 cplxValue1){
  const __m256i idx = _mm256_set_epi32(7,6,3,2,5,4,1,0);
  const __m256 squared0 = _mm256_mul_ps(cplxValue0, cplxValue0); // Square the values
  const __m256 squared1 = _mm256_mul_ps(cplxValue1, cplxValue1); // Square the Values
  const __m256 complex_result = _mm256_hadd_ps(squared0, squared1);
  return _mm256_permutevar8x32_ps(complex_result, idx);
}

static inline __m256
_mm256_scaled_norm_dist_ps_avx2(const __m256 symbols0, const __m256 symbols1, const __m256 points0, const __m256 points1, const __m256 scalar){
  /*
   * Calculate: |y - x|^2 * SNR_lin
   * Consider 'symbolsX' and 'pointsX' to be complex float
   * 'symbolsX' are 'y' and 'pointsX' are 'x'
   */
  const __m256 diff0 = _mm256_sub_ps(symbols0, points0);
  const __m256 diff1 = _mm256_sub_ps(symbols1, points1);
  const __m256 norms = _mm256_magnitudesquared_ps_avx2(diff0, diff1);
  return _mm256_mul_ps(norms, scalar);
}

#endif /* INCLUDE_VOLK_VOLK_AVX2_INTRINSICS_H_ */
