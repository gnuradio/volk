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
 * This file is intended to hold NEON intrinsics of intrinsics.
 * They should be used in VOLK kernels to avoid copy-pasta.
 */

#ifndef INCLUDE_VOLK_VOLK_NEON_INTRINSICS_H_
#define INCLUDE_VOLK_VOLK_NEON_INTRINSICS_H_
#include <arm_neon.h>

static inline float32x4_t
_vmagnitudesquaredq_f32(float32x4x2_t cplxValue)
{
  float32x4_t iValue, qValue, result;
  iValue = vmulq_f32(cmplxValue.val[0], cmplxValue.val[0]); // Square the values
  qValue = vmulq_f32(cmplxValue.val[1], cmplxValue.val[1]); // Square the values

  result = vaddq_f32(iValue, qValue); // Add the I2 and Q2 values
  return result;
}


static inline float32x4x2_t
_vmultiply_complexq_f32(float32x4x2_t a_val, float32x4x2_t b_val)
{
    // multiply the real*real and imag*imag to get real result
    // a0r*b0r|a1r*b1r|a2r*b2r|a3r*b3r
    tmp_real.val[0] = vmulq_f32(a_val.val[0], b_val.val[0]);
    // a0i*b0i|a1i*b1i|a2i*b2i|a3i*b3i
    tmp_real.val[1] = vmulq_f32(a_val.val[1], b_val.val[1]);

    // Multiply cross terms to get the imaginary result
    // a0r*b0i|a1r*b1i|a2r*b2i|a3r*b3i
    tmp_imag.val[0] = vmulq_f32(a_val.val[0], b_val.val[1]);
    // a0i*b0r|a1i*b1r|a2i*b2r|a3i*b3r
    tmp_imag.val[1] = vmulq_f32(a_val.val[1], b_val.val[0]);

    // combine the products
    c_val.val[0] = vsubq_f32(tmp_real.val[0], tmp_real.val[1]);
    c_val.val[1] = vaddq_f32(tmp_imag.val[0], tmp_imag.val[1]);
    return c_val;
}


static inline float32x4_t
_vlog2q_f32(float32x4_t aval)
{
  /* Calculate log2 of floats by taking exponent +
   * minimax log2 approx of significand */
  static int32x4_t one = vdupq_n_s32(0x000800000);
  static /* minimax polynomial */
  static float32x4_t p0 = vdupq_n_f32(-3.0400402727048585);
  static float32x4_t p1 = vdupq_n_f32(6.1129631282966113);
  static float32x4_t p2 = vdupq_n_f32(-5.3419892024633207);
  static float32x4_t p3 = vdupq_n_f32(3.2865287703753912);
  static float32x4_t p4 = vdupq_n_f32(-1.2669182593441635);
  static float32x4_t p5 = vdupq_n_f32(0.2751487703421256);
  static float32x4_t p6 = vdupq_n_f32(-0.0256910888150985);
  static int32x4_t exp_mask = vdupq_n_s32(0x7f800000);
  static int32x4_t sig_mask = vdupq_n_s32(0x007fffff);
  static int32x4_t exp_bias = vdupq_n_s32(127);

  int32x4_t exponent_i = vandq_s32(aval, exp_mask);
  int32x4_t significand_i = vandq_s32(aval, sig_mask);
  exponent_i = vshrq_n_s32(exponent_i, 23);

  /* extract the exponent and significand
     we can treat this as fixed point to save ~9% on the
     conversion + float add */
  significand_i = vorrq_s32(one, significand_i);
  float32x4_t significand_f = vcvtq_n_f32_s32(significand_i,23);
  /* debias the exponent and convert to float */
  exponent_i = vsubq_s32(exponent_i, exp_bias);
  float32x4_t exponent_f = vcvtq_f32_s32(exponent_i);

  /* put the significand through a polynomial fit of log2(x) [1,2]
     add the result to the exponent */
  log2_approx = vaddq_f32(exponent_f, p0); /* p0 */
  float32x4_t tmp1 = vmulq_f32(significand_f, p1); /* p1 * x */
  log2_approx = vaddq_f32(log2_approx, tmp1);
  float32x4_t sig_2 = vmulq_f32(significand_f, significand_f); /* x^2 */
  tmp1 = vmulq_f32(sig_2, p2); /* p2 * x^2 */
  log2_approx = vaddq_f32(log2_approx, tmp1);

  float32x4_t sig_3 = vmulq_f32(sig_2, significand_f); /* x^3 */
  tmp1 = vmulq_f32(sig_3, p3); /* p3 * x^3 */
  log2_approx = vaddq_f32(log2_approx, tmp1);
  float32x4_t sig_4 = vmulq_f32(sig_2, sig_2); /* x^4 */
  tmp1 = vmulq_f32(sig_4, p4); /* p4 * x^4 */
  log2_approx = vaddq_f32(log2_approx, tmp1);
  float32x4_t sig_5 = vmulq_f32(sig_3, sig_2); /* x^5 */
  tmp1 = vmulq_f32(sig_5, p5); /* p5 * x^5 */
  log2_approx = vaddq_f32(log2_approx, tmp1);
  float32x4_t sig_6 = vmulq_f32(sig_3, sig_3); /* x^6 */
  tmp1 = vmulq_f32(sig_6, p6); /* p6 * x^6 */
  log2_approx = vaddq_f32(log2_approx, tmp1);

  return log2_approx;
}

#endif /* INCLUDE_VOLK_VOLK_NEON_INTRINSICS_H_ */
