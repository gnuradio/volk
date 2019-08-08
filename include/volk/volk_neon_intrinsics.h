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

/* Copyright (C) 2011  Julien Pommier
 Copyright (C) 2019  Albin Stigo
 
 This software is provided 'as-is', without any express or implied
 warranty.  In no event will the authors be held liable for any damages
 arising from the use of this software.
 
 Permission is granted to anyone to use this software for any purpose,
 including commercial applications, and to alter it and redistribute it
 freely, subject to the following restrictions:
 
 1. The origin of this software must not be misrepresented; you must not
 claim that you wrote the original software. If you use this software
 in a product, an acknowledgment in the product documentation would be
 appreciated but is not required.
 2. Altered source versions must be plainly marked as such, and must not be
 misrepresented as being the original software.
 3. This notice may not be removed or altered from any source distribution.
 
 (this is the zlib license)
 */

/*
 * This file is intended to hold NEON intrinsics of intrinsics.
 * They should be used in VOLK kernels to avoid copy-pasta.
 */

#ifndef INCLUDE_VOLK_VOLK_NEON_INTRINSICS_H_
#define INCLUDE_VOLK_VOLK_NEON_INTRINSICS_H_
#include <arm_neon.h>


/* Magnitude squared for float32x4x2_t */
static inline float32x4_t
_vmagnitudesquaredq_f32(float32x4x2_t cmplxValue)
{
    float32x4_t iValue, qValue, result;
    iValue = vmulq_f32(cmplxValue.val[0], cmplxValue.val[0]); // Square the values
    qValue = vmulq_f32(cmplxValue.val[1], cmplxValue.val[1]); // Square the values
    result = vaddq_f32(iValue, qValue); // Add the I2 and Q2 values
    return result;
}

/* Inverse square root for float32x4_t */
static inline float32x4_t _vinvsqrtq_f32(float32x4_t x)
{
    float32x4_t sqrt_reciprocal = vrsqrteq_f32(x);
    sqrt_reciprocal = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x, sqrt_reciprocal), sqrt_reciprocal), sqrt_reciprocal);
    sqrt_reciprocal = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x, sqrt_reciprocal), sqrt_reciprocal), sqrt_reciprocal);
    
    return sqrt_reciprocal;
}

/* Complex multiplication for float32x4x2_t */
static inline float32x4x2_t
_vmultiply_complexq_f32(float32x4x2_t a_val, float32x4x2_t b_val)
{
    float32x4x2_t tmp_real;
    float32x4x2_t tmp_imag;
    float32x4x2_t c_val;
    
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

/* Inverse (1/x) */
static inline float32x4_t _vinvq_f32(float32x4_t x)
{
    // Newton's method
    float32x4_t recip = vrecpeq_f32(x);
    recip             = vmulq_f32(vrecpsq_f32(x, recip), recip);
    recip             = vmulq_f32(vrecpsq_f32(x, recip), recip);
    return recip;
}

/* Evaluation of 4 sines & cosines at once.
 * Optimized from here (zlib license)
 * http://gruntthepeon.free.fr/ssemath/ */
static inline void _vsincosq_f32(float32x4_t x,
                                 float32x4_t *ysin,
                                 float32x4_t *ycos) {
    float32x4_t y;
    
    const float32x4_t c_minus_cephes_DP1 = vdupq_n_f32(-0.78515625);
    const float32x4_t c_minus_cephes_DP2 = vdupq_n_f32(-2.4187564849853515625e-4);
    const float32x4_t c_minus_cephes_DP3 = vdupq_n_f32(-3.77489497744594108e-8);
    const float32x4_t c_sincof_p0 = vdupq_n_f32(-1.9515295891e-4);
    const float32x4_t c_sincof_p1  = vdupq_n_f32(8.3321608736e-3);
    const float32x4_t c_sincof_p2 = vdupq_n_f32(-1.6666654611e-1);
    const float32x4_t c_coscof_p0 = vdupq_n_f32(2.443315711809948e-005);
    const float32x4_t c_coscof_p1 = vdupq_n_f32(-1.388731625493765e-003);
    const float32x4_t c_coscof_p2 = vdupq_n_f32(4.166664568298827e-002);
    const float32x4_t c_cephes_FOPI = vdupq_n_f32(1.27323954473516); // 4 / M_PI
    
    const float32x4_t CONST_1 = vdupq_n_f32(1.f);
    const float32x4_t CONST_1_2 = vdupq_n_f32(0.5f);
    const float32x4_t CONST_0 = vdupq_n_f32(0.f);
    const uint32x4_t  CONST_2 = vdupq_n_u32(2);
    const uint32x4_t  CONST_4 = vdupq_n_u32(4);
    
    uint32x4_t emm2;
    
    uint32x4_t sign_mask_sin, sign_mask_cos;
    sign_mask_sin = vcltq_f32(x, CONST_0);
    x = vabsq_f32(x);
    // scale by 4/pi
    y = vmulq_f32(x, c_cephes_FOPI);
    
    // store the integer part of y in mm0
    emm2 = vcvtq_u32_f32(y);
    /* j=(j+1) & (~1) (see the cephes sources) */
    emm2 = vaddq_u32(emm2, vdupq_n_u32(1));
    emm2 = vandq_u32(emm2, vdupq_n_u32(~1));
    y = vcvtq_f32_u32(emm2);
    
    /* get the polynom selection mask
     there is one polynom for 0 <= x <= Pi/4
     and another one for Pi/4<x<=Pi/2
     Both branches will be computed. */
    const uint32x4_t poly_mask = vtstq_u32(emm2, CONST_2);
    
    // The magic pass: "Extended precision modular arithmetic"
    x = vmlaq_f32(x, y, c_minus_cephes_DP1);
    x = vmlaq_f32(x, y, c_minus_cephes_DP2);
    x = vmlaq_f32(x, y, c_minus_cephes_DP3);
    
    sign_mask_sin = veorq_u32(sign_mask_sin, vtstq_u32(emm2, CONST_4));
    sign_mask_cos = vtstq_u32(vsubq_u32(emm2, CONST_2), CONST_4);
    
    /* Evaluate the first polynom  (0 <= x <= Pi/4) in y1,
     and the second polynom      (Pi/4 <= x <= 0) in y2 */
    
    float32x4_t y1, y2;
    float32x4_t z = vmulq_f32(x,x);
    
    y1 = vmlaq_f32(c_coscof_p1, z, c_coscof_p0);
    y1 = vmlaq_f32(c_coscof_p2, z, y1);
    y1 = vmulq_f32(y1, z);
    y1 = vmulq_f32(y1, z);
    y1 = vmlsq_f32(y1, z, CONST_1_2);
    y1 = vaddq_f32(y1, CONST_1);
    
    y2 = vmlaq_f32(c_sincof_p1, z, c_sincof_p0);
    y2 = vmlaq_f32(c_sincof_p2, z, y2);
    y2 = vmulq_f32(y2, z);
    y2 = vmlaq_f32(x, x, y2);
    
    /* select the correct result from the two polynoms */
    const float32x4_t ys = vbslq_f32(poly_mask, y1, y2);
    const float32x4_t yc = vbslq_f32(poly_mask, y2, y1);
    *ysin = vbslq_f32(sign_mask_sin, vnegq_f32(ys), ys);
    *ycos = vbslq_f32(sign_mask_cos, yc, vnegq_f32(yc));
}

static inline float32x4_t _vsinq_f32(float32x4_t x) {
    float32x4_t ysin, ycos;
    _vsincosq_f32(x, &ysin, &ycos);
    return ysin;
}

static inline float32x4_t _vcosq_f32(float32x4_t x) {
    float32x4_t ysin, ycos;
    _vsincosq_f32(x, &ysin, &ycos);
    return ycos;
}

static inline float32x4_t _vtanq_f32(float32x4_t x) {
    float32x4_t ysin, ycos;
    _vsincosq_f32(x, &ysin, &ycos);
    return vmulq_f32(ysin, _vinvq_f32(ycos));
}


#endif /* INCLUDE_VOLK_VOLK_NEON_INTRINSICS_H_ */
