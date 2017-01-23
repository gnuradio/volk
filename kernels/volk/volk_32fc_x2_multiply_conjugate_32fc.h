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
 * \page volk_32fc_x2_multiply_conjugate_32fc
 *
 * \b Overview
 *
 * Multiplies a complex vector by the conjugate of a secod complex
 * vector and returns the complex result.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_x2_multiply_conjugate_32fc(lv_32fc_t* cVector, const lv_32fc_t* aVector, const lv_32fc_t* bVector, unsigned int num_points);
 * \endcode
 *
 * \b Inputs
 * \li aVector: The first input vector of complex floats.
 * \li bVector: The second input vector of complex floats that is conjugated.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li outputVector: The output vector complex floats.
 *
 * \b Example
 * Calculate mag^2 of a signal using x * conj(x).
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   lv_32fc_t* sig_1  = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *   lv_32fc_t* out = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *
 *   float delta = 2.f*M_PI / (float)N;
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       float real_1 = std::cos(0.3f * (float)ii);
 *       float imag_1 = std::sin(0.3f * (float)ii);
 *       sig_1[ii] = lv_cmake(real_1, imag_1);
 *   }
 *
 *   volk_32fc_x2_multiply_conjugate_32fc(out, sig_1, sig_1, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("%1.4f%+1.4fj,", lv_creal(out[ii]), lv_cimag(out[ii]));
 *   }
 *   printf("\n");
 *
 *   volk_free(sig_1);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_x2_multiply_conjugate_32fc_u_H
#define INCLUDED_volk_32fc_x2_multiply_conjugate_32fc_u_H

#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_complex.h>
#include <float.h>

#ifdef LV_HAVE_AVX
#include <immintrin.h>
#include <volk/volk_avx_intrinsics.h>

static inline void
volk_32fc_x2_multiply_conjugate_32fc_u_avx(lv_32fc_t* cVector, const lv_32fc_t* aVector,
                                           const lv_32fc_t* bVector, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int quarterPoints = num_points / 4;

  __m256 x, y, z;
  lv_32fc_t* c = cVector;
  const lv_32fc_t* a = aVector;
  const lv_32fc_t* b = bVector;

  for(; number < quarterPoints; number++){
    x = _mm256_loadu_ps((float*) a); // Load the ar + ai, br + bi ... as ar,ai,br,bi ...
    y = _mm256_loadu_ps((float*) b); // Load the cr + ci, dr + di ... as cr,ci,dr,di ...
    z = _mm256_complexconjugatemul_ps(x, y);
    _mm256_storeu_ps((float*) c, z); // Store the results back into the C container

    a += 4;
    b += 4;
    c += 4;
  }

  number = quarterPoints * 4;

  for(; number < num_points; number++){
    *c++ = (*a++) * lv_conj(*b++);
  }
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>
#include <volk/volk_sse3_intrinsics.h>

static inline void
volk_32fc_x2_multiply_conjugate_32fc_u_sse3(lv_32fc_t* cVector, const lv_32fc_t* aVector,
                                            const lv_32fc_t* bVector, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int halfPoints = num_points / 2;

  __m128 x, y, z;
  lv_32fc_t* c = cVector;
  const lv_32fc_t* a = aVector;
  const lv_32fc_t* b = bVector;

  for(; number < halfPoints; number++){
    x = _mm_loadu_ps((float*) a); // Load the ar + ai, br + bi as ar,ai,br,bi
    y = _mm_loadu_ps((float*) b); // Load the cr + ci, dr + di as cr,ci,dr,di
    z = _mm_complexconjugatemul_ps(x, y);
    _mm_storeu_ps((float*) c, z); // Store the results back into the C container

    a += 2;
    b += 2;
    c += 2;
  }

  if((num_points % 2) != 0){
    *c = (*a) * lv_conj(*b);
  }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_GENERIC

static inline void
volk_32fc_x2_multiply_conjugate_32fc_generic(lv_32fc_t* cVector, const lv_32fc_t* aVector,
                                             const lv_32fc_t* bVector, unsigned int num_points)
{
  lv_32fc_t* cPtr = cVector;
  const lv_32fc_t* aPtr = aVector;
  const lv_32fc_t* bPtr=  bVector;
  unsigned int number = 0;

  for(number = 0; number < num_points; number++){
    *cPtr++ = (*aPtr++) * lv_conj(*bPtr++);
  }
}
#endif /* LV_HAVE_GENERIC */



#endif /* INCLUDED_volk_32fc_x2_multiply_conjugate_32fc_u_H */
#ifndef INCLUDED_volk_32fc_x2_multiply_conjugate_32fc_a_H
#define INCLUDED_volk_32fc_x2_multiply_conjugate_32fc_a_H

#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_complex.h>
#include <float.h>

#ifdef LV_HAVE_AVX
#include <immintrin.h>
#include <volk/volk_avx_intrinsics.h>

static inline void
volk_32fc_x2_multiply_conjugate_32fc_a_avx(lv_32fc_t* cVector, const lv_32fc_t* aVector,
                                           const lv_32fc_t* bVector, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int quarterPoints = num_points / 4;

  __m256 x, y, z;
  lv_32fc_t* c = cVector;
  const lv_32fc_t* a = aVector;
  const lv_32fc_t* b = bVector;

  for(; number < quarterPoints; number++){
    x = _mm256_load_ps((float*) a); // Load the ar + ai, br + bi ... as ar,ai,br,bi ...
    y = _mm256_load_ps((float*) b); // Load the cr + ci, dr + di ... as cr,ci,dr,di ...
    z = _mm256_complexconjugatemul_ps(x, y);
    _mm256_store_ps((float*) c, z); // Store the results back into the C container

    a += 4;
    b += 4;
    c += 4;
  }

  number = quarterPoints * 4;

  for(; number < num_points; number++){
    *c++ = (*a++) * lv_conj(*b++);
  }
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>
#include <volk/volk_sse3_intrinsics.h>

static inline void
volk_32fc_x2_multiply_conjugate_32fc_a_sse3(lv_32fc_t* cVector, const lv_32fc_t* aVector,
                                            const lv_32fc_t* bVector, unsigned int num_points)
{
  unsigned int number = 0;
  const unsigned int halfPoints = num_points / 2;

  __m128 x, y, z;
  lv_32fc_t* c = cVector;
  const lv_32fc_t* a = aVector;
  const lv_32fc_t* b = bVector;

  for(; number < halfPoints; number++){
    x = _mm_load_ps((float*) a); // Load the ar + ai, br + bi as ar,ai,br,bi
    y = _mm_load_ps((float*) b); // Load the cr + ci, dr + di as cr,ci,dr,di
    z = _mm_complexconjugatemul_ps(x, y);
    _mm_store_ps((float*) c, z); // Store the results back into the C container

    a += 2;
    b += 2;
    c += 2;
  }

  if((num_points % 2) != 0){
    *c = (*a) * lv_conj(*b);
  }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void
volk_32fc_x2_multiply_conjugate_32fc_neon(lv_32fc_t* cVector, const lv_32fc_t* aVector,
                                          const lv_32fc_t* bVector, unsigned int num_points)
{
  lv_32fc_t *a_ptr = (lv_32fc_t*) aVector;
  lv_32fc_t *b_ptr = (lv_32fc_t*) bVector;
  unsigned int quarter_points = num_points / 4;
  float32x4x2_t a_val, b_val, c_val;
  float32x4x2_t tmp_real, tmp_imag;
  unsigned int number = 0;

  for(number = 0; number < quarter_points; ++number) {
    a_val = vld2q_f32((float*)a_ptr); // a0r|a1r|a2r|a3r || a0i|a1i|a2i|a3i
    b_val = vld2q_f32((float*)b_ptr); // b0r|b1r|b2r|b3r || b0i|b1i|b2i|b3i
    b_val.val[1] = vnegq_f32(b_val.val[1]);
    __VOLK_PREFETCH(a_ptr+4);
    __VOLK_PREFETCH(b_ptr+4);

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

    // store the results
    c_val.val[0] = vsubq_f32(tmp_real.val[0], tmp_real.val[1]);
    c_val.val[1] = vaddq_f32(tmp_imag.val[0], tmp_imag.val[1]);
    vst2q_f32((float*)cVector, c_val);

    a_ptr += 4;
    b_ptr += 4;
    cVector += 4;
    }

  for(number = quarter_points*4; number < num_points; number++){
    *cVector++ = (*a_ptr++) * conj(*b_ptr++);
  }
}
#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_GENERIC

static inline void
volk_32fc_x2_multiply_conjugate_32fc_a_generic(lv_32fc_t* cVector, const lv_32fc_t* aVector,
                                               const lv_32fc_t* bVector, unsigned int num_points)
{
  lv_32fc_t* cPtr = cVector;
  const lv_32fc_t* aPtr = aVector;
  const lv_32fc_t* bPtr=  bVector;
  unsigned int number = 0;

  for(number = 0; number < num_points; number++){
    *cPtr++ = (*aPtr++) * lv_conj(*bPtr++);
  }
}
#endif /* LV_HAVE_GENERIC */


#endif /* INCLUDED_volk_32fc_x2_multiply_conjugate_32fc_a_H */
