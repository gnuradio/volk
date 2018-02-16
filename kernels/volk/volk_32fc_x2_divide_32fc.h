/* -*- c++ -*- */
/*
 * Copyright 2016 Free Software Foundation, Inc.
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
 * \page volk_32fc_x2_divide_32fc
 *
 * \b Overview
 *
 * Divide first vector of complexes element-wise by second.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_x2_divide_32fc(lv_32fc_t* cVector, const lv_32fc_t* numeratorVector, const lv_32fc_t* denumeratorVector, unsigned int num_points);
 * \endcode
 *
 * \b Inputs
 * \li numeratorVector: The numerator complex values.
 * \li numeratorVector: The denumerator complex values.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li outputVector: The output vector complex floats.
 *
 * \b Example
 * divide a complex vector by itself, demonstrating the result should be pretty close to 1+0j.
 *
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   lv_32fc_t* input_vector  = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *   lv_32fc_t* out = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *
 *   float delta = 2.f*M_PI / (float)N;
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       float real_1 = std::cos(0.3f * (float)ii);
 *       float imag_1 = std::sin(0.3f * (float)ii);
 *       input_vector[ii] = lv_cmake(real_1, imag_1);
 *   }
 *
 *   volk_32fc_x2_divide_32fc(out, input_vector, input_vector, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("%1.4f%+1.4fj,", lv_creal(out[ii]), lv_cimag(out[ii]));
 *   }
 *   printf("\n");
 *
 *   volk_free(input_vector);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_x2_divide_32fc_u_H
#define INCLUDED_volk_32fc_x2_divide_32fc_u_H

#include <inttypes.h>
#include <volk/volk_complex.h>
#include <float.h>

#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>
#include <volk/volk_sse3_intrinsics.h>

static inline void
volk_32fc_x2_divide_32fc_u_sse3(lv_32fc_t* cVector, const lv_32fc_t* numeratorVector,
                                            const lv_32fc_t* denumeratorVector, unsigned int num_points)
{
    /*
     * we'll do the "classical"
     *  a      a b*
     * --- = -------
     *  b     |b|^2
     * */
  unsigned int number = 0;
  const unsigned int quarterPoints = num_points / 4;

  __m128 num01, num23, den01, den23, norm, result;
  lv_32fc_t* c = cVector;
  const lv_32fc_t* a = numeratorVector;
  const lv_32fc_t* b = denumeratorVector;

  for(; number < quarterPoints; number++){
    num01 = _mm_loadu_ps((float*) a);    // first pair
    den01 = _mm_loadu_ps((float*) b);    // first pair
    num01 = _mm_complexconjugatemul_ps(num01, den01);   // a conj(b)
    a += 2;
    b += 2;

    num23 = _mm_loadu_ps((float*) a);    // second pair
    den23 = _mm_loadu_ps((float*) b);    // second pair
    num23 = _mm_complexconjugatemul_ps(num23, den23);   // a conj(b)
    a += 2;
    b += 2;

    norm = _mm_magnitudesquared_ps_sse3(den01, den23);
    den01 = _mm_unpacklo_ps(norm,norm);
    den23 = _mm_unpackhi_ps(norm,norm);

    result = _mm_div_ps(num01, den01);
    _mm_storeu_ps((float*) c, result); // Store the results back into the C container
    c += 2;
    result = _mm_div_ps(num23, den23);
    _mm_storeu_ps((float*) c, result); // Store the results back into the C container
    c += 2;
  }

  number *= 4;
  for(;number < num_points; number++){
    *c = (*a) / (*b);
    a++; b++; c++;
  }
}
#endif /* LV_HAVE_SSE3 */


#ifdef LV_HAVE_AVX
#include <immintrin.h>
#include <volk/volk_avx_intrinsics.h>

static inline void
volk_32fc_x2_divide_32fc_u_avx(lv_32fc_t* cVector, const lv_32fc_t* numeratorVector,
                                            const lv_32fc_t* denumeratorVector, unsigned int num_points)
{
    /*
     * we'll do the "classical"
     *  a      a b*
     * --- = -------
     *  b     |b|^2
     * */
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    __m256 num, denum, mul_conj, sq, mag_sq, mag_sq_un, div;
    lv_32fc_t* c = cVector;
    const lv_32fc_t* a = numeratorVector;
    const lv_32fc_t* b = denumeratorVector;

    for(; number < quarterPoints; number++){
        num = _mm256_loadu_ps((float*) a); // Load the ar + ai, br + bi ... as ar,ai,br,bi ...
        denum = _mm256_loadu_ps((float*) b); // Load the cr + ci, dr + di ... as cr,ci,dr,di ...
        mul_conj = _mm256_complexconjugatemul_ps(num, denum);
        sq = _mm256_mul_ps(denum, denum); // Square the values
        mag_sq_un = _mm256_hadd_ps(sq,sq); // obtain the actual squared magnitude, although out of order
        mag_sq = _mm256_permute_ps(mag_sq_un, 0xd8); // I order them
        // best guide I found on using these functions: https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=2738,2059,2738,2738,3875,3874,3875,2738,3870
        div = _mm256_div_ps(mul_conj,mag_sq);

        _mm256_storeu_ps((float*) c, div); // Store the results back into the C container

        a += 4;
        b += 4;
        c += 4;
    }

    number = quarterPoints * 4;

    for(; number < num_points; number++){
        *c++ = (*a++) / (*b++);
    }

}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_GENERIC

static inline void
volk_32fc_x2_divide_32fc_generic(lv_32fc_t* cVector, const lv_32fc_t* aVector,
                                             const lv_32fc_t* bVector, unsigned int num_points)
{
  lv_32fc_t* cPtr = cVector;
  const lv_32fc_t* aPtr = aVector;
  const lv_32fc_t* bPtr=  bVector;
  unsigned int number = 0;

  for(number = 0; number < num_points; number++){
    *cPtr++ = (*aPtr++) / (*bPtr++);
  }
}
#endif /* LV_HAVE_GENERIC */



#endif /* INCLUDED_volk_32fc_x2_divide_32fc_u_H */


#ifndef INCLUDED_volk_32fc_x2_divide_32fc_a_H
#define INCLUDED_volk_32fc_x2_divide_32fc_a_H

#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_complex.h>
#include <float.h>

#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>
#include <volk/volk_sse3_intrinsics.h>

static inline void
volk_32fc_x2_divide_32fc_a_sse3(lv_32fc_t* cVector, const lv_32fc_t* numeratorVector,
                                            const lv_32fc_t* denumeratorVector, unsigned int num_points)
{
    /*
     * we'll do the "classical"
     *  a      a b*
     * --- = -------
     *  b     |b|^2
     * */
  unsigned int number = 0;
  const unsigned int quarterPoints = num_points / 4;

  __m128 num01, num23, den01, den23, norm, result;
  lv_32fc_t* c = cVector;
  const lv_32fc_t* a = numeratorVector;
  const lv_32fc_t* b = denumeratorVector;

  for(; number < quarterPoints; number++){
    num01 = _mm_load_ps((float*) a);    // first pair
    den01 = _mm_load_ps((float*) b);    // first pair
    num01 = _mm_complexconjugatemul_ps(num01, den01);   // a conj(b)
    a += 2;
    b += 2;

    num23 = _mm_load_ps((float*) a);    // second pair
    den23 = _mm_load_ps((float*) b);    // second pair
    num23 = _mm_complexconjugatemul_ps(num23, den23);   // a conj(b)
    a += 2;
    b += 2;

    norm = _mm_magnitudesquared_ps_sse3(den01, den23);

    den01 = _mm_unpacklo_ps(norm,norm); // select the lower floats twice
    den23 = _mm_unpackhi_ps(norm,norm); // select the upper floats twice

    result = _mm_div_ps(num01, den01);
    _mm_store_ps((float*) c, result); // Store the results back into the C container
    c += 2;
    result = _mm_div_ps(num23, den23);
    _mm_store_ps((float*) c, result); // Store the results back into the C container
    c += 2;
  }

  number *= 4;
  for(;number < num_points; number++){
    *c = (*a) / (*b);
    a++; b++; c++;
  }
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_AVX
#include <immintrin.h>
#include <volk/volk_avx_intrinsics.h>

static inline void
volk_32fc_x2_divide_32fc_a_avx(lv_32fc_t* cVector, const lv_32fc_t* numeratorVector,
                                            const lv_32fc_t* denumeratorVector, unsigned int num_points)
{
    /*
     * we'll do the "classical"
     *  a      a b*
     * --- = -------
     *  b     |b|^2
     * */
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    __m256 num, denum, mul_conj, sq, mag_sq, mag_sq_un, div;
    lv_32fc_t* c = cVector;
    const lv_32fc_t* a = numeratorVector;
    const lv_32fc_t* b = denumeratorVector;

    for(; number < quarterPoints; number++){
        num = _mm256_load_ps((float*) a); // Load the ar + ai, br + bi ... as ar,ai,br,bi ...
        denum = _mm256_load_ps((float*) b); // Load the cr + ci, dr + di ... as cr,ci,dr,di ...
        mul_conj = _mm256_complexconjugatemul_ps(num, denum);
        sq = _mm256_mul_ps(denum, denum); // Square the values
        mag_sq_un = _mm256_hadd_ps(sq,sq); // obtain the actual squared magnitude, although out of order
        mag_sq = _mm256_permute_ps(mag_sq_un, 0xd8); // I order them
        // best guide I found on using these functions: https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=2738,2059,2738,2738,3875,3874,3875,2738,3870
        div = _mm256_div_ps(mul_conj,mag_sq);

        _mm256_store_ps((float*) c, div); // Store the results back into the C container

        a += 4;
        b += 4;
        c += 4;
    }

    number = quarterPoints * 4;

    for(; number < num_points; number++){
        *c++ = (*a++) / (*b++);
    }


}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void
volk_32fc_x2_divide_32fc_neon(lv_32fc_t* cVector, const lv_32fc_t* aVector,
			      const lv_32fc_t* bVector, unsigned int num_points)
{
  lv_32fc_t* cPtr = cVector;
  const lv_32fc_t* aPtr = aVector;
  const lv_32fc_t* bPtr = bVector;

  float32x4x2_t aVal, bVal, cVal;
  float32x4_t bAbs, bAbsInv;

  const unsigned int quarterPoints = num_points / 4;
  unsigned int number = 0;
  for(; number < quarterPoints; number++){
    aVal = vld2q_f32((const float*)(aPtr));
    bVal = vld2q_f32((const float*)(bPtr));
    aPtr += 4;
    bPtr += 4;
    __VOLK_PREFETCH(aPtr+4);
    __VOLK_PREFETCH(bPtr+4);

    bAbs = vmulq_f32(      bVal.val[0], bVal.val[0]);
    bAbs = vmlaq_f32(bAbs, bVal.val[1], bVal.val[1]);

    bAbsInv = vrecpeq_f32(bAbs);
    bAbsInv = vmulq_f32(bAbsInv, vrecpsq_f32(bAbsInv, bAbs));
    bAbsInv = vmulq_f32(bAbsInv, vrecpsq_f32(bAbsInv, bAbs));

    cVal.val[0] = vmulq_f32(             aVal.val[0], bVal.val[0]);
    cVal.val[0] = vmlaq_f32(cVal.val[0], aVal.val[1], bVal.val[1]);
    cVal.val[0] = vmulq_f32(cVal.val[0], bAbsInv);

    cVal.val[1] = vmulq_f32(             aVal.val[1], bVal.val[0]);
    cVal.val[1] = vmlsq_f32(cVal.val[1], aVal.val[0], bVal.val[1]);
    cVal.val[1] = vmulq_f32(cVal.val[1], bAbsInv);

    vst2q_f32((float*)(cPtr), cVal);
    cPtr += 4;
  }

  for(number = quarterPoints * 4; number < num_points; number++){
    *cPtr++ = (*aPtr++) / (*bPtr++);
  }
}
#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_GENERIC

static inline void
volk_32fc_x2_divide_32fc_a_generic(lv_32fc_t* cVector, const lv_32fc_t* aVector,
                                               const lv_32fc_t* bVector, unsigned int num_points)
{
  lv_32fc_t* cPtr = cVector;
  const lv_32fc_t* aPtr = aVector;
  const lv_32fc_t* bPtr=  bVector;
  unsigned int number = 0;

  for(number = 0; number < num_points; number++){
    *cPtr++ = (*aPtr++)  / (*bPtr++);
  }
}
#endif /* LV_HAVE_GENERIC */


#endif /* INCLUDED_volk_32fc_x2_divide_32fc_a_H */
