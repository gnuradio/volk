/* -*- c++ -*- */
/*
 * Copyright 2012, 2014, 2016 Free Software Foundation, Inc.
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
  const unsigned int eighthPoints = num_points / 8;

  __m256 num0123, num4567, den0123, den4567, norm, result;
  __m256 tmp0123, tmp4567;
  lv_32fc_t* c = cVector;
  const lv_32fc_t* a = numeratorVector;
  const lv_32fc_t* b = denumeratorVector;

  for(; number < eighthPoints; number++){
    num0123 = _mm256_loadu_ps((float*) a);    // first quad
    den0123 = _mm256_loadu_ps((float*) b);    // first quad
    num0123 = _mm256_complexconjugatemul_ps(num0123, den0123);   // a conj(b)
    a += 4;
    b += 4;

    num4567 = _mm256_loadu_ps((float*) a);    // second quad
    den4567 = _mm256_loadu_ps((float*) b);    // second quad
    num4567 = _mm256_complexconjugatemul_ps(num4567, den4567);   // a conj(b)
    a += 4;
    b += 4;

    norm = _mm256_magnitudesquared_ps(den0123, den4567);

    tmp0123 = _mm256_permute_ps(norm, _MM_SHUFFLE(1,1,0,0));
    tmp4567 = _mm256_permute_ps(norm, _MM_SHUFFLE(3,3,2,2));
    den0123 = _mm256_permute2f128_ps(tmp0123, tmp4567, (0<<0)| (2<<4)); // select the lower 128bit from tmp0123, lower from tmp4567
    den4567 = _mm256_permute2f128_ps(tmp0123, tmp4567, (1<<0)| (3<<4)); // select the upper 128bit from tmp0123, upper from tmp4567

    result = _mm256_div_ps(num0123, den0123);
    _mm256_storeu_ps((float*) c, result); // Store the results back into the C container
    c += 4;
    result = _mm256_div_ps(num4567, den4567);
    _mm256_storeu_ps((float*) c, result); // Store the results back into the C container
    c += 4;
  }

  number *= 8;
  for(;number < num_points; number++){
    *c = (*a) / (*b);
    a++; b++; c++;
  }

}
#endif /* LV_HAVE_AVX */


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
#endif /* LV_HAVE_SSE */


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
  const unsigned int eighthPoints = num_points / 8;

  __m256 num0123, num4567, den0123, den4567, norm, result;
  __m256 tmp0123, tmp4567;
  lv_32fc_t* c = cVector;
  const lv_32fc_t* a = numeratorVector;
  const lv_32fc_t* b = denumeratorVector;

  for(; number < eighthPoints; number++){
    num0123 = _mm256_load_ps((float*) a);    // first quad
    den0123 = _mm256_load_ps((float*) b);    // first quad
    num0123 = _mm256_complexconjugatemul_ps(num0123, den0123);   // a conj(b)
    a += 4;
    b += 4;

    num4567 = _mm256_load_ps((float*) a);    // second quad
    den4567 = _mm256_load_ps((float*) b);    // second quad
    num4567 = _mm256_complexconjugatemul_ps(num4567, den4567);   // a conj(b)
    a += 4;
    b += 4;

    norm = _mm256_magnitudesquared_ps(den0123, den4567);

    tmp0123 = _mm256_permute_ps(norm, _MM_SHUFFLE(1,1,0,0));
    tmp4567 = _mm256_permute_ps(norm, _MM_SHUFFLE(3,3,2,2));
    den0123 = _mm256_permute2f128_ps(tmp0123, tmp4567, (0<<0)| (2<<4)); // select the lower 128bit from tmp0123, lower from tmp4567
    den4567 = _mm256_permute2f128_ps(tmp0123, tmp4567, (1<<0)| (3<<4)); // select the upper 128bit from tmp0123, upper from tmp4567

    result = _mm256_div_ps(num0123, den0123);
    _mm256_store_ps((float*) c, result); // Store the results back into the C container
    c += 4;
    result = _mm256_div_ps(num4567, den4567);
    _mm256_store_ps((float*) c, result); // Store the results back into the C container
    c += 4;
  }

  number *= 8;
  for(;number < num_points; number++){
    *c = (*a) / (*b);
    a++; b++; c++;
  }

}
#endif /* LV_HAVE_AVX */

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
