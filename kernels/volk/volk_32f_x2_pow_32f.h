/* -*- c++ -*- */
/*
 * Copyright 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32f_x2_pow_32f
 *
 * \b Overview
 *
 * Raises the sample in aVector to the power of the number in bVector.
 *
 * c[i] = pow(a[i], b[i])
 *
 * Note that the aVector values must be positive; otherwise the output may be inaccurate.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32f_x2_pow_32f(float* cVector, const float* bVector, const float* aVector,
 * unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li bVector: The input vector of indices (power values).
 * \li aVector: The input vector of base values.
 * \li num_points: The number of values in both input vectors.
 *
 * \b Outputs
 * \li cVector: The output vector.
 *
 * \b Example
 * Calculate the first two powers of two (2^x).
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   float* increasing = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* twos = (float*)volk_malloc(sizeof(float)*N, alignment);
 *   float* out = (float*)volk_malloc(sizeof(float)*N, alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       increasing[ii] = (float)ii;
 *       twos[ii] = 2.f;
 *   }
 *
 *   volk_32f_x2_pow_32f(out, increasing, twos, N);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("out[%u] = %1.2f\n", ii, out[ii]);
 *   }
 *
 *   volk_free(increasing);
 *   volk_free(twos);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32f_x2_pow_32f_a_H
#define INCLUDED_volk_32f_x2_pow_32f_a_H

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define POW_POLY_DEGREE 3

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>

#define POLY0_AVX2_FMA(x, c0) _mm256_set1_ps(c0)
#define POLY1_AVX2_FMA(x, c0, c1) \
    _mm256_fmadd_ps(POLY0_AVX2_FMA(x, c1), x, _mm256_set1_ps(c0))
#define POLY2_AVX2_FMA(x, c0, c1, c2) \
    _mm256_fmadd_ps(POLY1_AVX2_FMA(x, c1, c2), x, _mm256_set1_ps(c0))
#define POLY3_AVX2_FMA(x, c0, c1, c2, c3) \
    _mm256_fmadd_ps(POLY2_AVX2_FMA(x, c1, c2, c3), x, _mm256_set1_ps(c0))
#define POLY4_AVX2_FMA(x, c0, c1, c2, c3, c4) \
    _mm256_fmadd_ps(POLY3_AVX2_FMA(x, c1, c2, c3, c4), x, _mm256_set1_ps(c0))
#define POLY5_AVX2_FMA(x, c0, c1, c2, c3, c4, c5) \
    _mm256_fmadd_ps(POLY4_AVX2_FMA(x, c1, c2, c3, c4, c5), x, _mm256_set1_ps(c0))

static inline void volk_32f_x2_pow_32f_a_avx2_fma(float* cVector,
                                                  const float* bVector,
                                                  const float* aVector,
                                                  unsigned int num_points)
{
    float* cPtr = cVector;
    const float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    __m256 aVal, bVal, cVal, logarithm, mantissa, frac, leadingOne;
    __m256 tmp, fx, mask, pow2n, z, y;
    __m256 one, exp_hi, exp_lo, ln2, log2EF, half, exp_C1, exp_C2;
    __m256 exp_p0, exp_p1, exp_p2, exp_p3, exp_p4, exp_p5;
    __m256i bias, exp, emm0, pi32_0x7f;

    one = _mm256_set1_ps(1.0);
    exp_hi = _mm256_set1_ps(88.3762626647949);
    exp_lo = _mm256_set1_ps(-88.3762626647949);
    ln2 = _mm256_set1_ps(0.6931471805);
    log2EF = _mm256_set1_ps(1.44269504088896341);
    half = _mm256_set1_ps(0.5);
    exp_C1 = _mm256_set1_ps(0.693359375);
    exp_C2 = _mm256_set1_ps(-2.12194440e-4);
    pi32_0x7f = _mm256_set1_epi32(0x7f);

    exp_p0 = _mm256_set1_ps(1.9875691500e-4);
    exp_p1 = _mm256_set1_ps(1.3981999507e-3);
    exp_p2 = _mm256_set1_ps(8.3334519073e-3);
    exp_p3 = _mm256_set1_ps(4.1665795894e-2);
    exp_p4 = _mm256_set1_ps(1.6666665459e-1);
    exp_p5 = _mm256_set1_ps(5.0000001201e-1);

    for (; number < eighthPoints; number++) {
        // First compute the logarithm
        aVal = _mm256_load_ps(aPtr);
        bias = _mm256_set1_epi32(127);
        leadingOne = _mm256_set1_ps(1.0f);
        exp = _mm256_sub_epi32(
            _mm256_srli_epi32(_mm256_and_si256(_mm256_castps_si256(aVal),
                                               _mm256_set1_epi32(0x7f800000)),
                              23),
            bias);
        logarithm = _mm256_cvtepi32_ps(exp);

        frac = _mm256_or_ps(
            leadingOne,
            _mm256_and_ps(aVal, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffff))));

#if POW_POLY_DEGREE == 6
        mantissa = POLY5_AVX2_FMA(frac,
                                  3.1157899f,
                                  -3.3241990f,
                                  2.5988452f,
                                  -1.2315303f,
                                  3.1821337e-1f,
                                  -3.4436006e-2f);
#elif POW_POLY_DEGREE == 5
        mantissa = POLY4_AVX2_FMA(frac,
                                  2.8882704548164776201f,
                                  -2.52074962577807006663f,
                                  1.48116647521213171641f,
                                  -0.465725644288844778798f,
                                  0.0596515482674574969533f);
#elif POW_POLY_DEGREE == 4
        mantissa = POLY3_AVX2_FMA(frac,
                                  2.61761038894603480148f,
                                  -1.75647175389045657003f,
                                  0.688243882994381274313f,
                                  -0.107254423828329604454f);
#elif POW_POLY_DEGREE == 3
        mantissa = POLY2_AVX2_FMA(frac,
                                  2.28330284476918490682f,
                                  -1.04913055217340124191f,
                                  0.204446009836232697516f);
#else
#error
#endif

        logarithm = _mm256_fmadd_ps(mantissa, _mm256_sub_ps(frac, leadingOne), logarithm);
        logarithm = _mm256_mul_ps(logarithm, ln2);

        // Now calculate b*lna
        bVal = _mm256_load_ps(bPtr);
        bVal = _mm256_mul_ps(bVal, logarithm);

        // Now compute exp(b*lna)
        bVal = _mm256_max_ps(_mm256_min_ps(bVal, exp_hi), exp_lo);

        fx = _mm256_fmadd_ps(bVal, log2EF, half);

        emm0 = _mm256_cvttps_epi32(fx);
        tmp = _mm256_cvtepi32_ps(emm0);

        mask = _mm256_and_ps(_mm256_cmp_ps(tmp, fx, _CMP_GT_OS), one);
        fx = _mm256_sub_ps(tmp, mask);

        tmp = _mm256_fnmadd_ps(fx, exp_C1, bVal);
        bVal = _mm256_fnmadd_ps(fx, exp_C2, tmp);
        z = _mm256_mul_ps(bVal, bVal);

        y = _mm256_fmadd_ps(exp_p0, bVal, exp_p1);
        y = _mm256_fmadd_ps(y, bVal, exp_p2);
        y = _mm256_fmadd_ps(y, bVal, exp_p3);
        y = _mm256_fmadd_ps(y, bVal, exp_p4);
        y = _mm256_fmadd_ps(y, bVal, exp_p5);
        y = _mm256_fmadd_ps(y, z, bVal);
        y = _mm256_add_ps(y, one);

        emm0 =
            _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(fx), pi32_0x7f), 23);

        pow2n = _mm256_castsi256_ps(emm0);
        cVal = _mm256_mul_ps(y, pow2n);

        _mm256_store_ps(cPtr, cVal);

        aPtr += 8;
        bPtr += 8;
        cPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *cPtr++ = pow(*aPtr++, *bPtr++);
    }
}

#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA for aligned */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

#define POLY0_AVX2(x, c0) _mm256_set1_ps(c0)
#define POLY1_AVX2(x, c0, c1) \
    _mm256_add_ps(_mm256_mul_ps(POLY0_AVX2(x, c1), x), _mm256_set1_ps(c0))
#define POLY2_AVX2(x, c0, c1, c2) \
    _mm256_add_ps(_mm256_mul_ps(POLY1_AVX2(x, c1, c2), x), _mm256_set1_ps(c0))
#define POLY3_AVX2(x, c0, c1, c2, c3) \
    _mm256_add_ps(_mm256_mul_ps(POLY2_AVX2(x, c1, c2, c3), x), _mm256_set1_ps(c0))
#define POLY4_AVX2(x, c0, c1, c2, c3, c4) \
    _mm256_add_ps(_mm256_mul_ps(POLY3_AVX2(x, c1, c2, c3, c4), x), _mm256_set1_ps(c0))
#define POLY5_AVX2(x, c0, c1, c2, c3, c4, c5) \
    _mm256_add_ps(_mm256_mul_ps(POLY4_AVX2(x, c1, c2, c3, c4, c5), x), _mm256_set1_ps(c0))

static inline void volk_32f_x2_pow_32f_a_avx2(float* cVector,
                                              const float* bVector,
                                              const float* aVector,
                                              unsigned int num_points)
{
    float* cPtr = cVector;
    const float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    __m256 aVal, bVal, cVal, logarithm, mantissa, frac, leadingOne;
    __m256 tmp, fx, mask, pow2n, z, y;
    __m256 one, exp_hi, exp_lo, ln2, log2EF, half, exp_C1, exp_C2;
    __m256 exp_p0, exp_p1, exp_p2, exp_p3, exp_p4, exp_p5;
    __m256i bias, exp, emm0, pi32_0x7f;

    one = _mm256_set1_ps(1.0);
    exp_hi = _mm256_set1_ps(88.3762626647949);
    exp_lo = _mm256_set1_ps(-88.3762626647949);
    ln2 = _mm256_set1_ps(0.6931471805);
    log2EF = _mm256_set1_ps(1.44269504088896341);
    half = _mm256_set1_ps(0.5);
    exp_C1 = _mm256_set1_ps(0.693359375);
    exp_C2 = _mm256_set1_ps(-2.12194440e-4);
    pi32_0x7f = _mm256_set1_epi32(0x7f);

    exp_p0 = _mm256_set1_ps(1.9875691500e-4);
    exp_p1 = _mm256_set1_ps(1.3981999507e-3);
    exp_p2 = _mm256_set1_ps(8.3334519073e-3);
    exp_p3 = _mm256_set1_ps(4.1665795894e-2);
    exp_p4 = _mm256_set1_ps(1.6666665459e-1);
    exp_p5 = _mm256_set1_ps(5.0000001201e-1);

    for (; number < eighthPoints; number++) {
        // First compute the logarithm
        aVal = _mm256_load_ps(aPtr);
        bias = _mm256_set1_epi32(127);
        leadingOne = _mm256_set1_ps(1.0f);
        exp = _mm256_sub_epi32(
            _mm256_srli_epi32(_mm256_and_si256(_mm256_castps_si256(aVal),
                                               _mm256_set1_epi32(0x7f800000)),
                              23),
            bias);
        logarithm = _mm256_cvtepi32_ps(exp);

        frac = _mm256_or_ps(
            leadingOne,
            _mm256_and_ps(aVal, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffff))));

#if POW_POLY_DEGREE == 6
        mantissa = POLY5_AVX2(frac,
                              3.1157899f,
                              -3.3241990f,
                              2.5988452f,
                              -1.2315303f,
                              3.1821337e-1f,
                              -3.4436006e-2f);
#elif POW_POLY_DEGREE == 5
        mantissa = POLY4_AVX2(frac,
                              2.8882704548164776201f,
                              -2.52074962577807006663f,
                              1.48116647521213171641f,
                              -0.465725644288844778798f,
                              0.0596515482674574969533f);
#elif POW_POLY_DEGREE == 4
        mantissa = POLY3_AVX2(frac,
                              2.61761038894603480148f,
                              -1.75647175389045657003f,
                              0.688243882994381274313f,
                              -0.107254423828329604454f);
#elif POW_POLY_DEGREE == 3
        mantissa = POLY2_AVX2(frac,
                              2.28330284476918490682f,
                              -1.04913055217340124191f,
                              0.204446009836232697516f);
#else
#error
#endif

        logarithm = _mm256_add_ps(
            _mm256_mul_ps(mantissa, _mm256_sub_ps(frac, leadingOne)), logarithm);
        logarithm = _mm256_mul_ps(logarithm, ln2);

        // Now calculate b*lna
        bVal = _mm256_load_ps(bPtr);
        bVal = _mm256_mul_ps(bVal, logarithm);

        // Now compute exp(b*lna)
        bVal = _mm256_max_ps(_mm256_min_ps(bVal, exp_hi), exp_lo);

        fx = _mm256_add_ps(_mm256_mul_ps(bVal, log2EF), half);

        emm0 = _mm256_cvttps_epi32(fx);
        tmp = _mm256_cvtepi32_ps(emm0);

        mask = _mm256_and_ps(_mm256_cmp_ps(tmp, fx, _CMP_GT_OS), one);
        fx = _mm256_sub_ps(tmp, mask);

        tmp = _mm256_sub_ps(bVal, _mm256_mul_ps(fx, exp_C1));
        bVal = _mm256_sub_ps(tmp, _mm256_mul_ps(fx, exp_C2));
        z = _mm256_mul_ps(bVal, bVal);

        y = _mm256_add_ps(_mm256_mul_ps(exp_p0, bVal), exp_p1);
        y = _mm256_add_ps(_mm256_mul_ps(y, bVal), exp_p2);
        y = _mm256_add_ps(_mm256_mul_ps(y, bVal), exp_p3);
        y = _mm256_add_ps(_mm256_mul_ps(y, bVal), exp_p4);
        y = _mm256_add_ps(_mm256_mul_ps(y, bVal), exp_p5);
        y = _mm256_add_ps(_mm256_mul_ps(y, z), bVal);
        y = _mm256_add_ps(y, one);

        emm0 =
            _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(fx), pi32_0x7f), 23);

        pow2n = _mm256_castsi256_ps(emm0);
        cVal = _mm256_mul_ps(y, pow2n);

        _mm256_store_ps(cPtr, cVal);

        aPtr += 8;
        bPtr += 8;
        cPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *cPtr++ = pow(*aPtr++, *bPtr++);
    }
}

#endif /* LV_HAVE_AVX2 for aligned */


#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

#define POLY0(x, c0) _mm_set1_ps(c0)
#define POLY1(x, c0, c1) _mm_add_ps(_mm_mul_ps(POLY0(x, c1), x), _mm_set1_ps(c0))
#define POLY2(x, c0, c1, c2) _mm_add_ps(_mm_mul_ps(POLY1(x, c1, c2), x), _mm_set1_ps(c0))
#define POLY3(x, c0, c1, c2, c3) \
    _mm_add_ps(_mm_mul_ps(POLY2(x, c1, c2, c3), x), _mm_set1_ps(c0))
#define POLY4(x, c0, c1, c2, c3, c4) \
    _mm_add_ps(_mm_mul_ps(POLY3(x, c1, c2, c3, c4), x), _mm_set1_ps(c0))
#define POLY5(x, c0, c1, c2, c3, c4, c5) \
    _mm_add_ps(_mm_mul_ps(POLY4(x, c1, c2, c3, c4, c5), x), _mm_set1_ps(c0))

static inline void volk_32f_x2_pow_32f_a_sse4_1(float* cVector,
                                                const float* bVector,
                                                const float* aVector,
                                                unsigned int num_points)
{
    float* cPtr = cVector;
    const float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    __m128 aVal, bVal, cVal, logarithm, mantissa, frac, leadingOne;
    __m128 tmp, fx, mask, pow2n, z, y;
    __m128 one, exp_hi, exp_lo, ln2, log2EF, half, exp_C1, exp_C2;
    __m128 exp_p0, exp_p1, exp_p2, exp_p3, exp_p4, exp_p5;
    __m128i bias, exp, emm0, pi32_0x7f;

    one = _mm_set1_ps(1.0);
    exp_hi = _mm_set1_ps(88.3762626647949);
    exp_lo = _mm_set1_ps(-88.3762626647949);
    ln2 = _mm_set1_ps(0.6931471805);
    log2EF = _mm_set1_ps(1.44269504088896341);
    half = _mm_set1_ps(0.5);
    exp_C1 = _mm_set1_ps(0.693359375);
    exp_C2 = _mm_set1_ps(-2.12194440e-4);
    pi32_0x7f = _mm_set1_epi32(0x7f);

    exp_p0 = _mm_set1_ps(1.9875691500e-4);
    exp_p1 = _mm_set1_ps(1.3981999507e-3);
    exp_p2 = _mm_set1_ps(8.3334519073e-3);
    exp_p3 = _mm_set1_ps(4.1665795894e-2);
    exp_p4 = _mm_set1_ps(1.6666665459e-1);
    exp_p5 = _mm_set1_ps(5.0000001201e-1);

    for (; number < quarterPoints; number++) {
        // First compute the logarithm
        aVal = _mm_load_ps(aPtr);
        bias = _mm_set1_epi32(127);
        leadingOne = _mm_set1_ps(1.0f);
        exp = _mm_sub_epi32(
            _mm_srli_epi32(
                _mm_and_si128(_mm_castps_si128(aVal), _mm_set1_epi32(0x7f800000)), 23),
            bias);
        logarithm = _mm_cvtepi32_ps(exp);

        frac = _mm_or_ps(leadingOne,
                         _mm_and_ps(aVal, _mm_castsi128_ps(_mm_set1_epi32(0x7fffff))));

#if POW_POLY_DEGREE == 6
        mantissa = POLY5(frac,
                         3.1157899f,
                         -3.3241990f,
                         2.5988452f,
                         -1.2315303f,
                         3.1821337e-1f,
                         -3.4436006e-2f);
#elif POW_POLY_DEGREE == 5
        mantissa = POLY4(frac,
                         2.8882704548164776201f,
                         -2.52074962577807006663f,
                         1.48116647521213171641f,
                         -0.465725644288844778798f,
                         0.0596515482674574969533f);
#elif POW_POLY_DEGREE == 4
        mantissa = POLY3(frac,
                         2.61761038894603480148f,
                         -1.75647175389045657003f,
                         0.688243882994381274313f,
                         -0.107254423828329604454f);
#elif POW_POLY_DEGREE == 3
        mantissa = POLY2(frac,
                         2.28330284476918490682f,
                         -1.04913055217340124191f,
                         0.204446009836232697516f);
#else
#error
#endif

        logarithm =
            _mm_add_ps(logarithm, _mm_mul_ps(mantissa, _mm_sub_ps(frac, leadingOne)));
        logarithm = _mm_mul_ps(logarithm, ln2);


        // Now calculate b*lna
        bVal = _mm_load_ps(bPtr);
        bVal = _mm_mul_ps(bVal, logarithm);

        // Now compute exp(b*lna)
        bVal = _mm_max_ps(_mm_min_ps(bVal, exp_hi), exp_lo);

        fx = _mm_add_ps(_mm_mul_ps(bVal, log2EF), half);

        emm0 = _mm_cvttps_epi32(fx);
        tmp = _mm_cvtepi32_ps(emm0);

        mask = _mm_and_ps(_mm_cmpgt_ps(tmp, fx), one);
        fx = _mm_sub_ps(tmp, mask);

        tmp = _mm_mul_ps(fx, exp_C1);
        z = _mm_mul_ps(fx, exp_C2);
        bVal = _mm_sub_ps(_mm_sub_ps(bVal, tmp), z);
        z = _mm_mul_ps(bVal, bVal);

        y = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(exp_p0, bVal), exp_p1), bVal);
        y = _mm_add_ps(_mm_mul_ps(_mm_add_ps(y, exp_p2), bVal), exp_p3);
        y = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(y, bVal), exp_p4), bVal);
        y = _mm_add_ps(_mm_mul_ps(_mm_add_ps(y, exp_p5), z), bVal);
        y = _mm_add_ps(y, one);

        emm0 = _mm_slli_epi32(_mm_add_epi32(_mm_cvttps_epi32(fx), pi32_0x7f), 23);

        pow2n = _mm_castsi128_ps(emm0);
        cVal = _mm_mul_ps(y, pow2n);

        _mm_store_ps(cPtr, cVal);

        aPtr += 4;
        bPtr += 4;
        cPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *cPtr++ = powf(*aPtr++, *bPtr++);
    }
}

#endif /* LV_HAVE_SSE4_1 for aligned */

#endif /* INCLUDED_volk_32f_x2_pow_32f_a_H */

#ifndef INCLUDED_volk_32f_x2_pow_32f_u_H
#define INCLUDED_volk_32f_x2_pow_32f_u_H

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define POW_POLY_DEGREE 3

#ifdef LV_HAVE_GENERIC

static inline void volk_32f_x2_pow_32f_generic(float* cVector,
                                               const float* bVector,
                                               const float* aVector,
                                               unsigned int num_points)
{
    float* cPtr = cVector;
    const float* bPtr = bVector;
    const float* aPtr = aVector;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        *cPtr++ = powf(*aPtr++, *bPtr++);
    }
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32f_x2_pow_32f_neon(float* cVector,
                                            const float* bVector,
                                            const float* aVector,
                                            unsigned int num_points)
{
    float* cPtr = cVector;
    const float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    // Constants
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t exp_hi = vdupq_n_f32(88.3762626647949f);
    float32x4_t exp_lo = vdupq_n_f32(-88.3762626647949f);
    float32x4_t ln2 = vdupq_n_f32(0.6931471805f);
    float32x4_t log2EF = vdupq_n_f32(1.44269504088896341f);
    float32x4_t half = vdupq_n_f32(0.5f);
    float32x4_t exp_C1 = vdupq_n_f32(0.693359375f);
    float32x4_t exp_C2 = vdupq_n_f32(-2.12194440e-4f);
    int32x4_t pi32_0x7f = vdupq_n_s32(0x7f);
    int32x4_t bias = vdupq_n_s32(127);
    int32x4_t mantMask = vdupq_n_s32(0x7fffff);
    int32x4_t expMask = vdupq_n_s32(0x7f800000);

    // Polynomial coefficients for log
    float32x4_t log_c0 = vdupq_n_f32(2.28330284476918490682f);
    float32x4_t log_c1 = vdupq_n_f32(-1.04913055217340124191f);
    float32x4_t log_c2 = vdupq_n_f32(0.204446009836232697516f);

    // Polynomial coefficients for exp
    float32x4_t exp_p0 = vdupq_n_f32(1.9875691500e-4f);
    float32x4_t exp_p1 = vdupq_n_f32(1.3981999507e-3f);
    float32x4_t exp_p2 = vdupq_n_f32(8.3334519073e-3f);
    float32x4_t exp_p3 = vdupq_n_f32(4.1665795894e-2f);
    float32x4_t exp_p4 = vdupq_n_f32(1.6666665459e-1f);
    float32x4_t exp_p5 = vdupq_n_f32(5.0000001201e-1f);

    for (; number < quarterPoints; number++) {
        float32x4_t aVal = vld1q_f32(aPtr);

        // First compute log(a)
        int32x4_t aInt = vreinterpretq_s32_f32(aVal);
        int32x4_t expPart = vsubq_s32(vshrq_n_s32(vandq_s32(aInt, expMask), 23), bias);
        float32x4_t logarithm = vcvtq_f32_s32(expPart);

        int32x4_t mantPart =
            vorrq_s32(vandq_s32(aInt, mantMask), vreinterpretq_s32_f32(one));
        float32x4_t frac = vreinterpretq_f32_s32(mantPart);

        // Polynomial for log mantissa (degree 3)
        float32x4_t mantissa = vmlaq_f32(log_c1, log_c2, frac);
        mantissa = vmlaq_f32(log_c0, mantissa, frac);

        float32x4_t fracMinusOne = vsubq_f32(frac, one);
        logarithm = vmlaq_f32(logarithm, mantissa, fracMinusOne);
        logarithm = vmulq_f32(logarithm, ln2);

        // Now calculate b*log(a)
        float32x4_t bVal = vld1q_f32(bPtr);
        bVal = vmulq_f32(bVal, logarithm);

        // Now compute exp(b*log(a))
        bVal = vmaxq_f32(vminq_f32(bVal, exp_hi), exp_lo);

        float32x4_t fx = vmlaq_f32(half, bVal, log2EF);

        int32x4_t emm0 = vcvtq_s32_f32(fx);
        float32x4_t tmp = vcvtq_f32_s32(emm0);

        uint32x4_t mask = vcgtq_f32(tmp, fx);
        float32x4_t mask_one = vbslq_f32(mask, one, vdupq_n_f32(0.0f));
        fx = vsubq_f32(tmp, mask_one);

        tmp = vmulq_f32(fx, exp_C1);
        float32x4_t z = vmulq_f32(fx, exp_C2);
        bVal = vsubq_f32(vsubq_f32(bVal, tmp), z);
        z = vmulq_f32(bVal, bVal);

        float32x4_t y = vmlaq_f32(exp_p1, exp_p0, bVal);
        y = vmulq_f32(y, bVal);
        y = vaddq_f32(y, exp_p2);
        y = vmulq_f32(y, bVal);
        y = vaddq_f32(y, exp_p3);
        y = vmlaq_f32(exp_p4, y, bVal);
        y = vmulq_f32(y, bVal);
        y = vaddq_f32(y, exp_p5);
        y = vmlaq_f32(bVal, y, z);
        y = vaddq_f32(y, one);

        emm0 = vcvtq_s32_f32(fx);
        emm0 = vaddq_s32(emm0, pi32_0x7f);
        emm0 = vshlq_n_s32(emm0, 23);
        float32x4_t pow2n = vreinterpretq_f32_s32(emm0);

        float32x4_t cVal = vmulq_f32(y, pow2n);
        vst1q_f32(cPtr, cVal);

        aPtr += 4;
        bPtr += 4;
        cPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *cPtr++ = powf(*aPtr++, *bPtr++);
    }
}

#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32f_x2_pow_32f_neonv8(float* cVector,
                                              const float* bVector,
                                              const float* aVector,
                                              unsigned int num_points)
{
    float* cPtr = cVector;
    const float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    // Constants
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t exp_hi = vdupq_n_f32(88.3762626647949f);
    float32x4_t exp_lo = vdupq_n_f32(-88.3762626647949f);
    float32x4_t ln2 = vdupq_n_f32(0.6931471805f);
    float32x4_t log2EF = vdupq_n_f32(1.44269504088896341f);
    float32x4_t half = vdupq_n_f32(0.5f);
    float32x4_t exp_C1 = vdupq_n_f32(0.693359375f);
    float32x4_t exp_C2 = vdupq_n_f32(-2.12194440e-4f);
    int32x4_t pi32_0x7f = vdupq_n_s32(0x7f);
    int32x4_t bias = vdupq_n_s32(127);
    int32x4_t mantMask = vdupq_n_s32(0x7fffff);
    int32x4_t expMask = vdupq_n_s32(0x7f800000);

    // Polynomial coefficients for log
    float32x4_t log_c0 = vdupq_n_f32(2.28330284476918490682f);
    float32x4_t log_c1 = vdupq_n_f32(-1.04913055217340124191f);
    float32x4_t log_c2 = vdupq_n_f32(0.204446009836232697516f);

    // Polynomial coefficients for exp
    float32x4_t exp_p0 = vdupq_n_f32(1.9875691500e-4f);
    float32x4_t exp_p1 = vdupq_n_f32(1.3981999507e-3f);
    float32x4_t exp_p2 = vdupq_n_f32(8.3334519073e-3f);
    float32x4_t exp_p3 = vdupq_n_f32(4.1665795894e-2f);
    float32x4_t exp_p4 = vdupq_n_f32(1.6666665459e-1f);
    float32x4_t exp_p5 = vdupq_n_f32(5.0000001201e-1f);

    for (; number < quarterPoints; number++) {
        __VOLK_PREFETCH(aPtr + 8);
        __VOLK_PREFETCH(bPtr + 8);

        float32x4_t aVal = vld1q_f32(aPtr);

        // First compute log(a)
        int32x4_t aInt = vreinterpretq_s32_f32(aVal);
        int32x4_t expPart = vsubq_s32(vshrq_n_s32(vandq_s32(aInt, expMask), 23), bias);
        float32x4_t logarithm = vcvtq_f32_s32(expPart);

        int32x4_t mantPart =
            vorrq_s32(vandq_s32(aInt, mantMask), vreinterpretq_s32_f32(one));
        float32x4_t frac = vreinterpretq_f32_s32(mantPart);

        // Polynomial for log mantissa (degree 3)
        float32x4_t mantissa = vfmaq_f32(log_c1, log_c2, frac);
        mantissa = vfmaq_f32(log_c0, mantissa, frac);

        float32x4_t fracMinusOne = vsubq_f32(frac, one);
        logarithm = vfmaq_f32(logarithm, mantissa, fracMinusOne);
        logarithm = vmulq_f32(logarithm, ln2);

        // Now calculate b*log(a)
        float32x4_t bVal = vld1q_f32(bPtr);
        bVal = vmulq_f32(bVal, logarithm);

        // Now compute exp(b*log(a))
        bVal = vmaxq_f32(vminq_f32(bVal, exp_hi), exp_lo);

        float32x4_t fx = vfmaq_f32(half, bVal, log2EF);

        int32x4_t emm0 = vcvtq_s32_f32(fx);
        float32x4_t tmp = vcvtq_f32_s32(emm0);

        uint32x4_t mask = vcgtq_f32(tmp, fx);
        float32x4_t mask_one = vbslq_f32(mask, one, vdupq_n_f32(0.0f));
        fx = vsubq_f32(tmp, mask_one);

        tmp = vmulq_f32(fx, exp_C1);
        float32x4_t z = vmulq_f32(fx, exp_C2);
        bVal = vsubq_f32(vsubq_f32(bVal, tmp), z);
        z = vmulq_f32(bVal, bVal);

        float32x4_t y = vfmaq_f32(exp_p1, exp_p0, bVal);
        y = vmulq_f32(y, bVal);
        y = vaddq_f32(y, exp_p2);
        y = vmulq_f32(y, bVal);
        y = vaddq_f32(y, exp_p3);
        y = vfmaq_f32(exp_p4, y, bVal);
        y = vmulq_f32(y, bVal);
        y = vaddq_f32(y, exp_p5);
        y = vfmaq_f32(bVal, y, z);
        y = vaddq_f32(y, one);

        emm0 = vcvtq_s32_f32(fx);
        emm0 = vaddq_s32(emm0, pi32_0x7f);
        emm0 = vshlq_n_s32(emm0, 23);
        float32x4_t pow2n = vreinterpretq_f32_s32(emm0);

        float32x4_t cVal = vmulq_f32(y, pow2n);
        vst1q_f32(cPtr, cVal);

        aPtr += 4;
        bPtr += 4;
        cPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *cPtr++ = powf(*aPtr++, *bPtr++);
    }
}

#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>

#define POLY0(x, c0) _mm_set1_ps(c0)
#define POLY1(x, c0, c1) _mm_add_ps(_mm_mul_ps(POLY0(x, c1), x), _mm_set1_ps(c0))
#define POLY2(x, c0, c1, c2) _mm_add_ps(_mm_mul_ps(POLY1(x, c1, c2), x), _mm_set1_ps(c0))
#define POLY3(x, c0, c1, c2, c3) \
    _mm_add_ps(_mm_mul_ps(POLY2(x, c1, c2, c3), x), _mm_set1_ps(c0))
#define POLY4(x, c0, c1, c2, c3, c4) \
    _mm_add_ps(_mm_mul_ps(POLY3(x, c1, c2, c3, c4), x), _mm_set1_ps(c0))
#define POLY5(x, c0, c1, c2, c3, c4, c5) \
    _mm_add_ps(_mm_mul_ps(POLY4(x, c1, c2, c3, c4, c5), x), _mm_set1_ps(c0))

static inline void volk_32f_x2_pow_32f_u_sse4_1(float* cVector,
                                                const float* bVector,
                                                const float* aVector,
                                                unsigned int num_points)
{
    float* cPtr = cVector;
    const float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    __m128 aVal, bVal, cVal, logarithm, mantissa, frac, leadingOne;
    __m128 tmp, fx, mask, pow2n, z, y;
    __m128 one, exp_hi, exp_lo, ln2, log2EF, half, exp_C1, exp_C2;
    __m128 exp_p0, exp_p1, exp_p2, exp_p3, exp_p4, exp_p5;
    __m128i bias, exp, emm0, pi32_0x7f;

    one = _mm_set1_ps(1.0);
    exp_hi = _mm_set1_ps(88.3762626647949);
    exp_lo = _mm_set1_ps(-88.3762626647949);
    ln2 = _mm_set1_ps(0.6931471805);
    log2EF = _mm_set1_ps(1.44269504088896341);
    half = _mm_set1_ps(0.5);
    exp_C1 = _mm_set1_ps(0.693359375);
    exp_C2 = _mm_set1_ps(-2.12194440e-4);
    pi32_0x7f = _mm_set1_epi32(0x7f);

    exp_p0 = _mm_set1_ps(1.9875691500e-4);
    exp_p1 = _mm_set1_ps(1.3981999507e-3);
    exp_p2 = _mm_set1_ps(8.3334519073e-3);
    exp_p3 = _mm_set1_ps(4.1665795894e-2);
    exp_p4 = _mm_set1_ps(1.6666665459e-1);
    exp_p5 = _mm_set1_ps(5.0000001201e-1);

    for (; number < quarterPoints; number++) {
        // First compute the logarithm
        aVal = _mm_loadu_ps(aPtr);
        bias = _mm_set1_epi32(127);
        leadingOne = _mm_set1_ps(1.0f);
        exp = _mm_sub_epi32(
            _mm_srli_epi32(
                _mm_and_si128(_mm_castps_si128(aVal), _mm_set1_epi32(0x7f800000)), 23),
            bias);
        logarithm = _mm_cvtepi32_ps(exp);

        frac = _mm_or_ps(leadingOne,
                         _mm_and_ps(aVal, _mm_castsi128_ps(_mm_set1_epi32(0x7fffff))));

#if POW_POLY_DEGREE == 6
        mantissa = POLY5(frac,
                         3.1157899f,
                         -3.3241990f,
                         2.5988452f,
                         -1.2315303f,
                         3.1821337e-1f,
                         -3.4436006e-2f);
#elif POW_POLY_DEGREE == 5
        mantissa = POLY4(frac,
                         2.8882704548164776201f,
                         -2.52074962577807006663f,
                         1.48116647521213171641f,
                         -0.465725644288844778798f,
                         0.0596515482674574969533f);
#elif POW_POLY_DEGREE == 4
        mantissa = POLY3(frac,
                         2.61761038894603480148f,
                         -1.75647175389045657003f,
                         0.688243882994381274313f,
                         -0.107254423828329604454f);
#elif POW_POLY_DEGREE == 3
        mantissa = POLY2(frac,
                         2.28330284476918490682f,
                         -1.04913055217340124191f,
                         0.204446009836232697516f);
#else
#error
#endif

        logarithm =
            _mm_add_ps(logarithm, _mm_mul_ps(mantissa, _mm_sub_ps(frac, leadingOne)));
        logarithm = _mm_mul_ps(logarithm, ln2);


        // Now calculate b*lna
        bVal = _mm_loadu_ps(bPtr);
        bVal = _mm_mul_ps(bVal, logarithm);

        // Now compute exp(b*lna)
        bVal = _mm_max_ps(_mm_min_ps(bVal, exp_hi), exp_lo);

        fx = _mm_add_ps(_mm_mul_ps(bVal, log2EF), half);

        emm0 = _mm_cvttps_epi32(fx);
        tmp = _mm_cvtepi32_ps(emm0);

        mask = _mm_and_ps(_mm_cmpgt_ps(tmp, fx), one);
        fx = _mm_sub_ps(tmp, mask);

        tmp = _mm_mul_ps(fx, exp_C1);
        z = _mm_mul_ps(fx, exp_C2);
        bVal = _mm_sub_ps(_mm_sub_ps(bVal, tmp), z);
        z = _mm_mul_ps(bVal, bVal);

        y = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(exp_p0, bVal), exp_p1), bVal);
        y = _mm_add_ps(_mm_mul_ps(_mm_add_ps(y, exp_p2), bVal), exp_p3);
        y = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(y, bVal), exp_p4), bVal);
        y = _mm_add_ps(_mm_mul_ps(_mm_add_ps(y, exp_p5), z), bVal);
        y = _mm_add_ps(y, one);

        emm0 = _mm_slli_epi32(_mm_add_epi32(_mm_cvttps_epi32(fx), pi32_0x7f), 23);

        pow2n = _mm_castsi128_ps(emm0);
        cVal = _mm_mul_ps(y, pow2n);

        _mm_storeu_ps(cPtr, cVal);

        aPtr += 4;
        bPtr += 4;
        cPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *cPtr++ = powf(*aPtr++, *bPtr++);
    }
}

#endif /* LV_HAVE_SSE4_1 for unaligned */

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>

#define POLY0_AVX2_FMA(x, c0) _mm256_set1_ps(c0)
#define POLY1_AVX2_FMA(x, c0, c1) \
    _mm256_fmadd_ps(POLY0_AVX2_FMA(x, c1), x, _mm256_set1_ps(c0))
#define POLY2_AVX2_FMA(x, c0, c1, c2) \
    _mm256_fmadd_ps(POLY1_AVX2_FMA(x, c1, c2), x, _mm256_set1_ps(c0))
#define POLY3_AVX2_FMA(x, c0, c1, c2, c3) \
    _mm256_fmadd_ps(POLY2_AVX2_FMA(x, c1, c2, c3), x, _mm256_set1_ps(c0))
#define POLY4_AVX2_FMA(x, c0, c1, c2, c3, c4) \
    _mm256_fmadd_ps(POLY3_AVX2_FMA(x, c1, c2, c3, c4), x, _mm256_set1_ps(c0))
#define POLY5_AVX2_FMA(x, c0, c1, c2, c3, c4, c5) \
    _mm256_fmadd_ps(POLY4_AVX2_FMA(x, c1, c2, c3, c4, c5), x, _mm256_set1_ps(c0))

static inline void volk_32f_x2_pow_32f_u_avx2_fma(float* cVector,
                                                  const float* bVector,
                                                  const float* aVector,
                                                  unsigned int num_points)
{
    float* cPtr = cVector;
    const float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    __m256 aVal, bVal, cVal, logarithm, mantissa, frac, leadingOne;
    __m256 tmp, fx, mask, pow2n, z, y;
    __m256 one, exp_hi, exp_lo, ln2, log2EF, half, exp_C1, exp_C2;
    __m256 exp_p0, exp_p1, exp_p2, exp_p3, exp_p4, exp_p5;
    __m256i bias, exp, emm0, pi32_0x7f;

    one = _mm256_set1_ps(1.0);
    exp_hi = _mm256_set1_ps(88.3762626647949);
    exp_lo = _mm256_set1_ps(-88.3762626647949);
    ln2 = _mm256_set1_ps(0.6931471805);
    log2EF = _mm256_set1_ps(1.44269504088896341);
    half = _mm256_set1_ps(0.5);
    exp_C1 = _mm256_set1_ps(0.693359375);
    exp_C2 = _mm256_set1_ps(-2.12194440e-4);
    pi32_0x7f = _mm256_set1_epi32(0x7f);

    exp_p0 = _mm256_set1_ps(1.9875691500e-4);
    exp_p1 = _mm256_set1_ps(1.3981999507e-3);
    exp_p2 = _mm256_set1_ps(8.3334519073e-3);
    exp_p3 = _mm256_set1_ps(4.1665795894e-2);
    exp_p4 = _mm256_set1_ps(1.6666665459e-1);
    exp_p5 = _mm256_set1_ps(5.0000001201e-1);

    for (; number < eighthPoints; number++) {
        // First compute the logarithm
        aVal = _mm256_loadu_ps(aPtr);
        bias = _mm256_set1_epi32(127);
        leadingOne = _mm256_set1_ps(1.0f);
        exp = _mm256_sub_epi32(
            _mm256_srli_epi32(_mm256_and_si256(_mm256_castps_si256(aVal),
                                               _mm256_set1_epi32(0x7f800000)),
                              23),
            bias);
        logarithm = _mm256_cvtepi32_ps(exp);

        frac = _mm256_or_ps(
            leadingOne,
            _mm256_and_ps(aVal, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffff))));

#if POW_POLY_DEGREE == 6
        mantissa = POLY5_AVX2_FMA(frac,
                                  3.1157899f,
                                  -3.3241990f,
                                  2.5988452f,
                                  -1.2315303f,
                                  3.1821337e-1f,
                                  -3.4436006e-2f);
#elif POW_POLY_DEGREE == 5
        mantissa = POLY4_AVX2_FMA(frac,
                                  2.8882704548164776201f,
                                  -2.52074962577807006663f,
                                  1.48116647521213171641f,
                                  -0.465725644288844778798f,
                                  0.0596515482674574969533f);
#elif POW_POLY_DEGREE == 4
        mantissa = POLY3_AVX2_FMA(frac,
                                  2.61761038894603480148f,
                                  -1.75647175389045657003f,
                                  0.688243882994381274313f,
                                  -0.107254423828329604454f);
#elif POW_POLY_DEGREE == 3
        mantissa = POLY2_AVX2_FMA(frac,
                                  2.28330284476918490682f,
                                  -1.04913055217340124191f,
                                  0.204446009836232697516f);
#else
#error
#endif

        logarithm = _mm256_fmadd_ps(mantissa, _mm256_sub_ps(frac, leadingOne), logarithm);
        logarithm = _mm256_mul_ps(logarithm, ln2);


        // Now calculate b*lna
        bVal = _mm256_loadu_ps(bPtr);
        bVal = _mm256_mul_ps(bVal, logarithm);

        // Now compute exp(b*lna)
        bVal = _mm256_max_ps(_mm256_min_ps(bVal, exp_hi), exp_lo);

        fx = _mm256_fmadd_ps(bVal, log2EF, half);

        emm0 = _mm256_cvttps_epi32(fx);
        tmp = _mm256_cvtepi32_ps(emm0);

        mask = _mm256_and_ps(_mm256_cmp_ps(tmp, fx, _CMP_GT_OS), one);
        fx = _mm256_sub_ps(tmp, mask);

        tmp = _mm256_fnmadd_ps(fx, exp_C1, bVal);
        bVal = _mm256_fnmadd_ps(fx, exp_C2, tmp);
        z = _mm256_mul_ps(bVal, bVal);

        y = _mm256_fmadd_ps(exp_p0, bVal, exp_p1);
        y = _mm256_fmadd_ps(y, bVal, exp_p2);
        y = _mm256_fmadd_ps(y, bVal, exp_p3);
        y = _mm256_fmadd_ps(y, bVal, exp_p4);
        y = _mm256_fmadd_ps(y, bVal, exp_p5);
        y = _mm256_fmadd_ps(y, z, bVal);
        y = _mm256_add_ps(y, one);

        emm0 =
            _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(fx), pi32_0x7f), 23);

        pow2n = _mm256_castsi256_ps(emm0);
        cVal = _mm256_mul_ps(y, pow2n);

        _mm256_storeu_ps(cPtr, cVal);

        aPtr += 8;
        bPtr += 8;
        cPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *cPtr++ = pow(*aPtr++, *bPtr++);
    }
}

#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA for unaligned */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

#define POLY0_AVX2(x, c0) _mm256_set1_ps(c0)
#define POLY1_AVX2(x, c0, c1) \
    _mm256_add_ps(_mm256_mul_ps(POLY0_AVX2(x, c1), x), _mm256_set1_ps(c0))
#define POLY2_AVX2(x, c0, c1, c2) \
    _mm256_add_ps(_mm256_mul_ps(POLY1_AVX2(x, c1, c2), x), _mm256_set1_ps(c0))
#define POLY3_AVX2(x, c0, c1, c2, c3) \
    _mm256_add_ps(_mm256_mul_ps(POLY2_AVX2(x, c1, c2, c3), x), _mm256_set1_ps(c0))
#define POLY4_AVX2(x, c0, c1, c2, c3, c4) \
    _mm256_add_ps(_mm256_mul_ps(POLY3_AVX2(x, c1, c2, c3, c4), x), _mm256_set1_ps(c0))
#define POLY5_AVX2(x, c0, c1, c2, c3, c4, c5) \
    _mm256_add_ps(_mm256_mul_ps(POLY4_AVX2(x, c1, c2, c3, c4, c5), x), _mm256_set1_ps(c0))

static inline void volk_32f_x2_pow_32f_u_avx2(float* cVector,
                                              const float* bVector,
                                              const float* aVector,
                                              unsigned int num_points)
{
    float* cPtr = cVector;
    const float* bPtr = bVector;
    const float* aPtr = aVector;

    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    __m256 aVal, bVal, cVal, logarithm, mantissa, frac, leadingOne;
    __m256 tmp, fx, mask, pow2n, z, y;
    __m256 one, exp_hi, exp_lo, ln2, log2EF, half, exp_C1, exp_C2;
    __m256 exp_p0, exp_p1, exp_p2, exp_p3, exp_p4, exp_p5;
    __m256i bias, exp, emm0, pi32_0x7f;

    one = _mm256_set1_ps(1.0);
    exp_hi = _mm256_set1_ps(88.3762626647949);
    exp_lo = _mm256_set1_ps(-88.3762626647949);
    ln2 = _mm256_set1_ps(0.6931471805);
    log2EF = _mm256_set1_ps(1.44269504088896341);
    half = _mm256_set1_ps(0.5);
    exp_C1 = _mm256_set1_ps(0.693359375);
    exp_C2 = _mm256_set1_ps(-2.12194440e-4);
    pi32_0x7f = _mm256_set1_epi32(0x7f);

    exp_p0 = _mm256_set1_ps(1.9875691500e-4);
    exp_p1 = _mm256_set1_ps(1.3981999507e-3);
    exp_p2 = _mm256_set1_ps(8.3334519073e-3);
    exp_p3 = _mm256_set1_ps(4.1665795894e-2);
    exp_p4 = _mm256_set1_ps(1.6666665459e-1);
    exp_p5 = _mm256_set1_ps(5.0000001201e-1);

    for (; number < eighthPoints; number++) {
        // First compute the logarithm
        aVal = _mm256_loadu_ps(aPtr);
        bias = _mm256_set1_epi32(127);
        leadingOne = _mm256_set1_ps(1.0f);
        exp = _mm256_sub_epi32(
            _mm256_srli_epi32(_mm256_and_si256(_mm256_castps_si256(aVal),
                                               _mm256_set1_epi32(0x7f800000)),
                              23),
            bias);
        logarithm = _mm256_cvtepi32_ps(exp);

        frac = _mm256_or_ps(
            leadingOne,
            _mm256_and_ps(aVal, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffff))));

#if POW_POLY_DEGREE == 6
        mantissa = POLY5_AVX2(frac,
                              3.1157899f,
                              -3.3241990f,
                              2.5988452f,
                              -1.2315303f,
                              3.1821337e-1f,
                              -3.4436006e-2f);
#elif POW_POLY_DEGREE == 5
        mantissa = POLY4_AVX2(frac,
                              2.8882704548164776201f,
                              -2.52074962577807006663f,
                              1.48116647521213171641f,
                              -0.465725644288844778798f,
                              0.0596515482674574969533f);
#elif POW_POLY_DEGREE == 4
        mantissa = POLY3_AVX2(frac,
                              2.61761038894603480148f,
                              -1.75647175389045657003f,
                              0.688243882994381274313f,
                              -0.107254423828329604454f);
#elif POW_POLY_DEGREE == 3
        mantissa = POLY2_AVX2(frac,
                              2.28330284476918490682f,
                              -1.04913055217340124191f,
                              0.204446009836232697516f);
#else
#error
#endif

        logarithm = _mm256_add_ps(
            _mm256_mul_ps(mantissa, _mm256_sub_ps(frac, leadingOne)), logarithm);
        logarithm = _mm256_mul_ps(logarithm, ln2);

        // Now calculate b*lna
        bVal = _mm256_loadu_ps(bPtr);
        bVal = _mm256_mul_ps(bVal, logarithm);

        // Now compute exp(b*lna)
        bVal = _mm256_max_ps(_mm256_min_ps(bVal, exp_hi), exp_lo);

        fx = _mm256_add_ps(_mm256_mul_ps(bVal, log2EF), half);

        emm0 = _mm256_cvttps_epi32(fx);
        tmp = _mm256_cvtepi32_ps(emm0);

        mask = _mm256_and_ps(_mm256_cmp_ps(tmp, fx, _CMP_GT_OS), one);
        fx = _mm256_sub_ps(tmp, mask);

        tmp = _mm256_sub_ps(bVal, _mm256_mul_ps(fx, exp_C1));
        bVal = _mm256_sub_ps(tmp, _mm256_mul_ps(fx, exp_C2));
        z = _mm256_mul_ps(bVal, bVal);

        y = _mm256_add_ps(_mm256_mul_ps(exp_p0, bVal), exp_p1);
        y = _mm256_add_ps(_mm256_mul_ps(y, bVal), exp_p2);
        y = _mm256_add_ps(_mm256_mul_ps(y, bVal), exp_p3);
        y = _mm256_add_ps(_mm256_mul_ps(y, bVal), exp_p4);
        y = _mm256_add_ps(_mm256_mul_ps(y, bVal), exp_p5);
        y = _mm256_add_ps(_mm256_mul_ps(y, z), bVal);
        y = _mm256_add_ps(y, one);

        emm0 =
            _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(fx), pi32_0x7f), 23);

        pow2n = _mm256_castsi256_ps(emm0);
        cVal = _mm256_mul_ps(y, pow2n);

        _mm256_storeu_ps(cPtr, cVal);

        aPtr += 8;
        bPtr += 8;
        cPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *cPtr++ = pow(*aPtr++, *bPtr++);
    }
}

#endif /* LV_HAVE_AVX2 for unaligned */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32f_x2_pow_32f_rvv(float* cVector,
                                           const float* bVector,
                                           const float* aVector,
                                           unsigned int num_points)
{
    size_t vlmax = __riscv_vsetvlmax_e32m1();

#if POW_POLY_DEGREE == 6
    const vfloat32m1_t cl5 = __riscv_vfmv_v_f_f32m1(3.1157899f, vlmax);
    const vfloat32m1_t cl4 = __riscv_vfmv_v_f_f32m1(-3.3241990f, vlmax);
    const vfloat32m1_t cl3 = __riscv_vfmv_v_f_f32m1(2.5988452f, vlmax);
    const vfloat32m1_t cl2 = __riscv_vfmv_v_f_f32m1(-1.2315303f, vlmax);
    const vfloat32m1_t cl1 = __riscv_vfmv_v_f_f32m1(3.1821337e-1f, vlmax);
    const vfloat32m1_t cl0 = __riscv_vfmv_v_f_f32m1(-3.4436006e-2f, vlmax);
#elif POW_POLY_DEGREE == 5
    const vfloat32m1_t cl4 = __riscv_vfmv_v_f_f32m1(2.8882704548164776201f, vlmax);
    const vfloat32m1_t cl3 = __riscv_vfmv_v_f_f32m1(-2.52074962577807006663f, vlmax);
    const vfloat32m1_t cl2 = __riscv_vfmv_v_f_f32m1(1.48116647521213171641f, vlmax);
    const vfloat32m1_t cl1 = __riscv_vfmv_v_f_f32m1(-0.465725644288844778798f, vlmax);
    const vfloat32m1_t cl0 = __riscv_vfmv_v_f_f32m1(0.0596515482674574969533f, vlmax);
#elif POW_POLY_DEGREE == 4
    const vfloat32m1_t cl3 = __riscv_vfmv_v_f_f32m1(2.61761038894603480148f, vlmax);
    const vfloat32m1_t cl2 = __riscv_vfmv_v_f_f32m1(-1.75647175389045657003f, vlmax);
    const vfloat32m1_t cl1 = __riscv_vfmv_v_f_f32m1(0.688243882994381274313f, vlmax);
    const vfloat32m1_t cl0 = __riscv_vfmv_v_f_f32m1(-0.107254423828329604454f, vlmax);
#elif POW_POLY_DEGREE == 3
    const vfloat32m1_t cl2 = __riscv_vfmv_v_f_f32m1(2.28330284476918490682f, vlmax);
    const vfloat32m1_t cl1 = __riscv_vfmv_v_f_f32m1(-1.04913055217340124191f, vlmax);
    const vfloat32m1_t cl0 = __riscv_vfmv_v_f_f32m1(0.204446009836232697516f, vlmax);
#else
#error
#endif

    const vfloat32m1_t exp_hi = __riscv_vfmv_v_f_f32m1(88.376259f, vlmax);
    const vfloat32m1_t exp_lo = __riscv_vfmv_v_f_f32m1(-88.376259f, vlmax);
    const vfloat32m1_t log2EF = __riscv_vfmv_v_f_f32m1(1.442695f, vlmax);
    const vfloat32m1_t exp_C1 = __riscv_vfmv_v_f_f32m1(-0.6933594f, vlmax);
    const vfloat32m1_t exp_C2 = __riscv_vfmv_v_f_f32m1(0.000212194f, vlmax);
    const vfloat32m1_t cf1 = __riscv_vfmv_v_f_f32m1(1.0f, vlmax);
    const vfloat32m1_t cf1o2 = __riscv_vfmv_v_f_f32m1(0.5f, vlmax);
    const vfloat32m1_t ln2 = __riscv_vfmv_v_f_f32m1(0.6931471805f, vlmax);

    const vfloat32m1_t ce0 = __riscv_vfmv_v_f_f32m1(1.9875691500e-4, vlmax);
    const vfloat32m1_t ce1 = __riscv_vfmv_v_f_f32m1(1.3981999507e-3, vlmax);
    const vfloat32m1_t ce2 = __riscv_vfmv_v_f_f32m1(8.3334519073e-3, vlmax);
    const vfloat32m1_t ce3 = __riscv_vfmv_v_f_f32m1(4.1665795894e-2, vlmax);
    const vfloat32m1_t ce4 = __riscv_vfmv_v_f_f32m1(1.6666665459e-1, vlmax);
    const vfloat32m1_t ce5 = __riscv_vfmv_v_f_f32m1(5.0000001201e-1, vlmax);

    const vint32m1_t m1 = __riscv_vreinterpret_i32m1(cf1);
    const vint32m1_t m2 = __riscv_vmv_v_x_i32m1(0x7FFFFF, vlmax);
    const vint32m1_t c127 = __riscv_vmv_v_x_i32m1(127, vlmax);

    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, aVector += vl, bVector += vl, cVector += vl) {
        vl = __riscv_vsetvl_e32m1(n);
        vfloat32m1_t va = __riscv_vle32_v_f32m1(aVector, vl);
        vfloat32m1_t log;

        { /* log(a) */
            vfloat32m1_t a = __riscv_vfabs(va, vl);
            vfloat32m1_t exp = __riscv_vfcvt_f(
                __riscv_vsub(
                    __riscv_vsra(__riscv_vreinterpret_i32m1(a), 23, vl), c127, vl),
                vl);
            vfloat32m1_t frac = __riscv_vreinterpret_f32m1(__riscv_vor(
                __riscv_vand(__riscv_vreinterpret_i32m1(va), m2, vl), m1, vl));

            vfloat32m1_t mant = cl0;
            mant = __riscv_vfmadd(mant, frac, cl1, vl);
            mant = __riscv_vfmadd(mant, frac, cl2, vl);
#if POW_POLY_DEGREE >= 4
            mant = __riscv_vfmadd(mant, frac, cl3, vl);
#if POW_POLY_DEGREE >= 5
            mant = __riscv_vfmadd(mant, frac, cl4, vl);
#if POW_POLY_DEGREE >= 6
            mant = __riscv_vfmadd(mant, frac, cl5, vl);
#endif
#endif
#endif
            log = __riscv_vfmacc(exp, mant, __riscv_vfsub(frac, cf1, vl), vl);
            log = __riscv_vfmul(log, ln2, vl);
        }

        vfloat32m1_t vb = __riscv_vle32_v_f32m1(bVector, vl);
        vb = __riscv_vfmul(vb, log, vl); /* b*log(a) */
        vfloat32m1_t exp;

        { /* exp(b*log(a)) */
            vb = __riscv_vfmin(vb, exp_hi, vl);
            vb = __riscv_vfmax(vb, exp_lo, vl);
            vfloat32m1_t fx = __riscv_vfmadd(vb, log2EF, cf1o2, vl);

            vfloat32m1_t rtz = __riscv_vfcvt_f(__riscv_vfcvt_rtz_x(fx, vl), vl);
            fx = __riscv_vfsub_mu(__riscv_vmfgt(rtz, fx, vl), rtz, rtz, cf1, vl);
            vb = __riscv_vfmacc(vb, exp_C1, fx, vl);
            vb = __riscv_vfmacc(vb, exp_C2, fx, vl);
            vfloat32m1_t vv = __riscv_vfmul(vb, vb, vl);

            vfloat32m1_t y = ce0;
            y = __riscv_vfmadd(y, vb, ce1, vl);
            y = __riscv_vfmadd(y, vb, ce2, vl);
            y = __riscv_vfmadd(y, vb, ce3, vl);
            y = __riscv_vfmadd(y, vb, ce4, vl);
            y = __riscv_vfmadd(y, vb, ce5, vl);
            y = __riscv_vfmadd(y, vv, vb, vl);
            y = __riscv_vfadd(y, cf1, vl);

            vfloat32m1_t pow2n = __riscv_vreinterpret_f32m1(__riscv_vsll(
                __riscv_vadd(__riscv_vfcvt_rtz_x(fx, vl), c127, vl), 23, vl));

            exp = __riscv_vfmul(y, pow2n, vl);
        }

        __riscv_vse32(cVector, exp, vl);
    }
}

#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32f_x2_log2_32f_u_H */
