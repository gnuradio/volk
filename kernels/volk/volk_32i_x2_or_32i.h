/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32i_x2_or_32i
 *
 * \b Overview
 *
 * Computes the Boolean OR operation between two input 32-bit integer vectors.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32i_x2_or_32i(int32_t* cVector, const int32_t* aVector, const int32_t*
 * bVector, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li aVector: Input vector of samples.
 * \li bVector: Input vector of samples.
 * \li num_points: The number of values.
 *
 * \b Outputs
 * \li cVector: The output vector.
 *
 * \b Example
 * This example generates a Karnaugh map for the first two bits of x OR y
 * \code
 *   int N = 1<<4;
 *   unsigned int alignment = volk_get_alignment();
 *
 *   int32_t* x = (int32_t*)volk_malloc(N*sizeof(int32_t), alignment);
 *   int32_t* y = (int32_t*)volk_malloc(N*sizeof(int32_t), alignment);
 *   int32_t* z = (int32_t*)volk_malloc(N*sizeof(int32_t), alignment);
 *   int32_t in_seq[] = {0,1,3,2};
 *   unsigned int jj=0;
 *   for(unsigned int ii=0; ii<N; ++ii){
 *       x[ii] = in_seq[ii%4];
 *       y[ii] = in_seq[jj];
 *       if(((ii+1) % 4) == 0) jj++;
 *   }
 *
 *   volk_32i_x2_or_32i(z, x, y, N);
 *
 *   printf("Karnaugh map for x OR y\n");
 *   printf("y\\x|");
 *   for(unsigned int ii=0; ii<4; ++ii){
 *       printf(" %.2x ", in_seq[ii]);
 *   }
 *   printf("\n---|---------------\n");
 *   jj = 0;
 *   for(unsigned int ii=0; ii<N; ++ii){
 *       if(((ii+1) % 4) == 1){
 *           printf("%.2x | ", in_seq[jj++]);
 *       }
 *       printf("%.2x  ", z[ii]);
 *       if(!((ii+1) % 4)){
 *           printf("\n");
 *       }
 *   }
 * \endcode
 */

#ifndef INCLUDED_volk_32i_x2_or_32i_a_H
#define INCLUDED_volk_32i_x2_or_32i_a_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32i_x2_or_32i_a_avx512f(int32_t* cVector,
                                                const int32_t* aVector,
                                                const int32_t* bVector,
                                                unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    int32_t* cPtr = (int32_t*)cVector;
    const int32_t* aPtr = (int32_t*)aVector;
    const int32_t* bPtr = (int32_t*)bVector;

    __m512i aVal, bVal, cVal;
    for (; number < sixteenthPoints; number++) {

        aVal = _mm512_load_si512(aPtr);
        bVal = _mm512_load_si512(bPtr);

        cVal = _mm512_or_si512(aVal, bVal);

        _mm512_store_si512(cPtr, cVal); // Store the results back into the C container

        aPtr += 16;
        bPtr += 16;
        cPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        cVector[number] = aVector[number] | bVector[number];
    }
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32i_x2_or_32i_a_avx2(int32_t* cVector,
                                             const int32_t* aVector,
                                             const int32_t* bVector,
                                             unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int oneEightPoints = num_points / 8;

    int32_t* cPtr = cVector;
    const int32_t* aPtr = aVector;
    const int32_t* bPtr = bVector;

    __m256i aVal, bVal, cVal;
    for (; number < oneEightPoints; number++) {

        aVal = _mm256_load_si256((__m256i*)aPtr);
        bVal = _mm256_load_si256((__m256i*)bPtr);

        cVal = _mm256_or_si256(aVal, bVal);

        _mm256_store_si256((__m256i*)cPtr,
                           cVal); // Store the results back into the C container

        aPtr += 8;
        bPtr += 8;
        cPtr += 8;
    }

    number = oneEightPoints * 8;
    for (; number < num_points; number++) {
        cVector[number] = aVector[number] | bVector[number];
    }
}
#endif /* LV_HAVE_AVX2 */


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32i_x2_or_32i_a_sse(int32_t* cVector,
                                            const int32_t* aVector,
                                            const int32_t* bVector,
                                            unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    float* cPtr = (float*)cVector;
    const float* aPtr = (float*)aVector;
    const float* bPtr = (float*)bVector;

    __m128 aVal, bVal, cVal;
    for (; number < quarterPoints; number++) {
        aVal = _mm_load_ps(aPtr);
        bVal = _mm_load_ps(bPtr);

        cVal = _mm_or_ps(aVal, bVal);

        _mm_store_ps(cPtr, cVal); // Store the results back into the C container

        aPtr += 4;
        bPtr += 4;
        cPtr += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        cVector[number] = aVector[number] | bVector[number];
    }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32i_x2_or_32i_neon(int32_t* cVector,
                                           const int32_t* aVector,
                                           const int32_t* bVector,
                                           unsigned int num_points)
{
    int32_t* cPtr = cVector;
    const int32_t* aPtr = aVector;
    const int32_t* bPtr = bVector;
    unsigned int number = 0;
    unsigned int quarter_points = num_points / 4;

    int32x4_t a_val, b_val, c_val;

    for (number = 0; number < quarter_points; number++) {
        a_val = vld1q_s32(aPtr);
        b_val = vld1q_s32(bPtr);
        c_val = vorrq_s32(a_val, b_val);
        vst1q_s32(cPtr, c_val);
        aPtr += 4;
        bPtr += 4;
        cPtr += 4;
    }

    for (number = quarter_points * 4; number < num_points; number++) {
        *cPtr++ = (*aPtr++) | (*bPtr++);
    }
}
#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_GENERIC

static inline void volk_32i_x2_or_32i_generic(int32_t* cVector,
                                              const int32_t* aVector,
                                              const int32_t* bVector,
                                              unsigned int num_points)
{
    int32_t* cPtr = cVector;
    const int32_t* aPtr = aVector;
    const int32_t* bPtr = bVector;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        *cPtr++ = (*aPtr++) | (*bPtr++);
    }
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_ORC
extern void volk_32i_x2_or_32i_a_orc_impl(int32_t* cVector,
                                          const int32_t* aVector,
                                          const int32_t* bVector,
                                          int num_points);

static inline void volk_32i_x2_or_32i_u_orc(int32_t* cVector,
                                            const int32_t* aVector,
                                            const int32_t* bVector,
                                            unsigned int num_points)
{
    volk_32i_x2_or_32i_a_orc_impl(cVector, aVector, bVector, num_points);
}
#endif /* LV_HAVE_ORC */


#endif /* INCLUDED_volk_32i_x2_or_32i_a_H */


#ifndef INCLUDED_volk_32i_x2_or_32i_u_H
#define INCLUDED_volk_32i_x2_or_32i_u_H

#include <inttypes.h>
#include <stdio.h>

#ifdef LV_HAVE_AVX512F
#include <immintrin.h>

static inline void volk_32i_x2_or_32i_u_avx512f(int32_t* cVector,
                                                const int32_t* aVector,
                                                const int32_t* bVector,
                                                unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int sixteenthPoints = num_points / 16;

    int32_t* cPtr = (int32_t*)cVector;
    const int32_t* aPtr = (int32_t*)aVector;
    const int32_t* bPtr = (int32_t*)bVector;

    __m512i aVal, bVal, cVal;
    for (; number < sixteenthPoints; number++) {

        aVal = _mm512_loadu_si512(aPtr);
        bVal = _mm512_loadu_si512(bPtr);

        cVal = _mm512_or_si512(aVal, bVal);

        _mm512_storeu_si512(cPtr, cVal); // Store the results back into the C container

        aPtr += 16;
        bPtr += 16;
        cPtr += 16;
    }

    number = sixteenthPoints * 16;
    for (; number < num_points; number++) {
        cVector[number] = aVector[number] | bVector[number];
    }
}
#endif /* LV_HAVE_AVX512F */

#ifdef LV_HAVE_AVX2
#include <immintrin.h>

static inline void volk_32i_x2_or_32i_u_avx2(int32_t* cVector,
                                             const int32_t* aVector,
                                             const int32_t* bVector,
                                             unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int oneEightPoints = num_points / 8;

    int32_t* cPtr = cVector;
    const int32_t* aPtr = aVector;
    const int32_t* bPtr = bVector;

    __m256i aVal, bVal, cVal;
    for (; number < oneEightPoints; number++) {

        aVal = _mm256_loadu_si256((__m256i*)aPtr);
        bVal = _mm256_loadu_si256((__m256i*)bPtr);

        cVal = _mm256_or_si256(aVal, bVal);

        _mm256_storeu_si256((__m256i*)cPtr,
                            cVal); // Store the results back into the C container

        aPtr += 8;
        bPtr += 8;
        cPtr += 8;
    }

    number = oneEightPoints * 8;
    for (; number < num_points; number++) {
        cVector[number] = aVector[number] | bVector[number];
    }
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32i_x2_or_32i_rvv(int32_t* cVector,
                                          const int32_t* aVector,
                                          const int32_t* bVector,
                                          unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, aVector += vl, bVector += vl, cVector += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vint32m8_t va = __riscv_vle32_v_i32m8(aVector, vl);
        vint32m8_t vb = __riscv_vle32_v_i32m8(bVector, vl);
        __riscv_vse32(cVector, __riscv_vor(va, vb, vl), vl);
    }
}
#endif /*LV_HAVE_RVV*/

#endif /* INCLUDED_volk_32i_x2_or_32i_u_H */
