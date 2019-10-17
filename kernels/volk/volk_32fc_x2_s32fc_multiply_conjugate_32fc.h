/* -*- c++ -*- */
/*
 * Copyright 2019 Free Software Foundation, Inc.
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
 * \page volk_32fc_x2_s32fc_multiply_conjugate_32fc
 *
 * \b Overview
 *
 * Conjugate the input complex vector, multiply them by a complex scalar,
 * add the another input complex vector and returns the results.
 *
 * c[i] = a[i] + conj(b[i]) * scalar
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_x2_s32fc_multiply_conjugate_32fc(lv_32fc_t* cVector, const lv_32fc_t* aVector, const lv_32fc_t* bVector, const lv_32fc_t scalar, unsigned int num_points);
 * \endcode
 *
 * \b Inputs
 * \li aVector: The input vector to be added.
 * \li bVector: The input vector to be conjugate and multiplied.
 * \li scalar The complex scalar to multiply against conjugated bVector.
 * \li num_points: The number of complex values in aVector.
 *
 * \b Outputs
 * \li cVector: The vector where the results will be updated.
 *
 * \b Example
 *
 * \code
 *
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_x2_s32fc_multiply_conjugate_32fc_a_H
#define INCLUDED_volk_32fc_x2_s32fc_multiply_conjugate_32fc_a_H

#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_complex.h>
#include <float.h>


#ifdef LV_HAVE_AVX
#include <immintrin.h>

static inline void volk_32fc_x2_s32fc_multiply_conjugate_32fc_a_avx(lv_32fc_t* cVector, const lv_32fc_t* aVector, const lv_32fc_t* bVector, const lv_32fc_t scalar, unsigned int num_points) {
    unsigned int number = 0;
    unsigned int i = 0;
    const unsigned int quarterPoints = num_points / 4;
    unsigned int isodd = num_points & 3;

    __m256 x, sl, sh, y, z, tmp1, tmp2;

    const lv_32fc_t* a = aVector;
    const lv_32fc_t* b = bVector;
    lv_32fc_t* c = cVector;

    __m256 conjugator = _mm256_setr_ps(0, -0.f, 0, -0.f, 0, -0.f, 0, -0.f);

    // Set up constant scalar vector
    sl = _mm256_set1_ps(lv_creal(scalar));
    sh = _mm256_set1_ps(lv_cimag(scalar));

    for(;number < quarterPoints; number++) {
        x = _mm256_load_ps((float*)b);
        y = _mm256_load_ps((float*)a);

        x = _mm256_xor_ps(x, conjugator);

        tmp1 = _mm256_mul_ps(x,sl);

        x = _mm256_shuffle_ps(x,x,0xB1);

        tmp2 = _mm256_mul_ps(x,sh);

        z = _mm256_addsub_ps(tmp1,tmp2);

        z = _mm256_add_ps(y, z);

        _mm256_store_ps((float*)c,z);

        a += 4;
        b += 4;
        c += 4;
    }

    for(i = num_points-isodd; i < num_points; i++) {
        *c++ = (*a++) + lv_conj(*b++) * scalar;
    }

}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>

static inline void volk_32fc_x2_s32fc_multiply_conjugate_32fc_a_sse3(lv_32fc_t* cVector, const lv_32fc_t* aVector, const lv_32fc_t* bVector, const lv_32fc_t scalar, unsigned int num_points) {
    unsigned int number = 0;
    const unsigned int halfPoints = num_points / 2;

    __m128 x, sl, sh, y, z, tmp1, tmp2;

    const lv_32fc_t* a = aVector;
    const lv_32fc_t* b = bVector;
    lv_32fc_t* c = cVector;

    __m128 conjugator = _mm_setr_ps(0, -0.f, 0, -0.f);

    // Set up constant scalar vector
    sl = _mm_set_ps1(lv_creal(scalar));
    sh = _mm_set_ps1(lv_cimag(scalar));

    for(;number < halfPoints; number++){
        x = _mm_load_ps((float*)b);
        y = _mm_load_ps((float*)a);

        x = _mm_xor_ps(x, conjugator);

        tmp1 = _mm_mul_ps(x,sl);

        x = _mm_shuffle_ps(x,x,0xB1);

        tmp2 = _mm_mul_ps(x,sh);

        z = _mm_addsub_ps(tmp1,tmp2);

        z = _mm_add_ps(y, z);

        _mm_store_ps((float*)c,z);

        a += 2;
        b += 2;
        c += 2;
    }

    if((num_points % 2) != 0) {
        *c = *a + lv_conj(*b) * scalar;
    }
}
#endif /* LV_HAVE_SSE */

#endif /* INCLUDED_volk_32fc_x2_s32fc_multiply_conjugate_32fc_a_H */


#ifndef INCLUDED_volk_32fc_x2_s32fc_multiply_conjugate_32fc_u_H
#define INCLUDED_volk_32fc_x2_s32fc_multiply_conjugate_32fc_u_H

#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_complex.h>
#include <float.h>


#ifdef LV_HAVE_NEON
#include  <arm_neon.h>

static inline void volk_32fc_x2_s32fc_multiply_conjugate_32fc_neon(lv_32fc_t* cVector, const lv_32fc_t* aVector, const lv_32fc_t* bVector, const lv_32fc_t scalar, unsigned int num_points){
    const lv_32fc_t* bPtr = bVector;
    const lv_32fc_t* aPtr = aVector;
    lv_32fc_t* cPtr = cVector;
    unsigned int number = num_points;
    unsigned int quarter_points = num_points / 4;

    float32x4x2_t a_val, b_val, c_val, scalar_val;
    float32x4x2_t tmp_val;

    scalar_val.val[0] = vld1q_dup_f32((const float*)&scalar);
    scalar_val.val[1] = vld1q_dup_f32(((const float*)&scalar) + 1);

    for(number = 0; number < quarter_points; ++number) {
        a_val = vld2q_f32((float*)aPtr);
        b_val = vld2q_f32((float*)bPtr);
        b_val.val[1] = vnegq_f32(b_val.val[1]);
        __VOLK_PREFETCH(aPtr + 8);
        __VOLK_PREFETCH(bPtr + 8);

        tmp_val.val[1] = vmulq_f32(b_val.val[1], scalar_val.val[0]);
        tmp_val.val[0] = vmulq_f32(b_val.val[0], scalar_val.val[0]);

        tmp_val.val[1] = vmlaq_f32(tmp_val.val[1], b_val.val[0], scalar_val.val[1]);
        tmp_val.val[0] = vmlsq_f32(tmp_val.val[0], b_val.val[1], scalar_val.val[1]);

        c_val.val[1] = vaddq_f32(a_val.val[1], tmp_val.val[1]);
        c_val.val[0] = vaddq_f32(a_val.val[0], tmp_val.val[0]);

        vst2q_f32((float*)cPtr, c_val);

        aPtr += 4;
        bPtr += 4;
        cPtr += 4;
    }

    for(number = quarter_points*4; number < num_points; number++){
        *cPtr++ = (*aPtr++) + lv_conj(*bPtr++) * scalar;
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_GENERIC

static inline void volk_32fc_x2_s32fc_multiply_conjugate_32fc_generic(lv_32fc_t* cVector, const lv_32fc_t* aVector, const lv_32fc_t* bVector, const lv_32fc_t scalar, unsigned int num_points){
    const lv_32fc_t* aPtr = aVector;
    const lv_32fc_t* bPtr = bVector;
    lv_32fc_t* cPtr = cVector;
    unsigned int number = num_points;

    // unwrap loop
    while (number >= 8){
        *cPtr++ = (*aPtr++) + lv_conj(*bPtr++) * scalar;
        *cPtr++ = (*aPtr++) + lv_conj(*bPtr++) * scalar;
        *cPtr++ = (*aPtr++) + lv_conj(*bPtr++) * scalar;
        *cPtr++ = (*aPtr++) + lv_conj(*bPtr++) * scalar;
        *cPtr++ = (*aPtr++) + lv_conj(*bPtr++) * scalar;
        *cPtr++ = (*aPtr++) + lv_conj(*bPtr++) * scalar;
        *cPtr++ = (*aPtr++) + lv_conj(*bPtr++) * scalar;
        *cPtr++ = (*aPtr++) + lv_conj(*bPtr++) * scalar;
        number -= 8;
    }

    // clean up any remaining
    while (number-- > 0)
        *cPtr++ = (*aPtr++) + lv_conj(*bPtr++) * scalar;
}
#endif /* LV_HAVE_GENERIC */

#endif /* INCLUDED_volk_32fc_x2_s32fc_multiply_conjugate_32fc_u_H */
