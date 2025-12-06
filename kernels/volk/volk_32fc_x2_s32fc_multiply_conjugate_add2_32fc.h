/* -*- c++ -*- */
/*
 * Copyright 2019 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc
 *
 * \b Overview
 *
 * Conjugate the input complex vector, multiply them by a complex scalar,
 * add the another input complex vector and returns the results.
 *
 * c[i] = a[i] + conj(b[i]) * (*scalar)
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc(lv_32fc_t* cVector, const
 * lv_32fc_t* aVector, const lv_32fc_t* bVector, const lv_32fc_t* scalar, unsigned int
 * num_points); \endcode
 *
 * \b Inputs
 * \li aVector: The input vector to be added.
 * \li bVector: The input vector to be conjugate and multiplied.
 * \li scalar: The complex scalar to multiply against conjugated bVector.
 * \li num_points: The number of complex values in aVector and bVector to be conjugate,
 * multiplied and stored into cVector.
 *
 * \b Outputs
 * \li cVector: The vector where the results will be stored.
 *
 * \b Example
 * Calculate coefficients.
 *
 * \code
 * int n_filter = 2 * N + 1;
 * unsigned int alignment = volk_get_alignment();
 *
 * // state is a queue of input IQ data.
 * lv_32fc_t* state = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * n_filter, alignment);
 * // weight and next one.
 * lv_32fc_t* weight = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * n_filter, alignment);
 * lv_32fc_t* next = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t) * n_filter, alignment);
 * ...
 * // push back input IQ data into state.
 * foo_push_back_queue(state, input);
 *
 * // get filtered output.
 * lv_32fc_t output = lv_cmake(0.f,0.f);
 * for (int i = 0; i < n_filter; i++) {
 *   output += state[i] * weight[i];
 * }
 *
 * // update weight using output.
 * float real = lv_creal(output) * (1.0 - std::norm(output)) * MU;
 * lv_32fc_t factor = lv_cmake(real, 0.f);
 * volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc(
 *     next, weight, state, &factor, n_filter);
 * lv_32fc_t *tmp = next;
 * next = weight;
 * weight = tmp;
 * weight[N + 1] = lv_cmake(lv_creal(weight[N + 1]), 0.f);
 * ...
 * volk_free(state);
 * volk_free(weight);
 * volk_free(next);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_H
#define INCLUDED_volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_H

#include <float.h>
#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_complex.h>


#ifdef LV_HAVE_GENERIC

static inline void
volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_generic(lv_32fc_t* cVector,
                                                        const lv_32fc_t* aVector,
                                                        const lv_32fc_t* bVector,
                                                        const lv_32fc_t* scalar,
                                                        unsigned int num_points)
{
    const lv_32fc_t* aPtr = aVector;
    const lv_32fc_t* bPtr = bVector;
    lv_32fc_t* cPtr = cVector;
    unsigned int number = num_points;

    // unwrap loop
    while (number >= 8) {
        *cPtr++ = (*aPtr++) + lv_conj(*bPtr++) * (*scalar);
        *cPtr++ = (*aPtr++) + lv_conj(*bPtr++) * (*scalar);
        *cPtr++ = (*aPtr++) + lv_conj(*bPtr++) * (*scalar);
        *cPtr++ = (*aPtr++) + lv_conj(*bPtr++) * (*scalar);
        *cPtr++ = (*aPtr++) + lv_conj(*bPtr++) * (*scalar);
        *cPtr++ = (*aPtr++) + lv_conj(*bPtr++) * (*scalar);
        *cPtr++ = (*aPtr++) + lv_conj(*bPtr++) * (*scalar);
        *cPtr++ = (*aPtr++) + lv_conj(*bPtr++) * (*scalar);
        number -= 8;
    }

    // clean up any remaining
    while (number-- > 0) {
        *cPtr++ = (*aPtr++) + lv_conj(*bPtr++) * (*scalar);
    }
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_AVX
#include <immintrin.h>
#include <volk/volk_avx_intrinsics.h>

static inline void
volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_u_avx(lv_32fc_t* cVector,
                                                      const lv_32fc_t* aVector,
                                                      const lv_32fc_t* bVector,
                                                      const lv_32fc_t* scalar,
                                                      unsigned int num_points)
{
    unsigned int number = 0;
    unsigned int i = 0;
    const unsigned int quarterPoints = num_points / 4;
    unsigned int isodd = num_points & 3;

    __m256 x, y, s, z;
    lv_32fc_t v_scalar[4] = { *scalar, *scalar, *scalar, *scalar };

    const lv_32fc_t* a = aVector;
    const lv_32fc_t* b = bVector;
    lv_32fc_t* c = cVector;

    // Set up constant scalar vector
    s = _mm256_loadu_ps((float*)v_scalar);

    for (; number < quarterPoints; number++) {
        x = _mm256_loadu_ps((float*)b);
        y = _mm256_loadu_ps((float*)a);
        z = _mm256_complexconjugatemul_ps(s, x);
        z = _mm256_add_ps(y, z);
        _mm256_storeu_ps((float*)c, z);

        a += 4;
        b += 4;
        c += 4;
    }

    for (i = num_points - isodd; i < num_points; i++) {
        *c++ = (*a++) + lv_conj(*b++) * (*scalar);
    }
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>
#include <volk/volk_sse3_intrinsics.h>

static inline void
volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_u_sse3(lv_32fc_t* cVector,
                                                       const lv_32fc_t* aVector,
                                                       const lv_32fc_t* bVector,
                                                       const lv_32fc_t* scalar,
                                                       unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int halfPoints = num_points / 2;

    __m128 x, y, s, z;
    lv_32fc_t v_scalar[2] = { *scalar, *scalar };

    const lv_32fc_t* a = aVector;
    const lv_32fc_t* b = bVector;
    lv_32fc_t* c = cVector;

    // Set up constant scalar vector
    s = _mm_loadu_ps((float*)v_scalar);

    for (; number < halfPoints; number++) {
        x = _mm_loadu_ps((float*)b);
        y = _mm_loadu_ps((float*)a);
        z = _mm_complexconjugatemul_ps(s, x);
        z = _mm_add_ps(y, z);
        _mm_storeu_ps((float*)c, z);

        a += 2;
        b += 2;
        c += 2;
    }

    if ((num_points % 2) != 0) {
        *c = *a + lv_conj(*b) * (*scalar);
    }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_AVX
#include <immintrin.h>
#include <volk/volk_avx_intrinsics.h>

static inline void
volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_a_avx(lv_32fc_t* cVector,
                                                      const lv_32fc_t* aVector,
                                                      const lv_32fc_t* bVector,
                                                      const lv_32fc_t* scalar,
                                                      unsigned int num_points)
{
    unsigned int number = 0;
    unsigned int i = 0;
    const unsigned int quarterPoints = num_points / 4;
    unsigned int isodd = num_points & 3;

    __m256 x, y, s, z;
    lv_32fc_t v_scalar[4] = { *scalar, *scalar, *scalar, *scalar };

    const lv_32fc_t* a = aVector;
    const lv_32fc_t* b = bVector;
    lv_32fc_t* c = cVector;

    // Set up constant scalar vector
    s = _mm256_loadu_ps((float*)v_scalar);

    for (; number < quarterPoints; number++) {
        x = _mm256_load_ps((float*)b);
        y = _mm256_load_ps((float*)a);
        z = _mm256_complexconjugatemul_ps(s, x);
        z = _mm256_add_ps(y, z);
        _mm256_store_ps((float*)c, z);

        a += 4;
        b += 4;
        c += 4;
    }

    for (i = num_points - isodd; i < num_points; i++) {
        *c++ = (*a++) + lv_conj(*b++) * (*scalar);
    }
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>
#include <volk/volk_sse3_intrinsics.h>

static inline void
volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_a_sse3(lv_32fc_t* cVector,
                                                       const lv_32fc_t* aVector,
                                                       const lv_32fc_t* bVector,
                                                       const lv_32fc_t* scalar,
                                                       unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int halfPoints = num_points / 2;

    __m128 x, y, s, z;
    lv_32fc_t v_scalar[2] = { *scalar, *scalar };

    const lv_32fc_t* a = aVector;
    const lv_32fc_t* b = bVector;
    lv_32fc_t* c = cVector;

    // Set up constant scalar vector
    s = _mm_loadu_ps((float*)v_scalar);

    for (; number < halfPoints; number++) {
        x = _mm_load_ps((float*)b);
        y = _mm_load_ps((float*)a);
        z = _mm_complexconjugatemul_ps(s, x);
        z = _mm_add_ps(y, z);
        _mm_store_ps((float*)c, z);

        a += 2;
        b += 2;
        c += 2;
    }

    if ((num_points % 2) != 0) {
        *c = *a + lv_conj(*b) * (*scalar);
    }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void
volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_neon(lv_32fc_t* cVector,
                                                     const lv_32fc_t* aVector,
                                                     const lv_32fc_t* bVector,
                                                     const lv_32fc_t* scalar,
                                                     unsigned int num_points)
{
    const lv_32fc_t* bPtr = bVector;
    const lv_32fc_t* aPtr = aVector;
    lv_32fc_t* cPtr = cVector;
    unsigned int number = num_points;
    unsigned int quarter_points = num_points / 4;

    float32x4x2_t a_val, b_val, c_val, scalar_val;
    float32x4x2_t tmp_val;

    scalar_val.val[0] = vld1q_dup_f32((const float*)scalar);
    scalar_val.val[1] = vld1q_dup_f32(((const float*)scalar) + 1);

    for (number = 0; number < quarter_points; ++number) {
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

    for (number = quarter_points * 4; number < num_points; number++) {
        *cPtr++ = (*aPtr++) + lv_conj(*bPtr++) * (*scalar);
    }
}
#endif /* LV_HAVE_NEON */

#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void
volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_neonv8(lv_32fc_t* cVector,
                                                       const lv_32fc_t* aVector,
                                                       const lv_32fc_t* bVector,
                                                       const lv_32fc_t* scalar,
                                                       unsigned int num_points)
{
    const lv_32fc_t* bPtr = bVector;
    const lv_32fc_t* aPtr = aVector;
    lv_32fc_t* cPtr = cVector;
    unsigned int number = num_points;
    unsigned int quarter_points = num_points / 4;

    float32x4x2_t a_val, b_val, c_val;
    float32x4x2_t tmp_val;

    float32x4_t scalar_real = vdupq_n_f32(lv_creal(*scalar));
    float32x4_t scalar_imag = vdupq_n_f32(lv_cimag(*scalar));

    for (number = 0; number < quarter_points; ++number) {
        a_val = vld2q_f32((float*)aPtr);
        b_val = vld2q_f32((float*)bPtr);
        __VOLK_PREFETCH(aPtr + 8);
        __VOLK_PREFETCH(bPtr + 8);

        /* Conjugate b: negate imaginary part */
        /* tmp_real = br*sr - (-bi)*si = br*sr + bi*si */
        tmp_val.val[0] =
            vfmaq_f32(vmulq_f32(b_val.val[0], scalar_real), b_val.val[1], scalar_imag);

        /* tmp_imag = (-bi)*sr + br*si = br*si - bi*sr */
        tmp_val.val[1] =
            vfmsq_f32(vmulq_f32(b_val.val[0], scalar_imag), b_val.val[1], scalar_real);

        /* c = a + tmp */
        c_val.val[0] = vaddq_f32(a_val.val[0], tmp_val.val[0]);
        c_val.val[1] = vaddq_f32(a_val.val[1], tmp_val.val[1]);

        vst2q_f32((float*)cPtr, c_val);

        aPtr += 4;
        bPtr += 4;
        cPtr += 4;
    }

    for (number = quarter_points * 4; number < num_points; number++) {
        *cPtr++ = (*aPtr++) + lv_conj(*bPtr++) * (*scalar);
    }
}
#endif /* LV_HAVE_NEONV8 */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void
volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_rvv(lv_32fc_t* cVector,
                                                    const lv_32fc_t* aVector,
                                                    const lv_32fc_t* bVector,
                                                    const lv_32fc_t* scalar,
                                                    unsigned int num_points)
{
    vfloat32m2_t vbr =
        __riscv_vfmv_v_f_f32m2(lv_creal(*scalar), __riscv_vsetvlmax_e32m2());
    vfloat32m2_t vbi =
        __riscv_vfmv_v_f_f32m2(lv_cimag(*scalar), __riscv_vsetvlmax_e32m2());
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, bVector += vl, aVector += vl, cVector += vl) {
        vl = __riscv_vsetvl_e32m2(n);
        vuint64m4_t va = __riscv_vle64_v_u64m4((const uint64_t*)bVector, vl);
        vuint64m4_t vc = __riscv_vle64_v_u64m4((const uint64_t*)aVector, vl);
        vfloat32m2_t var = __riscv_vreinterpret_f32m2(__riscv_vnsrl(va, 0, vl));
        vfloat32m2_t vcr = __riscv_vreinterpret_f32m2(__riscv_vnsrl(vc, 0, vl));
        vfloat32m2_t vai = __riscv_vreinterpret_f32m2(__riscv_vnsrl(va, 32, vl));
        vfloat32m2_t vci = __riscv_vreinterpret_f32m2(__riscv_vnsrl(vc, 32, vl));
        vfloat32m2_t vr = __riscv_vfmacc(__riscv_vfmul(var, vbr, vl), vai, vbi, vl);
        vfloat32m2_t vi = __riscv_vfnmsac(__riscv_vfmul(var, vbi, vl), vai, vbr, vl);
        vuint32m2_t vru = __riscv_vreinterpret_u32m2(__riscv_vfadd(vr, vcr, vl));
        vuint32m2_t viu = __riscv_vreinterpret_u32m2(__riscv_vfadd(vi, vci, vl));
        vuint64m4_t v =
            __riscv_vwmaccu(__riscv_vwaddu_vv(vru, viu, vl), 0xFFFFFFFF, viu, vl);
        __riscv_vse64((uint64_t*)cVector, v, vl);
    }
}
#endif /*LV_HAVE_RVV*/

#ifdef LV_HAVE_RVVSEG
#include <riscv_vector.h>

static inline void
volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_rvvseg(lv_32fc_t* cVector,
                                                       const lv_32fc_t* aVector,
                                                       const lv_32fc_t* bVector,
                                                       const lv_32fc_t* scalar,
                                                       unsigned int num_points)
{
    vfloat32m4_t vbr =
        __riscv_vfmv_v_f_f32m4(lv_creal(*scalar), __riscv_vsetvlmax_e32m4());
    vfloat32m4_t vbi =
        __riscv_vfmv_v_f_f32m4(lv_cimag(*scalar), __riscv_vsetvlmax_e32m4());
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, aVector += vl, bVector += vl, cVector += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4x2_t vc = __riscv_vlseg2e32_v_f32m4x2((const float*)aVector, vl);
        vfloat32m4x2_t va = __riscv_vlseg2e32_v_f32m4x2((const float*)bVector, vl);
        vfloat32m4_t vcr = __riscv_vget_f32m4(vc, 0), vci = __riscv_vget_f32m4(vc, 1);
        vfloat32m4_t var = __riscv_vget_f32m4(va, 0), vai = __riscv_vget_f32m4(va, 1);
        vfloat32m4_t vr = __riscv_vfmacc(__riscv_vfmul(var, vbr, vl), vai, vbi, vl);
        vfloat32m4_t vi = __riscv_vfnmsac(__riscv_vfmul(var, vbi, vl), vai, vbr, vl);
        vr = __riscv_vfadd(vr, vcr, vl);
        vi = __riscv_vfadd(vi, vci, vl);
        __riscv_vsseg2e32_v_f32m4x2(
            (float*)cVector, __riscv_vcreate_v_f32m4x2(vr, vi), vl);
    }
}
#endif /*LV_HAVE_RVVSEG*/

#endif /* INCLUDED_volk_32fc_x2_s32fc_multiply_conjugate_add2_32fc_H */
