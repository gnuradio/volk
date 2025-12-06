/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_x2_multiply_32fc
 *
 * \b Overview
 *
 * Multiplies two complex vectors and returns the complex result.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_x2_multiply_32fc(lv_32fc_t* cVector, const lv_32fc_t* aVector, const
 * lv_32fc_t* bVector, unsigned int num_points); \endcode
 *
 * \b Inputs
 * \li aVector: The first input vector of complex floats.
 * \li bVector: The second input vector of complex floats.
 * \li num_points: The number of data points.
 *
 * \b Outputs
 * \li outputVector: The output vector complex floats.
 *
 * \b Example
 * Mix two signals at f=0.3 and 0.1.
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   lv_32fc_t* sig_1  = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *   lv_32fc_t* sig_2  = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *   lv_32fc_t* out = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       // Generate two tones
 *       float real_1 = std::cos(0.3f * (float)ii);
 *       float imag_1 = std::sin(0.3f * (float)ii);
 *       sig_1[ii] = lv_cmake(real_1, imag_1);
 *       float real_2 = std::cos(0.1f * (float)ii);
 *       float imag_2 = std::sin(0.1f * (float)ii);
 *       sig_2[ii] = lv_cmake(real_2, imag_2);
 *   }
 *
 *   volk_32fc_x2_multiply_32fc(out, sig_1, sig_2, N);
 * *
 *   volk_free(sig_1);
 *   volk_free(sig_2);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_x2_multiply_32fc_u_H
#define INCLUDED_volk_32fc_x2_multiply_32fc_u_H

#include <float.h>
#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_complex.h>

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>
/*!
  \brief Multiplies the two input complex vectors and stores their results in the third
  vector \param cVector The vector where the results will be stored \param aVector One of
  the vectors to be multiplied \param bVector One of the vectors to be multiplied \param
  num_points The number of complex values in aVector and bVector to be multiplied together
  and stored into cVector
*/
static inline void volk_32fc_x2_multiply_32fc_u_avx2_fma(lv_32fc_t* cVector,
                                                         const lv_32fc_t* aVector,
                                                         const lv_32fc_t* bVector,
                                                         unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    lv_32fc_t* c = cVector;
    const lv_32fc_t* a = aVector;
    const lv_32fc_t* b = bVector;

    for (; number < quarterPoints; number++) {

        const __m256 x =
            _mm256_loadu_ps((float*)a); // Load the ar + ai, br + bi as ar,ai,br,bi
        const __m256 y =
            _mm256_loadu_ps((float*)b); // Load the cr + ci, dr + di as cr,ci,dr,di

        const __m256 yl = _mm256_moveldup_ps(y); // Load yl with cr,cr,dr,dr
        const __m256 yh = _mm256_movehdup_ps(y); // Load yh with ci,ci,di,di

        const __m256 tmp2x = _mm256_permute_ps(x, 0xB1); // Re-arrange x to be ai,ar,bi,br

        const __m256 tmp2 = _mm256_mul_ps(tmp2x, yh); // tmp2 = ai*ci,ar*ci,bi*di,br*di

        const __m256 z = _mm256_fmaddsub_ps(
            x, yl, tmp2); // ar*cr-ai*ci, ai*cr+ar*ci, br*dr-bi*di, bi*dr+br*di

        _mm256_storeu_ps((float*)c, z); // Store the results back into the C container

        a += 4;
        b += 4;
        c += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *c++ = (*a++) * (*b++);
    }
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */


#ifdef LV_HAVE_AVX
#include <immintrin.h>
#include <volk/volk_avx_intrinsics.h>

static inline void volk_32fc_x2_multiply_32fc_u_avx(lv_32fc_t* cVector,
                                                    const lv_32fc_t* aVector,
                                                    const lv_32fc_t* bVector,
                                                    unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    __m256 x, y, z;
    lv_32fc_t* c = cVector;
    const lv_32fc_t* a = aVector;
    const lv_32fc_t* b = bVector;

    for (; number < quarterPoints; number++) {
        x = _mm256_loadu_ps(
            (float*)a); // Load the ar + ai, br + bi ... as ar,ai,br,bi ...
        y = _mm256_loadu_ps(
            (float*)b); // Load the cr + ci, dr + di ... as cr,ci,dr,di ...
        z = _mm256_complexmul_ps(x, y);
        _mm256_storeu_ps((float*)c, z); // Store the results back into the C container

        a += 4;
        b += 4;
        c += 4;
    }

    number = quarterPoints * 4;

    for (; number < num_points; number++) {
        *c++ = (*a++) * (*b++);
    }
}
#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>
#include <volk/volk_sse3_intrinsics.h>

static inline void volk_32fc_x2_multiply_32fc_u_sse3(lv_32fc_t* cVector,
                                                     const lv_32fc_t* aVector,
                                                     const lv_32fc_t* bVector,
                                                     unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int halfPoints = num_points / 2;

    __m128 x, y, z;
    lv_32fc_t* c = cVector;
    const lv_32fc_t* a = aVector;
    const lv_32fc_t* b = bVector;

    for (; number < halfPoints; number++) {
        x = _mm_loadu_ps((float*)a); // Load the ar + ai, br + bi as ar,ai,br,bi
        y = _mm_loadu_ps((float*)b); // Load the cr + ci, dr + di as cr,ci,dr,di
        z = _mm_complexmul_ps(x, y);
        _mm_storeu_ps((float*)c, z); // Store the results back into the C container

        a += 2;
        b += 2;
        c += 2;
    }

    if ((num_points % 2) != 0) {
        *c = (*a) * (*b);
    }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_GENERIC

static inline void volk_32fc_x2_multiply_32fc_generic(lv_32fc_t* cVector,
                                                      const lv_32fc_t* aVector,
                                                      const lv_32fc_t* bVector,
                                                      unsigned int num_points)
{
    lv_32fc_t* cPtr = cVector;
    const lv_32fc_t* aPtr = aVector;
    const lv_32fc_t* bPtr = bVector;
    unsigned int number = 0;

    for (number = 0; number < num_points; number++) {
        *cPtr++ = (*aPtr++) * (*bPtr++);
    }
}
#endif /* LV_HAVE_GENERIC */


#endif /* INCLUDED_volk_32fc_x2_multiply_32fc_u_H */
#ifndef INCLUDED_volk_32fc_x2_multiply_32fc_a_H
#define INCLUDED_volk_32fc_x2_multiply_32fc_a_H

#include <float.h>
#include <inttypes.h>
#include <stdio.h>
#include <volk/volk_complex.h>

#if LV_HAVE_AVX2 && LV_HAVE_FMA
#include <immintrin.h>
/*!
  \brief Multiplies the two input complex vectors and stores their results in the third
  vector \param cVector The vector where the results will be stored \param aVector One of
  the vectors to be multiplied \param bVector One of the vectors to be multiplied \param
  num_points The number of complex values in aVector and bVector to be multiplied together
  and stored into cVector
*/
static inline void volk_32fc_x2_multiply_32fc_a_avx2_fma(lv_32fc_t* cVector,
                                                         const lv_32fc_t* aVector,
                                                         const lv_32fc_t* bVector,
                                                         unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    lv_32fc_t* c = cVector;
    const lv_32fc_t* a = aVector;
    const lv_32fc_t* b = bVector;

    for (; number < quarterPoints; number++) {

        const __m256 x =
            _mm256_load_ps((float*)a); // Load the ar + ai, br + bi as ar,ai,br,bi
        const __m256 y =
            _mm256_load_ps((float*)b); // Load the cr + ci, dr + di as cr,ci,dr,di

        const __m256 yl = _mm256_moveldup_ps(y); // Load yl with cr,cr,dr,dr
        const __m256 yh = _mm256_movehdup_ps(y); // Load yh with ci,ci,di,di

        const __m256 tmp2x = _mm256_permute_ps(x, 0xB1); // Re-arrange x to be ai,ar,bi,br

        const __m256 tmp2 = _mm256_mul_ps(tmp2x, yh); // tmp2 = ai*ci,ar*ci,bi*di,br*di

        const __m256 z = _mm256_fmaddsub_ps(
            x, yl, tmp2); // ar*cr-ai*ci, ai*cr+ar*ci, br*dr-bi*di, bi*dr+br*di

        _mm256_store_ps((float*)c, z); // Store the results back into the C container

        a += 4;
        b += 4;
        c += 4;
    }

    number = quarterPoints * 4;
    for (; number < num_points; number++) {
        *c++ = (*a++) * (*b++);
    }
}
#endif /* LV_HAVE_AVX2 && LV_HAVE_FMA */


#ifdef LV_HAVE_AVX
#include <immintrin.h>
#include <volk/volk_avx_intrinsics.h>

static inline void volk_32fc_x2_multiply_32fc_a_avx(lv_32fc_t* cVector,
                                                    const lv_32fc_t* aVector,
                                                    const lv_32fc_t* bVector,
                                                    unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int quarterPoints = num_points / 4;

    __m256 x, y, z;
    lv_32fc_t* c = cVector;
    const lv_32fc_t* a = aVector;
    const lv_32fc_t* b = bVector;

    for (; number < quarterPoints; number++) {
        x = _mm256_load_ps((float*)a); // Load the ar + ai, br + bi ... as ar,ai,br,bi ...
        y = _mm256_load_ps((float*)b); // Load the cr + ci, dr + di ... as cr,ci,dr,di ...
        z = _mm256_complexmul_ps(x, y);
        _mm256_store_ps((float*)c, z); // Store the results back into the C container

        a += 4;
        b += 4;
        c += 4;
    }

    number = quarterPoints * 4;

    for (; number < num_points; number++) {
        *c++ = (*a++) * (*b++);
    }
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>
#include <volk/volk_sse3_intrinsics.h>

static inline void volk_32fc_x2_multiply_32fc_a_sse3(lv_32fc_t* cVector,
                                                     const lv_32fc_t* aVector,
                                                     const lv_32fc_t* bVector,
                                                     unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int halfPoints = num_points / 2;

    __m128 x, y, z;
    lv_32fc_t* c = cVector;
    const lv_32fc_t* a = aVector;
    const lv_32fc_t* b = bVector;

    for (; number < halfPoints; number++) {
        x = _mm_load_ps((float*)a); // Load the ar + ai, br + bi as ar,ai,br,bi
        y = _mm_load_ps((float*)b); // Load the cr + ci, dr + di as cr,ci,dr,di
        z = _mm_complexmul_ps(x, y);
        _mm_store_ps((float*)c, z); // Store the results back into the C container

        a += 2;
        b += 2;
        c += 2;
    }

    if ((num_points % 2) != 0) {
        *c = (*a) * (*b);
    }
}
#endif /* LV_HAVE_SSE */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>

static inline void volk_32fc_x2_multiply_32fc_neon(lv_32fc_t* cVector,
                                                   const lv_32fc_t* aVector,
                                                   const lv_32fc_t* bVector,
                                                   unsigned int num_points)
{
    lv_32fc_t* a_ptr = (lv_32fc_t*)aVector;
    lv_32fc_t* b_ptr = (lv_32fc_t*)bVector;
    unsigned int quarter_points = num_points / 4;
    float32x4x2_t a_val, b_val, c_val;
    float32x4x2_t tmp_real, tmp_imag;
    unsigned int number = 0;

    for (number = 0; number < quarter_points; ++number) {
        a_val = vld2q_f32((float*)a_ptr); // a0r|a1r|a2r|a3r || a0i|a1i|a2i|a3i
        b_val = vld2q_f32((float*)b_ptr); // b0r|b1r|b2r|b3r || b0i|b1i|b2i|b3i
        __VOLK_PREFETCH(a_ptr + 4);
        __VOLK_PREFETCH(b_ptr + 4);

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

    for (number = quarter_points * 4; number < num_points; number++) {
        *cVector++ = (*a_ptr++) * (*b_ptr++);
    }
}
#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_NEONV7

extern void volk_32fc_x2_multiply_32fc_a_neonasm(lv_32fc_t* cVector,
                                                 const lv_32fc_t* aVector,
                                                 const lv_32fc_t* bVector,
                                                 unsigned int num_points);
#endif /* LV_HAVE_NEONV7 */


#ifdef LV_HAVE_NEONV8
#include <arm_neon.h>

static inline void volk_32fc_x2_multiply_32fc_neonv8(lv_32fc_t* cVector,
                                                     const lv_32fc_t* aVector,
                                                     const lv_32fc_t* bVector,
                                                     unsigned int num_points)
{
    unsigned int n = num_points;
    lv_32fc_t* c = cVector;
    const lv_32fc_t* a = aVector;
    const lv_32fc_t* b = bVector;

    /* Process 8 complex numbers per iteration (2x unroll) */
    while (n >= 8) {
        /* Load and deinterleave: ar,ai separated */
        float32x4x2_t a0 = vld2q_f32((const float*)a);
        float32x4x2_t b0 = vld2q_f32((const float*)b);
        float32x4x2_t a1 = vld2q_f32((const float*)(a + 4));
        float32x4x2_t b1 = vld2q_f32((const float*)(b + 4));
        __VOLK_PREFETCH(a + 8);
        __VOLK_PREFETCH(b + 8);

        /* Complex multiply: (ar + ai*j)(br + bi*j) = (ar*br - ai*bi) + (ar*bi + ai*br)*j
         */
        /* Using FMA: real = ar*br - ai*bi, imag = ar*bi + ai*br */
        float32x4x2_t c0, c1;

        /* real part: ar*br - ai*bi = fms(ar*br, ai, bi) */
        c0.val[0] = vfmsq_f32(vmulq_f32(a0.val[0], b0.val[0]), a0.val[1], b0.val[1]);
        c1.val[0] = vfmsq_f32(vmulq_f32(a1.val[0], b1.val[0]), a1.val[1], b1.val[1]);

        /* imag part: ar*bi + ai*br = fma(ar*bi, ai, br) */
        c0.val[1] = vfmaq_f32(vmulq_f32(a0.val[0], b0.val[1]), a0.val[1], b0.val[0]);
        c1.val[1] = vfmaq_f32(vmulq_f32(a1.val[0], b1.val[1]), a1.val[1], b1.val[0]);

        vst2q_f32((float*)c, c0);
        vst2q_f32((float*)(c + 4), c1);

        a += 8;
        b += 8;
        c += 8;
        n -= 8;
    }

    /* Process remaining 4 complex numbers */
    if (n >= 4) {
        float32x4x2_t a0 = vld2q_f32((const float*)a);
        float32x4x2_t b0 = vld2q_f32((const float*)b);
        float32x4x2_t c0;
        c0.val[0] = vfmsq_f32(vmulq_f32(a0.val[0], b0.val[0]), a0.val[1], b0.val[1]);
        c0.val[1] = vfmaq_f32(vmulq_f32(a0.val[0], b0.val[1]), a0.val[1], b0.val[0]);
        vst2q_f32((float*)c, c0);
        a += 4;
        b += 4;
        c += 4;
        n -= 4;
    }

    /* Scalar tail */
    while (n > 0) {
        *c++ = (*a++) * (*b++);
        n--;
    }
}

#endif /* LV_HAVE_NEONV8 */


#ifdef LV_HAVE_ORC

extern void volk_32fc_x2_multiply_32fc_a_orc_impl(lv_32fc_t* cVector,
                                                  const lv_32fc_t* aVector,
                                                  const lv_32fc_t* bVector,
                                                  int num_points);

static inline void volk_32fc_x2_multiply_32fc_u_orc(lv_32fc_t* cVector,
                                                    const lv_32fc_t* aVector,
                                                    const lv_32fc_t* bVector,
                                                    unsigned int num_points)
{
    volk_32fc_x2_multiply_32fc_a_orc_impl(cVector, aVector, bVector, num_points);
}

#endif /* LV_HAVE_ORC */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32fc_x2_multiply_32fc_rvv(lv_32fc_t* cVector,
                                                  const lv_32fc_t* aVector,
                                                  const lv_32fc_t* bVector,
                                                  unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, aVector += vl, bVector += vl, cVector += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vuint64m8_t va = __riscv_vle64_v_u64m8((const uint64_t*)aVector, vl);
        vuint64m8_t vb = __riscv_vle64_v_u64m8((const uint64_t*)bVector, vl);
        vfloat32m4_t var = __riscv_vreinterpret_f32m4(__riscv_vnsrl(va, 0, vl));
        vfloat32m4_t vbr = __riscv_vreinterpret_f32m4(__riscv_vnsrl(vb, 0, vl));
        vfloat32m4_t vai = __riscv_vreinterpret_f32m4(__riscv_vnsrl(va, 32, vl));
        vfloat32m4_t vbi = __riscv_vreinterpret_f32m4(__riscv_vnsrl(vb, 32, vl));
        vfloat32m4_t vr = __riscv_vfnmsac(__riscv_vfmul(var, vbr, vl), vai, vbi, vl);
        vfloat32m4_t vi = __riscv_vfmacc(__riscv_vfmul(var, vbi, vl), vai, vbr, vl);
        vuint32m4_t vru = __riscv_vreinterpret_u32m4(vr);
        vuint32m4_t viu = __riscv_vreinterpret_u32m4(vi);
        vuint64m8_t v =
            __riscv_vwmaccu(__riscv_vwaddu_vv(vru, viu, vl), 0xFFFFFFFF, viu, vl);
        __riscv_vse64((uint64_t*)cVector, v, vl);
    }
}
#endif /*LV_HAVE_RVV*/

#ifdef LV_HAVE_RVVSEG
#include <riscv_vector.h>

static inline void volk_32fc_x2_multiply_32fc_rvvseg(lv_32fc_t* cVector,
                                                     const lv_32fc_t* aVector,
                                                     const lv_32fc_t* bVector,
                                                     unsigned int num_points)
{
    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, aVector += vl, bVector += vl, cVector += vl) {
        vl = __riscv_vsetvl_e32m4(n);
        vfloat32m4x2_t va = __riscv_vlseg2e32_v_f32m4x2((const float*)aVector, vl);
        vfloat32m4x2_t vb = __riscv_vlseg2e32_v_f32m4x2((const float*)bVector, vl);
        vfloat32m4_t var = __riscv_vget_f32m4(va, 0), vai = __riscv_vget_f32m4(va, 1);
        vfloat32m4_t vbr = __riscv_vget_f32m4(vb, 0), vbi = __riscv_vget_f32m4(vb, 1);
        vfloat32m4_t vr = __riscv_vfnmsac(__riscv_vfmul(var, vbr, vl), vai, vbi, vl);
        vfloat32m4_t vi = __riscv_vfmacc(__riscv_vfmul(var, vbi, vl), vai, vbr, vl);
        __riscv_vsseg2e32_v_f32m4x2(
            (float*)cVector, __riscv_vcreate_v_f32m4x2(vr, vi), vl);
    }
}
#endif /*LV_HAVE_RVVSEG*/

#endif /* INCLUDED_volk_32fc_x2_multiply_32fc_a_H */
