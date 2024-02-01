/* -*- c++ -*- */
/*
 * Copyright 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_8u_x4_conv_k7_r2_8u
 *
 * \b Overview
 *
 * Performs convolutional decoding for a K=7, rate 1/2 convolutional
 * code. The polynomials user defined.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_8u_x4_conv_k7_r2_8u(unsigned char* Y, unsigned char* X, unsigned char* syms,
 * unsigned char* dec, unsigned int framebits, unsigned int excess, unsigned char*
 * Branchtab) \endcode
 *
 * \b Inputs
 * \li X: <FIXME>
 * \li syms: <FIXME>
 * \li dec: <FIXME>
 * \li framebits: size of the frame to decode in bits.
 * \li excess: <FIXME>
 * \li Branchtab: <FIXME>
 *
 * \b Outputs
 * \li Y: The decoded output bits.
 *
 * \b Example
 * \code
 * int N = 10000;
 *
 * volk_8u_x4_conv_k7_r2_8u();
 *
 * volk_free(x);
 * \endcode
 */

#ifndef INCLUDED_volk_8u_x4_conv_k7_r2_8u_H
#define INCLUDED_volk_8u_x4_conv_k7_r2_8u_H

typedef union {
    unsigned char /*DECISIONTYPE*/ t[64 /*NUMSTATES*/ / 8 /*DECISIONTYPE_BITSIZE*/];
    unsigned int w[64 /*NUMSTATES*/ / 32];
    unsigned short s[64 /*NUMSTATES*/ / 16];
    unsigned char c[64 /*NUMSTATES*/ / 8];
#ifdef _MSC_VER
} decision_t;
#else
} decision_t __attribute__((aligned(16)));
#endif


static inline void renormalize(unsigned char* X)
{
    int NUMSTATES = 64;
    int i;

    unsigned char min = X[0];
    for (i = 0; i < NUMSTATES; i++)
        if (min > X[i])
            min = X[i];
    for (i = 0; i < NUMSTATES; i++)
        X[i] -= min;
}


// helper BFLY for GENERIC version
static inline void BFLY(int i,
                        int s,
                        unsigned char* syms,
                        unsigned char* Y,
                        unsigned char* X,
                        decision_t* d,
                        unsigned char* Branchtab)
{
    int j;
    unsigned int decision0, decision1;
    unsigned char metric, m0, m1, m2, m3;
    unsigned short metricsum;

    int NUMSTATES = 64;
    int RATE = 2;
    int METRICSHIFT = 1;
    int PRECISIONSHIFT = 2;

    metricsum = 1;
    for (j = 0; j < RATE; j++)
        metricsum += (Branchtab[i + j * NUMSTATES / 2] ^ syms[s * RATE + j]);
    metric = (metricsum >> METRICSHIFT) >> PRECISIONSHIFT;

    unsigned char max = ((RATE * ((256 - 1) >> METRICSHIFT)) >> PRECISIONSHIFT);

    m0 = X[i] + metric;
    m1 = X[i + NUMSTATES / 2] + (max - metric);
    m2 = X[i] + (max - metric);
    m3 = X[i + NUMSTATES / 2] + metric;

    decision0 = (signed int)(m0 - m1) >= 0;
    decision1 = (signed int)(m2 - m3) >= 0;

    Y[2 * i] = decision0 ? m1 : m0;
    Y[2 * i + 1] = decision1 ? m3 : m2;

    d->w[i / (sizeof(unsigned int) * 8 / 2) +
         s * (sizeof(decision_t) / sizeof(unsigned int))] |=
        (decision0 | decision1 << 1) << ((2 * i) & (sizeof(unsigned int) * 8 - 1));
}


#if LV_HAVE_AVX2

#include <immintrin.h>
#include <stdio.h>

static inline void volk_8u_x4_conv_k7_r2_8u_avx2(unsigned char* Y,
                                                 unsigned char* X,
                                                 unsigned char* syms,
                                                 unsigned char* dec,
                                                 unsigned int framebits,
                                                 unsigned int excess,
                                                 unsigned char* Branchtab)
{
    unsigned int i;
    for (i = 0; i < framebits + excess; i++) {
        unsigned char* tmp;
        unsigned int* dec_int = (unsigned int*)dec;
        __m256i a76, a78, a79, a82, a84, a85, a86, a88, a89, a90, d10, d9, m23, m24, m25,
            m26, s18, s19, s22, s23, t14, t15;

        // Butterfly
        s18 = ((__m256i*)X)[0];
        s19 = ((__m256i*)X)[1];
        a76 = _mm256_set1_epi8(syms[2 * i]);
        a78 = ((__m256i*)Branchtab)[0];
        a79 = _mm256_xor_si256(a76, a78);
        a82 = _mm256_set1_epi8(syms[2 * i + 1]);
        a84 = ((__m256i*)Branchtab)[1];
        a85 = _mm256_xor_si256(a82, a84);
        a86 = _mm256_avg_epu8(a79, a85);
        a88 = _mm256_srli_epi16(a86, 2);
        t14 = _mm256_and_si256(a88, _mm256_set1_epi8(63));
        t15 = _mm256_subs_epu8(_mm256_set1_epi8(63), t14);
        m23 = _mm256_adds_epu8(s18, t14);
        m24 = _mm256_adds_epu8(s19, t15);
        m25 = _mm256_adds_epu8(s18, t15);
        m26 = _mm256_adds_epu8(s19, t14);
        a89 = _mm256_min_epu8(m24, m23);
        d9 = _mm256_cmpeq_epi8(a89, m24);
        a90 = _mm256_min_epu8(m26, m25);
        d10 = _mm256_cmpeq_epi8(a90, m26);
        s22 = _mm256_unpacklo_epi8(d9, d10);
        s23 = _mm256_unpackhi_epi8(d9, d10);
        dec_int[2 * i] = _mm256_movemask_epi8(_mm256_permute2x128_si256(s22, s23, 0x20));
        dec_int[2 * i + 1] =
            _mm256_movemask_epi8(_mm256_permute2x128_si256(s22, s23, 0x31));
        s22 = _mm256_unpacklo_epi8(a89, a90);
        s23 = _mm256_unpackhi_epi8(a89, a90);
        ((__m256i*)Y)[0] = _mm256_permute2x128_si256(s22, s23, 0x20);
        ((__m256i*)Y)[1] = _mm256_permute2x128_si256(s22, s23, 0x31);

        // Renormalize
        __m256i m5, m6;
        m5 = ((__m256i*)Y)[0];
        m5 = _mm256_min_epu8(m5, ((__m256i*)Y)[1]);
        m5 = ((__m256i)_mm256_min_epu8(_mm256_permute2x128_si256(m5, m5, 0x21), m5));
        __m256i m7;
        m7 = _mm256_min_epu8(_mm256_srli_si256(m5, 8), m5);
        m7 = ((__m256i)_mm256_min_epu8(((__m256i)_mm256_srli_epi64(m7, 32)),
                                       ((__m256i)m7)));
        m7 = ((__m256i)_mm256_min_epu8(((__m256i)_mm256_srli_epi64(m7, 16)),
                                       ((__m256i)m7)));
        m7 = ((__m256i)_mm256_min_epu8(((__m256i)_mm256_srli_epi64(m7, 8)),
                                       ((__m256i)m7)));
        m7 = _mm256_unpacklo_epi8(m7, m7);
        m7 = _mm256_shufflelo_epi16(m7, 0);
        m6 = _mm256_unpacklo_epi64(m7, m7);
        m6 = _mm256_permute2x128_si256(
            m6, m6, 0); // copy lower half of m6 to upper half, since above ops
                        // operate on 128 bit lanes
        ((__m256i*)Y)[0] = _mm256_subs_epu8(((__m256i*)Y)[0], m6);
        ((__m256i*)Y)[1] = _mm256_subs_epu8(((__m256i*)Y)[1], m6);

        // Swap pointers to old and new metrics
        tmp = X;
        X = Y;
        Y = tmp;
    }
}

#endif /*LV_HAVE_AVX2*/


#if LV_HAVE_SSE3

#include <emmintrin.h>
#include <mmintrin.h>
#include <pmmintrin.h>
#include <stdio.h>
#include <xmmintrin.h>

static inline void volk_8u_x4_conv_k7_r2_8u_spiral(unsigned char* Y,
                                                   unsigned char* X,
                                                   unsigned char* syms,
                                                   unsigned char* dec,
                                                   unsigned int framebits,
                                                   unsigned int excess,
                                                   unsigned char* Branchtab)
{
    unsigned int i;
    for (i = 0; i < framebits + excess; i++) {
        unsigned char* tmp;
        unsigned short* dec_short = (unsigned short*)dec;
        __m128i a100, a101, a103, a104, a105, a107, a108, a109, a76, a78, a79, a82, a84,
            a85, a86, a88, a89, a90, d10, d11, d12, d9, m23, m24, m25, m26, m27, m28, m29,
            m30, s18, s19, s24, s25, t14, t15, t17, t18;

        // First half of butterfly
        s18 = ((__m128i*)X)[0];
        s19 = ((__m128i*)X)[2];
        a76 = _mm_set1_epi8(syms[2 * i]);
        a78 = ((__m128i*)Branchtab)[0];
        a79 = _mm_xor_si128(a76, a78);
        a82 = _mm_set1_epi8(syms[2 * i + 1]);
        a84 = ((__m128i*)Branchtab)[2];
        a85 = _mm_xor_si128(a82, a84);
        a86 = _mm_avg_epu8(a79, a85);
        a88 = _mm_srli_epi16(a86, 2);
        t14 = _mm_and_si128(a88, _mm_set1_epi8(63));
        t15 = _mm_subs_epu8(_mm_set1_epi8(63), t14);
        m23 = _mm_adds_epu8(s18, t14);
        m24 = _mm_adds_epu8(s19, t15);
        m25 = _mm_adds_epu8(s18, t15);
        m26 = _mm_adds_epu8(s19, t14);
        a89 = _mm_min_epu8(m24, m23);
        d9 = _mm_cmpeq_epi8(a89, m24);
        a90 = _mm_min_epu8(m26, m25);
        d10 = _mm_cmpeq_epi8(a90, m26);
        dec_short[4 * i] = _mm_movemask_epi8(_mm_unpacklo_epi8(d9, d10));
        dec_short[4 * i + 1] = _mm_movemask_epi8(_mm_unpackhi_epi8(d9, d10));
        ((__m128i*)Y)[0] = _mm_unpacklo_epi8(a89, a90);
        ((__m128i*)Y)[1] = _mm_unpackhi_epi8(a89, a90);

        // Second half of butterfly
        s24 = ((__m128i*)X)[1];
        s25 = ((__m128i*)X)[3];
        a100 = ((__m128i*)Branchtab)[1];
        a101 = _mm_xor_si128(a76, a100);
        a103 = ((__m128i*)Branchtab)[3];
        a104 = _mm_xor_si128(a82, a103);
        a105 = _mm_avg_epu8(a101, a104);
        a107 = _mm_srli_epi16(a105, 2);
        t17 = _mm_and_si128(a107, _mm_set1_epi8(63));
        t18 = _mm_subs_epu8(_mm_set1_epi8(63), t17);
        m27 = _mm_adds_epu8(s24, t17);
        m28 = _mm_adds_epu8(s25, t18);
        m29 = _mm_adds_epu8(s24, t18);
        m30 = _mm_adds_epu8(s25, t17);
        a108 = _mm_min_epu8(m28, m27);
        d11 = _mm_cmpeq_epi8(a108, m28);
        a109 = _mm_min_epu8(m30, m29);
        d12 = _mm_cmpeq_epi8(a109, m30);
        dec_short[4 * i + 2] = _mm_movemask_epi8(_mm_unpacklo_epi8(d11, d12));
        dec_short[4 * i + 3] = _mm_movemask_epi8(_mm_unpackhi_epi8(d11, d12));
        ((__m128i*)Y)[2] = _mm_unpacklo_epi8(a108, a109);
        ((__m128i*)Y)[3] = _mm_unpackhi_epi8(a108, a109);

        // Renormalize
        __m128i m5, m6;
        m5 = ((__m128i*)Y)[0];
        m5 = _mm_min_epu8(m5, ((__m128i*)Y)[1]);
        m5 = _mm_min_epu8(m5, ((__m128i*)Y)[2]);
        m5 = _mm_min_epu8(m5, ((__m128i*)Y)[3]);
        __m128i m7;
        m7 = _mm_min_epu8(_mm_srli_si128(m5, 8), m5);
        m7 = ((__m128i)_mm_min_epu8(((__m128i)_mm_srli_epi64(m7, 32)), ((__m128i)m7)));
        m7 = ((__m128i)_mm_min_epu8(((__m128i)_mm_srli_epi64(m7, 16)), ((__m128i)m7)));
        m7 = ((__m128i)_mm_min_epu8(((__m128i)_mm_srli_epi64(m7, 8)), ((__m128i)m7)));
        m7 = _mm_unpacklo_epi8(m7, m7);
        m7 = _mm_shufflelo_epi16(m7, _MM_SHUFFLE(0, 0, 0, 0));
        m6 = _mm_unpacklo_epi64(m7, m7);
        ((__m128i*)Y)[0] = _mm_subs_epu8(((__m128i*)Y)[0], m6);
        ((__m128i*)Y)[1] = _mm_subs_epu8(((__m128i*)Y)[1], m6);
        ((__m128i*)Y)[2] = _mm_subs_epu8(((__m128i*)Y)[2], m6);
        ((__m128i*)Y)[3] = _mm_subs_epu8(((__m128i*)Y)[3], m6);

        // Swap pointers to old and new metrics
        tmp = X;
        X = Y;
        Y = tmp;
    }
}

#endif /*LV_HAVE_SSE3*/

#if LV_HAVE_NEON

#include <arm_neon.h>

static inline void volk_8u_x4_conv_k7_r2_8u_neonspiral(unsigned char* Y,
                                                       unsigned char* X,
                                                       unsigned char* syms,
                                                       unsigned char* dec,
                                                       unsigned int framebits,
                                                       unsigned int excess,
                                                       unsigned char* Branchtab)
{
    unsigned int i;
    for (i = 0; i < framebits + excess; i++) {
        unsigned char* tmp;
        unsigned int* dec_int = (unsigned int*)dec;
        uint8x16_t a100, a101, a103, a104, a105, a108, a109, a76, a78, a79, a82, a84, a85,
            a86, a89, a90, d10, d11, d12, d9, m23, m24, m25, m26, m27, m28, m29, m30, s18,
            s19, s24, s25, t14, t15, t17, t18;
        uint16x8_t high_bits;
        uint32x4_t paired16;
        uint8x16_t paired32;
        uint8x8_t left, right;
        uint8x8x2_t both;

        // First half of butterfly
        s18 = ((uint8x16_t*)X)[0];
        s19 = ((uint8x16_t*)X)[2];
        a76 = vdupq_n_u8(syms[2 * i]);
        a78 = ((uint8x16_t*)Branchtab)[0];
        a79 = veorq_u8(a76, a78);
        a82 = vdupq_n_u8(syms[2 * i + 1]);
        a84 = ((uint8x16_t*)Branchtab)[2];
        a85 = veorq_u8(a82, a84);
        a86 = vrhaddq_u8(a79, a85);
        t14 = vshrq_n_u8(a86, 2);
        t15 = vqsubq_u8(vdupq_n_u8(63), t14);
        m23 = vqaddq_u8(s18, t14);
        m24 = vqaddq_u8(s19, t15);
        m25 = vqaddq_u8(s18, t15);
        m26 = vqaddq_u8(s19, t14);
        a89 = vminq_u8(m24, m23);
        d9 = vceqq_u8(a89, m24);
        a90 = vminq_u8(m26, m25);
        d10 = vceqq_u8(a90, m26);
        high_bits = vreinterpretq_u16_u8(vshrq_n_u8(d9, 7));
        paired16 = vreinterpretq_u32_u16(vsraq_n_u16(high_bits, high_bits, 6));
        paired32 = vreinterpretq_u8_u32(vsraq_n_u32(paired16, paired16, 12));
        dec_int[2 * i] = ((unsigned int)vgetq_lane_u8(paired32, 0) << 0) |
                         ((unsigned int)vgetq_lane_u8(paired32, 4) << 8) |
                         ((unsigned int)vgetq_lane_u8(paired32, 8) << 16) |
                         ((unsigned int)vgetq_lane_u8(paired32, 12) << 24);
        high_bits = vreinterpretq_u16_u8(vshrq_n_u8(d10, 7));
        paired16 = vreinterpretq_u32_u16(vsraq_n_u16(high_bits, high_bits, 6));
        paired32 = vreinterpretq_u8_u32(vsraq_n_u32(paired16, paired16, 12));
        dec_int[2 * i] |= ((unsigned int)vgetq_lane_u8(paired32, 0) << 1) |
                          ((unsigned int)vgetq_lane_u8(paired32, 4) << 9) |
                          ((unsigned int)vgetq_lane_u8(paired32, 8) << 17) |
                          ((unsigned int)vgetq_lane_u8(paired32, 12) << 25);
        left = vget_low_u8(a89);
        right = vget_low_u8(a90);
        both = vzip_u8(left, right);
        ((uint8x16_t*)Y)[0] = vcombine_u8(both.val[0], both.val[1]);
        left = vget_high_u8(a89);
        right = vget_high_u8(a90);
        both = vzip_u8(left, right);
        ((uint8x16_t*)Y)[1] = vcombine_u8(both.val[0], both.val[1]);

        // Second half of butterfly
        s24 = ((uint8x16_t*)X)[1];
        s25 = ((uint8x16_t*)X)[3];
        a100 = ((uint8x16_t*)Branchtab)[1];
        a101 = veorq_u8(a76, a100);
        a103 = ((uint8x16_t*)Branchtab)[3];
        a104 = veorq_u8(a82, a103);
        a105 = vrhaddq_u8(a101, a104);
        t17 = vshrq_n_u8(a105, 2);
        t18 = vqsubq_u8(vdupq_n_u8(63), t17);
        m27 = vqaddq_u8(s24, t17);
        m28 = vqaddq_u8(s25, t18);
        m29 = vqaddq_u8(s24, t18);
        m30 = vqaddq_u8(s25, t17);
        a108 = vminq_u8(m28, m27);
        d11 = vceqq_u8(a108, m28);
        a109 = vminq_u8(m30, m29);
        d12 = vceqq_u8(a109, m30);
        high_bits = vreinterpretq_u16_u8(vshrq_n_u8(d11, 7));
        paired16 = vreinterpretq_u32_u16(vsraq_n_u16(high_bits, high_bits, 6));
        paired32 = vreinterpretq_u8_u32(vsraq_n_u32(paired16, paired16, 12));
        dec_int[2 * i + 1] = ((unsigned int)vgetq_lane_u8(paired32, 0) << 0) |
                             ((unsigned int)vgetq_lane_u8(paired32, 4) << 8) |
                             ((unsigned int)vgetq_lane_u8(paired32, 8) << 16) |
                             ((unsigned int)vgetq_lane_u8(paired32, 12) << 24);
        high_bits = vreinterpretq_u16_u8(vshrq_n_u8(d12, 7));
        paired16 = vreinterpretq_u32_u16(vsraq_n_u16(high_bits, high_bits, 6));
        paired32 = vreinterpretq_u8_u32(vsraq_n_u32(paired16, paired16, 12));
        dec_int[2 * i + 1] |= ((unsigned int)vgetq_lane_u8(paired32, 0) << 1) |
                              ((unsigned int)vgetq_lane_u8(paired32, 4) << 9) |
                              ((unsigned int)vgetq_lane_u8(paired32, 8) << 17) |
                              ((unsigned int)vgetq_lane_u8(paired32, 12) << 25);
        left = vget_low_u8(a108);
        right = vget_low_u8(a109);
        both = vzip_u8(left, right);
        ((uint8x16_t*)Y)[2] = vcombine_u8(both.val[0], both.val[1]);
        left = vget_high_u8(a108);
        right = vget_high_u8(a109);
        both = vzip_u8(left, right);
        ((uint8x16_t*)Y)[3] = vcombine_u8(both.val[0], both.val[1]);

        // Renormalize
        uint8x16_t m5, m6;
        m5 = ((uint8x16_t*)Y)[0];
        m5 = vminq_u8(m5, ((uint8x16_t*)Y)[1]);
        m5 = vminq_u8(m5, ((uint8x16_t*)Y)[2]);
        m5 = vminq_u8(m5, ((uint8x16_t*)Y)[3]);
        uint8x8_t m7;
        m7 = vpmin_u8(vget_low_u8(m5), vget_high_u8(m5));
        m7 = vpmin_u8(m7, m7);
        m7 = vpmin_u8(m7, m7);
        m7 = vpmin_u8(m7, m7);
        m6 = vcombine_u8(m7, m7);
        ((uint8x16_t*)Y)[0] = vqsubq_u8(((uint8x16_t*)Y)[0], m6);
        ((uint8x16_t*)Y)[1] = vqsubq_u8(((uint8x16_t*)Y)[1], m6);
        ((uint8x16_t*)Y)[2] = vqsubq_u8(((uint8x16_t*)Y)[2], m6);
        ((uint8x16_t*)Y)[3] = vqsubq_u8(((uint8x16_t*)Y)[3], m6);

        // Swap pointers to old and new metrics
        tmp = X;
        X = Y;
        Y = tmp;
    }
}

#endif /*LV_HAVE_NEON*/

#if LV_HAVE_GENERIC

static inline void volk_8u_x4_conv_k7_r2_8u_generic(unsigned char* Y,
                                                    unsigned char* X,
                                                    unsigned char* syms,
                                                    unsigned char* dec,
                                                    unsigned int framebits,
                                                    unsigned int excess,
                                                    unsigned char* Branchtab)
{
    int nbits = framebits + excess;
    int NUMSTATES = 64;

    int s, i;
    for (s = 0; s < nbits; s++) {
        void* tmp;
        for (i = 0; i < NUMSTATES / 2; i++) {
            BFLY(i, s, syms, Y, X, (decision_t*)dec, Branchtab);
        }

        renormalize(Y);

        ///     Swap pointers to old and new metrics
        tmp = (void*)X;
        X = Y;
        Y = (unsigned char*)tmp;
    }
}

#endif /* LV_HAVE_GENERIC */

#endif /*INCLUDED_volk_8u_x4_conv_k7_r2_8u_H*/
