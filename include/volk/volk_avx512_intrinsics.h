/* -*- c++ -*- */
/*
 * Copyright 2024 Magnus Lundmark <magnuslundmark@gmail.com>
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*
 * This file is intended to hold AVX512 intrinsics.
 * They should be used in VOLK kernels to avoid copy-paste.
 */

#ifndef INCLUDE_VOLK_VOLK_AVX512_INTRINSICS_H_
#define INCLUDE_VOLK_VOLK_AVX512_INTRINSICS_H_
#include <immintrin.h>

////////////////////////////////////////////////////////////////////////
// Place real parts of two complex vectors in output
// Requires AVX512F
////////////////////////////////////////////////////////////////////////
static inline __m512 _mm512_real(const __m512 z1, const __m512 z2)
{
    const __m512i idx =
        _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    return _mm512_permutex2var_ps(z1, idx, z2);
}

////////////////////////////////////////////////////////////////////////
// Place imaginary parts of two complex vectors in output
// Requires AVX512F
////////////////////////////////////////////////////////////////////////
static inline __m512 _mm512_imag(const __m512 z1, const __m512 z2)
{
    const __m512i idx =
        _mm512_set_epi32(31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1);
    return _mm512_permutex2var_ps(z1, idx, z2);
}

////////////////////////////////////////////////////////////////////////
// Approximate arctan(x) via polynomial expansion on the interval [-1, 1]
// Maximum relative error ~6.5e-7
// Polynomial evaluated via Horner's method
// Requires AVX512F
////////////////////////////////////////////////////////////////////////
static inline __m512 _mm512_arctan_poly_avx512(const __m512 x)
{
    const __m512 a1 = _mm512_set1_ps(+0x1.ffffeap-1f);
    const __m512 a3 = _mm512_set1_ps(-0x1.55437p-2f);
    const __m512 a5 = _mm512_set1_ps(+0x1.972be6p-3f);
    const __m512 a7 = _mm512_set1_ps(-0x1.1436ap-3f);
    const __m512 a9 = _mm512_set1_ps(+0x1.5785aap-4f);
    const __m512 a11 = _mm512_set1_ps(-0x1.2f3004p-5f);
    const __m512 a13 = _mm512_set1_ps(+0x1.01a37cp-7f);

    const __m512 x_times_x = _mm512_mul_ps(x, x);
    __m512 arctan;
    arctan = a13;
    arctan = _mm512_fmadd_ps(x_times_x, arctan, a11);
    arctan = _mm512_fmadd_ps(x_times_x, arctan, a9);
    arctan = _mm512_fmadd_ps(x_times_x, arctan, a7);
    arctan = _mm512_fmadd_ps(x_times_x, arctan, a5);
    arctan = _mm512_fmadd_ps(x_times_x, arctan, a3);
    arctan = _mm512_fmadd_ps(x_times_x, arctan, a1);
    arctan = _mm512_mul_ps(x, arctan);

    return arctan;
}

#endif /* INCLUDE_VOLK_VOLK_AVX512_INTRINSICS_H_ */
