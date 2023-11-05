/* -*- c++ -*- */
/*
 * Copyright 2023 Magnus Lundmark <magnuslundmark@gmail.com>
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*
 * This file is intended to hold AVX2 FMA intrinsics of intrinsics.
 * They should be used in VOLK kernels to avoid copy-paste.
 */

#ifndef INCLUDE_VOLK_VOLK_AVX2_FMA_INTRINSICS_H_
#define INCLUDE_VOLK_VOLK_AVX2_FMA_INTRINSICS_H_
#include <immintrin.h>

/*
 * Approximate arctan(x) via polynomial expansion
 * on the interval [-1, 1]
 *
 * Maximum relative error ~6.5e-7
 * Polynomial evaluated via Horner's method
 */
static inline __m256 _m256_arctan_poly_avx2_fma(const __m256 x)
{
    const __m256 a1 = _mm256_set1_ps(+0x1.ffffeap-1f);
    const __m256 a3 = _mm256_set1_ps(-0x1.55437p-2f);
    const __m256 a5 = _mm256_set1_ps(+0x1.972be6p-3f);
    const __m256 a7 = _mm256_set1_ps(-0x1.1436ap-3f);
    const __m256 a9 = _mm256_set1_ps(+0x1.5785aap-4f);
    const __m256 a11 = _mm256_set1_ps(-0x1.2f3004p-5f);
    const __m256 a13 = _mm256_set1_ps(+0x1.01a37cp-7f);

    const __m256 x_times_x = _mm256_mul_ps(x, x);
    __m256 arctan;
    arctan = a13;
    arctan = _mm256_fmadd_ps(x_times_x, arctan, a11);
    arctan = _mm256_fmadd_ps(x_times_x, arctan, a9);
    arctan = _mm256_fmadd_ps(x_times_x, arctan, a7);
    arctan = _mm256_fmadd_ps(x_times_x, arctan, a5);
    arctan = _mm256_fmadd_ps(x_times_x, arctan, a3);
    arctan = _mm256_fmadd_ps(x_times_x, arctan, a1);
    arctan = _mm256_mul_ps(x, arctan);

    return arctan;
}

#endif /* INCLUDE_VOLK_VOLK_AVX2_FMA_INTRINSICS_H_ */
