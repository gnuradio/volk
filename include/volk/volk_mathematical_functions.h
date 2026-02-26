/* -*- c++ -*- */
/*
 * Copyright 2010, 2011, 2015-2017, 2019, 2020 Free Software Foundation, Inc.
 * Copyright 2023 - 2025 Magnus Lundmark <magnuslundmark@gmail.com>
 * Copyright 2026 Marcus Müller <mmueller@gnuradio.org>
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */
#ifndef INCLUDED_LIBVOLK_MATHFUN_H
#define INCLUDED_LIBVOLK_MATHFUN_H

#include <volk/volk_common.h>
#if defined(__cplusplus)
/* BSD-hosted GCC (and potentially other compilers) might, when including this
 * in C++ mode, not automatically make std::isinf, std::isnan visible as "bare"
 * isinf. Including C-style <math.h> doesn't work, either, as reported by Greg
 * Troxel. Solution: actually behave like C++ when C++, and then use `using
 * std::…` to make things explicitly visible. Better than polluting global
 * namespaces or, even worse, #defines.
 */
#include <cmath>
/* make this C-compatible in C++ mode.  Closed at end-of-file. */
#else
#include <math.h>
#endif
__VOLK_DECL_BEGIN
////////////////////////////////////////////////////////////////////////
// log2f
////////////////////////////////////////////////////////////////////////
// +-Inf -> +-127.0f in order to match the behaviour of the SIMD kernels
// NaN -> NaN (preserved for consistency)
static inline float volk_log2f_non_ieee(float f)
{
#if defined(__cplusplus)
    using std::copysignf;
    using std::isinf;
    using std::isnan;
    using std::log2f;
#endif
    float const result = log2f(f);
    // Return NaN for NaN inputs or negative values (preserves IEEE behavior for invalid
    // inputs)
    if (isnan(result))
        return result;
    // Map ±Inf to ±127.0f to match SIMD kernel behavior
    return isinf(result) ? copysignf(127.0f, result) : result;
}

////////////////////////////////////////////////////////////////////////
// Constant used to do log10 calculations as faster log2
////////////////////////////////////////////////////////////////////////
// precalculated 10.0 / log2f_non_ieee(10.0) to allow for constexpr
#define VOLK_LOG2TO10FACTOR (0x1.815182p1) // 3.01029995663981209120

////////////////////////////////////////////////////////////////////////
// arctan(x) polynomial expansion
////////////////////////////////////////////////////////////////////////
static inline float volk_arctan_poly(const float x)
{
#if defined(__cplusplus)
    using std::fmaf;
#endif
    /*
     * arctan(x) polynomial expansion on the interval [-1, 1]
     * Maximum relative error < 6.6e-7
     */
    const float a1 = +0x1.ffffeap-1f;
    const float a3 = -0x1.55437p-2f;
    const float a5 = +0x1.972be6p-3f;
    const float a7 = -0x1.1436ap-3f;
    const float a9 = +0x1.5785aap-4f;
    const float a11 = -0x1.2f3004p-5f;
    const float a13 = +0x1.01a37cp-7f;

    const float x_times_x = x * x;
    float arctan = a13;
    arctan = fmaf(x_times_x, arctan, a11);
    arctan = fmaf(x_times_x, arctan, a9);
    arctan = fmaf(x_times_x, arctan, a7);
    arctan = fmaf(x_times_x, arctan, a5);
    arctan = fmaf(x_times_x, arctan, a3);
    arctan = fmaf(x_times_x, arctan, a1);
    arctan *= x;

    return arctan;
}
////////////////////////////////////////////////////////////////////////
// sin(x) polynomial expansion
////////////////////////////////////////////////////////////////////////
static inline float volk_sin_poly(const float x)
{
#if defined(__cplusplus)
    using std::fmaf;
#endif
    /*
     * Minimax polynomial for sin(x) on [-pi/4, pi/4]
     * Coefficients via Remez algorithm (Sollya)
     * Max |error| < 7.3e-9
     * sin(x) = x + x^3 * (s1 + x^2 * (s2 + x^2 * s3))
     */
    const float s1 = -0x1.555552p-3f;
    const float s2 = +0x1.110be2p-7f;
    const float s3 = -0x1.9ab22ap-13f;

    const float x2 = x * x;
    const float x3 = x2 * x;

    float poly = fmaf(x2, s3, s2);
    poly = fmaf(x2, poly, s1);
    return fmaf(x3, poly, x);
}
////////////////////////////////////////////////////////////////////////
// cos(x) polynomial expansion
////////////////////////////////////////////////////////////////////////
static inline float volk_cos_poly(const float x)
{
#if defined(__cplusplus)
    using std::fmaf;
#endif
    /*
     * Minimax polynomial for cos(x) on [-pi/4, pi/4]
     * Coefficients via Remez algorithm (Sollya)
     * Max |error| < 1.1e-7
     * cos(x) = 1 + x^2 * (c1 + x^2 * (c2 + x^2 * c3))
     */
    const float c1 = -0x1.fffff4p-2f;
    const float c2 = +0x1.554a46p-5f;
    const float c3 = -0x1.661be2p-10f;

    const float x2 = x * x;

    float poly = fmaf(x2, c3, c2);
    poly = fmaf(x2, poly, c1);
    return fmaf(x2, poly, 1.0f);
}
////////////////////////////////////////////////////////////////////////
// sin(x) with Cody-Waite argument reduction
////////////////////////////////////////////////////////////////////////
static inline float volk_sin(const float x)
{
#if defined(__cplusplus)
    using std::fmaf;
    using std::rintf;
#endif
    /*
     * Cody-Waite argument reduction: n = round(x * 2/pi), r = x - n * pi/2
     * Then use sin/cos polynomials based on quadrant
     */
    const float two_over_pi = 0x1.45f306p-1f;
    const float pi_over_2_hi = 0x1.921fb6p+0f;
    const float pi_over_2_lo = -0x1.777a5cp-25f;

    float n_f = rintf(x * two_over_pi);
    int n = (int)n_f;

    float r = fmaf(-n_f, pi_over_2_hi, x);
    r = fmaf(-n_f, pi_over_2_lo, r);

    float sin_r = volk_sin_poly(r);
    float cos_r = volk_cos_poly(r);

    // Quadrant selection: n&1 swaps sin/cos, n&2 negates
    float result = (n & 1) ? cos_r : sin_r;
    return (n & 2) ? -result : result;
}
////////////////////////////////////////////////////////////////////////
// cos(x) with Cody-Waite argument reduction
////////////////////////////////////////////////////////////////////////
static inline float volk_cos(const float x)
{
#if defined(__cplusplus)
    using std::fmaf;
    using std::rintf;
#endif
    /*
     * Cody-Waite argument reduction: n = round(x * 2/pi), r = x - n * pi/2
     * Then use sin/cos polynomials based on quadrant
     */
    const float two_over_pi = 0x1.45f306p-1f;
    const float pi_over_2_hi = 0x1.921fb6p+0f;
    const float pi_over_2_lo = -0x1.777a5cp-25f;

    float n_f = rintf(x * two_over_pi);
    int n = (int)n_f;

    float r = fmaf(-n_f, pi_over_2_hi, x);
    r = fmaf(-n_f, pi_over_2_lo, r);

    float sin_r = volk_sin_poly(r);
    float cos_r = volk_cos_poly(r);

    // Quadrant selection: n&1 swaps sin/cos, (n+1)&2 negates
    float result = (n & 1) ? sin_r : cos_r;
    return ((n + 1) & 2) ? -result : result;
}
////////////////////////////////////////////////////////////////////////
// arctan(x)
////////////////////////////////////////////////////////////////////////
static inline float volk_arctan(const float x)
{
#if defined(__cplusplus)
    using std::copysignf;
    using std::fabs;
    using std::isinf;
    using std::isnan;
#endif
    /*
     *  arctan(x) + arctan(1 / x) == sign(x) * pi / 2
     */
    const float pi_2 = 0x1.921fb6p0f;

    // Propagate NaN
    if (isnan(x)) {
        return x;
    }

    // arctan(±∞) = ±π/2
    if (isinf(x)) {
        return copysignf(pi_2, x);
    }

    if (fabs(x) < 1.f) {
        return volk_arctan_poly(x);
    } else {
        return copysignf(pi_2, x) - volk_arctan_poly(1.f / x);
    }
}
////////////////////////////////////////////////////////////////////////
// arctan2(y, x)
////////////////////////////////////////////////////////////////////////
static inline float volk_atan2(const float y, const float x)
{
    /*
     *                /  arctan(y / x)         if x > 0
     *                |  arctan(y / x)  + PI   if x < 0 and y >= 0
     * atan2(y, x) =  |  arctan(y / x)  - PI   if x < 0 and y <  0
     *                |  sign(y) * PI / 2      if x = 0
     *                \  undefined             if x = 0 and y = 0
     * atan2f(0.f,  0.f) shall return  0.f
     * atan2f(0.f, -0.f) shall return -0.f
     */
#if defined(__cplusplus)
    using std::copysignf;
    using std::fabs;
    using std::isinf;
    using std::isnan;
#endif
    const float pi = 0x1.921fb6p1f;
    const float pi_2 = 0x1.921fb6p0f;


    // Propagate NaN from inputs
    if (isnan(x) || isnan(y)) {
        return x + y;
    }

    // Handle infinity cases per IEEE 754
    if (isinf(y)) {
        if (isinf(x)) {
            // Both infinite: atan2(±∞, ±∞) = ±π/4 or ±3π/4
            const float angle = (x > 0.f) ? (pi_2 / 2.f) : (3.f * pi_2 / 2.f);
            return copysignf(angle, y);
        } else {
            // y infinite, x finite: atan2(±∞, x) = ±π/2
            return copysignf(pi_2, y);
        }
    }
    if (isinf(x)) {
        // x infinite, y finite: atan2(y, +∞) = ±0, atan2(y, -∞) = ±π
        return (x > 0.f) ? copysignf(0.f, y) : copysignf(pi, y);
    }

    if (fabs(x) == 0.f) {
        return (fabs(y) == 0.f) ? copysignf(0.f, y) : copysignf(pi_2, y);
    }
    const int swap = fabs(x) < fabs(y);
    const float numerator = swap ? x : y;
    const float denominator = swap ? y : x;
    float input = numerator / denominator;

    if (isnan(input)) {
        input = numerator;
    }

    float result = volk_arctan_poly(input);
    result = swap ? (input >= 0.f ? pi_2 : -pi_2) - result : result;
    if (x < 0.f) {
        result += copysignf(pi, y);
    }
    return result;
}

////////////////////////////////////////////////////////////////////////
// arcsin(x) polynomial expansion
// P(u) such that asin(x) = x * P(x^2) on |x| <= 0.5
// Maximum relative error ~1.5e-6
////////////////////////////////////////////////////////////////////////
static inline float volk_arcsin_poly(const float x)
{
#if defined(__cplusplus)
    using std::fmaf;
#endif
    const float c0 = 0x1.ffffcep-1f;
    const float c1 = 0x1.55b648p-3f;
    const float c2 = 0x1.24d192p-4f;
    const float c3 = 0x1.0a788p-4f;

    const float u = x * x;
    float p = c3;
    p = fmaf(u, p, c2);
    p = fmaf(u, p, c1);
    p = fmaf(u, p, c0);

    return x * p;
}
////////////////////////////////////////////////////////////////////////
// arcsin(x) using two-range algorithm
////////////////////////////////////////////////////////////////////////
static inline float volk_arcsin(const float x)
{
#if defined(__cplusplus)
    using std::copysignf;
    using std::fabsf;
    using std::sqrtf;
#endif
    const float pi_2 = 0x1.921fb6p0f;

    const float ax = fabsf(x);
    if (ax <= 0.5f) {
        // Small argument: direct polynomial
        return volk_arcsin_poly(x);
    } else {
        // Large argument: use identity asin(x) = pi/2 - 2*asin(sqrt((1-|x|)/2))
        const float t = (1.0f - ax) * 0.5f;
        const float s = sqrtf(t);
        const float inner = volk_arcsin_poly(s);
        const float result = pi_2 - 2.0f * inner;
        return copysignf(result, x);
    }
}
////////////////////////////////////////////////////////////////////////
// arccos(x) = pi/2 - arcsin(x)
////////////////////////////////////////////////////////////////////////
static inline float volk_arccos(const float x)
{
    const float pi_2 = 0x1.921fb6p0f;
    return pi_2 - volk_arcsin(x);
}

__VOLK_DECL_END
#endif
