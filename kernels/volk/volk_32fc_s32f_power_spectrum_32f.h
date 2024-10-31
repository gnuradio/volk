/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_s32f_power_spectrum_32f
 *
 * \b Overview
 *
 * Calculates the log10 power value for each input point.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_s32f_power_spectrum_32f(float* logPowerOutput, const lv_32fc_t*
 * complexFFTInput, const float normalizationFactor, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li complexFFTInput The complex data output from the FFT point.
 * \li normalizationFactor: This value is divided against all the input values before the
 * power is calculated. \li num_points: The number of fft data points.
 *
 * \b Outputs
 * \li logPowerOutput: The 10.0 * log10(r*r + i*i) for each data point.
 *
 * \b Example
 * \code
 * int N = 10000;
 *
 * volk_32fc_s32f_power_spectrum_32f();
 *
 * volk_free(x);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_s32f_power_spectrum_32f_a_H
#define INCLUDED_volk_32fc_s32f_power_spectrum_32f_a_H

#include <inttypes.h>
#include <math.h>
#include <stdio.h>

#ifdef LV_HAVE_GENERIC

static inline void
volk_32fc_s32f_power_spectrum_32f_generic(float* logPowerOutput,
                                          const lv_32fc_t* complexFFTInput,
                                          const float normalizationFactor,
                                          unsigned int num_points)
{
    // Calculate the Power of the complex point
    const float normFactSq = 1.0 / (normalizationFactor * normalizationFactor);

    // Calculate dBm
    // 50 ohm load assumption
    // 10 * log10 (v^2 / (2 * 50.0 * .001)) = 10 * log10( v^2 * 10)
    // 75 ohm load assumption
    // 10 * log10 (v^2 / (2 * 75.0 * .001)) = 10 * log10( v^2 * 15)

    /*
     * For generic reference, the code below is a volk-optimized
     * approach that also leverages a faster log2 calculation
     * to calculate the log10:
     * n*log10(x) = n*log2(x)/log2(10) = (n/log2(10)) * log2(x)
     *
     * Generic code:
     *
     * const float real = *inputPtr++ * iNormalizationFactor;
     * const float imag = *inputPtr++ * iNormalizationFactor;
     * realFFTDataPointsPtr = 10.0*log10f(((real * real) + (imag * imag)) + 1e-20);
     *  realFFTDataPointsPtr++;
     *
     */

    // Calc mag^2
    volk_32fc_magnitude_squared_32f(logPowerOutput, complexFFTInput, num_points);

    // Finish ((real * real) + (imag * imag)) calculation:
    volk_32f_s32f_multiply_32f(logPowerOutput, logPowerOutput, normFactSq, num_points);

    // The following calculates 10*log10(x) = 10*log2(x)/log2(10) = (10/log2(10))
    // * log2(x)
    volk_32f_log2_32f(logPowerOutput, logPowerOutput, num_points);
    volk_32f_s32f_multiply_32f(
        logPowerOutput, logPowerOutput, volk_log2to10factor, num_points);
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_NEON
#include <arm_neon.h>
#include <volk/volk_neon_intrinsics.h>

static inline void
volk_32fc_s32f_power_spectrum_32f_neon(float* logPowerOutput,
                                       const lv_32fc_t* complexFFTInput,
                                       const float normalizationFactor,
                                       unsigned int num_points)
{
    float* logPowerOutputPtr = logPowerOutput;
    const lv_32fc_t* complexFFTInputPtr = complexFFTInput;
    const float iNormalizationFactor = 1.0 / normalizationFactor;
    unsigned int number;
    unsigned int quarter_points = num_points / 4;
    float32x4x2_t fft_vec;
    float32x4_t log_pwr_vec;
    float32x4_t mag_squared_vec;

    const float inv_ln10_10 = 4.34294481903f; // 10.0/ln(10.)

    for (number = 0; number < quarter_points; number++) {
        // Load
        fft_vec = vld2q_f32((float*)complexFFTInputPtr);
        // Prefetch next 4
        __VOLK_PREFETCH(complexFFTInputPtr + 4);
        // Normalize
        fft_vec.val[0] = vmulq_n_f32(fft_vec.val[0], iNormalizationFactor);
        fft_vec.val[1] = vmulq_n_f32(fft_vec.val[1], iNormalizationFactor);
        mag_squared_vec = _vmagnitudesquaredq_f32(fft_vec);
        log_pwr_vec = vmulq_n_f32(_vlogq_f32(mag_squared_vec), inv_ln10_10);
        // Store
        vst1q_f32(logPowerOutputPtr, log_pwr_vec);
        // Move pointers ahead
        complexFFTInputPtr += 4;
        logPowerOutputPtr += 4;
    }

    // deal with the rest
    for (number = quarter_points * 4; number < num_points; number++) {
        const float real = lv_creal(*complexFFTInputPtr) * iNormalizationFactor;
        const float imag = lv_cimag(*complexFFTInputPtr) * iNormalizationFactor;

        *logPowerOutputPtr =
            volk_log2to10factor * log2f_non_ieee(((real * real) + (imag * imag)));
        complexFFTInputPtr++;
        logPowerOutputPtr++;
    }
}

#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32fc_s32f_power_spectrum_32f_rvv(float* logPowerOutput,
                                                         const lv_32fc_t* complexFFTInput,
                                                         const float normalizationFactor,
                                                         unsigned int num_points)
{
    size_t vlmax = __riscv_vsetvlmax_e32m2();

#if LOG_POLY_DEGREE == 6
    const vfloat32m2_t c5 = __riscv_vfmv_v_f_f32m2(3.1157899f, vlmax);
    const vfloat32m2_t c4 = __riscv_vfmv_v_f_f32m2(-3.3241990f, vlmax);
    const vfloat32m2_t c3 = __riscv_vfmv_v_f_f32m2(2.5988452f, vlmax);
    const vfloat32m2_t c2 = __riscv_vfmv_v_f_f32m2(-1.2315303f, vlmax);
    const vfloat32m2_t c1 = __riscv_vfmv_v_f_f32m2(3.1821337e-1f, vlmax);
    const vfloat32m2_t c0 = __riscv_vfmv_v_f_f32m2(-3.4436006e-2f, vlmax);
#elif LOG_POLY_DEGREE == 5
    const vfloat32m2_t c4 = __riscv_vfmv_v_f_f32m2(2.8882704548164776201f, vlmax);
    const vfloat32m2_t c3 = __riscv_vfmv_v_f_f32m2(-2.52074962577807006663f, vlmax);
    const vfloat32m2_t c2 = __riscv_vfmv_v_f_f32m2(1.48116647521213171641f, vlmax);
    const vfloat32m2_t c1 = __riscv_vfmv_v_f_f32m2(-0.465725644288844778798f, vlmax);
    const vfloat32m2_t c0 = __riscv_vfmv_v_f_f32m2(0.0596515482674574969533f, vlmax);
#elif LOG_POLY_DEGREE == 4
    const vfloat32m2_t c3 = __riscv_vfmv_v_f_f32m2(2.61761038894603480148f, vlmax);
    const vfloat32m2_t c2 = __riscv_vfmv_v_f_f32m2(-1.75647175389045657003f, vlmax);
    const vfloat32m2_t c1 = __riscv_vfmv_v_f_f32m2(0.688243882994381274313f, vlmax);
    const vfloat32m2_t c0 = __riscv_vfmv_v_f_f32m2(-0.107254423828329604454f, vlmax);
#elif LOG_POLY_DEGREE == 3
    const vfloat32m2_t c2 = __riscv_vfmv_v_f_f32m2(2.28330284476918490682f, vlmax);
    const vfloat32m2_t c1 = __riscv_vfmv_v_f_f32m2(-1.04913055217340124191f, vlmax);
    const vfloat32m2_t c0 = __riscv_vfmv_v_f_f32m2(0.204446009836232697516f, vlmax);
#else
#error
#endif

    const vfloat32m2_t cf1 = __riscv_vfmv_v_f_f32m2(1.0f, vlmax);
    const vint32m2_t m1 = __riscv_vreinterpret_i32m2(cf1);
    const vint32m2_t m2 = __riscv_vmv_v_x_i32m2(0x7FFFFF, vlmax);
    const vint32m2_t c127 = __riscv_vmv_v_x_i32m2(127, vlmax);

    const float normFactSq = 1.0 / (normalizationFactor * normalizationFactor);

    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, complexFFTInput += vl, logPowerOutput += vl) {
        vl = __riscv_vsetvl_e32m2(n);
        vuint64m4_t vc = __riscv_vle64_v_u64m4((const uint64_t*)complexFFTInput, vl);
        vfloat32m2_t vr = __riscv_vreinterpret_f32m2(__riscv_vnsrl(vc, 0, vl));
        vfloat32m2_t vi = __riscv_vreinterpret_f32m2(__riscv_vnsrl(vc, 32, vl));
        vfloat32m2_t v = __riscv_vfmacc(__riscv_vfmul(vi, vi, vl), vr, vr, vl);
        v = __riscv_vfmul(v, normFactSq, vl);

        vfloat32m2_t a = __riscv_vfabs(v, vl);
        vfloat32m2_t exp = __riscv_vfcvt_f(
            __riscv_vsub(__riscv_vsra(__riscv_vreinterpret_i32m2(a), 23, vl), c127, vl),
            vl);
        vfloat32m2_t frac = __riscv_vreinterpret_f32m2(
            __riscv_vor(__riscv_vand(__riscv_vreinterpret_i32m2(v), m2, vl), m1, vl));

        vfloat32m2_t mant = c0;
        mant = __riscv_vfmadd(mant, frac, c1, vl);
        mant = __riscv_vfmadd(mant, frac, c2, vl);
#if LOG_POLY_DEGREE >= 4
        mant = __riscv_vfmadd(mant, frac, c3, vl);
#if LOG_POLY_DEGREE >= 5
        mant = __riscv_vfmadd(mant, frac, c4, vl);
#if LOG_POLY_DEGREE >= 6
        mant = __riscv_vfmadd(mant, frac, c5, vl);
#endif
#endif
#endif
        v = __riscv_vfmacc(exp, mant, __riscv_vfsub(frac, cf1, vl), vl);
        v = __riscv_vfmul(v, volk_log2to10factor, vl);

        __riscv_vse32(logPowerOutput, v, vl);
    }
}
#endif /*LV_HAVE_RVV*/


#ifdef LV_HAVE_RVVSEG
#include <riscv_vector.h>

static inline void
volk_32fc_s32f_power_spectrum_32f_rvvseg(float* logPowerOutput,
                                         const lv_32fc_t* complexFFTInput,
                                         const float normalizationFactor,
                                         unsigned int num_points)
{
    size_t vlmax = __riscv_vsetvlmax_e32m2();

#if LOG_POLY_DEGREE == 6
    const vfloat32m2_t c5 = __riscv_vfmv_v_f_f32m2(3.1157899f, vlmax);
    const vfloat32m2_t c4 = __riscv_vfmv_v_f_f32m2(-3.3241990f, vlmax);
    const vfloat32m2_t c3 = __riscv_vfmv_v_f_f32m2(2.5988452f, vlmax);
    const vfloat32m2_t c2 = __riscv_vfmv_v_f_f32m2(-1.2315303f, vlmax);
    const vfloat32m2_t c1 = __riscv_vfmv_v_f_f32m2(3.1821337e-1f, vlmax);
    const vfloat32m2_t c0 = __riscv_vfmv_v_f_f32m2(-3.4436006e-2f, vlmax);
#elif LOG_POLY_DEGREE == 5
    const vfloat32m2_t c4 = __riscv_vfmv_v_f_f32m2(2.8882704548164776201f, vlmax);
    const vfloat32m2_t c3 = __riscv_vfmv_v_f_f32m2(-2.52074962577807006663f, vlmax);
    const vfloat32m2_t c2 = __riscv_vfmv_v_f_f32m2(1.48116647521213171641f, vlmax);
    const vfloat32m2_t c1 = __riscv_vfmv_v_f_f32m2(-0.465725644288844778798f, vlmax);
    const vfloat32m2_t c0 = __riscv_vfmv_v_f_f32m2(0.0596515482674574969533f, vlmax);
#elif LOG_POLY_DEGREE == 4
    const vfloat32m2_t c3 = __riscv_vfmv_v_f_f32m2(2.61761038894603480148f, vlmax);
    const vfloat32m2_t c2 = __riscv_vfmv_v_f_f32m2(-1.75647175389045657003f, vlmax);
    const vfloat32m2_t c1 = __riscv_vfmv_v_f_f32m2(0.688243882994381274313f, vlmax);
    const vfloat32m2_t c0 = __riscv_vfmv_v_f_f32m2(-0.107254423828329604454f, vlmax);
#elif LOG_POLY_DEGREE == 3
    const vfloat32m2_t c2 = __riscv_vfmv_v_f_f32m2(2.28330284476918490682f, vlmax);
    const vfloat32m2_t c1 = __riscv_vfmv_v_f_f32m2(-1.04913055217340124191f, vlmax);
    const vfloat32m2_t c0 = __riscv_vfmv_v_f_f32m2(0.204446009836232697516f, vlmax);
#else
#error
#endif

    const vfloat32m2_t cf1 = __riscv_vfmv_v_f_f32m2(1.0f, vlmax);
    const vint32m2_t m1 = __riscv_vreinterpret_i32m2(cf1);
    const vint32m2_t m2 = __riscv_vmv_v_x_i32m2(0x7FFFFF, vlmax);
    const vint32m2_t c127 = __riscv_vmv_v_x_i32m2(127, vlmax);

    const float normFactSq = 1.0 / (normalizationFactor * normalizationFactor);

    size_t n = num_points;
    for (size_t vl; n > 0; n -= vl, complexFFTInput += vl, logPowerOutput += vl) {
        vl = __riscv_vsetvl_e32m2(n);
        vfloat32m2x2_t vc =
            __riscv_vlseg2e32_v_f32m2x2((const float*)complexFFTInput, vl);
        vfloat32m2_t vr = __riscv_vget_f32m2(vc, 0);
        vfloat32m2_t vi = __riscv_vget_f32m2(vc, 1);
        vfloat32m2_t v = __riscv_vfmacc(__riscv_vfmul(vi, vi, vl), vr, vr, vl);
        v = __riscv_vfmul(v, normFactSq, vl);

        vfloat32m2_t a = __riscv_vfabs(v, vl);
        vfloat32m2_t exp = __riscv_vfcvt_f(
            __riscv_vsub(__riscv_vsra(__riscv_vreinterpret_i32m2(a), 23, vl), c127, vl),
            vl);
        vfloat32m2_t frac = __riscv_vreinterpret_f32m2(
            __riscv_vor(__riscv_vand(__riscv_vreinterpret_i32m2(v), m2, vl), m1, vl));

        vfloat32m2_t mant = c0;
        mant = __riscv_vfmadd(mant, frac, c1, vl);
        mant = __riscv_vfmadd(mant, frac, c2, vl);
#if LOG_POLY_DEGREE >= 4
        mant = __riscv_vfmadd(mant, frac, c3, vl);
#if LOG_POLY_DEGREE >= 5
        mant = __riscv_vfmadd(mant, frac, c4, vl);
#if LOG_POLY_DEGREE >= 6
        mant = __riscv_vfmadd(mant, frac, c5, vl);
#endif
#endif
#endif
        v = __riscv_vfmacc(exp, mant, __riscv_vfsub(frac, cf1, vl), vl);
        v = __riscv_vfmul(v, volk_log2to10factor, vl);

        __riscv_vse32(logPowerOutput, v, vl);
    }
}

#endif /*LV_HAVE_RVVSEG*/

#endif /* INCLUDED_volk_32fc_s32f_power_spectrum_32f_a_H */
