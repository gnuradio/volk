/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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

#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>

#ifdef LV_HAVE_LIB_SIMDMATH
#include <simdmath.h>
#endif /* LV_HAVE_LIB_SIMDMATH */

static inline void
volk_32fc_s32f_power_spectrum_32f_a_sse3(float* logPowerOutput,
                                         const lv_32fc_t* complexFFTInput,
                                         const float normalizationFactor,
                                         unsigned int num_points)
{
    const float* inputPtr = (const float*)complexFFTInput;
    float* destPtr = logPowerOutput;
    uint64_t number = 0;
    const float iNormalizationFactor = 1.0 / normalizationFactor;
#ifdef LV_HAVE_LIB_SIMDMATH
    __m128 magScalar = _mm_set_ps1(10.0);
    magScalar = _mm_div_ps(magScalar, logf4(magScalar));

    __m128 invNormalizationFactor = _mm_set_ps1(iNormalizationFactor);

    __m128 power;
    __m128 input1, input2;
    const uint64_t quarterPoints = num_points / 4;
    for (; number < quarterPoints; number++) {
        // Load the complex values
        input1 = _mm_load_ps(inputPtr);
        inputPtr += 4;
        input2 = _mm_load_ps(inputPtr);
        inputPtr += 4;

        // Apply the normalization factor
        input1 = _mm_mul_ps(input1, invNormalizationFactor);
        input2 = _mm_mul_ps(input2, invNormalizationFactor);

        // Multiply each value by itself
        // (r1*r1), (i1*i1), (r2*r2), (i2*i2)
        input1 = _mm_mul_ps(input1, input1);
        // (r3*r3), (i3*i3), (r4*r4), (i4*i4)
        input2 = _mm_mul_ps(input2, input2);

        // Horizontal add, to add (r*r) + (i*i) for each complex value
        // (r1*r1)+(i1*i1), (r2*r2) + (i2*i2), (r3*r3)+(i3*i3), (r4*r4)+(i4*i4)
        power = _mm_hadd_ps(input1, input2);

        // Calculate the natural log power
        power = logf4(power);

        // Convert to log10 and multiply by 10.0
        power = _mm_mul_ps(power, magScalar);

        // Store the floating point results
        _mm_store_ps(destPtr, power);

        destPtr += 4;
    }

    number = quarterPoints * 4;
#endif /* LV_HAVE_LIB_SIMDMATH */
    // Calculate the FFT for any remaining points

    for (; number < num_points; number++) {
        // Calculate dBm
        // 50 ohm load assumption
        // 10 * log10 (v^2 / (2 * 50.0 * .001)) = 10 * log10( v^2 * 10)
        // 75 ohm load assumption
        // 10 * log10 (v^2 / (2 * 75.0 * .001)) = 10 * log10( v^2 * 15)

        const float real = *inputPtr++ * iNormalizationFactor;
        const float imag = *inputPtr++ * iNormalizationFactor;

        *destPtr = volk_log2to10factor * log2f_non_ieee(((real * real) + (imag * imag)));

        destPtr++;
    }
}
#endif /* LV_HAVE_SSE3 */

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

#endif /* INCLUDED_volk_32fc_s32f_power_spectrum_32f_a_H */
