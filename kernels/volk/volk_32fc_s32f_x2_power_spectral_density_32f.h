/* -*- c++ -*- */
/*
 * Copyright 2012, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_s32f_x2_power_spectral_density_32f
 *
 * \b Overview
 *
 * Calculates the log10 power value divided by the RBW for each input point.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_s32f_x2_power_spectral_density_32f(float* logPowerOutput, const
 * lv_32fc_t* complexFFTInput, const float normalizationFactor, const float rbw, unsigned
 * int num_points) \endcode
 *
 * \b Inputs
 * \li complexFFTInput The complex data output from the FFT point.
 * \li normalizationFactor: This value is divided against all the input values before the
 * power is calculated. \li rbw: The resolution bandwidth of the fft spectrum \li
 * num_points: The number of fft data points.
 *
 * \b Outputs
 * \li logPowerOutput: The 10.0 * log10((r*r + i*i)/RBW) for each data point.
 *
 * \b Example
 * \code
 * int N = 10000;
 *
 * volk_32fc_s32f_x2_power_spectral_density_32f();
 *
 * volk_free(x);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_s32f_x2_power_spectral_density_32f_a_H
#define INCLUDED_volk_32fc_s32f_x2_power_spectral_density_32f_a_H

#include <inttypes.h>
#include <math.h>
#include <stdio.h>

#ifdef LV_HAVE_GENERIC

static inline void
volk_32fc_s32f_x2_power_spectral_density_32f_generic(float* logPowerOutput,
                                                     const lv_32fc_t* complexFFTInput,
                                                     const float normalizationFactor,
                                                     const float rbw,
                                                     unsigned int num_points)
{
    if (rbw != 1.0)
        volk_32fc_s32f_power_spectrum_32f(
            logPowerOutput, complexFFTInput, normalizationFactor * sqrt(rbw), num_points);
    else
        volk_32fc_s32f_power_spectrum_32f(
            logPowerOutput, complexFFTInput, normalizationFactor, num_points);
}

#endif /* LV_HAVE_GENERIC */

#endif /* INCLUDED_volk_32fc_s32f_x2_power_spectral_density_32f_a_H */
