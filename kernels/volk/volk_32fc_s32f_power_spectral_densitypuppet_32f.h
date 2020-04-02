/* -*- c++ -*- */
/*
 * Copyright 2020 Free Software Foundation, Inc.
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


#ifndef INCLUDED_volk_32fc_s32f_power_spectral_densitypuppet_32f_a_H
#define INCLUDED_volk_32fc_s32f_power_spectral_densitypuppet_32f_a_H


#include <volk/volk_32fc_s32f_x2_power_spectral_density_32f.h>


#ifdef LV_HAVE_AVX

static inline void
volk_32fc_s32f_power_spectral_densitypuppet_32f_a_avx(float* logPowerOutput,
                                                      const lv_32fc_t* complexFFTInput,
                                                      const float normalizationFactor,
                                                      unsigned int num_points)
{
    volk_32fc_s32f_x2_power_spectral_density_32f_a_avx(
        logPowerOutput, complexFFTInput, normalizationFactor, 2.5, num_points);
}

#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_SSE3

static inline void
volk_32fc_s32f_power_spectral_densitypuppet_32f_a_sse3(float* logPowerOutput,
                                                       const lv_32fc_t* complexFFTInput,
                                                       const float normalizationFactor,
                                                       unsigned int num_points)
{
    volk_32fc_s32f_x2_power_spectral_density_32f_a_sse3(
        logPowerOutput, complexFFTInput, normalizationFactor, 2.5, num_points);
}

#endif /* LV_HAVE_SSE3 */


#ifdef LV_HAVE_GENERIC

static inline void
volk_32fc_s32f_power_spectral_densitypuppet_32f_generic(float* logPowerOutput,
                                                        const lv_32fc_t* complexFFTInput,
                                                        const float normalizationFactor,
                                                        unsigned int num_points)
{
    volk_32fc_s32f_x2_power_spectral_density_32f_generic(
        logPowerOutput, complexFFTInput, normalizationFactor, 2.5, num_points);
}

#endif /* LV_HAVE_GENERIC */


#endif /* INCLUDED_volk_32fc_s32f_power_spectral_densitypuppet_32f_a_H */
