/* -*- c++ -*- */
/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */


#ifndef INCLUDED_volk_32fc_s32f_power_spectral_densitypuppet_32f_a_H
#define INCLUDED_volk_32fc_s32f_power_spectral_densitypuppet_32f_a_H


#include <volk/volk_32fc_s32f_x2_power_spectral_density_32f.h>


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
