/* -*- c++ -*- */
/*
 * Copyright 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * VOLK is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * VOLK is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with VOLK; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#ifndef INCLUDED_volk_32f_x2_fm_detectpuppet_32f_H
#define INCLUDED_volk_32f_x2_fm_detectpuppet_32f_H

#include "volk_32f_s32f_32f_fm_detect_32f.h"


#ifdef LV_HAVE_SSE
#include <xmmintrin.h>

static inline void volk_32f_x2_fm_detectpuppet_32f_a_sse(float* outputVector, const float* inputVector, float* saveValue, unsigned int num_points)
{
  const float bound = 1.0f;

  volk_32f_s32f_32f_fm_detect_32f_a_sse(outputVector, inputVector, bound, saveValue, num_points);
}
#endif /* LV_HAVE_SSE */

#ifdef LV_HAVE_GENERIC

static inline void volk_32f_x2_fm_detectpuppet_32f_generic(float* outputVector, const float* inputVector, float* saveValue, unsigned int num_points)
{
  const float bound = 1.0f;

  volk_32f_s32f_32f_fm_detect_32f_generic(outputVector, inputVector, bound, saveValue, num_points);
}
#endif /* LV_HAVE_GENERIC */


#endif /* INCLUDED_volk_32f_x2_fm_detectpuppet_32f_H */
