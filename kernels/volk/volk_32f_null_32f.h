/* -*- c++ -*- */
/*
 * Copyright 2014 Free Software Foundation, Inc.
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

#include <stdio.h>
#include <math.h>
#include <inttypes.h>

#ifndef INCLUDED_volk_32f_null_32f_a_H
#define INCLUDED_volk_32f_null_32f_a_H

#ifdef LV_HAVE_GENERIC

static inline void
volk_32f_null_32f_generic(float* bVector, const float* aVector, unsigned int num_points)
{
  float* bPtr = bVector;
  const float* aPtr = aVector;
  unsigned int number;

  for(number = 0; number < num_points; number++){
    *bPtr++ = *aPtr++;
  }
}
#endif /* LV_HAVE_GENERIC */

#endif /* INCLUDED_volk_32f_null_32f_u_H */
