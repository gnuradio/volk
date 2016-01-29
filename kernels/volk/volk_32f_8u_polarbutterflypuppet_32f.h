/* -*- c++ -*- */
/* 
 * Copyright 2015 Free Software Foundation, Inc.
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

/*
 * This puppet is for VOLK tests only.
 * For documentation see 'kernels/volk/volk_32f_8u_polarbutterfly_32f.h'
 */

#ifndef VOLK_KERNELS_VOLK_VOLK_32F_8U_POLARBUTTERFLYPUPPET_32F_H_
#define VOLK_KERNELS_VOLK_VOLK_32F_8U_POLARBUTTERFLYPUPPET_32F_H_

#include <volk/volk_32f_8u_polarbutterfly_32f.h>
#include <volk/volk_8u_x3_encodepolar_8u_x2.h>
#include <volk/volk_8u_x3_encodepolarpuppet_8u.h>


static inline void
sanitize_bytes(unsigned char* u, const int elements)
{
  int i;
  unsigned char* u_ptr = u;
  for(i = 0; i < elements; i++){
    *u_ptr = (*u_ptr & 0x01);
    u_ptr++;
  }
}

static inline void
clean_up_intermediate_values(float* llrs, unsigned char* u, const int frame_size, const int elements)
{
  memset(u + frame_size, 0, sizeof(unsigned char) * (elements - frame_size));
  memset(llrs + frame_size, 0, sizeof(float) * (elements - frame_size));
}

static inline void
generate_error_free_input_vector(float* llrs, unsigned char* u, const int frame_size)
{
  memset(u, 0, frame_size);
  unsigned char* target = u + frame_size;
  volk_8u_x2_encodeframepolar_8u_generic(target, u + 2 * frame_size, frame_size);
  float* ft = llrs;
  int i;
  for(i = 0; i < frame_size; i++){
    *ft = (-2 * ((float) *target++)) + 1.0f;
    ft++;
  }
}

static inline void
print_llr_tree(const float* llrs, const int frame_size, const int frame_exp)
{
  int s, e;
  for(s = 0; s < frame_size; s++){
    for(e = 0; e < frame_exp + 1; e++){
      printf("%+4.2f ", llrs[e * frame_size + s]);
    }
    printf("\n");
    if((s + 1) % 8 == 0){
      printf("\n");
    }
  }
}

static inline int
maximum_frame_size(const int elements)
{
  unsigned int frame_size = next_lower_power_of_two(elements);
  unsigned int frame_exp = log2_of_power_of_2(frame_size);
  return next_lower_power_of_two(frame_size / frame_exp);
}

#ifdef LV_HAVE_GENERIC
static inline void
volk_32f_8u_polarbutterflypuppet_32f_generic(float* llrs, const float* input, unsigned char* u, const int elements)
{
  unsigned int frame_size = maximum_frame_size(elements);
  unsigned int frame_exp = log2_of_power_of_2(frame_size);

  sanitize_bytes(u, elements);
  generate_error_free_input_vector(llrs + frame_exp * frame_size, u, frame_size);

  unsigned int u_num = 0;
  for(; u_num < frame_size; u_num++){
    volk_32f_8u_polarbutterfly_32f_generic(llrs, u, frame_size, frame_exp, 0, u_num, u_num);
    u[u_num] = llrs[u_num] > 0 ? 0 : 1;
  }

  clean_up_intermediate_values(llrs, u, frame_size, elements);
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_AVX
static inline void
volk_32f_8u_polarbutterflypuppet_32f_u_avx(float* llrs, const float* input, unsigned char* u, const int elements)
{
  unsigned int frame_size = maximum_frame_size(elements);
  unsigned int frame_exp = log2_of_power_of_2(frame_size);

  sanitize_bytes(u, elements);
  generate_error_free_input_vector(llrs + frame_exp * frame_size, u, frame_size);

  unsigned int u_num = 0;
  for(; u_num < frame_size; u_num++){
    volk_32f_8u_polarbutterfly_32f_u_avx(llrs, u, frame_size, frame_exp, 0, u_num, u_num);
    u[u_num] = llrs[u_num] > 0 ? 0 : 1;
  }

  clean_up_intermediate_values(llrs, u, frame_size, elements);
}
#endif /* LV_HAVE_AVX */



#endif /* VOLK_KERNELS_VOLK_VOLK_32F_8U_POLARBUTTERFLYPUPPET_32F_H_ */
