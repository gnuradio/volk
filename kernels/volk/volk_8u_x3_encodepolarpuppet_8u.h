/* -*- c++ -*- */
/*
 * Copyright 2015 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/* For documentation see 'kernels/volk/volk_8u_x3_encodepolar_8u_x2.h'
 * This file exists for test purposes only. Should not be used directly.
 */

#ifndef VOLK_KERNELS_VOLK_VOLK_8U_X3_ENCODEPOLARPUPPET_8U_H_
#define VOLK_KERNELS_VOLK_VOLK_8U_X3_ENCODEPOLARPUPPET_8U_H_
#include <volk/volk.h>
#include <volk/volk_8u_x3_encodepolar_8u_x2.h>

static inline unsigned int next_lower_power_of_two(const unsigned int val)
{
    // algorithm found and adopted from:
    // http://acius2.blogspot.de/2007/11/calculating-next-power-of-2.html
    unsigned int res = val;
    res = (res >> 1) | res;
    res = (res >> 2) | res;
    res = (res >> 4) | res;
    res = (res >> 8) | res;
    res = (res >> 16) | res;
    res += 1;
    return res >> 1;
}

static inline void adjust_frozen_mask(unsigned char* mask, const unsigned int frame_size)
{
    // just like the rest of the puppet this function exists for test purposes only.
    unsigned int i;
    for (i = 0; i < frame_size; ++i) {
        *mask = (*mask & 0x80) ? 0xFF : 0x00;
        mask++;
    }
}

#ifdef LV_HAVE_GENERIC
static inline void
volk_8u_x3_encodepolarpuppet_8u_generic(unsigned char* frame,
                                        unsigned char* frozen_bit_mask,
                                        const unsigned char* frozen_bits,
                                        const unsigned char* info_bits,
                                        unsigned int frame_size)
{
    if (frame_size < 1) {
        return;
    }

    frame_size = next_lower_power_of_two(frame_size);
    unsigned char* temp = (unsigned char*)volk_malloc(sizeof(unsigned char) * frame_size,
                                                      volk_get_alignment());
    adjust_frozen_mask(frozen_bit_mask, frame_size);
    volk_8u_x3_encodepolar_8u_x2_generic(
        frame, temp, frozen_bit_mask, frozen_bits, info_bits, frame_size);
    volk_free(temp);
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_SSSE3
static inline void
volk_8u_x3_encodepolarpuppet_8u_u_ssse3(unsigned char* frame,
                                        unsigned char* frozen_bit_mask,
                                        const unsigned char* frozen_bits,
                                        const unsigned char* info_bits,
                                        unsigned int frame_size)
{
    if (frame_size < 1) {
        return;
    }

    frame_size = next_lower_power_of_two(frame_size);
    unsigned char* temp = (unsigned char*)volk_malloc(sizeof(unsigned char) * frame_size,
                                                      volk_get_alignment());
    adjust_frozen_mask(frozen_bit_mask, frame_size);
    volk_8u_x3_encodepolar_8u_x2_u_ssse3(
        frame, temp, frozen_bit_mask, frozen_bits, info_bits, frame_size);
    volk_free(temp);
}
#endif /* LV_HAVE_SSSE3 */

#ifdef LV_HAVE_AVX2
static inline void
volk_8u_x3_encodepolarpuppet_8u_u_avx2(unsigned char* frame,
                                       unsigned char* frozen_bit_mask,
                                       const unsigned char* frozen_bits,
                                       const unsigned char* info_bits,
                                       unsigned int frame_size)
{
    if (frame_size < 1) {
        return;
    }

    frame_size = next_lower_power_of_two(frame_size);
    unsigned char* temp = (unsigned char*)volk_malloc(sizeof(unsigned char) * frame_size,
                                                      volk_get_alignment());
    adjust_frozen_mask(frozen_bit_mask, frame_size);
    volk_8u_x3_encodepolar_8u_x2_u_avx2(
        frame, temp, frozen_bit_mask, frozen_bits, info_bits, frame_size);
    volk_free(temp);
}
#endif /* LV_HAVE_AVX2 */

#endif /* VOLK_KERNELS_VOLK_VOLK_8U_X3_ENCODEPOLARPUPPET_8U_H_ */

#ifndef VOLK_KERNELS_VOLK_VOLK_8U_X3_ENCODEPOLARPUPPET_8U_A_H_
#define VOLK_KERNELS_VOLK_VOLK_8U_X3_ENCODEPOLARPUPPET_8U_A_H_

#ifdef LV_HAVE_SSSE3
static inline void
volk_8u_x3_encodepolarpuppet_8u_a_ssse3(unsigned char* frame,
                                        unsigned char* frozen_bit_mask,
                                        const unsigned char* frozen_bits,
                                        const unsigned char* info_bits,
                                        unsigned int frame_size)
{
    if (frame_size < 1) {
        return;
    }

    frame_size = next_lower_power_of_two(frame_size);
    unsigned char* temp = (unsigned char*)volk_malloc(sizeof(unsigned char) * frame_size,
                                                      volk_get_alignment());
    adjust_frozen_mask(frozen_bit_mask, frame_size);
    volk_8u_x3_encodepolar_8u_x2_a_ssse3(
        frame, temp, frozen_bit_mask, frozen_bits, info_bits, frame_size);
    volk_free(temp);
}
#endif /* LV_HAVE_SSSE3 */

#ifdef LV_HAVE_AVX2
static inline void
volk_8u_x3_encodepolarpuppet_8u_a_avx2(unsigned char* frame,
                                       unsigned char* frozen_bit_mask,
                                       const unsigned char* frozen_bits,
                                       const unsigned char* info_bits,
                                       unsigned int frame_size)
{
    if (frame_size < 1) {
        return;
    }

    frame_size = next_lower_power_of_two(frame_size);
    unsigned char* temp = (unsigned char*)volk_malloc(sizeof(unsigned char) * frame_size,
                                                      volk_get_alignment());
    adjust_frozen_mask(frozen_bit_mask, frame_size);
    volk_8u_x3_encodepolar_8u_x2_a_avx2(
        frame, temp, frozen_bit_mask, frozen_bits, info_bits, frame_size);
    volk_free(temp);
}
#endif /* LV_HAVE_AVX2 */


#endif /* VOLK_KERNELS_VOLK_VOLK_8U_X3_ENCODEPOLARPUPPET_8U_A_H_ */
