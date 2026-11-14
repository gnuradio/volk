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
#include <limits.h>
#include <volk/polar_helpers.h>
#include <volk/volk.h>
#include <volk/volk_8u_x3_encodepolar_8u_x2.h>

#ifdef LV_HAVE_GENERIC
static inline void
volk_8u_x3_encodepolarpuppet_8u_generic(unsigned char* frame,
                                        unsigned char* frozen_bit_mask,
                                        const unsigned char* frozen_bits,
                                        const unsigned char* info_bits,
                                        size_t frame_size)
{
    if (frame_size < 1 || frame_size > UINT32_MAX) {
        return;
    }

    frame_size = next_lower_power_of_two((uint32_t)frame_size);
    unsigned char* temp = (unsigned char*)volk_malloc(sizeof(unsigned char) * frame_size,
                                                      volk_get_alignment());
    adjust_frozen_mask(frozen_bit_mask, frame_size);
    volk_8u_x3_encodepolar_8u_x2_generic(
        frame, temp, frozen_bit_mask, frozen_bits, info_bits, (unsigned int)frame_size);
    volk_free(temp);
}
#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_SSSE3
static inline void
volk_8u_x3_encodepolarpuppet_8u_u_ssse3(unsigned char* frame,
                                        unsigned char* frozen_bit_mask,
                                        const unsigned char* frozen_bits,
                                        const unsigned char* info_bits,
                                        size_t frame_size)
{
    if (frame_size < 1 || frame_size > UINT32_MAX) {
        return;
    }

    frame_size = next_lower_power_of_two((uint32_t)frame_size);
    unsigned char* temp = (unsigned char*)volk_malloc(sizeof(unsigned char) * frame_size,
                                                      volk_get_alignment());
    adjust_frozen_mask(frozen_bit_mask, frame_size);
    volk_8u_x3_encodepolar_8u_x2_u_ssse3(
        frame, temp, frozen_bit_mask, frozen_bits, info_bits, (unsigned int)frame_size);
    volk_free(temp);
}
#endif /* LV_HAVE_SSSE3 */

#ifdef LV_HAVE_AVX2
static inline void
volk_8u_x3_encodepolarpuppet_8u_u_avx2(unsigned char* frame,
                                       unsigned char* frozen_bit_mask,
                                       const unsigned char* frozen_bits,
                                       const unsigned char* info_bits,
                                       size_t frame_size)
{
    if (frame_size < 1 || frame_size > UINT32_MAX) {
        return;
    }

    frame_size = next_lower_power_of_two((uint32_t)frame_size);
    unsigned char* temp = (unsigned char*)volk_malloc(sizeof(unsigned char) * frame_size,
                                                      volk_get_alignment());
    adjust_frozen_mask(frozen_bit_mask, frame_size);
    volk_8u_x3_encodepolar_8u_x2_u_avx2(
        frame, temp, frozen_bit_mask, frozen_bits, info_bits, (unsigned int)frame_size);
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
                                        size_t frame_size)
{
    if (frame_size < 1 || frame_size > UINT32_MAX) {
        return;
    }

    frame_size = next_lower_power_of_two((uint32_t)frame_size);
    unsigned char* temp = (unsigned char*)volk_malloc(sizeof(unsigned char) * frame_size,
                                                      volk_get_alignment());
    adjust_frozen_mask(frozen_bit_mask, frame_size);
    volk_8u_x3_encodepolar_8u_x2_a_ssse3(
        frame, temp, frozen_bit_mask, frozen_bits, info_bits, (unsigned int)frame_size);
    volk_free(temp);
}
#endif /* LV_HAVE_SSSE3 */

#ifdef LV_HAVE_AVX2
static inline void
volk_8u_x3_encodepolarpuppet_8u_a_avx2(unsigned char* frame,
                                       unsigned char* frozen_bit_mask,
                                       const unsigned char* frozen_bits,
                                       const unsigned char* info_bits,
                                       size_t frame_size)
{
    if (frame_size < 1 || frame_size > UINT32_MAX) {
        return;
    }

    frame_size = next_lower_power_of_two((uint32_t)frame_size);
    unsigned char* temp = (unsigned char*)volk_malloc(sizeof(unsigned char) * frame_size,
                                                      volk_get_alignment());
    adjust_frozen_mask(frozen_bit_mask, frame_size);
    volk_8u_x3_encodepolar_8u_x2_a_avx2(
        frame, temp, frozen_bit_mask, frozen_bits, info_bits, (unsigned int)frame_size);
    volk_free(temp);
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_RVV
static inline void volk_8u_x3_encodepolarpuppet_8u_rvv(unsigned char* frame,
                                                       unsigned char* frozen_bit_mask,
                                                       const unsigned char* frozen_bits,
                                                       const unsigned char* info_bits,
                                                       size_t frame_size)
{
    if (frame_size < 1 || frame_size > UINT32_MAX) {
        return;
    }

    frame_size = next_lower_power_of_two((uint32_t)frame_size);
    unsigned char* temp = (unsigned char*)volk_malloc(sizeof(unsigned char) * frame_size,
                                                      volk_get_alignment());
    adjust_frozen_mask(frozen_bit_mask, frame_size);
    volk_8u_x3_encodepolar_8u_x2_rvv(
        frame, temp, frozen_bit_mask, frozen_bits, info_bits, (unsigned int)frame_size);
    volk_free(temp);
}
#endif /* LV_HAVE_RVV */

#ifdef LV_HAVE_RVVSEG
static inline void
volk_8u_x3_encodepolarpuppet_8u_rvvseg(unsigned char* frame,
                                       unsigned char* frozen_bit_mask,
                                       const unsigned char* frozen_bits,
                                       const unsigned char* info_bits,
                                       size_t frame_size)
{
    if (frame_size < 1 || frame_size > UINT32_MAX) {
        return;
    }

    frame_size = next_lower_power_of_two((uint32_t)frame_size);
    unsigned char* temp = (unsigned char*)volk_malloc(sizeof(unsigned char) * frame_size,
                                                      volk_get_alignment());
    adjust_frozen_mask(frozen_bit_mask, frame_size);
    volk_8u_x3_encodepolar_8u_x2_rvvseg(
        frame, temp, frozen_bit_mask, frozen_bits, info_bits, (unsigned int)frame_size);
    volk_free(temp);
}
#endif /* LV_HAVE_RVVSEG */

#endif /* VOLK_KERNELS_VOLK_VOLK_8U_X3_ENCODEPOLARPUPPET_8U_A_H_ */
