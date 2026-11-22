/* -*- c++ -*- */
/*
 * Copyright 2015 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*
 * This puppet is for VOLK tests only.
 * For documentation see 'kernels/volk/volk_32f_8u_polarbutterfly_32f.h'
 */

#ifndef VOLK_KERNELS_VOLK_VOLK_32F_8U_POLARBUTTERFLYPUPPET_32F_H_
#define VOLK_KERNELS_VOLK_VOLK_32F_8U_POLARBUTTERFLYPUPPET_32F_H_

#include <volk/polar_helpers.h>
#include <volk/volk_32f_8u_polarbutterfly_32f.h>
#include <volk/volk_8u_x3_encodepolar_8u_x2.h>
#include <volk/volk_8u_x3_encodepolarpuppet_8u.h>


#ifdef LV_HAVE_GENERIC
static inline void volk_32f_8u_polarbutterflypuppet_32f_generic(float* llrs,
                                                                const float* input,
                                                                unsigned char* u,
                                                                const int elements)
{
    (void)input; // suppress unused parameter warning

    if (elements < 2) {
        return;
    }

    const uint32_t element_count = (uint32_t)elements;
    const uint32_t frame_size = maximum_frame_size(element_count);
    const uint32_t frame_exp = log2_of_power_of_2(frame_size);

    sanitize_bytes(u, element_count);
    clean_up_intermediate_values(llrs, u, frame_size, element_count);
    generate_error_free_input_vector(llrs + frame_exp * frame_size,
                                     u,
                                     frame_size,
                                     volk_8u_x2_encodeframepolar_8u_generic);

    for (size_t u_num = 0; u_num < frame_size; ++u_num) {
        volk_32f_8u_polarbutterfly_32f_generic(
            llrs, u, (int)frame_exp, 0, (int)u_num, (int)u_num);
        u[u_num] = llrs[u_num] > 0 ? 0 : 1;
    }

    clean_up_intermediate_values(llrs, u, frame_size, element_count);
}
#endif /* LV_HAVE_GENERIC */

#ifdef LV_HAVE_AVX
static inline void volk_32f_8u_polarbutterflypuppet_32f_u_avx(float* llrs,
                                                              const float* input,
                                                              unsigned char* u,
                                                              const int elements)
{
    (void)input; // suppress unused parameter warning

    if (elements < 2) {
        return;
    }

    const uint32_t element_count = (uint32_t)elements;
    const uint32_t frame_size = maximum_frame_size(element_count);
    const uint32_t frame_exp = log2_of_power_of_2(frame_size);

    sanitize_bytes(u, element_count);
    clean_up_intermediate_values(llrs, u, frame_size, element_count);
    generate_error_free_input_vector(llrs + frame_exp * frame_size,
                                     u,
                                     frame_size,
                                     volk_8u_x2_encodeframepolar_8u_generic);

    for (size_t u_num = 0; u_num < frame_size; ++u_num) {
        volk_32f_8u_polarbutterfly_32f_u_avx(
            llrs, u, (int)frame_exp, 0, (int)u_num, (int)u_num);
        u[u_num] = llrs[u_num] > 0 ? 0 : 1;
    }

    clean_up_intermediate_values(llrs, u, frame_size, element_count);
}
#endif /* LV_HAVE_AVX */

#ifdef LV_HAVE_AVX2
static inline void volk_32f_8u_polarbutterflypuppet_32f_u_avx2(float* llrs,
                                                               const float* input,
                                                               unsigned char* u,
                                                               const int elements)
{
    (void)input; // suppress unused parameter warning

    if (elements < 2) {
        return;
    }

    const uint32_t element_count = (uint32_t)elements;
    const uint32_t frame_size = maximum_frame_size(element_count);
    const uint32_t frame_exp = log2_of_power_of_2(frame_size);

    sanitize_bytes(u, element_count);
    clean_up_intermediate_values(llrs, u, frame_size, element_count);
    generate_error_free_input_vector(llrs + frame_exp * frame_size,
                                     u,
                                     frame_size,
                                     volk_8u_x2_encodeframepolar_8u_generic);

    for (size_t u_num = 0; u_num < frame_size; ++u_num) {
        volk_32f_8u_polarbutterfly_32f_u_avx2(
            llrs, u, (int)frame_exp, 0, (int)u_num, (int)u_num);
        u[u_num] = llrs[u_num] > 0 ? 0 : 1;
    }

    clean_up_intermediate_values(llrs, u, frame_size, element_count);
}
#endif /* LV_HAVE_AVX2 */

#ifdef LV_HAVE_RVV
static inline void volk_32f_8u_polarbutterflypuppet_32f_rvv(float* llrs,
                                                            const float* input,
                                                            unsigned char* u,
                                                            const int elements)
{
    (void)input; // suppress unused parameter warning

    if (elements < 2) {
        return;
    }

    const uint32_t element_count = (uint32_t)elements;
    const uint32_t frame_size = maximum_frame_size(element_count);
    const uint32_t frame_exp = log2_of_power_of_2(frame_size);

    sanitize_bytes(u, element_count);
    clean_up_intermediate_values(llrs, u, frame_size, element_count);
    generate_error_free_input_vector(llrs + frame_exp * frame_size,
                                     u,
                                     frame_size,
                                     volk_8u_x2_encodeframepolar_8u_generic);

    for (size_t u_num = 0; u_num < frame_size; ++u_num) {
        volk_32f_8u_polarbutterfly_32f_rvv(
            llrs, u, (int)frame_exp, 0, (int)u_num, (int)u_num);
        u[u_num] = llrs[u_num] > 0 ? 0 : 1;
    }

    clean_up_intermediate_values(llrs, u, frame_size, element_count);
}
#endif /* LV_HAVE_RVV */

#ifdef LV_HAVE_RVVSEG
static inline void volk_32f_8u_polarbutterflypuppet_32f_rvvseg(float* llrs,
                                                               const float* input,
                                                               unsigned char* u,
                                                               const int elements)
{
    (void)input; // suppress unused parameter warning

    if (elements < 2) {
        return;
    }

    const uint32_t element_count = (uint32_t)elements;
    const uint32_t frame_size = maximum_frame_size(element_count);
    const uint32_t frame_exp = log2_of_power_of_2(frame_size);

    sanitize_bytes(u, element_count);
    clean_up_intermediate_values(llrs, u, frame_size, element_count);
    generate_error_free_input_vector(llrs + frame_exp * frame_size,
                                     u,
                                     frame_size,
                                     volk_8u_x2_encodeframepolar_8u_generic);

    for (size_t u_num = 0; u_num < frame_size; ++u_num) {
        volk_32f_8u_polarbutterfly_32f_rvvseg(
            llrs, u, (int)frame_exp, 0, (int)u_num, (int)u_num);
        u[u_num] = llrs[u_num] > 0 ? 0 : 1;
    }

    clean_up_intermediate_values(llrs, u, frame_size, element_count);
}
#endif /* LV_HAVE_RVVSEG */

#endif /* VOLK_KERNELS_VOLK_VOLK_32F_8U_POLARBUTTERFLYPUPPET_32F_H_ */
