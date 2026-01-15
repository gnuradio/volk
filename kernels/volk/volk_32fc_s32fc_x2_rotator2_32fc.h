/* -*- c++ -*- */
/*
 * Copyright 2012, 2013, 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*!
 * \page volk_32fc_s32fc_x2_rotator2_32fc
 *
 * \b Overview
 *
 * Rotate input vector at fixed rate per sample from initial phase
 * offset.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_32fc_s32fc_x2_rotator2_32fc(lv_32fc_t* outVector, const lv_32fc_t* inVector,
 * const lv_32fc_t* phase_inc, lv_32fc_t* phase, unsigned int num_points) \endcode
 *
 * \b Inputs
 * \li inVector: Vector to be rotated.
 * \li phase_inc: rotational velocity (scalar, input).
 * \li phase: phase offset (scalar, input & output).
 * \li num_points: The number of values in inVector to be rotated and stored into
 * outVector.
 *
 * \b Outputs
 * \li outVector: The vector where the results will be stored.
 *
 * \b Example
 * Generate a tone at f=0.3 (normalized frequency) and use the rotator with
 * f=0.1 to shift the tone to f=0.4. Change this example to start with a DC
 * tone (initialize in with lv_cmake(1, 0)) to observe rotator signal generation.
 * \code
 *   int N = 10;
 *   unsigned int alignment = volk_get_alignment();
 *   lv_32fc_t* in  = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *   lv_32fc_t* out = (lv_32fc_t*)volk_malloc(sizeof(lv_32fc_t)*N, alignment);
 *
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       // Generate a tone at f=0.3
 *       float real = std::cos(0.3f * (float)ii);
 *       float imag = std::sin(0.3f * (float)ii);
 *       in[ii] = lv_cmake(real, imag);
 *   }
 *   // The oscillator rotates at f=0.1
 *   float frequency = 0.1f;
 *   lv_32fc_t phase_increment = lv_cmake(std::cos(frequency), std::sin(frequency));
 *   lv_32fc_t phase= lv_cmake(1.f, 0.0f); // start at 1 (0 rad phase)
 *
 *   // rotate so the output is a tone at f=0.4
 *   volk_32fc_s32fc_x2_rotator2_32fc(out, in, &phase_increment, &phase, N);
 *
 *   // print results for inspection
 *   for(unsigned int ii = 0; ii < N; ++ii){
 *       printf("out[%u] = %+1.2f %+1.2fj\n",
 *           ii, lv_creal(out[ii]), lv_cimag(out[ii]));
 *   }
 *
 *   volk_free(in);
 *   volk_free(out);
 * \endcode
 */

#ifndef INCLUDED_volk_32fc_s32fc_rotator2_32fc_a_H
#define INCLUDED_volk_32fc_s32fc_rotator2_32fc_a_H


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <volk/volk_complex.h>
#define ROTATOR_RELOAD 512
#define ROTATOR_RELOAD_2 (ROTATOR_RELOAD / 2)
#define ROTATOR_RELOAD_4 (ROTATOR_RELOAD / 4)
#define ROTATOR_RELOAD_8 (ROTATOR_RELOAD / 8)


#ifdef LV_HAVE_GENERIC

static inline void volk_32fc_s32fc_x2_rotator2_32fc_generic(lv_32fc_t* outVector,
                                                            const lv_32fc_t* inVector,
                                                            const lv_32fc_t* phase_inc,
                                                            lv_32fc_t* phase,
                                                            unsigned int num_points)
{
    unsigned int i = 0;
    int j = 0;
    for (i = 0; i < (unsigned int)(num_points / ROTATOR_RELOAD); ++i) {
        for (j = 0; j < ROTATOR_RELOAD; ++j) {
            *outVector++ = *inVector++ * (*phase);
            (*phase) *= *phase_inc;
        }

        (*phase) /= hypotf(lv_creal(*phase), lv_cimag(*phase));
    }
    for (i = 0; i < num_points % ROTATOR_RELOAD; ++i) {
        *outVector++ = *inVector++ * (*phase);
        (*phase) *= *phase_inc;
    }
    if (i) {
        // Make sure, we normalize phase on every call!
        (*phase) /= hypotf(lv_creal(*phase), lv_cimag(*phase));
    }
}

#endif /* LV_HAVE_GENERIC */


#ifdef LV_HAVE_NEON
#include <arm_neon.h>
#include <volk/volk_neon_intrinsics.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*!
 * \brief NEON implementation with angle-based resync for numerical stability.
 *
 * Uses Kahan summation for angle accumulation and periodic sincos resync
 * to eliminate accumulated phase error. Suitable for billion-sample stability.
 */
static inline void volk_32fc_s32fc_x2_rotator2_32fc_neon(lv_32fc_t* outVector,
                                                         const lv_32fc_t* inVector,
                                                         const lv_32fc_t* phase_inc,
                                                         lv_32fc_t* phase,
                                                         unsigned int num_points)
{
    lv_32fc_t* outputVectorPtr = outVector;
    const lv_32fc_t* inputVectorPtr = inVector;

    // Extract angles using double precision
    const double initial_angle =
        atan2((double)lv_cimag(*phase), (double)lv_creal(*phase));
    const double delta_angle =
        atan2((double)lv_cimag(*phase_inc), (double)lv_creal(*phase_inc));

    // Kahan summation state for angle accumulation
    double angle_sum = initial_angle;
    double angle_c = 0.0; // Compensation for lost low-order bits

    // Precompute block increment
    const double block_delta = (double)ROTATOR_RELOAD * delta_angle;

    // Compute incr = phase_inc^4 from exact angle (more accurate than multiplication)
    const double delta4 = 4.0 * delta_angle;
    lv_32fc_t incr = lv_cmake((float)cos(delta4), (float)sin(delta4));

    float32x4x2_t input_vec;
    float32x4x2_t output_vec;
    lv_32fc_t phasePtr[4];

    const lv_32fc_t incrPtr[4] = { incr, incr, incr, incr };
    const float32x4x2_t incr_vec = vld2q_f32((float*)incrPtr);
    float32x4x2_t phase_vec;

// Helper macro for angle reduction
#define REDUCE_ANGLE(a)              \
    do {                             \
        (a) = fmod((a), 2.0 * M_PI); \
        if ((a) > M_PI)              \
            (a) -= 2.0 * M_PI;       \
        else if ((a) < -M_PI)        \
            (a) += 2.0 * M_PI;       \
    } while (0)

    // Initialize phase vector from exact angles
    for (unsigned int k = 0; k < 4; ++k) {
        double a = angle_sum + (double)k * delta_angle;
        REDUCE_ANGLE(a);
        phasePtr[k] = lv_cmake((float)cos(a), (float)sin(a));
    }
    phase_vec = vld2q_f32((float*)phasePtr);

    unsigned int i, j;

    // Main loop with periodic resync
    for (i = 0; i < (unsigned int)(num_points / ROTATOR_RELOAD); i++) {
        // Inner loop: use fast SIMD multiply
        for (j = 0; j < ROTATOR_RELOAD_4; j++) {
            input_vec = vld2q_f32((float*)inputVectorPtr);
            __VOLK_PREFETCH(inputVectorPtr + 4);

            // Rotate
            output_vec = _vmultiply_complexq_f32(input_vec, phase_vec);
            // Increase phase
            phase_vec = _vmultiply_complexq_f32(phase_vec, incr_vec);
            // Store output
            vst2q_f32((float*)outputVectorPtr, output_vec);

            outputVectorPtr += 4;
            inputVectorPtr += 4;
        }

        // Advance angle using Kahan summation (compensated for rounding error)
        {
            double y = block_delta - angle_c;
            double t = angle_sum + y;
            angle_c = (t - angle_sum) - y;
            angle_sum = t;
        }

        // Resync: recompute phase from accumulated angle (eliminates drift)
        for (unsigned int k = 0; k < 4; ++k) {
            double a = angle_sum + (double)k * delta_angle;
            REDUCE_ANGLE(a);
            phasePtr[k] = lv_cmake((float)cos(a), (float)sin(a));
        }
        phase_vec = vld2q_f32((float*)phasePtr);
    }

    // Handle remainder
    for (i = 0; i < (num_points % ROTATOR_RELOAD) / 4; i++) {
        input_vec = vld2q_f32((float*)inputVectorPtr);
        __VOLK_PREFETCH(inputVectorPtr + 4);

        output_vec = _vmultiply_complexq_f32(input_vec, phase_vec);
        phase_vec = _vmultiply_complexq_f32(phase_vec, incr_vec);
        vst2q_f32((float*)outputVectorPtr, output_vec);

        outputVectorPtr += 4;
        inputVectorPtr += 4;

        // Update angle for remainder
        {
            double y = (4.0 * delta_angle) - angle_c;
            double t = angle_sum + y;
            angle_c = (t - angle_sum) - y;
            angle_sum = t;
        }
    }

    // Final phase output from exact angle
    {
        double a = angle_sum;
        REDUCE_ANGLE(a);
        *phase = lv_cmake((float)cos(a), (float)sin(a));
    }

    // Scalar remainder
    for (i = 0; i < num_points % 4; i++) {
        *outputVectorPtr++ = *inputVectorPtr++ * (*phase);
        {
            double y = delta_angle - angle_c;
            double t = angle_sum + y;
            angle_c = (t - angle_sum) - y;
            angle_sum = t;
        }
        double a = angle_sum;
        REDUCE_ANGLE(a);
        *phase = lv_cmake((float)cos(a), (float)sin(a));
    }

#undef REDUCE_ANGLE
}

#endif /* LV_HAVE_NEON */


#ifdef LV_HAVE_AVX
#include <immintrin.h>
#include <volk/volk_avx_intrinsics.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*!
 * \brief AVX implementation with angle-based resync for numerical stability.
 *
 * Uses Kahan summation for angle accumulation and periodic sincos resync
 * to eliminate accumulated phase error. Suitable for billion-sample stability.
 */
static inline void volk_32fc_s32fc_x2_rotator2_32fc_a_avx(lv_32fc_t* outVector,
                                                          const lv_32fc_t* inVector,
                                                          const lv_32fc_t* phase_inc,
                                                          lv_32fc_t* phase,
                                                          unsigned int num_points)
{
    lv_32fc_t* cPtr = outVector;
    const lv_32fc_t* aPtr = inVector;

    // Extract angles using double precision
    const double initial_angle =
        atan2((double)lv_cimag(*phase), (double)lv_creal(*phase));
    const double delta_angle =
        atan2((double)lv_cimag(*phase_inc), (double)lv_creal(*phase_inc));

    // Kahan summation state for angle accumulation
    double angle_sum = initial_angle;
    double angle_c = 0.0; // Compensation for lost low-order bits

    // Precompute block increment
    const double block_delta = (double)ROTATOR_RELOAD * delta_angle;

    // Compute incr = phase_inc^4 from exact angle (more accurate than multiplication)
    const double delta4 = 4.0 * delta_angle;
    lv_32fc_t incr = lv_cmake((float)cos(delta4), (float)sin(delta4));

    __m256 aVal, phase_Val, z;
    lv_32fc_t phase_Ptr[4];

    const __m256 inc_Val = _mm256_set_ps(lv_cimag(incr),
                                         lv_creal(incr),
                                         lv_cimag(incr),
                                         lv_creal(incr),
                                         lv_cimag(incr),
                                         lv_creal(incr),
                                         lv_cimag(incr),
                                         lv_creal(incr));

// Helper macro for angle reduction
#define REDUCE_ANGLE(a)              \
    do {                             \
        (a) = fmod((a), 2.0 * M_PI); \
        if ((a) > M_PI)              \
            (a) -= 2.0 * M_PI;       \
        else if ((a) < -M_PI)        \
            (a) += 2.0 * M_PI;       \
    } while (0)

    // Initialize phase vector from exact angles
    for (unsigned int k = 0; k < 4; ++k) {
        double a = angle_sum + (double)k * delta_angle;
        REDUCE_ANGLE(a);
        phase_Ptr[k] = lv_cmake((float)cos(a), (float)sin(a));
    }
    phase_Val = _mm256_loadu_ps((float*)phase_Ptr);

    unsigned int i, j;

    // Main loop with periodic resync
    for (i = 0; i < (unsigned int)(num_points / ROTATOR_RELOAD); ++i) {
        // Inner loop: use fast SIMD multiply
        for (j = 0; j < ROTATOR_RELOAD_4; ++j) {
            aVal = _mm256_loadu_ps((float*)aPtr);

            z = _mm256_complexmul_ps(aVal, phase_Val);
            phase_Val = _mm256_complexmul_ps(phase_Val, inc_Val);

            _mm256_storeu_ps((float*)cPtr, z);

            aPtr += 4;
            cPtr += 4;
        }

        // Advance angle using Kahan summation (compensated for rounding error)
        {
            double y = block_delta - angle_c;
            double t = angle_sum + y;
            angle_c = (t - angle_sum) - y;
            angle_sum = t;
        }

        // Resync: recompute phase from accumulated angle (eliminates drift)
        for (unsigned int k = 0; k < 4; ++k) {
            double a = angle_sum + (double)k * delta_angle;
            REDUCE_ANGLE(a);
            phase_Ptr[k] = lv_cmake((float)cos(a), (float)sin(a));
        }
        phase_Val = _mm256_loadu_ps((float*)phase_Ptr);
    }

    // Handle remainder
    for (i = 0; i < (num_points % ROTATOR_RELOAD) / 4; ++i) {
        aVal = _mm256_loadu_ps((float*)aPtr);

        z = _mm256_complexmul_ps(aVal, phase_Val);
        phase_Val = _mm256_complexmul_ps(phase_Val, inc_Val);

        _mm256_storeu_ps((float*)cPtr, z);

        aPtr += 4;
        cPtr += 4;

        // Update angle for remainder
        {
            double y = (4.0 * delta_angle) - angle_c;
            double t = angle_sum + y;
            angle_c = (t - angle_sum) - y;
            angle_sum = t;
        }
    }

    // Final phase output from exact angle
    {
        double a = angle_sum;
        REDUCE_ANGLE(a);
        *phase = lv_cmake((float)cos(a), (float)sin(a));
    }

    // Scalar remainder
    for (i = 0; i < num_points % 4; ++i) {
        *cPtr++ = *aPtr++ * (*phase);
        {
            double y = delta_angle - angle_c;
            double t = angle_sum + y;
            angle_c = (t - angle_sum) - y;
            angle_sum = t;
        }
        double a = angle_sum;
        REDUCE_ANGLE(a);
        *phase = lv_cmake((float)cos(a), (float)sin(a));
    }

#undef REDUCE_ANGLE
}

#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_AVX
#include <immintrin.h>

/*!
 * \brief Unaligned AVX implementation with angle-based resync for numerical stability.
 *
 * Uses Kahan summation for angle accumulation and periodic sincos resync
 * to eliminate accumulated phase error. Suitable for billion-sample stability.
 */
static inline void volk_32fc_s32fc_x2_rotator2_32fc_u_avx(lv_32fc_t* outVector,
                                                          const lv_32fc_t* inVector,
                                                          const lv_32fc_t* phase_inc,
                                                          lv_32fc_t* phase,
                                                          unsigned int num_points)
{
    lv_32fc_t* cPtr = outVector;
    const lv_32fc_t* aPtr = inVector;

    // Extract angles using double precision
    const double initial_angle =
        atan2((double)lv_cimag(*phase), (double)lv_creal(*phase));
    const double delta_angle =
        atan2((double)lv_cimag(*phase_inc), (double)lv_creal(*phase_inc));

    // Kahan summation state for angle accumulation
    double angle_sum = initial_angle;
    double angle_c = 0.0;

    // Precompute block increment
    const double block_delta = (double)ROTATOR_RELOAD * delta_angle;

    // Compute incr = phase_inc^4 from exact angle
    const double delta4 = 4.0 * delta_angle;
    lv_32fc_t incr = lv_cmake((float)cos(delta4), (float)sin(delta4));

    __m256 aVal, phase_Val, z;
    lv_32fc_t phase_Ptr[4];

    const __m256 inc_Val = _mm256_set_ps(lv_cimag(incr),
                                         lv_creal(incr),
                                         lv_cimag(incr),
                                         lv_creal(incr),
                                         lv_cimag(incr),
                                         lv_creal(incr),
                                         lv_cimag(incr),
                                         lv_creal(incr));

#define REDUCE_ANGLE(a)              \
    do {                             \
        (a) = fmod((a), 2.0 * M_PI); \
        if ((a) > M_PI)              \
            (a) -= 2.0 * M_PI;       \
        else if ((a) < -M_PI)        \
            (a) += 2.0 * M_PI;       \
    } while (0)

    // Initialize phase vector from exact angles
    for (unsigned int k = 0; k < 4; ++k) {
        double a = angle_sum + (double)k * delta_angle;
        REDUCE_ANGLE(a);
        phase_Ptr[k] = lv_cmake((float)cos(a), (float)sin(a));
    }
    phase_Val = _mm256_loadu_ps((float*)phase_Ptr);

    unsigned int i, j;

    // Main loop with periodic resync
    for (i = 0; i < (unsigned int)(num_points / ROTATOR_RELOAD); ++i) {
        for (j = 0; j < ROTATOR_RELOAD_4; ++j) {
            aVal = _mm256_loadu_ps((float*)aPtr);
            z = _mm256_complexmul_ps(aVal, phase_Val);
            phase_Val = _mm256_complexmul_ps(phase_Val, inc_Val);
            _mm256_storeu_ps((float*)cPtr, z);
            aPtr += 4;
            cPtr += 4;
        }

        // Advance angle using Kahan summation
        {
            double y = block_delta - angle_c;
            double t = angle_sum + y;
            angle_c = (t - angle_sum) - y;
            angle_sum = t;
        }

        // Resync: recompute phase from accumulated angle
        for (unsigned int k = 0; k < 4; ++k) {
            double a = angle_sum + (double)k * delta_angle;
            REDUCE_ANGLE(a);
            phase_Ptr[k] = lv_cmake((float)cos(a), (float)sin(a));
        }
        phase_Val = _mm256_loadu_ps((float*)phase_Ptr);
    }

    // Handle remainder
    for (i = 0; i < (num_points % ROTATOR_RELOAD) / 4; ++i) {
        aVal = _mm256_loadu_ps((float*)aPtr);
        z = _mm256_complexmul_ps(aVal, phase_Val);
        phase_Val = _mm256_complexmul_ps(phase_Val, inc_Val);
        _mm256_storeu_ps((float*)cPtr, z);
        aPtr += 4;
        cPtr += 4;

        {
            double y = (4.0 * delta_angle) - angle_c;
            double t = angle_sum + y;
            angle_c = (t - angle_sum) - y;
            angle_sum = t;
        }
    }

    // Final phase output
    {
        double a = angle_sum;
        REDUCE_ANGLE(a);
        *phase = lv_cmake((float)cos(a), (float)sin(a));
    }

    // Scalar remainder
    for (i = 0; i < num_points % 4; ++i) {
        *cPtr++ = *aPtr++ * (*phase);
        {
            double y = delta_angle - angle_c;
            double t = angle_sum + y;
            angle_c = (t - angle_sum) - y;
            angle_sum = t;
        }
        double a = angle_sum;
        REDUCE_ANGLE(a);
        *phase = lv_cmake((float)cos(a), (float)sin(a));
    }

#undef REDUCE_ANGLE
}

#endif /* LV_HAVE_AVX */


#ifdef LV_HAVE_AVX512F
#include <immintrin.h>
#include <volk/volk_avx512_intrinsics.h>

/*!
 * \brief Aligned AVX512F implementation with angle-based resync for numerical stability.
 *
 * Uses Kahan summation for angle accumulation and periodic sincos resync
 * to eliminate accumulated phase error. Suitable for billion-sample stability.
 */
static inline void volk_32fc_s32fc_x2_rotator2_32fc_a_avx512f(lv_32fc_t* outVector,
                                                              const lv_32fc_t* inVector,
                                                              const lv_32fc_t* phase_inc,
                                                              lv_32fc_t* phase,
                                                              unsigned int num_points)
{
    lv_32fc_t* cPtr = outVector;
    const lv_32fc_t* aPtr = inVector;

    // Extract angles using double precision
    const double initial_angle =
        atan2((double)lv_cimag(*phase), (double)lv_creal(*phase));
    const double delta_angle =
        atan2((double)lv_cimag(*phase_inc), (double)lv_creal(*phase_inc));

    // Kahan summation state for angle accumulation
    double angle_sum = initial_angle;
    double angle_c = 0.0;

    // Precompute block increment
    const double block_delta = (double)ROTATOR_RELOAD * delta_angle;

    // Compute incr = phase_inc^8 from exact angle
    const double delta8 = 8.0 * delta_angle;
    lv_32fc_t incr = lv_cmake((float)cos(delta8), (float)sin(delta8));

    __m512 aVal, phase_Val, z;
    __VOLK_ATTR_ALIGNED(64)
    lv_32fc_t phase_Ptr[8];

    const __m512 inc_Val = _mm512_set_ps(lv_cimag(incr),
                                         lv_creal(incr),
                                         lv_cimag(incr),
                                         lv_creal(incr),
                                         lv_cimag(incr),
                                         lv_creal(incr),
                                         lv_cimag(incr),
                                         lv_creal(incr),
                                         lv_cimag(incr),
                                         lv_creal(incr),
                                         lv_cimag(incr),
                                         lv_creal(incr),
                                         lv_cimag(incr),
                                         lv_creal(incr),
                                         lv_cimag(incr),
                                         lv_creal(incr));

#define REDUCE_ANGLE(a)              \
    do {                             \
        (a) = fmod((a), 2.0 * M_PI); \
        if ((a) > M_PI)              \
            (a) -= 2.0 * M_PI;       \
        else if ((a) < -M_PI)        \
            (a) += 2.0 * M_PI;       \
    } while (0)

    // Initialize phase vector from exact angles
    for (unsigned int k = 0; k < 8; ++k) {
        double a = angle_sum + (double)k * delta_angle;
        REDUCE_ANGLE(a);
        phase_Ptr[k] = lv_cmake((float)cos(a), (float)sin(a));
    }
    phase_Val = _mm512_load_ps((float*)phase_Ptr);

    unsigned int i, j;

    // Main loop with periodic resync
    for (i = 0; i < (unsigned int)(num_points / ROTATOR_RELOAD); ++i) {
        for (j = 0; j < ROTATOR_RELOAD_8; ++j) {
            aVal = _mm512_load_ps((float*)aPtr);
            z = _mm512_complexmul_ps(aVal, phase_Val);
            phase_Val = _mm512_complexmul_ps(phase_Val, inc_Val);
            _mm512_store_ps((float*)cPtr, z);
            aPtr += 8;
            cPtr += 8;
        }

        // Advance angle using Kahan summation
        {
            double y = block_delta - angle_c;
            double t = angle_sum + y;
            angle_c = (t - angle_sum) - y;
            angle_sum = t;
        }

        // Resync: recompute phase from accumulated angle
        for (unsigned int k = 0; k < 8; ++k) {
            double a = angle_sum + (double)k * delta_angle;
            REDUCE_ANGLE(a);
            phase_Ptr[k] = lv_cmake((float)cos(a), (float)sin(a));
        }
        phase_Val = _mm512_load_ps((float*)phase_Ptr);
    }

    // Handle remainder
    for (i = 0; i < (num_points % ROTATOR_RELOAD) / 8; ++i) {
        aVal = _mm512_load_ps((float*)aPtr);
        z = _mm512_complexmul_ps(aVal, phase_Val);
        phase_Val = _mm512_complexmul_ps(phase_Val, inc_Val);
        _mm512_store_ps((float*)cPtr, z);
        aPtr += 8;
        cPtr += 8;

        {
            double y = (8.0 * delta_angle) - angle_c;
            double t = angle_sum + y;
            angle_c = (t - angle_sum) - y;
            angle_sum = t;
        }
    }

    // Final phase output
    {
        double a = angle_sum;
        REDUCE_ANGLE(a);
        *phase = lv_cmake((float)cos(a), (float)sin(a));
    }

    // Scalar remainder
    for (i = 0; i < num_points % 8; ++i) {
        *cPtr++ = *aPtr++ * (*phase);
        {
            double y = delta_angle - angle_c;
            double t = angle_sum + y;
            angle_c = (t - angle_sum) - y;
            angle_sum = t;
        }
        double a = angle_sum;
        REDUCE_ANGLE(a);
        *phase = lv_cmake((float)cos(a), (float)sin(a));
    }

#undef REDUCE_ANGLE
}

#endif /* LV_HAVE_AVX512F */


#ifdef LV_HAVE_AVX512F
#include <immintrin.h>
#include <volk/volk_avx512_intrinsics.h>

/*!
 * \brief Unaligned AVX512F implementation with angle-based resync for numerical
 * stability.
 */
static inline void volk_32fc_s32fc_x2_rotator2_32fc_u_avx512f(lv_32fc_t* outVector,
                                                              const lv_32fc_t* inVector,
                                                              const lv_32fc_t* phase_inc,
                                                              lv_32fc_t* phase,
                                                              unsigned int num_points)
{
    lv_32fc_t* cPtr = outVector;
    const lv_32fc_t* aPtr = inVector;

    const double initial_angle =
        atan2((double)lv_cimag(*phase), (double)lv_creal(*phase));
    const double delta_angle =
        atan2((double)lv_cimag(*phase_inc), (double)lv_creal(*phase_inc));

    double angle_sum = initial_angle;
    double angle_c = 0.0;

    const double block_delta = (double)ROTATOR_RELOAD * delta_angle;

    const double delta8 = 8.0 * delta_angle;
    lv_32fc_t incr = lv_cmake((float)cos(delta8), (float)sin(delta8));

    __m512 aVal, phase_Val, z;
    lv_32fc_t phase_Ptr[8];

    const __m512 inc_Val = _mm512_set_ps(lv_cimag(incr),
                                         lv_creal(incr),
                                         lv_cimag(incr),
                                         lv_creal(incr),
                                         lv_cimag(incr),
                                         lv_creal(incr),
                                         lv_cimag(incr),
                                         lv_creal(incr),
                                         lv_cimag(incr),
                                         lv_creal(incr),
                                         lv_cimag(incr),
                                         lv_creal(incr),
                                         lv_cimag(incr),
                                         lv_creal(incr),
                                         lv_cimag(incr),
                                         lv_creal(incr));

#define REDUCE_ANGLE(a)              \
    do {                             \
        (a) = fmod((a), 2.0 * M_PI); \
        if ((a) > M_PI)              \
            (a) -= 2.0 * M_PI;       \
        else if ((a) < -M_PI)        \
            (a) += 2.0 * M_PI;       \
    } while (0)

    for (unsigned int k = 0; k < 8; ++k) {
        double a = angle_sum + (double)k * delta_angle;
        REDUCE_ANGLE(a);
        phase_Ptr[k] = lv_cmake((float)cos(a), (float)sin(a));
    }
    phase_Val = _mm512_loadu_ps((float*)phase_Ptr);

    unsigned int i, j;

    for (i = 0; i < (unsigned int)(num_points / ROTATOR_RELOAD); ++i) {
        for (j = 0; j < ROTATOR_RELOAD_8; ++j) {
            aVal = _mm512_loadu_ps((float*)aPtr);
            z = _mm512_complexmul_ps(aVal, phase_Val);
            phase_Val = _mm512_complexmul_ps(phase_Val, inc_Val);
            _mm512_storeu_ps((float*)cPtr, z);
            aPtr += 8;
            cPtr += 8;
        }

        {
            double y = block_delta - angle_c;
            double t = angle_sum + y;
            angle_c = (t - angle_sum) - y;
            angle_sum = t;
        }

        for (unsigned int k = 0; k < 8; ++k) {
            double a = angle_sum + (double)k * delta_angle;
            REDUCE_ANGLE(a);
            phase_Ptr[k] = lv_cmake((float)cos(a), (float)sin(a));
        }
        phase_Val = _mm512_loadu_ps((float*)phase_Ptr);
    }

    for (i = 0; i < (num_points % ROTATOR_RELOAD) / 8; ++i) {
        aVal = _mm512_loadu_ps((float*)aPtr);
        z = _mm512_complexmul_ps(aVal, phase_Val);
        phase_Val = _mm512_complexmul_ps(phase_Val, inc_Val);
        _mm512_storeu_ps((float*)cPtr, z);
        aPtr += 8;
        cPtr += 8;

        {
            double y = (8.0 * delta_angle) - angle_c;
            double t = angle_sum + y;
            angle_c = (t - angle_sum) - y;
            angle_sum = t;
        }
    }

    {
        double a = angle_sum;
        REDUCE_ANGLE(a);
        *phase = lv_cmake((float)cos(a), (float)sin(a));
    }

    for (i = 0; i < num_points % 8; ++i) {
        *cPtr++ = *aPtr++ * (*phase);
        {
            double y = delta_angle - angle_c;
            double t = angle_sum + y;
            angle_c = (t - angle_sum) - y;
            angle_sum = t;
        }
        double a = angle_sum;
        REDUCE_ANGLE(a);
        *phase = lv_cmake((float)cos(a), (float)sin(a));
    }

#undef REDUCE_ANGLE
}

#endif /* LV_HAVE_AVX512F */


/* Note on the RVV implementation:
 * The complex multiply was expanded, because we don't care about the corner cases.
 * Otherwise, without -ffast-math, the compiler would inserts function calls,
 * which invalidates all vector registers and spills them on each loop iteration. */

#ifdef LV_HAVE_RVV
#include <riscv_vector.h>

static inline void volk_32fc_s32fc_x2_rotator2_32fc_rvv(lv_32fc_t* outVector,
                                                        const lv_32fc_t* inVector,
                                                        const lv_32fc_t* phase_inc,
                                                        lv_32fc_t* phase,
                                                        unsigned int num_points)
{
    size_t vlmax = __riscv_vsetvlmax_e32m2();
    vlmax = vlmax < ROTATOR_RELOAD ? vlmax : ROTATOR_RELOAD;

    lv_32fc_t inc = 1.0f;
    vfloat32m2_t phr = __riscv_vfmv_v_f_f32m2(0, vlmax), phi = phr;
    for (size_t i = 0; i < vlmax; ++i) {
        lv_32fc_t ph =
            lv_cmake(lv_creal(*phase) * lv_creal(inc) - lv_cimag(*phase) * lv_cimag(inc),
                     lv_creal(*phase) * lv_cimag(inc) + lv_cimag(*phase) * lv_creal(inc));
        phr = __riscv_vfslide1down(phr, lv_creal(ph), vlmax);
        phi = __riscv_vfslide1down(phi, lv_cimag(ph), vlmax);
        inc = lv_cmake(
            lv_creal(*phase_inc) * lv_creal(inc) - lv_cimag(*phase_inc) * lv_cimag(inc),
            lv_creal(*phase_inc) * lv_cimag(inc) + lv_cimag(*phase_inc) * lv_creal(inc));
    }
    vfloat32m2_t incr = __riscv_vfmv_v_f_f32m2(lv_creal(inc), vlmax);
    vfloat32m2_t inci = __riscv_vfmv_v_f_f32m2(lv_cimag(inc), vlmax);

    size_t vl = 0;
    if (num_points > 0)
        while (1) {
            size_t n = num_points < ROTATOR_RELOAD ? num_points : ROTATOR_RELOAD;
            num_points -= n;

            for (; n > 0; n -= vl, inVector += vl, outVector += vl) {
                // vl<vlmax can only happen on the last iteration of the loops
                vl = __riscv_vsetvl_e32m2(n < vlmax ? n : vlmax);

                vuint64m4_t va = __riscv_vle64_v_u64m4((const uint64_t*)inVector, vl);
                vfloat32m2_t var = __riscv_vreinterpret_f32m2(__riscv_vnsrl(va, 0, vl));
                vfloat32m2_t vai = __riscv_vreinterpret_f32m2(__riscv_vnsrl(va, 32, vl));

                vfloat32m2_t vr =
                    __riscv_vfnmsac(__riscv_vfmul(var, phr, vl), vai, phi, vl);
                vfloat32m2_t vi =
                    __riscv_vfmacc(__riscv_vfmul(var, phi, vl), vai, phr, vl);

                vuint32m2_t vru = __riscv_vreinterpret_u32m2(vr);
                vuint32m2_t viu = __riscv_vreinterpret_u32m2(vi);
                vuint64m4_t res =
                    __riscv_vwmaccu(__riscv_vwaddu_vv(vru, viu, vl), 0xFFFFFFFF, viu, vl);
                __riscv_vse64((uint64_t*)outVector, res, vl);

                vfloat32m2_t tmp = phr;
                phr = __riscv_vfnmsac(__riscv_vfmul(tmp, incr, vl), phi, inci, vl);
                phi = __riscv_vfmacc(__riscv_vfmul(tmp, inci, vl), phi, incr, vl);
            }

            if (num_points <= 0)
                break;

            // normalize
            vfloat32m2_t scale =
                __riscv_vfmacc(__riscv_vfmul(phr, phr, vl), phi, phi, vl);
            scale = __riscv_vfsqrt(scale, vl);
            phr = __riscv_vfdiv(phr, scale, vl);
            phi = __riscv_vfdiv(phi, scale, vl);
        }

    lv_32fc_t ph = lv_cmake(__riscv_vfmv_f(phr), __riscv_vfmv_f(phi));
    for (size_t i = 0; i < vlmax - vl; ++i) {
        ph /= *phase_inc; // we're going backwards
    }
    *phase = ph * 1.0f / hypotf(lv_creal(ph), lv_cimag(ph));
}
#endif /*LV_HAVE_RVV*/

#ifdef LV_HAVE_RVVSEG
#include <riscv_vector.h>

static inline void volk_32fc_s32fc_x2_rotator2_32fc_rvvseg(lv_32fc_t* outVector,
                                                           const lv_32fc_t* inVector,
                                                           const lv_32fc_t* phase_inc,
                                                           lv_32fc_t* phase,
                                                           unsigned int num_points)
{
    size_t vlmax = __riscv_vsetvlmax_e32m2();
    vlmax = vlmax < ROTATOR_RELOAD ? vlmax : ROTATOR_RELOAD;

    lv_32fc_t inc = 1.0f;
    vfloat32m2_t phr = __riscv_vfmv_v_f_f32m2(0, vlmax), phi = phr;
    for (size_t i = 0; i < vlmax; ++i) {
        lv_32fc_t ph =
            lv_cmake(lv_creal(*phase) * lv_creal(inc) - lv_cimag(*phase) * lv_cimag(inc),
                     lv_creal(*phase) * lv_cimag(inc) + lv_cimag(*phase) * lv_creal(inc));
        phr = __riscv_vfslide1down(phr, lv_creal(ph), vlmax);
        phi = __riscv_vfslide1down(phi, lv_cimag(ph), vlmax);
        inc = lv_cmake(
            lv_creal(*phase_inc) * lv_creal(inc) - lv_cimag(*phase_inc) * lv_cimag(inc),
            lv_creal(*phase_inc) * lv_cimag(inc) + lv_cimag(*phase_inc) * lv_creal(inc));
    }
    vfloat32m2_t incr = __riscv_vfmv_v_f_f32m2(lv_creal(inc), vlmax);
    vfloat32m2_t inci = __riscv_vfmv_v_f_f32m2(lv_cimag(inc), vlmax);

    size_t vl = 0;
    if (num_points > 0)
        while (1) {
            size_t n = num_points < ROTATOR_RELOAD ? num_points : ROTATOR_RELOAD;
            num_points -= n;

            for (; n > 0; n -= vl, inVector += vl, outVector += vl) {
                // vl<vlmax can only happen on the last iteration of the loops
                vl = __riscv_vsetvl_e32m2(n < vlmax ? n : vlmax);

                vfloat32m2x2_t va =
                    __riscv_vlseg2e32_v_f32m2x2((const float*)inVector, vl);
                vfloat32m2_t var = __riscv_vget_f32m2(va, 0);
                vfloat32m2_t vai = __riscv_vget_f32m2(va, 1);

                vfloat32m2_t vr =
                    __riscv_vfnmsac(__riscv_vfmul(var, phr, vl), vai, phi, vl);
                vfloat32m2_t vi =
                    __riscv_vfmacc(__riscv_vfmul(var, phi, vl), vai, phr, vl);
                vfloat32m2x2_t vc = __riscv_vcreate_v_f32m2x2(vr, vi);
                __riscv_vsseg2e32_v_f32m2x2((float*)outVector, vc, vl);

                vfloat32m2_t tmp = phr;
                phr = __riscv_vfnmsac(__riscv_vfmul(tmp, incr, vl), phi, inci, vl);
                phi = __riscv_vfmacc(__riscv_vfmul(tmp, inci, vl), phi, incr, vl);
            }

            if (num_points <= 0)
                break;

            // normalize
            vfloat32m2_t scale =
                __riscv_vfmacc(__riscv_vfmul(phr, phr, vl), phi, phi, vl);
            scale = __riscv_vfsqrt(scale, vl);
            phr = __riscv_vfdiv(phr, scale, vl);
            phi = __riscv_vfdiv(phi, scale, vl);
        }

    lv_32fc_t ph = lv_cmake(__riscv_vfmv_f(phr), __riscv_vfmv_f(phi));
    for (size_t i = 0; i < vlmax - vl; ++i) {
        ph /= *phase_inc; // we're going backwards
    }
    *phase = ph * 1.0f / hypotf(lv_creal(ph), lv_cimag(ph));
}
#endif /*LV_HAVE_RVVSEG*/

#endif /* INCLUDED_volk_32fc_s32fc_rotator2_32fc_a_H */
