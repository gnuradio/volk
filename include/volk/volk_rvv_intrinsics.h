/* -*- c++ -*- */
/*
 * Copyright 2024 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*
 * This file is intended to hold RVV intrinsics of intrinsics.
 * They should be used in VOLK kernels to avoid copy-paste.
 */

#ifndef INCLUDE_VOLK_VOLK_RVV_INTRINSICS_H_
#define INCLUDE_VOLK_VOLK_RVV_INTRINSICS_H_
#include <riscv_vector.h>

#define RISCV_SHRINK2(op, T, S, v)              \
    __riscv_##op(__riscv_vget_##T##S##m1(v, 0), \
                 __riscv_vget_##T##S##m1(v, 1), \
                 __riscv_vsetvlmax_e##S##m1())

#define RISCV_SHRINK4(op, T, S, v)                           \
    __riscv_##op(__riscv_##op(__riscv_vget_##T##S##m1(v, 0), \
                              __riscv_vget_##T##S##m1(v, 1), \
                              __riscv_vsetvlmax_e##S##m1()), \
                 __riscv_##op(__riscv_vget_##T##S##m1(v, 2), \
                              __riscv_vget_##T##S##m1(v, 3), \
                              __riscv_vsetvlmax_e##S##m1()), \
                 __riscv_vsetvlmax_e##S##m1())

#define RISCV_SHRINK8(op, T, S, v)                                        \
    __riscv_##op(__riscv_##op(__riscv_##op(__riscv_vget_##T##S##m1(v, 0), \
                                           __riscv_vget_##T##S##m1(v, 1), \
                                           __riscv_vsetvlmax_e##S##m1()), \
                              __riscv_##op(__riscv_vget_##T##S##m1(v, 2), \
                                           __riscv_vget_##T##S##m1(v, 3), \
                                           __riscv_vsetvlmax_e##S##m1()), \
                              __riscv_vsetvlmax_e##S##m1()),              \
                 __riscv_##op(__riscv_##op(__riscv_vget_##T##S##m1(v, 4), \
                                           __riscv_vget_##T##S##m1(v, 5), \
                                           __riscv_vsetvlmax_e##S##m1()), \
                              __riscv_##op(__riscv_vget_##T##S##m1(v, 6), \
                                           __riscv_vget_##T##S##m1(v, 7), \
                                           __riscv_vsetvlmax_e##S##m1()), \
                              __riscv_vsetvlmax_e##S##m1()),              \
                 __riscv_vsetvlmax_e##S##m1())

#define RISCV_PERM4(f, v, vidx)                                     \
    __riscv_vcreate_v_u8m1_u8m4(                                    \
        f(__riscv_vget_u8m1(v, 0), vidx, __riscv_vsetvlmax_e8m1()), \
        f(__riscv_vget_u8m1(v, 1), vidx, __riscv_vsetvlmax_e8m1()), \
        f(__riscv_vget_u8m1(v, 2), vidx, __riscv_vsetvlmax_e8m1()), \
        f(__riscv_vget_u8m1(v, 3), vidx, __riscv_vsetvlmax_e8m1()))

#define RISCV_LUT4(f, vtbl, v)                                      \
    __riscv_vcreate_v_u8m1_u8m4(                                    \
        f(vtbl, __riscv_vget_u8m1(v, 0), __riscv_vsetvlmax_e8m1()), \
        f(vtbl, __riscv_vget_u8m1(v, 1), __riscv_vsetvlmax_e8m1()), \
        f(vtbl, __riscv_vget_u8m1(v, 2), __riscv_vsetvlmax_e8m1()), \
        f(vtbl, __riscv_vget_u8m1(v, 3), __riscv_vsetvlmax_e8m1()))

#define RISCV_PERM8(f, v, vidx)                                     \
    __riscv_vcreate_v_u8m1_u8m8(                                    \
        f(__riscv_vget_u8m1(v, 0), vidx, __riscv_vsetvlmax_e8m1()), \
        f(__riscv_vget_u8m1(v, 1), vidx, __riscv_vsetvlmax_e8m1()), \
        f(__riscv_vget_u8m1(v, 2), vidx, __riscv_vsetvlmax_e8m1()), \
        f(__riscv_vget_u8m1(v, 3), vidx, __riscv_vsetvlmax_e8m1()), \
        f(__riscv_vget_u8m1(v, 4), vidx, __riscv_vsetvlmax_e8m1()), \
        f(__riscv_vget_u8m1(v, 5), vidx, __riscv_vsetvlmax_e8m1()), \
        f(__riscv_vget_u8m1(v, 6), vidx, __riscv_vsetvlmax_e8m1()), \
        f(__riscv_vget_u8m1(v, 7), vidx, __riscv_vsetvlmax_e8m1()))

#define RISCV_VMFLTZ(T, v, vl) __riscv_vmslt(__riscv_vreinterpret_i##T(v), 0, vl)

/*
 * Polynomial coefficients for log2(x)/(x-1) on [1, 2]
 * Generated with Sollya: remez(log2(x)/(x-1), 6, [1+1b-20, 2])
 * Max error: ~1.55e-6
 *
 * Usage: log2(x) ≈ poly(x) * (x - 1) for x ∈ [1, 2]
 * Polynomial evaluated via Horner's method with FMA
 *
 * Parameters:
 *   x: mantissa values in [1, 2)
 *   vl: vector length for operations
 *   vlmax: maximum vector length used for creating coefficient vectors
 */
static inline vfloat32m2_t
__riscv_vlog2_poly_f32m2(vfloat32m2_t x, size_t vl, size_t vlmax)
{
    const vfloat32m2_t c0 = __riscv_vfmv_v_f_f32m2(+0x1.a8a726p+1f, vlmax);
    const vfloat32m2_t c1 = __riscv_vfmv_v_f_f32m2(-0x1.0b7f7ep+2f, vlmax);
    const vfloat32m2_t c2 = __riscv_vfmv_v_f_f32m2(+0x1.05d9ccp+2f, vlmax);
    const vfloat32m2_t c3 = __riscv_vfmv_v_f_f32m2(-0x1.4d476cp+1f, vlmax);
    const vfloat32m2_t c4 = __riscv_vfmv_v_f_f32m2(+0x1.04fc3ap+0f, vlmax);
    const vfloat32m2_t c5 = __riscv_vfmv_v_f_f32m2(-0x1.c97982p-3f, vlmax);
    const vfloat32m2_t c6 = __riscv_vfmv_v_f_f32m2(+0x1.57aa42p-6f, vlmax);

    // Horner's method with FMA: c0 + x*(c1 + x*(c2 + ...))
    vfloat32m2_t poly = c6;
    poly = __riscv_vfmadd(poly, x, c5, vl);
    poly = __riscv_vfmadd(poly, x, c4, vl);
    poly = __riscv_vfmadd(poly, x, c3, vl);
    poly = __riscv_vfmadd(poly, x, c2, vl);
    poly = __riscv_vfmadd(poly, x, c1, vl);
    poly = __riscv_vfmadd(poly, x, c0, vl);
    return poly;
}

#endif /* INCLUDE_VOLK_VOLK_RVV_INTRINSICS_H_ */
