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

#endif /* INCLUDE_VOLK_VOLK_RVV_INTRINSICS_H_ */
