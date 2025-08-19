/* -*- c++ -*- */
/*
 * Copyright 2011-2012 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include <volk/volk_cpu.h>
#include <volk/volk_config_fixed.h>
#include <stdlib.h>
#include <string.h>


#if defined(VOLK_CPU_FEATURES)
#include "cpu_features_macros.h"
#if defined(CPU_FEATURES_ARCH_X86)
#include "cpuinfo_x86.h"
#elif defined(CPU_FEATURES_ARCH_ARM)
#include "cpuinfo_arm.h"
#elif defined(CPU_FEATURES_ARCH_AARCH64)
#include "cpuinfo_aarch64.h"
#elif defined(CPU_FEATURES_ARCH_MIPS)
#include "cpuinfo_mips.h"
#elif defined(CPU_FEATURES_ARCH_PPC)
#include "cpuinfo_ppc.h"
#elif defined(CPU_FEATURES_ARCH_RISCV)
#include "cpuinfo_riscv.h"
#endif

// This is required for MSVC
#if defined(__cplusplus)
using namespace cpu_features;
#endif
#endif


struct VOLK_CPU volk_cpu;

%for arch in archs:
static int i_can_has_${arch.name} (void) {
    %for check, params in arch.checks:
        %if "neon" in arch.name:
#if defined(CPU_FEATURES_ARCH_ARM)
    if (GetArmInfo().features.${check} == 0){ return 0; }
#endif
        %elif "neon" in arch.name or "sve" in arch.name:
#if defined(CPU_FEATURES_ARCH_AARCH64)
    if (GetAarch64Info().features.${check} == 0){ return 0; }
#endif
        %elif "mips" in arch.name:
#if defined(CPU_FEATURES_ARCH_MIPS)
    if (GetMipsInfo().features.${check} == 0){ return 0; }
#endif
        %elif "riscv" in arch.name or arch.name[:2] == "rv":
#if defined(CPU_FEATURES_ARCH_RISCV)
    if (GetRiscvInfo().features.${check} == 0){ return 0; }
#endif
        %else:
#if defined(CPU_FEATURES_ARCH_X86)
    if (GetX86Info().features.${check} == 0){ return 0; }
#endif
        %endif
    %endfor
    return 1;
}

%endfor

#if defined(HAVE_FENV_H)
    #if defined(FE_TONEAREST)
        #include <fenv.h>
        static inline void set_float_rounding(void){
            fesetround(FE_TONEAREST);
        }
    #else
        static inline void set_float_rounding(void){
            //do nothing
        }
    #endif
#elif defined(_MSC_VER)
    #include <float.h>
    static inline void set_float_rounding(void){
        unsigned int cwrd;
        _controlfp_s(&cwrd, 0, 0);
        _controlfp_s(&cwrd, _RC_NEAR, _MCW_RC);
    }
#else
    static inline void set_float_rounding(void){
        //do nothing
    }
#endif


void volk_cpu_init() {
    %for arch in archs:
    volk_cpu.has_${arch.name} = &i_can_has_${arch.name};
    %endfor
    set_float_rounding();
}

unsigned int volk_get_lvarch() {
    unsigned int retval = 0;
    volk_cpu_init();
    %for arch in archs:
    retval += volk_cpu.has_${arch.name}() << LV_${arch.name.upper()};
    %endfor
    return retval;
}
