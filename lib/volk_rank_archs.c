/* -*- c++ -*- */
/*
 * Copyright 2011-2012 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */


#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <volk/volk_prefs.h>
#include <volk_rank_archs.h>

int volk_get_index(const char* impl_names[], // list of implementations by name
                   const size_t n_impls,     // number of implementations available
                   const char* impl_name     // the implementation name to find
)
{
    if (n_impls == 0) {
        fprintf(stderr, "Volk error: no implementations available\n");
        return -1;
    }
    unsigned int i;
    for (i = 0; i < n_impls; i++) {
        if (!strcmp(impl_names[i], impl_name)) {
            return i;
        }
    }
    // requested impl not found — try falling back to "generic"
    if (strcmp(impl_name, "generic") != 0) {
        fprintf(stderr,
                "Volk warning: arch '%s' not found, returning generic impl\n",
                impl_name);
        for (i = 0; i < n_impls; i++) {
            if (!strcmp(impl_names[i], "generic")) {
                return i;
            }
        }
    }
    // neither requested impl nor "generic" found — return first available
    fprintf(stderr, "Volk warning: no generic impl found, returning index 0\n");
    return 0;
}

int volk_rank_archs(const char* kern_name,       // name of the kernel to rank
                    const char* impl_names[],    // list of implementations by name
                    const uint64_t* impl_deps,   // requirement mask per implementation
                    const bool* alignment,       // alignment status of each implementation
                    size_t n_impls,              // number of implementations available
                    const bool align             // if false, filter aligned implementations
)
{
    if (n_impls == 0) {
        fprintf(stderr, "Volk error: %s has no implementations\n", kern_name);
        return 0;
    }
    size_t i;
    static volk_arch_pref_t* volk_arch_prefs;
    static size_t n_arch_prefs = 0;
    static int prefs_loaded = 0;
    if (!prefs_loaded) {
        n_arch_prefs = volk_load_preferences(&volk_arch_prefs);
        prefs_loaded = 1;
    }

    // If we've defined VOLK_GENERIC to be anything, always return the
    // 'generic' kernel. Used in GR's QA code.
    char* gen_env = getenv("VOLK_GENERIC");
    if (gen_env) {
        return volk_get_index(impl_names, n_impls, "generic");
    }

    // If we've defined the kernel name as an environment variable, always return
    // the 'overridden' kernel. Used for manually overring config kernels at runtime.
    char* override_env = getenv(kern_name);
    if (override_env) {
        return volk_get_index(impl_names, n_impls, override_env);
    }

    // now look for the function name in the prefs list
    for (i = 0; i < n_arch_prefs; i++) {
        if (!strncmp(kern_name,
                     volk_arch_prefs[i].name,
                     sizeof(volk_arch_prefs[i].name))) // found it
        {
            const char* impl_name =
                align ? volk_arch_prefs[i].impl_a : volk_arch_prefs[i].impl_u;
            return volk_get_index(impl_names, n_impls, impl_name);
        }
    }

    // return the best index with the largest deps
    size_t best_index_a = 0;
    size_t best_index_u = 0;
    int64_t best_value_a = -1;
    int64_t best_value_u = -1;
    for (i = 0; i < n_impls; i++) {
        const int64_t val = (int64_t)impl_deps[i];
        if (alignment[i] && val > best_value_a) {
            best_index_a = i;
            best_value_a = val;
        }
        if (!alignment[i] && val > best_value_u) {
            best_index_u = i;
            best_value_u = val;
        }
    }

    // when align and we found a best aligned, use it
    if (align && best_value_a != -1)
        return best_index_a;

    // otherwise return the best unaligned
    return best_index_u;
}
