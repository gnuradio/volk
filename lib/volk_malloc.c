/* -*- c -*- */
/*
 * Copyright 2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <volk/volk_malloc.h>

/*
 * C11 features:
 * see: https://en.cppreference.com/w/c/memory/aligned_alloc
 *
 * MSVC is broken
 * see:
 * https://learn.microsoft.com/en-us/cpp/overview/visual-cpp-language-conformance?view=msvc-170
 * This section:
 * C11 The Universal CRT implemented the parts of the
 * C11 Standard Library that are required by C++17,
 * with the exception of C99 strftime() E/O alternative
 * conversion specifiers, C11 fopen() exclusive mode,
 * and C11 aligned_alloc(). The latter is unlikely to
 * be implemented, because C11 specified aligned_alloc()
 * in a way that's incompatible with the Microsoft
 * implementation of free():
 * namely, that free() must be able to handle highly aligned allocations.
 *
 * We must work around this problem because MSVC is non-compliant!
 */


void* volk_malloc(size_t size, size_t alignment)
{
    if ((size == 0) || (alignment == 0)) {
        return NULL;
    }
    // Tweak size to satisfy ASAN (the GCC address sanitizer).
    // Calling 'volk_malloc' might therefor result in the allocation of more memory than
    // requested for correct alignment. Any allocation size change here will in general
    // not impact the end result since initial size alignment is required either way.
    if (size % alignment) {
        size += alignment - (size % alignment);
    }
#if HAVE_POSIX_MEMALIGN
    // quoting posix_memalign() man page:
    // "alignment must be a power of two and a multiple of sizeof(void *)"
    // volk_get_alignment() could return 1 for some machines (e.g. generic_orc)
    if (alignment == 1) {
        return malloc(size);
    }
    void* ptr;
    int err = posix_memalign(&ptr, alignment, size);
    if (err != 0) {
        ptr = NULL;
    }
#elif defined(_MSC_VER) || defined(__MINGW32__)
    void* ptr = _aligned_malloc(size, alignment);
#else
    void* ptr = aligned_alloc(alignment, size);
#endif
    return ptr;
}

void volk_free(void* ptr)
{
#if defined(_MSC_VER) || defined(__MINGW32__)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}
