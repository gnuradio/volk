/* -*- c -*- */
/*
 * Copyright 2014 Free Software Foundation, Inc.
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
 * https://docs.microsoft.com/en-us/cpp/overview/visual-cpp-language-conformance?view=vs-2019
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
        fprintf(stderr,
                "VOLK: Error allocating memory "
                "(posix_memalign: error %d: %s)\n",
                err,
                strerror(err));
    }
#elif defined(_MSC_VER)
    void* ptr = _aligned_malloc(size, alignment);
#else
    void* ptr = aligned_alloc(alignment, size);
#endif
    if (ptr == NULL) {
        fprintf(stderr,
                "VOLK: Error allocating memory (aligned_alloc/_aligned_malloc)\n");
    }
    return ptr;
}

void volk_free(void* ptr)
{
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}
