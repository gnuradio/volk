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
 * For #defines used to determine support for allocation functions,
 * see: http://linux.die.net/man/3/aligned_alloc
*/

// Otherwise, test if we are a POSIX or X/Open system
// This only has a restriction that alignment be a power of 2 and a
// multiple of sizeof(void *).
#if _POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600 || HAVE_POSIX_MEMALIGN

void *volk_malloc(size_t size, size_t alignment)
{
  void *ptr;

  // quoting posix_memalign() man page:
  // "alignment must be a power of two and a multiple of sizeof(void *)"
  // volk_get_alignment() could return 1 for some machines (e.g. generic_orc)
  if (alignment == 1)
    return malloc(size);

  int err = posix_memalign(&ptr, alignment, size);
  if(err == 0) {
    return ptr;
  }
  else {
    fprintf(stderr,
            "VOLK: Error allocating memory "
            "(posix_memalign: error %d: %s)\n", err, strerror(err));
    return NULL;
  }
}

void volk_free(void *ptr)
{
  free(ptr);
}

// _aligned_malloc has no restriction on size,
// available on Windows since Visual C++ 2005
#elif _MSC_VER >= 1400

void *volk_malloc(size_t size, size_t alignment)
{
  void *ptr = _aligned_malloc(size, alignment);
  if(ptr == NULL) {
    fprintf(stderr, "VOLK: Error allocating memory (_aligned_malloc)\n");
  }
  return ptr;
}

void volk_free(void *ptr)
{
  _aligned_free(ptr);
}

// No standard handlers; we'll do it ourselves.
#else // _POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600 || HAVE_POSIX_MEMALIGN

struct block_info
{
  void *real;
};

void *
volk_malloc(size_t size, size_t alignment)
{
  void *real, *user;
  struct block_info *info;

  /* At least align to sizeof our struct */
  if (alignment < sizeof(struct block_info))
    alignment = sizeof(struct block_info);

  /* Alloc */
  real = malloc(size + (2 * alignment - 1));

  /* Get pointer to the various zones */
  user = (void *)((((uintptr_t) real) + sizeof(struct block_info) + alignment - 1) & ~(alignment - 1));
  info = (struct block_info *)(((uintptr_t)user) - sizeof(struct block_info));

  /* Store the info for the free */
  info->real = real;

  /* Return pointer to user */
  return user;
}

void
volk_free(void *ptr)
{
  struct block_info *info;

  /* Get the real pointer */
  info = (struct block_info *)(((uintptr_t)ptr) - sizeof(struct block_info));

  /* Release real pointer */
  free(info->real);
}

#endif // _POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600 || HAVE_POSIX_MEMALIGN

//#endif // _ISOC11_SOURCE
