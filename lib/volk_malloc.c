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
 *
 * C11 features:
 * see: https://en.cppreference.com/w/c/memory/aligned_alloc
*/


void *volk_malloc(size_t size, size_t alignment)
{
  if (alignment == 1){
    return malloc(size);
  }

  void *ptr = aligned_alloc(alignment, size);
  if(ptr == NULL) {
    fprintf(stderr, "VOLK: Error allocating memory (aligned_alloc was POSIX)\n");
  }
  return ptr;
}

void volk_free(void *ptr)
{
  free(ptr);
}



//#endif // _ISOC11_SOURCE
