/* -*- c++ -*- */
/*
 * Copyright 2013, 2016, 2018 Free Software Foundation, Inc.
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

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include <volk/constants.h>       // for volk_available_machines, volk_c_com...
#include <iostream>               // for operator<<, endl, cout, ostream
#include <string>                 // for string

#include "volk/volk.h"            // for volk_get_alignment, volk_get_machine
#include "volk_option_helpers.h"  // for option_list, option_t

void print_alignment()
{
  std::cout << "Alignment in bytes: " << volk_get_alignment() << std::endl;
}

void print_malloc()
{
  // You don't want to change the volk_malloc code, so just copy the if/else
  // structure from there and give an explanation for the implementations
  std::cout << "Used malloc implementation: ";
  #if _POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600 || HAVE_POSIX_MEMALIGN
  std::cout << "posix_memalign" << std::endl;
  #elif _MSC_VER >= 1400
  std::cout << "aligned_malloc" << std::endl;
  #else
      std::cout << "No standard handler available, using own implementation." << std::endl;
  #endif
}


int
main(int argc, char **argv)
{

  option_list our_options("volk-config-info");
  our_options.add(option_t("prefix", "", "print the VOLK installation prefix", volk_prefix()));
  our_options.add(option_t("cc", "", "print the VOLK C compiler version", volk_c_compiler()));
  our_options.add(option_t("cflags", "", "print the VOLK CFLAGS", volk_compiler_flags()));
  our_options.add(option_t("all-machines", "", "print VOLK machines built", volk_available_machines()));
  our_options.add(option_t("avail-machines", "", "print VOLK machines on the current "
      "platform", volk_list_machines));
  our_options.add(option_t("machine", "", "print the current VOLK machine that will be used",
                           volk_get_machine()));
  our_options.add(option_t("alignment", "", "print the memory alignment", print_alignment));
  our_options.add(option_t("malloc", "", "print the malloc implementation used in volk_malloc",
                           print_malloc));
  our_options.add(option_t("version", "v", "print the VOLK version", volk_version()));

  our_options.parse(argc, argv);

  return 0;
}
