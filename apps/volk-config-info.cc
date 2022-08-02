/* -*- c++ -*- */
/*
 * Copyright 2013, 2016, 2018 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include <volk/constants.h> // for volk_available_machines, volk_c_com...
#include <iostream>         // for operator<<, endl, cout, ostream
#include <string>           // for string

#include "volk/volk.h"           // for volk_get_alignment, volk_get_machine
#include "volk_option_helpers.h" // for option_list, option_t

void print_alignment()
{
    std::cout << "Alignment in bytes: " << volk_get_alignment() << std::endl;
}

void print_malloc()
{
    // You don't want to change the volk_malloc code, so just copy the if/else
    // structure from there and give an explanation for the implementations
    std::cout << "Used malloc implementation: ";
#if HAVE_POSIX_MEMALIGN
    std::cout << "posix_memalign" << std::endl;
#elif defined(_MSC_VER)
    std::cout << "_aligned_malloc" << std::endl;
#else
    std::cout << "C11 aligned_alloc" << std::endl;
#endif
}


int main(int argc, char** argv)
{

    option_list our_options("volk-config-info");
    our_options.add(
        option_t("prefix", "", "print the VOLK installation prefix", volk_prefix()));
    our_options.add(
        option_t("cc", "", "print the VOLK C compiler version", volk_c_compiler()));
    our_options.add(
        option_t("cflags", "", "print the VOLK CFLAGS", volk_compiler_flags()));
    our_options.add(option_t(
        "all-machines", "", "print VOLK machines built", volk_available_machines()));
    our_options.add(option_t("avail-machines",
                             "",
                             "print VOLK machines on the current "
                             "platform",
                             volk_list_machines));
    our_options.add(option_t("machine",
                             "",
                             "print the current VOLK machine that will be used",
                             volk_get_machine()));
    our_options.add(
        option_t("alignment", "", "print the memory alignment", print_alignment));
    our_options.add(option_t("malloc",
                             "",
                             "print the malloc implementation used in volk_malloc",
                             print_malloc));
    our_options.add(option_t("version", "v", "print the VOLK version", volk_version()));

    our_options.parse(argc, argv);

    return 0;
}
