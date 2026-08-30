/* -*- c++ -*- */
/*
 * Copyright 2022 Johannes Demel
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include <gtest/gtest.h>
#include <volk/volk.h>
#include <algorithm>
#include <tuple>


std::vector<std::string> get_kernel_implementation_name_list(const volk_func_desc_t desc)
{
    std::vector<std::string> names;
    for (size_t i = 0; i < desc.n_impls; i++) {
        names.push_back(std::string(desc.impl_names[i]));
    }
    std::sort(names.begin(), names.end());
    return names;
}

bool is_aligned_implementation_name(const std::string& name)
{
    return name.rfind("a_", 0) == 0;
}

std::tuple<std::vector<std::string>, std::vector<std::string>>
separate_implementations_by_alignment(const std::vector<std::string>& names)
{
    std::vector<std::string> aligned;
    std::vector<std::string> unaligned;
    for (auto name : names) {
        if (is_aligned_implementation_name(name)) {
            aligned.push_back(name);
        } else {
            unaligned.push_back(name);
        }
    }
    return { aligned, unaligned };
}

std::vector<std::string>
get_aligned_kernel_implementation_names(const volk_func_desc_t desc)
{
    auto impls = get_kernel_implementation_name_list(desc);
    auto [aligned, unaligned] = separate_implementations_by_alignment(impls);
    return aligned;
}

std::vector<std::string>
get_unaligned_kernel_implementation_names(const volk_func_desc_t desc)
{
    auto impls = get_kernel_implementation_name_list(desc);
    auto [aligned, unaligned] = separate_implementations_by_alignment(impls);
    return unaligned;
}
