/* -*- c++ -*- */
/*
 * Copyright 2012-2014 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include <stddef.h>          // for size_t
#include <sys/stat.h>        // for stat
#include <volk/volk_prefs.h> // for volk_get_config_path
#include <filesystem>
#include <fstream>  // IWYU pragma: keep
#include <sstream>  // for istringstream
#include <iomanip>  // for setprecision, fixed
#include <iostream> // for operator<<, basic_ostream
#include <map>      // for map, map<>::iterator
#include <utility>  // for pair
#include <vector>   // for vector, vector<>::const_...

#include "kernel_tests.h"        // for init_test_list
#include "qa_utils.h"            // for volk_test_results_t, vol...
#include "volk/volk_complex.h"   // for lv_32fc_t
#include "volk_option_helpers.h" // for option_list, option_t
#include "volk_profile.h"

namespace fs = std::filesystem;

volk_test_params_t test_params(1e-6f, 327.f, 131071, 1987, false, "");

void set_benchmark(bool val) { test_params.set_benchmark(val); }
void set_tolerance(float val) { test_params.set_tol(val); }
void set_vlen(int val) { test_params.set_vlen((unsigned int)val); }
void set_iter(int val) { test_params.set_iter((unsigned int)val); }
std::vector<std::string> kernel_patterns;
void set_substr(std::string val) { kernel_patterns.push_back(val); }
bool update_mode = false;
void set_update(bool val) { update_mode = val; }
bool dry_run = false;
void set_dryrun(bool val) { dry_run = val; }
std::string json_filename("");
void set_json(std::string val) { json_filename = val; }
std::string volk_config_path("");
void set_volk_config(std::string val) { volk_config_path = val; }
void set_warmup(int val) { volk_test_set_warmup_ms((double)val); }
void set_trials(int val) { test_params.set_trials((unsigned int)val); }

int main(int argc, char* argv[])
{
    option_list profile_options("volk_profile");
    
    profile_options.add(
        option_t("benchmark", "b", "Run all kernels (benchmark mode)", set_benchmark));
    profile_options.add(
        option_t("tol", "t", "Set the default tolerance for all tests", set_tolerance));
    profile_options.add(
        option_t("vlen", "v", "Set the default vector length for tests", set_vlen));
    profile_options.add((option_t(
        "iter", "i", "Set the default number of test iterations per kernel", set_iter)));
    profile_options.add((option_t("tests-substr",
                                  "R",
                                  "Run tests matching substring (can be repeated)",
                                  set_substr)));
    profile_options.add(
        (option_t("update", "u", "Run only kernels missing from config", set_update)));
    profile_options.add(
        (option_t("dry-run",
                  "n",
                  "Dry run. Respect other options, but don't write to file",
                  set_dryrun)));
    profile_options.add((option_t(
        "json", "j", "Write results to JSON file named as argument value", set_json)));
    profile_options.add(
        (option_t("path", "p", "Specify the volk_config path", set_volk_config)));
    profile_options.add(
        (option_t("warmup", "w", "Set warmup time in ms (default 2000)", set_warmup)));
    profile_options.add(
        (option_t("trials", "T", "Timing trials per arch (-T alone defaults to 10)", set_trials)));
    profile_options.parse(argc, argv);

    if (profile_options.present("help")) {
        return 0;
    }

    // Resolve trials default: 0 means the user didn't provide a value
    if (test_params.trials() == 0) {
        if (profile_options.present("trials")) {
            // -T was given without a numeric argument; default to 10
            test_params.set_trials(10);
            std::cout << "Using default of 10 timing trials" << std::endl;
        } else {
            test_params.set_trials(1);
        }
    }

    if (dry_run) {
        std::cout << "Warning: this IS a dry-run. Config will not be written!"
                  << std::endl;
    }

    // Adding program options
    std::ofstream json_file;
    std::string config_file;

    if (json_filename != "") {
        json_file.open(json_filename.c_str());
    }

    if (volk_config_path != "") {
        config_file = volk_config_path + "/volk_config";
    }

    // Run tests
    std::vector<volk_test_results_t> results;
    if (update_mode) {
        if (config_file != "")
            read_results(&results, config_file);
        else
            read_results(&results);
    }

    // Initialize the list of tests
    std::vector<volk_test_case_t> test_cases = init_test_list(test_params);

    // Iterate through list of tests running each one
    for (unsigned int ii = 0; ii < test_cases.size(); ++ii) {
        volk_test_case_t test_case = test_cases[ii];
        std::string test_case_name = test_case.name();

        // Check if kernel name matches any of the patterns (or all if no patterns given)
        bool regex_match = kernel_patterns.empty();
        for (const auto& pattern : kernel_patterns) {
            if (test_case_name.find(pattern) != std::string::npos) {
                regex_match = true;
                break;
            }
        }

        // if we are in update mode check if we've already got results
        // if we have any, then no need to test that kernel
        bool update = true;
        if (update_mode) {
            for (unsigned int jj = 0; jj < results.size(); ++jj) {
                if (results[jj].name == test_case.name() ||
                    results[jj].name == test_case.puppet_master_name()) {
                    update = false;
                    break;
                }
            }
        }

        if (regex_match && update) {
            try {
                run_volk_tests(test_case.desc(),
                               test_case.kernel_ptr(),
                               test_case.name(),
                               test_case.test_parameters(),
                               &results,
                               test_case.puppet_master_name());
            } catch (std::string& error) {
                std::cerr << "Caught Exception in 'run_volk_tests': " << error
                          << std::endl;
            }
        }
    }


    // Print session summary
    if (results.size() > 0) {
        double sum_speedup = 0.0;
        double max_speedup = 0.0;
        std::string max_speedup_kernel;
        int kernel_count = 0;

        for (const auto& r : results) {
            auto generic_it = r.results.find("generic");
            if (generic_it == r.results.end())
                continue;
            double generic_time = generic_it->second.time;

            // Use best aligned arch for speedup calculation
            auto best_it = r.results.find(r.best_arch_a);
            if (best_it != r.results.end()) {
                double speedup = generic_time / best_it->second.time;
                sum_speedup += speedup;
                if (speedup > max_speedup) {
                    max_speedup = speedup;
                    max_speedup_kernel = r.name;
                }
                kernel_count++;
            }
        }

        if (kernel_count > 0) {
            std::cout << std::endl;
            std::cout << "Session summary (" << kernel_count << " kernels):" << std::endl;
            std::cout << "  Average speedup vs generic: " << std::fixed
                      << std::setprecision(2) << (sum_speedup / kernel_count) << "x"
                      << std::endl;
            std::cout << "  Max speedup vs generic: " << max_speedup << "x ("
                      << max_speedup_kernel << ")" << std::endl;
        }
    }

    // Output results according to provided options
    if (json_filename != "") {
        write_json(json_file, results);
        json_file.close();
    }

    if (!dry_run) {
        if (config_file != "")
            write_results(&results, false, config_file);
        else
            write_results(&results, false);
    } else {
        std::cout << "Warning: this was a dry-run. Config not generated" << std::endl;
    }
    return 0;
}

void read_results(std::vector<volk_test_results_t>* results)
{
    char path[1024];
    volk_get_config_path(path, true);
    if (path[0] == 0) {
        std::cout << "No prior test results found ..." << std::endl;
        return;
    }

    read_results(results, std::string(path));
}

void read_results(std::vector<volk_test_results_t>* results, std::string path)
{
    struct stat buffer;
    bool config_status = (stat(path.c_str(), &buffer) == 0);

    if (config_status) {
        // a config exists and we are reading results from it
        std::ifstream config(path.c_str());
        std::string line;
        while (std::getline(config, line)) {
            // tokenize the input line by kernel_name aligned unaligned
            std::istringstream iss(line);
            std::string name, arch_a, arch_u;
            if ((iss >> name >> arch_a >> arch_u) && name.size() < 128) {
                volk_test_results_t kernel_result;
                kernel_result.name = name;
                kernel_result.config_name = name;
                kernel_result.best_arch_u = arch_a;
                kernel_result.best_arch_a = arch_u;
                results->push_back(kernel_result);
            }
        }
    }
}

void write_results(const std::vector<volk_test_results_t>* results, bool update_result)
{
    char path[1024];
    volk_get_config_path(path, false);
    if (path[0] == 0) {
        std::cout << "Aborting 'No config save path found' ..." << std::endl;
        return;
    }

    write_results(results, update_result, std::string(path));
}

void write_results(const std::vector<volk_test_results_t>* results,
                   bool update_result,
                   const std::string path)
{
    //    struct stat buffer;
    //    bool config_status = (stat (path.c_str(), &buffer) == 0);

    /*
     * These
     */
    const fs::path config_path(path);
    if (!fs::exists(config_path.parent_path())) {
        std::cout << "Creating " << config_path.parent_path() << "..." << std::endl;
        fs::create_directories(config_path.parent_path());
    }

    std::ofstream config;
    if (update_result) {
        std::cout << "Updating " << path << "..." << std::endl;
        config.open(path.c_str(), std::ofstream::app);
        if (!config.is_open()) { // either we don't have write access or we don't have the
                                 // dir yet
            std::cout << "Error opening file " << path << std::endl;
        }
    } else {
        std::cout << "Writing " << path << "..." << std::endl;
        config.open(path.c_str());
        if (!config.is_open()) { // either we don't have write access or we don't have the
                                 // dir yet
            std::cout << "Error opening file " << path << std::endl;
        }

        config << "\
#this file is generated by volk_profile.\n\
#the function name is followed by the preferred architecture.\n\
";
    }

    std::vector<volk_test_results_t>::const_iterator profile_results;
    for (profile_results = results->begin(); profile_results != results->end();
         ++profile_results) {
        config << profile_results->config_name << " " << profile_results->best_arch_a
               << " " << profile_results->best_arch_u << std::endl;
    }
    config.close();
}

void write_json(std::ofstream& json_file, std::vector<volk_test_results_t> results)
{
    json_file << "{" << std::endl;
    json_file << " \"volk_tests\": [" << std::endl;
    size_t len = results.size();
    size_t i = 0;
    std::vector<volk_test_results_t>::iterator result;
    for (result = results.begin(); result != results.end(); ++result) {
        json_file << "  {" << std::endl;
        json_file << "   \"name\": \"" << result->name << "\"," << std::endl;
        json_file << "   \"vlen\": " << (int)(result->vlen) << "," << std::endl;
        json_file << "   \"iter\": " << result->iter << "," << std::endl;
        json_file << "   \"best_arch_a\": \"" << result->best_arch_a << "\","
                  << std::endl;
        json_file << "   \"best_arch_u\": \"" << result->best_arch_u << "\","
                  << std::endl;
        json_file << "   \"results\": {" << std::endl;
        size_t results_len = result->results.size();
        size_t ri = 0;

        std::map<std::string, volk_test_time_t>::iterator kernel_time_pair;
        for (kernel_time_pair = result->results.begin();
             kernel_time_pair != result->results.end();
             ++kernel_time_pair) {
            volk_test_time_t time = kernel_time_pair->second;
            json_file << "    \"" << time.name << "\": {" << std::endl;
            json_file << "     \"name\": \"" << time.name << "\"," << std::endl;
            json_file << "     \"time\": " << time.time << "," << std::endl;
            json_file << "     \"time_stddev\": " << time.time_stddev << ","
                      << std::endl;
            json_file << "     \"time_min\": " << time.time_min << "," << std::endl;
            if (!time.trial_times.empty()) {
                json_file << "     \"trial_times\": [";
                for (size_t ti = 0; ti < time.trial_times.size(); ti++) {
                    if (ti > 0)
                        json_file << ", ";
                    json_file << time.trial_times[ti];
                }
                json_file << "]," << std::endl;
            }
            json_file << "     \"units\": \"" << time.units << "\"" << std::endl;
            json_file << "    }";
            if (ri + 1 != results_len) {
                json_file << ",";
            }
            json_file << std::endl;
            ri++;
        }
        json_file << "   }" << std::endl;
        json_file << "  }";
        if (i + 1 != len) {
            json_file << ",";
        }
        json_file << std::endl;
        i++;
    }
    json_file << " ]" << std::endl;
    json_file << "}" << std::endl;
}
