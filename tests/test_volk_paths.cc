// SPDX-License-Identifier: LGPL-3.0-or-later
#include <gtest/gtest.h>
#include <volk/volk_prefs.h>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>

// Filesystem compatibility
#if defined(__has_include)
#  if __has_include(<filesystem>)
#    include <filesystem>
#    namespace fs = std::filesystem;
#  elif __has_include(<experimental/filesystem>)
#    include <experimental/filesystem>
#    namespace fs = std::experimental::filesystem;
#  else
#    error "<filesystem> or <experimental/filesystem> is required"
#  endif
#else
#  include <filesystem>
#  namespace fs = std::filesystem;
#endif

static void set_env(const std::string& name, const char* value)
{
#if defined(_MSC_VER)
    if (value)
        _putenv_s(name.c_str(), value);
    else
        _putenv_s(name.c_str(), "");
#else
    if (value)
        setenv(name.c_str(), value, 1);
    else
        unsetenv(name.c_str());
#endif
}

TEST(VolkPaths, EnvOverrideTakesPrecedence)
{
    auto tmp = fs::temp_directory_path() / "volk_test_override";
    fs::remove_all(tmp);
    fs::create_directories(tmp / "volk");
    auto cfg = tmp / "volk" / "volk_config";
    std::ofstream(cfg.string()) << "# override" << std::endl;

    set_env("VOLK_CONFIGPATH", tmp.string().c_str());
    set_env("XDG_CONFIG_HOME", nullptr);
    set_env("HOME", nullptr);

    char path[512] = { 0 };
    volk_get_config_path(path, true);
    EXPECT_EQ(std::string(path), cfg.string());

    fs::remove_all(tmp);
}

TEST(VolkPaths, XdgPrefersXdgWhenPresent)
{
    auto tmp = fs::temp_directory_path() / "volk_test_xdg";
    fs::remove_all(tmp);
    fs::create_directories(tmp / "volk");
    auto cfg = tmp / "volk" / "volk_config";
    std::ofstream(cfg.string()) << "# test" << std::endl;

    set_env("XDG_CONFIG_HOME", tmp.string().c_str());
    set_env("HOME", nullptr);
    set_env("VOLK_CONFIGPATH", nullptr);

    char path[512] = { 0 };
    volk_get_config_path(path, true);
    EXPECT_EQ(std::string(path), cfg.string());

    fs::remove_all(tmp);
}

TEST(VolkPaths, ReadFallbackReturnsXdgWhenNoneExist)
{
    auto tmp = fs::temp_directory_path() / "volk_test_xdg_fallback";
    fs::remove_all(tmp);
    fs::create_directories(tmp);

    set_env("XDG_CONFIG_HOME", tmp.string().c_str());
    set_env("VOLK_CONFIGPATH", nullptr);
    set_env("HOME", nullptr);

    char path[512] = { 0 };
    volk_get_config_path(path, true);
    std::string got(path);
    std::string expect = (tmp / "volk" / "volk_config").string();
    EXPECT_EQ(got, expect);

    fs::remove_all(tmp);
}

TEST(VolkPaths, ReadFallbackReturnsHomeWhenXdgUnset)
{
    auto tmp = fs::temp_directory_path() / "volk_test_home_fallback";
    fs::remove_all(tmp);
    fs::create_directories(tmp);

    set_env("XDG_CONFIG_HOME", nullptr);
    set_env("VOLK_CONFIGPATH", nullptr);
    set_env("HOME", tmp.string().c_str());

    char path[512] = { 0 };
    volk_get_config_path(path, true);
    std::string got(path);
    std::string expect = (tmp / ".config" / "volk" / "volk_config").string();
    EXPECT_EQ(got, expect);

    fs::remove_all(tmp);
}

