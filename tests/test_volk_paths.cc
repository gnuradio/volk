// SPDX-License-Identifier: LGPL-3.0-or-later
#include <gtest/gtest.h>
#include <volk/volk_prefs.h>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

#if defined(_MSC_VER)
static void set_env(const char* name, const char* value)
{
    _putenv_s(name, value ? value : "");
}
#else
static void set_env(const char* name, const char* value)
{
    if (value)
        setenv(name, value, 1);
    else
        unsetenv(name);
}
#endif

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
    std::string got(path);
    EXPECT_EQ(got, cfg.string());

    fs::remove_all(tmp);
}

TEST(VolkPaths, WriteCreatesHomeConfigWhenXdgMissing)
{
    auto tmp = fs::temp_directory_path() / "volk_test_home";
    fs::remove_all(tmp);
    fs::create_directories(tmp);

    set_env("XDG_CONFIG_HOME", nullptr);
    set_env("VOLK_CONFIGPATH", nullptr);
    set_env("HOME", tmp.string().c_str());

    char path[512] = { 0 };
    // write mode should create ~/.config/volk and return its config path
    volk_get_config_path(path, false);
    std::string got(path);

    fs::path expected = tmp / ".config" / "volk" / "volk_config";
    EXPECT_EQ(got, expected.string());
    EXPECT_TRUE(fs::exists(expected.parent_path()));

    fs::remove_all(tmp);
}

// Ensure VOLK_CONFIGPATH env override takes precedence
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

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
