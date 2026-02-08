// Tests for volk config path resolution (XDG preference, legacy fallback)
#include <gtest/gtest.h>
#include <volk/volk_prefs.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <cstdlib>
#include <random>
#include <sstream>

#if defined(_WIN32)
static void set_env(std::string_view name, const char* value)
{
    std::string n(name);
    if (value)
        _putenv_s(n.c_str(), value);
    else
        _putenv_s(n.c_str(), "");
}
#else
static void set_env(std::string_view name, const char* value)
{
    std::string n(name);
    if (value)
        setenv(n.c_str(), value, 1);
    else
        unsetenv(n.c_str());
}
#endif

static std::string volk_config_path_read()
{
    char buf[512] = {0};
    volk_get_config_path(buf, true);
    return std::string(buf);
}

static std::string volk_config_path_write()
{
    char buf[512] = {0};
    volk_get_config_path(buf, false);
    return std::string(buf);
}

// Helper to generate unique temp directory names
static std::filesystem::path unique_temp_path(const char* prefix)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(100000, 999999);
    std::ostringstream oss;
    oss << prefix << "_" << distrib(gen);
    return std::filesystem::temp_directory_path() / oss.str();
}

TEST(VolkPaths, OverrideEnv)
{
    namespace fs = std::filesystem;
    fs::path tmp = unique_temp_path("volk_test_override");
    fs::create_directories(tmp / "volk");
    std::ofstream(tmp / "volk" / "volk_config") << "test";

    set_env("VOLK_CONFIGPATH", tmp.string().c_str());

    std::string expect = (tmp / "volk" / "volk_config").generic_string();
    std::string got = volk_config_path_read();
    EXPECT_EQ(expect, got);

    // cleanup
    set_env("VOLK_CONFIGPATH", nullptr);
    fs::remove_all(tmp);
}

TEST(VolkPaths, XDGPreference)
{
    namespace fs = std::filesystem;
    fs::path tmp = unique_temp_path("volk_test_xdg");
    fs::create_directories(tmp / "volk");
    std::ofstream(tmp / "volk" / "volk_config") << "test";

    set_env("VOLK_CONFIGPATH", nullptr);
    set_env("XDG_CONFIG_HOME", tmp.string().c_str());

    std::string expect = (tmp / "volk" / "volk_config").generic_string();
    std::string got = volk_config_path_read();
    EXPECT_EQ(expect, got);

    set_env("XDG_CONFIG_HOME", nullptr);
    fs::remove_all(tmp);
}

TEST(VolkPaths, HomeDotConfigFallback)
{
    namespace fs = std::filesystem;
    fs::path tmp = unique_temp_path("volk_test_homecfg");
    fs::create_directories(tmp / ".config" / "volk");
    std::ofstream(tmp / ".config" / "volk" / "volk_config") << "test";

    set_env("VOLK_CONFIGPATH", nullptr);
    set_env("XDG_CONFIG_HOME", nullptr);
    set_env("HOME", tmp.string().c_str());

    std::string expect = (tmp / ".config" / "volk" / "volk_config").generic_string();
    std::string got = volk_config_path_read();
    EXPECT_EQ(expect, got);

    set_env("HOME", nullptr);
    fs::remove_all(tmp);
}

TEST(VolkPaths, LegacyFallback)
{
    namespace fs = std::filesystem;
    fs::path tmp = unique_temp_path("volk_test_legacy");
    fs::create_directories(tmp / ".volk");
    std::ofstream(tmp / ".volk" / "volk_config") << "test";

    set_env("VOLK_CONFIGPATH", nullptr);
    set_env("XDG_CONFIG_HOME", nullptr);
    set_env("HOME", tmp.string().c_str());

    // Ensure .config/volk does not exist to force legacy path selection
    fs::remove_all(tmp / ".config");

    std::string expect = (tmp / ".volk" / "volk_config").generic_string();
    std::string got = volk_config_path_read();
    EXPECT_EQ(expect, got);

    set_env("HOME", nullptr);
    fs::remove_all(tmp);
}

TEST(VolkPaths, WriteCreatesXDGDir)
{
    namespace fs = std::filesystem;
    fs::path tmp = unique_temp_path("volk_test_write");

    set_env("VOLK_CONFIGPATH", nullptr);
    set_env("XDG_CONFIG_HOME", tmp.string().c_str());

    // Ensure directory does not exist yet
    fs::remove_all(tmp);
    EXPECT_FALSE(fs::exists(tmp / "volk"));

    // Call write-mode; this should create the XDG/volk directory
    volk_config_path_write();
    EXPECT_TRUE(fs::exists(tmp / "volk"));

    set_env("XDG_CONFIG_HOME", nullptr);
    fs::remove_all(tmp);
}
