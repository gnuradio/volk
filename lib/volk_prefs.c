/* -*- c++ -*- */
/*
 * Copyright 2011, 2012, 2015, 2016, 2019, 2020 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_MSC_VER)
#include <direct.h>
#include <io.h>
#define access _access
#define F_OK 0
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif
#include <volk/volk_prefs.h>

void volk_get_config_path(char* path, bool read)
{
    if (!path)
        return;
    const char* legacy_suffix = "/.volk/volk_config";
    const char* nonhidden_suffix = "/volk/volk_config"; // non-hidden
    char tmp[512] = { 0 };

    /* helper to ensure directory exists for write-mode */
    {
#if defined(_MSC_VER)
        /* use _mkdir on MSVC */
#else
        /* ensure sys/stat available */
#endif
    }

    /* 1) explicit override via VOLK_CONFIGPATH */
    const char* env_override = getenv("VOLK_CONFIGPATH");
    if (env_override) {
        snprintf(tmp, sizeof(tmp), "%s%s", env_override, nonhidden_suffix);
        if (!read || access(tmp, F_OK) == 0) {
            strncpy(path, tmp, 512);
            return;
        }
    }

    /* 2) XDG_CONFIG_HOME/volk if XDG set */
    const char* xdg = getenv("XDG_CONFIG_HOME");
    if (xdg) {
        snprintf(tmp, sizeof(tmp), "%s/volk/volk_config", xdg);
        if (!read) {
            char dir[512];
            snprintf(dir, sizeof(dir), "%s/volk", xdg);
#if defined(_MSC_VER)
            _mkdir(dir);
#else
            struct stat st = { 0 };
            if (stat(dir, &st) == -1) {
                mkdir(dir, 0755);
            }
#endif
        }
        if (!read || access(tmp, F_OK) == 0) {
            strncpy(path, tmp, 512);
            return;
        }
    }

    /* 3) $HOME/.config/volk */
    const char* home = getenv("HOME");
    if (home) {
        snprintf(tmp, sizeof(tmp), "%s/.config/volk/volk_config", home);
        if (!read) {
            char dir[512];
            snprintf(dir, sizeof(dir), "%s/.config/volk", home);
#if defined(_MSC_VER)
            _mkdir(dir);
#else
            struct stat st = { 0 };
            if (stat(dir, &st) == -1) {
                mkdir(dir, 0755);
            }
#endif
        }
        if (!read || access(tmp, F_OK) == 0) {
            strncpy(path, tmp, 512);
            return;
        }
    }

    /* 4) legacy $HOME/.volk */
    if (home) {
        snprintf(tmp, sizeof(tmp), "%s%s", home, legacy_suffix);
        if (!read || access(tmp, F_OK) == 0) {
            strncpy(path, tmp, 512);
            return;
        }
    }

    /* 5) Windows APPDATA fallback */
    const char* appdata = getenv("APPDATA");
    if (appdata) {
        snprintf(tmp, sizeof(tmp), "%s%s", appdata, legacy_suffix);
        if (!read || access(tmp, F_OK) == 0) {
            strncpy(path, tmp, 512);
            return;
        }
    }

    /* System-wide */
    if (access("/etc/volk/volk_config", F_OK) == 0) {
        strncpy(path, "/etc/volk/volk_config", 512);
        return;
    }
    /* If nothing found, follow the XDG-first fallback behavior:
       - if XDG_CONFIG_HOME is set, return XDG_CONFIG_HOME/volk/volk_config
       - else if HOME is set, return HOME/.config/volk/volk_config
       - otherwise return empty string
       This ensures read-mode returns the new XDG location as fallback. */
    if (xdg) {
        snprintf(tmp, sizeof(tmp), "%s/volk/volk_config", xdg);
        strncpy(path, tmp, 512);
        return;
    }
    if (home) {
        snprintf(tmp, sizeof(tmp), "%s/.config/volk/volk_config", home);
        strncpy(path, tmp, 512);
        return;
    }

    path[0] = '\0';
    return;
}

size_t volk_load_preferences(volk_arch_pref_t** prefs_res)
{
    FILE* config_file;
    char path[512], line[512];
    size_t n_arch_prefs = 0;
    volk_arch_pref_t* prefs = NULL;

    // get the config path
    volk_get_config_path(path, true);
    if (!path[0])
        return n_arch_prefs; // no prefs found
    config_file = fopen(path, "r");
    if (!config_file)
        return n_arch_prefs; // no prefs found

    // reset the file pointer and write the prefs into volk_arch_prefs
    while (fgets(line, sizeof(line), config_file) != NULL) {
        void* new_prefs = realloc(prefs, (n_arch_prefs + 1) * sizeof(*prefs));
        if (!new_prefs) {
            printf("volk_load_preferences: bad malloc\n");
            break;
        }
        prefs = (volk_arch_pref_t*)new_prefs;
        volk_arch_pref_t* p = prefs + n_arch_prefs;
        if (sscanf(line, "%s %s %s", p->name, p->impl_a, p->impl_u) == 3 &&
            !strncmp(p->name, "volk_", 5)) {
            n_arch_prefs++;
        }
    }
    fclose(config_file);
    *prefs_res = prefs;
    return n_arch_prefs;
}
