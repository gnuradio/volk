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
#include <io.h>
#define access _access
#define F_OK 0
#else
#include <unistd.h>
#endif
#include <threads.h>
#include <volk/volk_prefs.h>

void volk_get_config_path(char* path, bool read)
{
    if (!path)
        return;
    const char* suffix = "/.volk/volk_config";
    const char* suffix2 = "/volk/volk_config"; // non-hidden
    char* home = NULL;

    // allows config redirection via env variable
    home = getenv("VOLK_CONFIGPATH");
    if (home != NULL) {
        strncpy(path, home, 512);
        strcat(path, suffix2);
        if (!read || access(path, F_OK) != -1) {
            return;
        }
    }

    // check for user-local config file
    home = getenv("HOME");
    if (home != NULL) {
        strncpy(path, home, 512);
        strcat(path, suffix);
        if (!read || (access(path, F_OK) != -1)) {
            return;
        }
    }

    // check for config file in APPDATA (Windows)
    home = getenv("APPDATA");
    if (home != NULL) {
        strncpy(path, home, 512);
        strcat(path, suffix);
        if (!read || (access(path, F_OK) != -1)) {
            return;
        }
    }

    // check for system-wide config file
    if (access("/etc/volk/volk_config", F_OK) != -1) {
        strncpy(path, "/etc", 512);
        strcat(path, suffix2);
        if (!read || (access(path, F_OK) != -1)) {
            return;
        }
    }

    // If still no path was found set path[0] to '0' and fall through
    path[0] = 0;
    return;
}


static struct volk_preferences {
    volk_arch_pref_t* volk_arch_prefs;
    size_t n_arch_prefs;
    int initialized;
    mtx_t mutex;

} volk_preferences;

void init_struct_mutex(void)
{
    if (mtx_init(&volk_preferences.mutex, mtx_plain) != thrd_success) {
        printf("\n mutex init failed\n");
    }
}

static once_flag mutex_init_once_flag = ONCE_FLAG_INIT;
void initialize_mutex() { call_once(&mutex_init_once_flag, init_struct_mutex); }

void volk_initialize_preferences()
{
    initialize_mutex();
    mtx_lock(&volk_preferences.mutex);
    if (!volk_preferences.initialized) {
        volk_preferences.n_arch_prefs =
            volk_load_preferences(&volk_preferences.volk_arch_prefs);
        volk_preferences.initialized = 1;
    }
    mtx_unlock(&volk_preferences.mutex);
}


void volk_free_preferences()
{
    initialize_mutex();
    mtx_lock(&volk_preferences.mutex);
    if (volk_preferences.initialized) {
        free(volk_preferences.volk_arch_prefs);
        volk_preferences.n_arch_prefs = 0;
        volk_preferences.initialized = 0;
    }
    mtx_unlock(&volk_preferences.mutex);
}


const size_t volk_get_num_arch_prefs()
{
    volk_initialize_preferences();
    return volk_preferences.n_arch_prefs;
}


const volk_arch_pref_t* volk_get_arch_prefs()
{
    volk_initialize_preferences();
    return volk_preferences.volk_arch_prefs;
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
