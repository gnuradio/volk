#!/usr/bin/env python
# Copyright 2019 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# GNU Radio is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# GNU Radio is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GNU Radio; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.


# This script is used to compare the generic kernels to the highest performing kernel, for each operation
# Run:
#   ./volk_profile -j volk_results.json
# Then run this script under python3

import matplotlib.pyplot as plt
import numpy as np
import json

filename = 'volk_results.json'

operations = []
metrics = []
with open(filename) as json_file:
    data = json.load(json_file)
    for test in data['volk_tests']:
        if ('generic' in test['results']) or ('u_generic' in test['results']): # some dont have a generic kernel
            operations.append(test['name'][5:]) # remove volk_ prefix that they all have
            extension_performance = []
            for key, val in test['results'].items():
                if key not in ['generic', 'u_generic']: # exclude generic results, when trying to find fastest time
                    extension_performance.append(val['time'])
            try:
                generic_time = test['results']['generic']['time']
            except:
                generic_time = test['results']['u_generic']['time']
            metrics.append(extension_performance[np.argmin(extension_performance)]/generic_time)

plt.bar(np.arange(len(metrics)), metrics)
plt.hlines(1.0, -1, len(metrics), colors='r', linestyles='dashed')
plt.axis([-1, len(metrics), 0, 2])
plt.xticks(np.arange(len(operations)), operations, rotation=90)
plt.ylabel('Time taken of fastest kernel relative to generic kernel')
plt.show()
