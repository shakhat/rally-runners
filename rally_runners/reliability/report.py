# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

from __future__ import print_function

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


def process_one_run(data):

    table = []
    points = []
    etalon = []
    hook_start_time = 0
    shock_start_index = 0  # iteration # when the hook started
    shock_end_index = 0  # iteration # when the last error observed

    results = data['result']
    hooks = data['hooks']

    if not results:
        return  # skip empty

    start = results[0]['timestamp']  # start of the run
    hook_start_time = hooks[0]['started_at'] - start  # when the hook started
    hook_end_time = hooks[0]['finished_at'] - start  # when the hook finished

    for idx, result in enumerate(results):
        timestamp = result['timestamp'] - start
        duration = result['duration']

        row = {
            'timestamp': timestamp,
            'duration': duration,
            'error': bool(result['error']),
        }

        if timestamp + duration < hook_start_time:
            etalon.append(duration)
            shock_start_index = idx

        points.append(duration)
        table.append(row)

    etalon = np.array(etalon)
    e_median = np.median(etalon)
    e_p95 = np.percentile(etalon, 95)

    print('Hook time: %s' % hook_start_time)
    print('There are %s etalon samples' % len(etalon))
    print('Etalon median: %s' % e_median)
    print('Etalon 95%% percentile: %s' % e_p95)

    # find errors
    first_error_index = None
    first_error_timestamp = None
    last_error_index = None
    last_error_timestamp = None
    shock_end_index = shock_start_index
    error_count = 0

    for idx, point in enumerate(table):
        if point['error'] and not first_error_timestamp:
            first_error_timestamp = point['timestamp']
            first_error_index = idx
        if point['error']:
            last_error_timestamp = point['timestamp']
            last_error_index = idx
            shock_end_index = idx
            error_count += 1

    error_rate = None
    if error_count:
        error_rate = float(error_count /
                           (last_error_index - first_error_index + 1))

    downtime = None
    has_errors = first_error_timestamp and last_error_timestamp
    if has_errors:
        downtime = last_error_timestamp - first_error_timestamp

    print('Downtime: %s' % downtime)
    print('Error rate during downtime: %s%%' % (error_rate * 100))

    if has_errors:
        post = [p['duration'] for p in table
                if p['timestamp'] > last_error_timestamp]
    else:
        post = [p['duration'] for p in table
                if p['timestamp'] > hook_start_time]

    print('There are %s post samples' % len(post))

    post = np.array(post)

    window_size = 21
    medians_x = []
    medians_y = []

    for i in range(0, len(table) - window_size):
        cur_median = np.median(
            [p['duration'] for p in table[i: i + window_size]])
        cur_95p = np.percentile(
            [p['duration'] for p in table[i: i + window_size]], 95)
        medians_x.append(np.mean(
            [p['timestamp'] for p in table[i: i + window_size]]))
        medians_y.append(cur_median)

    out_liers = np.array([(p['timestamp'], p['duration']) for p in table
                          if p['duration'] > e_p95 * 2])
    print('Outliers: %s' % len(out_liers))
    out_liers_duration = 0
    if len(out_liers):
        out_liers_end = out_liers[-1][0]
        degradation_start = min(first_error_timestamp, out_liers[0][0])
        out_liers_duration = out_liers_end - degradation_start

        ps = [p['duration'] for p in table
              if degradation_start <= p['timestamp'] <= out_liers_end]
        ps_median = np.median(ps)
        degradation = ps_median / e_median
        print('Performance degradation: %s' % degradation)

    print('Outliers duration: %s' % out_liers_duration)

    x = [p['timestamp'] for p in table]
    y = [p['duration'] for p in table]
    # y2 = [1 if p['error'] else 0 for p in table]
    x2 = [p['timestamp'] for p in table if p['error']]
    y2 = [p['duration'] for p in table if p['error']]

    plt.plot(x, y, 'b.', x2, y2, 'r.')
    plt.axvline(hook_start_time, color='orange')
    plt.axvline(hook_end_time, color='orange')
    plt.axvspan(degradation_start, out_liers_end, color='cyan', alpha=0.1)
    plt.axvspan(first_error_timestamp, last_error_timestamp, color='red',
                alpha=0.25)
    # plt.axvline(first_error_timestamp, color='red')
    # plt.axvline(last_error_timestamp, color='red')
    plt.plot(out_liers[:, 0], out_liers[:, 1], 'c.')
    plt.plot(medians_x, medians_y, 'g')
    plt.grid(True)
    plt.xlabel('time, s')
    plt.ylabel('duration, s')
    plt.savefig("test.svg")
    plt.show()


def main():
    parser = argparse.ArgumentParser(prog='rally-reliability-report')
    parser.add_argument('-i', '--input', dest='input', required=True,
                        help='Rally raw json output')
    parser.add_argument('-b', '--book', dest='book',
                        help='folder where to write RST book')
    args = parser.parse_args()

    file_name = args.input
    with open(file_name) as fd:
        data = json.loads(fd.read())

    for one_run in data:
        process_one_run(one_run)
        break


if __name__ == '__main__':
    main()
