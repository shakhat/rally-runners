from __future__ import print_function

import json

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def main():
    file_name = 'data.json'
    with open(file_name) as fd:
        data = json.loads(fd.read())

    table = []
    points = []
    etalon = []
    hook_time = 0
    shock_start_index = 0
    shock_end_index = 0

    for one in data:
        results = one['result']
        hooks = one['hooks']

        hook_time = hooks[0]['started_at']

        for idx, result in enumerate(results):
            start = result['timestamp']
            duration = result['duration']

            row = {
                'timestamp': start,
                'duration': duration,
                'error': bool(result['error']),
            }

            if start + duration < hook_time:
                etalon.append(duration)
                shock_start_index = idx

            points.append(duration)
            table.append(row)

    etalon = np.array(etalon)
    e_median = np.median(etalon)
    e_p95 = np.percentile(etalon, 95)

    print('There are %s etalon samples' % len(etalon))
    print('Etalon median: %s' % e_median)
    print('Etalon 95%% percentile: %s' % e_p95)

    start = min(p['timestamp'] for p in table)
    end = max(p['timestamp'] for p in table)

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
        error_rate = float(error_count / (last_error_index - first_error_index + 1))

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
                if p['timestamp'] > hook_time]

    print('There are %s post samples' % len(post))

    post = np.array(post)

    window_size = 21
    medians_x = []
    medians_y = []

    for i in range(shock_end_index, len(table) - window_size):
        cur_median = np.median(
            [p['duration'] for p in table[i: i + window_size]])
        cur_95p = np.percentile(
            [p['duration'] for p in table[i: i + window_size]], 95)
        medians_x.append(np.mean(
            [p['timestamp'] - start for p in table[i: i + window_size]]))
        medians_y.append(cur_median)

    for i in range(0, len(post), window_size):
        chunk = post[i: i+window_size]
        print(stats.ks_2samp(etalon, chunk))

    x = [p['timestamp'] - start for p in table]
    y = [p['duration'] for p in table]
    # y2 = [1 if p['error'] else 0 for p in table]
    x2 = [p['timestamp'] - start for p in table if p['error']]
    y2 = [p['duration'] for p in table if p['error']]
    plt.plot(x, y, 'b', x2, y2, 'r', medians_x, medians_y, 'g')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()