# coding=utf-8

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
import collections
import json
import math
import os

import errno
from interval import interval
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn import cluster as skl
from tabulate import tabulate


MIN_CLUSTER_WIDTH = 3
MAX_CLUSTER_GAP = 6
WINDOW_SIZE = 21


MinMax = collections.namedtuple('MinMax', ('min', 'max'))
Mean = collections.namedtuple('Mean', ('statistic', 'minmax'))
MeanVar = collections.namedtuple('MeanVar', ('statistic', 'var'))
Cluster = collections.namedtuple('Cluster', ['start', 'end'])
ErrorClusterStats = collections.namedtuple(
    'ErrorClusterStats', ['start', 'end', 'duration', 'count'])
AnomalyClusterStats = collections.namedtuple(
    'AnomalyClusterStats',
    ['start', 'end', 'duration', 'count', 'difference'])
RunResult = collections.namedtuple(
    'RunResult', ['errors', 'anomalies', 'etalon', 'plot'])


def mkdir_tree(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def find_clusters(arr, filter_fn, max_gap=MAX_CLUSTER_GAP,
                  min_cluster_width=MIN_CLUSTER_WIDTH):
    # filter_fn: y -> [0, 1]
    clusters = interval()

    start = None
    end = None

    for i, y in enumerate(arr):
        v = filter_fn(y)
        if v:
            if not start:
                start = i
            end = i
        else:
            if end and i - end > max_gap:
                if end - start >= min_cluster_width:
                    clusters |= interval([start, end])
                start = end = None

    if end:
        if end - start >= MIN_CLUSTER_WIDTH:
            clusters |= interval([start, end])

    return clusters


def process_one_run(data):

    table = []
    etalon = []

    results = data['result']
    hooks = data['hooks']

    if not results:
        return  # skip empty

    start = results[0]['timestamp']  # start of the run

    if hooks:
        # when the hook started
        hook_start_time = hooks[0]['started_at'] - start
    else:
        # let all data be etalon
        hook_start_time = results[-1]['timestamp']

    # convert Rally data into our table
    for idx, result in enumerate(results):
        timestamp = result['timestamp'] - start
        duration = result['duration']

        row = {
            'idx': idx,
            'timestamp': timestamp,
            'duration': duration,
            'error': bool(result['error']),
        }

        if timestamp + duration < hook_start_time:
            etalon.append(duration)

        table.append(row)

    etalon = np.array(etalon)
    etalon_mean = np.mean(etalon)
    etalon_median = np.median(etalon)
    etalon_mean_sem = stats.sem(etalon)
    etalon_p95 = np.percentile(etalon, 95)
    etalon_var = np.var(etalon)
    etalon_s = np.std(etalon)

    print('Hook time: %s' % hook_start_time)
    print('There are %s etalon samples' % len(etalon))
    print('Etalon mean: %s (±%s)' % (etalon_mean, etalon_mean_sem))
    print('Etalon median: %s' % etalon_median)
    print('Etalon 95%% percentile: %s' % etalon_p95)
    print('Variance: %s' % etalon_var)
    print('Normal test: %s' % str(stats.normaltest(etalon)))
    print('Bayes: %s' % str(stats.bayes_mvs(etalon, 0.95)))

    # find errors
    error_clusters = find_clusters(
        (p['error'] for p in table),
        filter_fn=lambda x: 1 if x else 0,
        min_cluster_width=0
    )
    print('Error clusters: %s' % str(error_clusters))

    error_stats = []

    for cluster in error_clusters:
        start_idx = int(cluster.inf)
        end_idx = int(cluster.sup)
        d_start = (table[start_idx]['timestamp'] -
                   table[start_idx - 1]['timestamp']) / 2
        d_end = (table[end_idx + 1]['timestamp'] -
                 table[end_idx]['timestamp']) / 2
        start_ts = table[start_idx]['timestamp'] - d_start
        end_ts = table[end_idx]['timestamp'] + d_end
        var = d_start + d_end
        duration = end_ts - start_ts
        print('Error duration %s, variance: %s' % (duration, var))
        count = sum(1 if p['error'] else 0 for p in table)
        print('Count: %s' % count)

        error_stats.append(ErrorClusterStats(
            start=start_ts, end=end_ts, duration=MeanVar(duration, var),
            count=count))

    # process non-error data
    table_filtered = [p for p in table if not p['error']]  # rm errors

    # apply MeanShift clustering algorithm
    x = [p['duration'] for p in table_filtered]
    X = np.array(zip(x, np.zeros(len(x))), dtype=np.float)
    bandwidth = skl.estimate_bandwidth(X, quantile=0.9)
    ms = skl.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    lm = stats.mode(labels)

    for k in range(n_clusters_):
        my_members = labels == k
        print('cluster {0}: {1}'.format(k, X[my_members, 0]))

    vl = []
    for i, p in enumerate(x):
        vl.append(0 if labels[i] == lm.mode else 1)

    anomalies = find_clusters(
        vl,
        filter_fn=lambda y: y
    )

    if not hooks:  # disable anomalies when no hooks specified
        anomalies = []

    # calculate means

    mean_idx = []
    mean_derivative_y = []
    mean_x = []
    mean_y = []

    if len(table_filtered) <= WINDOW_SIZE:
        raise Exception('Not enough data points')

    for i in range(0, len(table_filtered) - WINDOW_SIZE):
        durations = [p['duration'] for p in table_filtered[i: i + WINDOW_SIZE]]

        mean = np.mean(durations)

        idx = table_filtered[i]['idx']  # current index of window start
        mean_idx.append(idx)
        mean_y.append(mean)
        mean_x.append(np.mean(
            [p['timestamp'] for p in table_filtered[i: i + WINDOW_SIZE]]))

        if len(mean_y) > 1:
            # calculate derivative
            mean_prev = mean_y[-2]
            loc = mean - mean_prev
            mean_derivative_y.append(loc)

    etalon_derivative_mean = np.mean(mean_derivative_y[:len(etalon)])
    etalon_derivative_s = np.std(mean_derivative_y[:len(etalon)])

    # find anomalies
    # anomalies = find_clusters(
    #     mean_derivative_y,
    #     filter_fn=lambda y: 0 if abs(y) < abs(etalon_derivative_mean +
    #                                           5 * etalon_derivative_s) else 1
    # )

    anomalies2 = find_clusters(
        mean_y,
        filter_fn=lambda y: 0 if abs(y) < abs(etalon_mean +
                                              5 * etalon_s) else 1
    )

    print('Anomalies: %s' % str(anomalies))
    print('Anomalies2: %s' % str(anomalies2))
    anomaly_stats = []
    filtered2original = dict()

    anomalies = anomalies | anomalies2

    for cluster in anomalies:
        # back to original indexing
        start_idx = table_filtered[int(cluster.inf)]['idx']
        end_idx = table_filtered[int(cluster.sup)]['idx'] -1

        # it means that this item impacted the mean value and caused window
        # to be distinguished
        # start_idx += WINDOW_SIZE - 1

        d_start = (table[start_idx]['timestamp'] -
                   table[start_idx - 1]['timestamp']) / 2
        d_end = (table[end_idx + 1]['timestamp'] -
                 table[end_idx]['timestamp']) / 2
        start_ts = table[start_idx]['timestamp'] - d_start
        end_ts = table[end_idx]['timestamp'] + d_end
        var = d_start + d_end
        duration = end_ts - start_ts
        print('Anomaly duration %s, variance: %s' % (duration, var))

        length = end_idx - start_idx + 1
        print('Anomaly length: %s' % length)

        durations = [p['duration'] for p in table[start_idx: end_idx + 1]]
        anomaly_mean = np.mean(durations)
        anomaly_var = np.var(durations)
        se = math.sqrt(anomaly_var / length + etalon_var / len(etalon))
        dof = len(etalon) + length - 2
        mean_diff = anomaly_mean - etalon_mean
        conf_interval = stats.t.interval(0.95, dof, loc=mean_diff, scale=se)

        difference = MeanVar(mean_diff,
                             np.mean([mean_diff - conf_interval[0],
                                      conf_interval[1] - mean_diff]))

        print('Mean diff: %s' % mean_diff)
        print('Conf int: %s' % str(conf_interval))

        anomaly_stats.append(AnomalyClusterStats(
            start=start_ts, end=end_ts, duration=MeanVar(duration, var),
            difference=difference, count=length
        ))

    # print stats
    print('Error clusters: %s' % error_stats)
    print('Anomaly clusters: %s' % anomaly_stats)

    # draw the plot
    x = [p['timestamp'] for p in table]
    y = [p['duration'] for p in table]

    x2 = [p['timestamp'] for p in table if p['error']]
    y2 = [p['duration'] for p in table if p['error']]

    plt.plot(x, y, 'b.', label='Successful operations')
    plt.plot(x2, y2, 'r.', label='Failed operations')
    plt.ylim(0)

    # highlight etalon
    if hooks:
        plt.axvspan(0, table[len(etalon) - 1]['timestamp'],
                    color='lime', alpha=0.1, label='Etalon area')

    # highlight errors
    for c in error_stats:
        plt.axvspan(c.start, c.end, color='red', alpha=0.4,
                    label='Errors area')

    # highlight anomalies
    for c in anomaly_stats:
        plt.axvspan(c.start, c.end, color='orange', alpha=0.2,
                    label='Anomaly area')

    # draw mean
    plt.plot(mean_x, mean_y, 'cyan', label='Mean duration')

    plt.grid(True)
    plt.xlabel('time, s')
    plt.ylabel('duration, s')

    # add legend
    legend = plt.legend(loc='right', shadow=True)
    for label in legend.get_texts():
        label.set_fontsize('small')

    return RunResult(
        errors=error_stats,
        anomalies=anomaly_stats,
        plot=plt,
        etalon=stats.bayes_mvs(etalon, 0.95),
    )


def round2(number, variance):
    return round(number, int(math.ceil(-(math.log10(variance)))))


def mean_var_to_str(mv):
    precision = int(math.ceil(-(math.log10(mv.var)))) + 1
    if precision > 0:
        pattern = '%%.%df' % precision
        pattern_1 = '%%.%df' % (precision)
    else:
        pattern = pattern_1 = '%d'

    return '%s ~%s' % (pattern % round(mv.statistic, precision),
                       pattern_1 % round(mv.var, precision + 1))


def tabulate2(*args, **kwargs):
    return str(tabulate(*args, **kwargs)).replace('~', '±')


def process(data, book_folder):
    downtime_statistic = []
    downtime_var = []
    ttr_statistic = []
    ttr_var = []
    slowdown_statistic = []
    slowdown_var = []

    for i, one_run in enumerate(data):
        res = process_one_run(one_run)
        print('Res: %s' % str(res))

        res.plot.savefig(os.path.join(book_folder, 'plot_%02d.svg' % i))
        res.plot.show()

        headers = ['#', 'Count', 'Downtime, s']
        t = []
        ds = 0
        for idx, stat in enumerate(res.errors):
            t.append([idx + 1, stat.count, mean_var_to_str(stat.duration)])
            ds += stat.duration.statistic
            downtime_var.append(stat.duration.var)

        if res.errors:
            downtime_statistic.append(ds)

            s = tabulate2(t, headers=headers, tablefmt="grid")
            print(s)

        headers = ['#', 'Count', 'Time to recover, s', 'Operation slowdown, s']
        t = []
        ts = ss = 0
        for idx, stat in enumerate(res.anomalies):
            t.append([idx + 1, stat.count, mean_var_to_str(stat.duration),
                      mean_var_to_str(stat.difference)])
            ts += stat.duration.statistic
            ttr_var.append(stat.duration.var)
            ss += stat.difference.statistic
            slowdown_var.append(stat.difference.var)

        if res.anomalies:
            ttr_statistic.append(ts)
            slowdown_statistic.append(ss)

            s = tabulate2(t, headers=headers, tablefmt="grid")
            print(s)

    downtime = None
    if downtime_statistic:
        downtime = MeanVar(np.mean(downtime_statistic), np.mean(downtime_var))
    mttr = None
    if ttr_statistic:
        mttr = MeanVar(np.mean(ttr_statistic), np.mean(ttr_var))
    slowdown = None
    if slowdown_statistic:
        slowdown = MeanVar(np.mean(slowdown_statistic), np.mean(slowdown_var))

    headers = ['Downtime, s', 'MTTR, s', 'Operation slowdown, s']
    t = [[mean_var_to_str(downtime) if downtime else 'N/A',
          mean_var_to_str(mttr) if mttr else 'N/A',
          mean_var_to_str(slowdown) if slowdown else 'N/A']]
    s = tabulate2(t, headers=headers, tablefmt="grid")
    print(s)


def make_report(file_name, book_folder):
    with open(file_name) as fd:
        data = json.loads(fd.read())

    mkdir_tree(book_folder)
    process(data, book_folder)


def main():
    parser = argparse.ArgumentParser(prog='rally-reliability-report')
    parser.add_argument('-i', '--input', dest='input', required=True,
                        help='Rally raw json output')
    parser.add_argument('-b', '--book', dest='book', required=True,
                        help='folder where to write RST book')
    args = parser.parse_args()
    make_report(args.input, args.book)


if __name__ == '__main__':
    main()
