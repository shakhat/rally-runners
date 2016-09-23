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
import functools
import json
import math
import os

from interval import interval
import jinja2
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn import cluster as skl
from tabulate import tabulate
import yaml

from rally_runners import utils


MIN_CLUSTER_WIDTH = 3
MAX_CLUSTER_GAP = 6
WINDOW_SIZE = 21

REPORT_TEMPLATE = 'rally_runners/reliability/templates/report.rst'
SCENARIOS_DIR = 'rally_runners/reliability/scenarios/'


MinMax = collections.namedtuple('MinMax', ('min', 'max'))
Mean = collections.namedtuple('Mean', ('statistic', 'minmax'))
MeanVar = collections.namedtuple('MeanVar', ('statistic', 'var'))
Cluster = collections.namedtuple('Cluster', ['start', 'end'])
ClusterStats = collections.namedtuple(
    'ClusterStats', ['start', 'end', 'duration', 'count'])
DegradationClusterStats = collections.namedtuple(
    'DegradationClusterStats',
    ['start', 'end', 'duration', 'count', 'degradation'])
RunResult = collections.namedtuple(
    'RunResult', ['errors', 'anomalies', 'degradation', 'etalon', 'plot'])
SmoothData = collections.namedtuple('SmoothData', ['time', 'duration', 'var'])
DataRow = collections.namedtuple(
    'DataRow', ['index', 'time', 'duration', 'error'])


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


def convert_rally_data(data):
    results = data['result']
    start = results[0]['timestamp']  # start of the run

    hooks = data['hooks']
    hook_index = 0

    if hooks:
        # when the hook started
        hook_start_time = hooks[0]['started_at'] - start
    else:
        # let all data be etalon
        hook_start_time = results[-1]['timestamp']

    table = []
    for index, result in enumerate(results):
        time = result['timestamp'] - start
        duration = result['duration']

        if time + duration < hook_start_time:
            hook_index = index

        table.append(DataRow(index=index, time=time, duration=duration, 
                             error=bool(result['error'])))

    return table, hook_index


def indexed_interval_to_time_interval(table, src_interval):
    start_index = int(src_interval.inf)
    end_index = int(src_interval.sup)

    if start_index > 0:
        d_start = (table[start_index].time - table[start_index - 1].time) / 2
    else:
        d_start = 0

    if end_index < len(table) - 1:
        d_end = (table[end_index + 1].time - table[end_index].time) / 2
    else:
        d_end = 0

    start_time = table[start_index].time - d_start
    end_time = table[end_index].time + d_end
    var = d_start + d_end
    duration = end_time - start_time
    count = sum(1 if start_time <= p.time <= end_time else 0 for p in table)

    return ClusterStats(start=start_time, end=end_time, count=count,
                        duration=MeanVar(duration, var))


def calculate_error_stats(table):
    """Calculates error statistics

    :param table:
    :return: list of time intervals where errors occur
    """
    error_clusters = find_clusters(
        (p.error for p in table),
        filter_fn=lambda x: 1 if x else 0,
        min_cluster_width=0
    )
    error_stats = [indexed_interval_to_time_interval(table, cluster)
                   for cluster in error_clusters]
    return error_stats


def calculate_anomalies_stats(table, quantile=0.9):
    """Find anomalies

    :param quantile: float, default 0.3
    :param table:
    :return: list of time intervals where anomalies occur
    """
    table = [p for p in table if not p.error]  # rm errors
    x = [p.duration for p in table]
    X = np.array(zip(x, np.zeros(len(x))), dtype=np.float)
    bandwidth = skl.estimate_bandwidth(X, quantile=quantile)
    mean_shift_algo = skl.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    mean_shift_algo.fit(X)
    labels = mean_shift_algo.labels_
    lm = stats.mode(labels)

    # filter out the largest cluster
    vl = [(0 if labels[i] == lm.mode else 1) for i, p in enumerate(x)]

    anomaly_clusters = find_clusters(vl, filter_fn=lambda y: y)
    anomaly_stats = [indexed_interval_to_time_interval(table, cluster)
                     for cluster in anomaly_clusters]
    return anomaly_stats


def smooth_data(table, window_size):
    """Calculate mean for the data

    :param table:
    :param window_size:
    :return: list of points in mean data
    """
    table = [p for p in table if not p.error]  # rm errors
    smooth = []

    for i in range(0, len(table) - window_size):
        durations = [p.duration for p in table[i: i + window_size]]

        time = np.mean([p.time for p in table[i: i + window_size]])
        duration = np.mean(durations)
        var = abs(
            time - np.mean([p.time for p in table[i + 1: i + window_size - 1]]))

        smooth.append(SmoothData(time=time, duration=duration, var=var))

    return smooth


def calculate_degradation_cluster_stats(table, smooth, etalon):
    table = [p for p in table if not p.error]  # rm errors
    if len(table) <= WINDOW_SIZE:
        return []

    etalon_mean = np.mean(etalon)
    etalon_var = np.var(etalon)
    etalon_s = np.std(etalon)
    threshold = etalon_mean + 5 * etalon_s

    mean_times = [p.time for p in smooth]
    mean_durations = [p.duration for p in smooth]
    mean_vars = [p.var for p in smooth]

    clusters = find_clusters(
        mean_durations, filter_fn=lambda y: 0 if abs(y) < threshold else 1)

    # calculate cluster duration
    cluster_stats = []
    for cluster in clusters:
        start_idx = int(cluster.inf)
        end_idx = int(cluster.sup)
        start_time = mean_times[start_idx]
        end_time = mean_times[end_idx]
        duration = end_time - start_time
        count = sum(1 if start_time < p.time < end_time else 0 for p in table)
        var = np.mean(mean_vars[start_idx: end_idx])

        cluster_stats.append(
            ClusterStats(start=start_time, end=end_time, count=count,
                         duration=MeanVar(duration, var)))

    # calculate degradation for every cluster
    degradation_cluster_stats = []
    for cluster in cluster_stats:
        start_time = cluster.start
        end_time = cluster.end

        # durations
        durations = []
        for p in table:
            if start_time < p.time < end_time:
                durations.append(p.duration)

        anomaly_mean = np.mean(durations)
        anomaly_var = np.var(durations)
        se = math.sqrt(anomaly_var / len(durations) + etalon_var / len(etalon))
        dof = len(etalon) + len(durations) - 2
        mean_diff = anomaly_mean - etalon_mean
        conf_interval = stats.t.interval(0.95, dof, loc=mean_diff, scale=se)

        degradation = MeanVar(
            mean_diff, np.mean([mean_diff - conf_interval[0],
                                conf_interval[1] - mean_diff]))

        print('Mean diff: %s' % mean_diff)
        print('Conf int: %s' % str(conf_interval))

        degradation_cluster_stats.append(DegradationClusterStats(
            start=start_time, end=end_time, duration=cluster.duration,
            degradation=degradation, count=cluster.count
        ))

    return degradation_cluster_stats


def process_one_run(data):

    table, hook_index = convert_rally_data(data)
    etalon = [p.duration for p in table[0:hook_index]]

    etalon = np.array(etalon)
    etalon_mean = np.mean(etalon)
    etalon_median = np.median(etalon)
    etalon_mean_sem = stats.sem(etalon)
    etalon_p95 = np.percentile(etalon, 95)
    etalon_var = np.var(etalon)
    etalon_s = np.std(etalon)

    print('Hook index: %s' % hook_index)
    print('There are %s etalon samples' % len(etalon))
    print('Etalon mean: %s (±%s)' % (etalon_mean, etalon_mean_sem))
    print('Etalon median: %s' % etalon_median)
    print('Etalon 95%% percentile: %s' % etalon_p95)
    print('Variance: %s' % etalon_var)
    print('Normal test: %s' % str(stats.normaltest(etalon)))
    print('Bayes: %s' % str(stats.bayes_mvs(etalon, 0.95)))

    # Calculate stats
    error_stats = calculate_error_stats(table)

    anomaly_stats = calculate_anomalies_stats(table)

    smooth = smooth_data(table, window_size=WINDOW_SIZE)
    degradation_cluster_stats = calculate_degradation_cluster_stats(
        table, smooth, etalon)

    # print stats
    print('Error clusters: %s' % error_stats)
    print('Anomaly clusters: %s' % anomaly_stats)
    print('Degradation cluster stats: %s' % degradation_cluster_stats)

    # draw the plot
    x = [p.time for p in table]
    y = [p.duration for p in table]

    x2 = [p.time for p in table if p.error]
    y2 = [p.duration for p in table if p.error]

    figure = plt.figure()
    plot = figure.add_subplot(111)
    plot.plot(x, y, 'b.', label='Successful operations')
    plot.plot(x2, y2, 'r.', label='Failed operations')
    plot.set_ylim(0)

    plot.axhline(abs(etalon_mean + 5 * etalon_s), color='violet')

    # highlight etalon
    if len(table) > hook_index:
        plot.axvspan(0, table[len(etalon) - 1].time,
                    color='#b0efa0', label='Etalon area')

    # highlight anomalies
    for c in anomaly_stats:
        plot.axvspan(c.start, c.end, color='#f0f0f0', label='Anomaly area')

    # highlight anomalies
    for c in anomaly_stats:
        plot.axvspan(c.start, c.end, color='#f8efa8', label='Anomaly area')

    # highlight errors
    for c in error_stats:
        plot.axvspan(c.start, c.end, color='#ffc0a7', label='Errors area')

    # draw mean
    plot.plot([p.time for p in smooth], [p.duration for p in smooth], 'cyan',
              label='Mean duration')

    plot.grid(True)
    plot.set_xlabel('time, s')
    plot.set_ylabel('duration, s')

    # add legend
    legend = plot.legend(loc='right', shadow=True)
    for label in legend.get_texts():
        label.set_fontsize('small')

    return RunResult(
        errors=error_stats,
        anomalies=anomaly_stats,
        degradation=degradation_cluster_stats,
        plot=figure,
        etalon=stats.bayes_mvs(etalon, 0.95),
    )


def round2(number, variance):
    return round(number, int(math.ceil(-(math.log10(variance)))))


def mean_var_to_str(mv):
    if mv.var == 0:
        precision = 4
    else:
        precision = int(math.ceil(-(math.log10(mv.var)))) + 1
    if precision > 0:
        pattern = '%%.%df' % precision
        pattern_1 = '%%.%df' % (precision)
    else:
        pattern = pattern_1 = '%d'

    return '%s ~%s' % (pattern % round(mv.statistic, precision),
                       pattern_1 % round(mv.var, precision + 1))


def tabulate2(*args, **kwargs):
    return (u'%s' % tabulate(*args, **kwargs)).replace('~', u'±')


def process(data, book_folder, scenario):
    scenario_text = '\n'.join('    %s' % line for line in scenario.split('\n'))
    report = dict(runs=[], scenario=scenario_text)
    downtime_statistic = []
    downtime_var = []
    ttr_statistic = []
    ttr_var = []
    slowdown_statistic = []
    slowdown_var = []

    for i, one_run in enumerate(data):
        report_one_run = {}
        res = process_one_run(one_run)
        print('Res: %s' % str(res))

        res.plot.savefig(os.path.join(book_folder, 'plot_%d.svg' % (i + 1)))
        # res.plot.show()

        headers = ['#', 'Count', 'Downtime, s']
        t = []
        ds = 0
        for index, stat in enumerate(res.errors):
            t.append([index + 1, stat.count, mean_var_to_str(stat.duration)])
            ds += stat.duration.statistic
            downtime_var.append(stat.duration.var)

        if res.errors:
            downtime_statistic.append(ds)

            s = tabulate2(t, headers=headers, tablefmt="grid")
            report_one_run['errors_table'] = s
            print(s)

        headers = ['#', 'Count', 'Time to recover, s', 'Operation slowdown, s']
        t = []
        ts = ss = 0
        for index, stat in enumerate(res.degradation):
            t.append([index + 1, stat.count, mean_var_to_str(stat.duration),
                      mean_var_to_str(stat.degradation)])
            ts += stat.duration.statistic
            ttr_var.append(stat.duration.var)
            ss += stat.degradation.statistic
            slowdown_var.append(stat.degradation.var)

        if res.degradation:
            ttr_statistic.append(ts)
            slowdown_statistic.append(ss)

            s = tabulate2(t, headers=headers, tablefmt="grid")
            report_one_run['anomalies_table'] = s
            print(s)

        report['runs'].append(report_one_run)

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
    report['summary_table'] = s
    print(s)

    jinja_env = jinja2.Environment()
    jinja_env.filters['json'] = json.dumps
    jinja_env.filters['yaml'] = functools.partial(
        yaml.safe_dump, indent=2, default_flow_style=False)

    path = utils.resolve_relative_path(REPORT_TEMPLATE)
    with open(path) as fd:
        template = fd.read()
        compiled_template = jinja_env.from_string(template)
        rendered_template = compiled_template.render(dict(report=report))

        index_path = os.path.join(book_folder, 'index.rst')
        with open(index_path, 'w') as fd2:
            fd2.write(rendered_template.encode('utf8'))


def make_report(scenario, file_name, book_folder):
    scenario_dir = utils.resolve_relative_path(SCENARIOS_DIR)
    scenario_path = os.path.join(scenario_dir, scenario)
    if not scenario_path.endswith('.yaml'):
        scenario_path += '.yaml'

    scenario = ''
    with open(scenario_path) as fd:
        scenario = fd.read()

    with open(file_name) as fd:
        data = json.loads(fd.read())

    utils.mkdir_tree(book_folder)
    process(data, book_folder, scenario)


def main():
    parser = argparse.ArgumentParser(prog='rally-reliability-report')
    parser.add_argument('-i', '--input', dest='input', required=True,
                        help='Rally raw json output')
    parser.add_argument('-b', '--book', dest='book', required=True,
                        help='folder where to write RST book')
    parser.add_argument('-s', '--scenario', dest='scenario', required=True,
                        help='Rally scenario')
    args = parser.parse_args()
    make_report(args.scenario, args.input, args.book)


if __name__ == '__main__':
    main()
