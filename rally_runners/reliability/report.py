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

import argparse
import collections
import functools
import json
import logging
import math
import os

import matplotlib as mpl
mpl.use('Agg')  # do not require X server

from interval import interval
import jinja2
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn import cluster as skl
from tabulate import tabulate
import yaml

from rally_runners import utils


MIN_CLUSTER_WIDTH = 3  # filter cluster with less items
MAX_CLUSTER_GAP = 6  # max allowed gap in the cluster (otherwise split them)
WINDOW_SIZE = 21  # window size for average duration calculation
WARM_UP_CUTOFF = 10  # drop first N points from etalon
DEGRADATION_THRESHOLD = 4  # how many sigmas duration differs from etalon mean

REPORT_TEMPLATE = 'rally_runners/reliability/templates/report.rst'
SCENARIOS_DIR = 'rally_runners/reliability/scenarios/'


MinMax = collections.namedtuple('MinMax', ('min', 'max'))
Mean = collections.namedtuple('Mean', ('statistic', 'minmax'))
MeanVar = collections.namedtuple('MeanVar', ('statistic', 'var'))
ArrayStats = collections.namedtuple(
    'ArrayStats', ['mean', 'median', 'p95', 'var', 'std', 'count'])
ClusterStats = collections.namedtuple(
    'ClusterStats', ['start', 'end', 'duration', 'count'])
DegradationClusterStats = collections.namedtuple(
    'DegradationClusterStats',
    ['start', 'end', 'duration', 'count', 'degradation', 'degradation_ratio'])
RunResult = collections.namedtuple(
    'RunResult', ['data', 'error_area', 'anomaly_area', 'degradation_area',
                  'etalon_stats', 'etalon_interval', 'etalon_threshold',
                  'smooth_data'])
SummaryResult = collections.namedtuple(
    'SummaryResult', ['run_results', 'mttr', 'degradation',
                      'degradation_ratio', 'downtime'])
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


def calculate_array_stats(data):
    data = np.array(data)
    return ArrayStats(mean=np.mean(data), median=np.median(data),
                      p95=np.percentile(data, 95), var=np.var(data),
                      std=np.std(data), count=len(data))


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


def calculate_error_area(table):
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


def calculate_anomaly_area(table, quantile=0.9):
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


def calculate_smooth_data(table, window_size):
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
        var = abs(time - np.mean(
            [p.time for p in table[i + 1: i + window_size - 1]]))

        smooth.append(SmoothData(time=time, duration=duration, var=var))

    return smooth


def calculate_degradation_area(table, smooth, etalon_stats,
                               etalon_threshold):
    table = [p for p in table if not p.error]  # rm errors
    if len(table) <= WINDOW_SIZE:
        return []

    mean_times = [p.time for p in smooth]
    mean_durations = [p.duration for p in smooth]
    mean_vars = [p.var for p in smooth]

    clusters = find_clusters(
        mean_durations,
        filter_fn=lambda y: 0 if abs(y) < etalon_threshold else 1)

    # calculate cluster duration
    degradation_cluster_stats = []
    for cluster in clusters:
        start_idx = int(cluster.inf)
        end_idx = int(cluster.sup)
        start_time = mean_times[start_idx]
        end_time = mean_times[end_idx]
        duration = end_time - start_time
        var = np.mean(mean_vars[start_idx: end_idx])

        # point durations
        point_durations = []
        for p in table:
            if start_time < p.time < end_time:
                point_durations.append(p.duration)

        # calculate difference between means
        # http://onlinestatbook.com/2/tests_of_means/difference_means.html
        anomaly_mean = np.mean(point_durations)
        anomaly_var = np.var(point_durations)
        se = math.sqrt(anomaly_var / len(point_durations) +
                       etalon_stats.var / etalon_stats.count)
        dof = etalon_stats.count + len(point_durations) - 2
        mean_diff = anomaly_mean - etalon_stats.mean
        conf_interval = stats.t.interval(0.95, dof, loc=mean_diff, scale=se)

        degradation = MeanVar(
            mean_diff, np.mean([mean_diff - conf_interval[0],
                                conf_interval[1] - mean_diff]))
        degradation_ratio = MeanVar(
            anomaly_mean / etalon_stats.mean,
            np.mean([(mean_diff - conf_interval[0]) / etalon_stats.mean,
                     (conf_interval[1] - mean_diff) / etalon_stats.mean]))

        logging.debug('Mean diff: %s' % mean_diff)
        logging.debug('Conf int: %s' % str(conf_interval))

        degradation_cluster_stats.append(DegradationClusterStats(
            start=start_time, end=end_time, duration=MeanVar(duration, var),
            degradation=degradation, degradation_ratio=degradation_ratio,
            count=len(point_durations)
        ))

    return degradation_cluster_stats


def draw_area(plot, area, color, label):
    for i, c in enumerate(area):
        plot.axvspan(c.start, c.end, color=color, label=label)
        label = None  # show label only once


def draw_plot(run_result, show_etalon=True, show_errors=True,
              show_anomalies=False, show_degradation=True):
    table = run_result.data
    x = [p.time for p in table]
    y = [p.duration for p in table]

    x2 = [p.time for p in table if p.error]
    y2 = [p.duration for p in table if p.error]

    figure = plt.figure()
    plot = figure.add_subplot(111)
    plot.plot(x, y, 'b.', label='Successful operations')
    plot.plot(x2, y2, 'r.', label='Failed operations')
    plot.set_ylim(0)

    plot.axhline(run_result.etalon_threshold, color='violet',
                 label='Degradation threshold')

    # highlight etalon
    if show_etalon:
        plot.axvspan(run_result.etalon_interval.inf,
                     run_result.etalon_interval.sup,
                     color='#b0efa0', label='Baseline')

    # highlight anomalies
    if show_anomalies:
        draw_area(plot, run_result.anomaly_area,
                  color='#f0f0f0', label='Anomaly')

    # highlight degradation
    if show_degradation:
        draw_area(plot, run_result.degradation_area,
                  color='#f8efa8', label='Degradation')

    # highlight errors
    if show_errors:
        draw_area(plot, run_result.error_area,
                  color='#ffc0a7', label='Downtime')

    # draw mean
    plot.plot([p.time for p in run_result.smooth_data],
              [p.duration for p in run_result.smooth_data],
              color='cyan', label='Mean duration')

    plot.grid(True)
    plot.set_xlabel('time, s')
    plot.set_ylabel('operation duration, s')

    # add legend
    legend = plot.legend(loc='right', shadow=True)
    for label in legend.get_texts():
        label.set_fontsize('small')

    return figure


def process_one_run(rally_data):
    data, hook_index = convert_rally_data(rally_data)
    etalon = [p.duration for p in data[WARM_UP_CUTOFF:hook_index]]

    etalon_stats = calculate_array_stats(etalon)
    etalon_threshold = abs(etalon_stats.mean +
                           DEGRADATION_THRESHOLD * etalon_stats.std)
    etalon_interval = interval([data[WARM_UP_CUTOFF].time,
                                data[hook_index].time])[0]

    logging.debug('Hook index: %s' % hook_index)
    logging.debug('Etalon stats: %s' % str(etalon_stats))

    # Calculate stats
    error_area = calculate_error_area(data)

    anomaly_area = calculate_anomaly_area(data)

    smooth_data = calculate_smooth_data(data, window_size=WINDOW_SIZE)

    degradation_area = calculate_degradation_area(
        data, smooth_data, etalon_stats, etalon_threshold)

    # logging.debug stats
    logging.debug('Error area: %s' % error_area)
    logging.debug('Anomaly area: %s' % anomaly_area)
    logging.debug('Degradation area: %s' % degradation_area)

    return RunResult(
        data=data,
        error_area=error_area,
        anomaly_area=anomaly_area,
        degradation_area=degradation_area,
        etalon_stats=etalon_stats,
        etalon_interval=etalon_interval,
        etalon_threshold=etalon_threshold,
        smooth_data=smooth_data,
    )


def process_all_runs(runs):

    run_results = []
    downtime_statistic = []
    downtime_var = []
    ttr_statistic = []
    ttr_var = []
    degradation_statistic = []
    degradation_var = []
    degradation_ratio_statistic = []
    degradation_ratio_var = []

    for i, one_run in enumerate(runs):
        run_result = process_one_run(one_run)
        run_results.append(run_result)

        ds = 0
        for index, stat in enumerate(run_result.error_area):
            ds += stat.duration.statistic
            downtime_var.append(stat.duration.var)

        if run_result.error_area:
            downtime_statistic.append(ds)

        ts = ss = sr = 0
        for index, stat in enumerate(run_result.degradation_area):
            ts += stat.duration.statistic
            ttr_var.append(stat.duration.var)
            ss += stat.degradation.statistic
            degradation_var.append(stat.degradation.var)
            sr += stat.degradation_ratio.statistic
            degradation_ratio_var.append(stat.degradation_ratio.var)

        if run_result.degradation_area:
            ttr_statistic.append(ts)
            degradation_statistic.append(ss)
            degradation_ratio_statistic.append(sr)

    downtime = None
    if downtime_statistic:
        downtime_mean = np.mean(downtime_statistic)
        se = math.sqrt((sum(downtime_var) +
                       np.var(downtime_statistic)) / len(downtime_statistic))
        downtime = MeanVar(downtime_mean, se)
    mttr = None
    if ttr_statistic:
        ttr_mean = np.mean(ttr_statistic)
        se = math.sqrt((sum(ttr_var) +
                        np.var(ttr_statistic)) / len(ttr_statistic))
        mttr = MeanVar(ttr_mean, se)
    degradation = None
    degradation_ratio = None
    if degradation_statistic:
        degradation = MeanVar(np.mean(degradation_statistic),
                              np.mean(degradation_var))
        degradation_ratio = MeanVar(np.mean(degradation_ratio_statistic),
                                    np.mean(degradation_ratio_var))

    return SummaryResult(run_results=run_results, mttr=mttr,
                         degradation=degradation,
                         degradation_ratio=degradation_ratio,
                         downtime=downtime)


def round2(number, variance=None):
    if not variance:
        variance = number
    return round(number, int(math.ceil(-(math.log10(variance)))) + 1)


def mean_var_to_str(mv):
    if not mv:
        return 'N/A'

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
    return (u'%s' % tabulate(*args, **kwargs)).replace(' ~', u'\u00A0±')


def get_runs(raw_rally_reports):
    for one_report in raw_rally_reports:
        for one_run in one_report:
            yield one_run


def indent(text, distance):
    return '\n'.join((' ' * distance + line) for line in text.split('\n'))


def process(raw_rally_reports, book_folder, scenario, scenario_name):
    scenario_text = indent(scenario, 4)
    report = dict(runs=[], scenario=scenario_text, scenario_name=scenario_name)

    summary = process_all_runs(get_runs(raw_rally_reports))
    logging.debug('Summary: %s', summary)

    for i, one_run in enumerate(summary.run_results):
        report_one_run = {}

        plot = draw_plot(one_run)
        plot.savefig(os.path.join(book_folder, 'plot_%d.svg' % (i + 1)))

        headers = ['Samples', 'Median, s', 'Mean, s', 'Std dev',
                   '95% percentile, s']
        t = [[one_run.etalon_stats.count,
              round2(one_run.etalon_stats.median),
              round2(one_run.etalon_stats.mean),
              round2(one_run.etalon_stats.std),
              round2(one_run.etalon_stats.p95)]]
        report_one_run['etalon_table'] = tabulate2(
            t, headers=headers, tablefmt='grid')

        headers = ['#', 'Downtime, s']
        t = []
        for index, stat in enumerate(one_run.error_area):
            t.append([index + 1, mean_var_to_str(stat.duration)])

        if one_run.error_area:
            report_one_run['errors_table'] = tabulate2(
                t, headers=headers, tablefmt='grid')

        headers = ['#', 'Time to recover, s', 'Absolute degradation, s',
                   'Relative degradation']
        t = []
        for index, stat in enumerate(one_run.degradation_area):
            t.append([index + 1,
                      mean_var_to_str(stat.duration),
                      mean_var_to_str(stat.degradation),
                      mean_var_to_str(stat.degradation_ratio)])

        if one_run.degradation_area:
            report_one_run['degradation_table'] = tabulate2(
                t, headers=headers, tablefmt="grid")

        report['runs'].append(report_one_run)

    headers = ['Service downtime, s', 'MTTR, s',
               'Absolute performance degradation, s',
               'Relative performance degradation, ratio']
    t = [[mean_var_to_str(summary.downtime),
          mean_var_to_str(summary.mttr),
          mean_var_to_str(summary.degradation),
          mean_var_to_str(summary.degradation_ratio)]]
    report['summary_table'] = tabulate2(t, headers=headers, tablefmt='grid')

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

    logging.info('The book is written to: %s', book_folder)


def make_report(scenario_name, raw_rally_file_names, book_folder):
    scenario_dir = utils.resolve_relative_path(SCENARIOS_DIR)
    scenario_path = os.path.join(scenario_dir, scenario_name)
    if not scenario_path.endswith('.yaml'):
        scenario_path += '.yaml'

    with open(scenario_path) as fd:
        scenario = fd.read()

    raw_rally_reports = []
    for file_name in raw_rally_file_names:
        with open(file_name) as fd:
            raw_rally_reports.append(json.loads(fd.read()))

    utils.mkdir_tree(book_folder)
    process(raw_rally_reports, book_folder, scenario, scenario_name)


def main():
    parser = argparse.ArgumentParser(prog='rally-reliability-report')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-i', '--input', dest='input', nargs='+',
                        help='Rally raw json output')
    parser.add_argument('-b', '--book', dest='book', required=True,
                        help='folder where to write RST book')
    parser.add_argument('-s', '--scenario', dest='scenario', required=True,
                        help='Rally scenario')
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.DEBUG if args.debug else logging.INFO)

    make_report(args.scenario, args.input, args.book)


if __name__ == '__main__':
    main()
