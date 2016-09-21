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
import os
import shlex

from oslo_concurrency import processutils

import rally_runners.reliability as me
import rally_runners.reliability.rally_plugins as plugins
from rally_runners.reliability import report


def main():
    parser = argparse.ArgumentParser(prog='rally-reliability')
    parser.add_argument('-s', '--scenario', dest='scenario', required=True,
                        help='Rally scenario')
    parser.add_argument('-o', '--output', dest='output', required=True,
                        help='raw Rally output')
    parser.add_argument('-b', '--book', dest='book', required=True,
                        help='folder where to write RST book')
    args = parser.parse_args()

    plugin_paths = os.path.dirname(plugins.__file__)
    scenario_dir = os.path.join(os.path.dirname(me.__file__), 'scenarios')
    scenario_path = os.path.join(scenario_dir, args.scenario)

    processutils.execute(
        ['rally', '--plugin-paths', plugin_paths, 'task', 'start',
         '--task', scenario_path])

    command_stdout, command_stderr = processutils.execute(
        *shlex.split('rally task list'))

    last_task_id = command_stdout.split('\n')[-2]

    processutils.execute(
        *shlex.split('rally task report %(id)s --out %(output)s' %
                     dict(id=last_task_id, output=args.output)))

    report.make_report(args.output, args.book)


if __name__ == '__main__':
    main()
