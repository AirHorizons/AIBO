__author__ = 'Hyunsoo Park (hspark8312@ncsoft.com), Game AI Lab, NCSOFT'

import logging
import os
from collections import OrderedDict, defaultdict, deque

import numpy as np
import pandas as pd
import tensorflow as tf
import visdom
from IPython import embed
from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator
from tensorboardX import SummaryWriter


class LogTools(object):
    def __init__(self, args):

        self.log_level = args.log_level
        self.log_path = os.path.join(args.log_path, args.session_id)
        os.makedirs(self.log_path, exist_ok=True)
        self.log_data_path = os.path.join(args.log_path, args.session_id,
                                          'data')
        os.makedirs(self.log_data_path, exist_ok=True)

        # csv
        self.csv_file_path = os.path.join(self.log_data_path, 'data.csv')

        # tensorboardX
        self.writer = SummaryWriter(self.log_path)

        # visdom
        self.visdom = args.visdom
        self.viz_server = args.visdom_server
        self.viz_envs = dict()
        self.viz_buff = dict()
        # if args.visdom:
        #     self.viz = visdom.Visdom(args.visdom_server, env='main')
        # else:
        #     self.viz = None
        #     self.viz_buff = None

    def get_viz_env(self, visdom_env):
        if visdom_env not in self.viz_envs:
            self.viz_envs[visdom_env] = visdom.Visdom(
                self.viz_server, env=visdom_env)
        return self.viz_envs[visdom_env]

    def table(self, tag, dict_obj, visdom_env='main', width=None, height=None):
        assert type(dict_obj) in (dict, OrderedDict)

        table = """
            <style>
                table {
                    width: 100%;
                    border: 1px solid #444444;
                }
                th, td {
                    border: 1px solid #444444;
                }
            </style>\n"""
        table += '<table>\n'
        table += '<tr><th><strong>Key</strong></th><th><strong>Value</strong></th></tr>\n'
        for k in sorted(dict_obj.keys()):
            v = dict_obj[k]
            table += f'<tr><td>{k}</td><td>{v}</td></tr>\n'
        table += '</table>'

        html_path = os.path.join(self.log_data_path, f'{tag}_table.html')
        with open(html_path, 'wt') as f:
            f.write(table)

        if self.visdom:
            viz = self.get_viz_env(visdom_env)
            viz.text(
                table,
                win=f'{tag}_table',
                opts=dict(title=tag, width=width, height=height))

    def line(self,
             tag,
             x,
             y,
             visdom_env='main',
             width=None,
             height=None,
             ymin=None,
             ymax=None,
             buffer_size=500,
             average=1):

        self._tb_line(tag, x, y)

        if self.visdom:
            self._viz_line(tag, x, y, visdom_env, width, height, ymin, ymax,
                           buffer_size, average)

    def _tb_line(self, tag, x, y):

        if type(y) in (dict, OrderedDict):
            for k, y_ in y.items():
                tag_ = f'{tag}/{k}'
                self.writer.add_scalar(tag_, y_, x)
        else:
            self.writer.add_scalar(tag, y, x)

    def _viz_line(self, tag, x, y, visdom_env, width, height, ymin, ymax,
                  buffer_size, average):
        buff = self.viz_buff

        def moving_average(xs, window_size=3):
            assert len(xs) > 0
            assert window_size > 0

            if window_size % 2 == 0:
                window_size += 1

            if window_size > 1 and len(xs) > window_size:
                weights = np.ones(window_size) / window_size
                xs_extended = [xs]
                for _ in range(window_size // 2):
                    xs_extended.insert(0, xs[:1])
                    xs_extended.insert(-1, xs[-1:])
                xs_extended = np.concatenate(xs_extended)
                return np.convolve(xs_extended, weights, mode='valid')
            else:
                return xs

        try:
            win_name = f'{tag}_line'

            if type(y) in (dict, OrderedDict):
                buff_name = f'{tag}_line'
                log_buffer = buff.get(
                    buff_name, pd.DataFrame(
                        columns=['step'], dtype=np.float32))
                log_buffer = log_buffer.append(
                    dict(step=x, **y), ignore_index=True)
                buff[buff_name] = log_buffer

                grouped_buffer = log_buffer.groupby('step').mean()
                index = np.array(grouped_buffer.index).reshape(-1, 1)
                index = index.repeat(len(grouped_buffer.columns), axis=1)
                viz = self.get_viz_env(visdom_env)
                opts = dict(
                    title=tag,
                    width=width,
                    height=height,
                    legend=grouped_buffer.columns.values.tolist())
                viz.line(
                    X=index, Y=grouped_buffer.values, win=win_name, opts=opts)
            else:
                buff_name = f'{tag}_line'
                log_buffer = buff.get(buff_name, deque(maxlen=buffer_size))
                if len(log_buffer) > 0 and log_buffer[-1][0] == x:
                    log_buffer[-1][1].append(y)
                else:
                    log_buffer.append((x, [y]))
                xs, ys = zip(*log_buffer)
                xs, ys = np.array(xs), np.array([np.mean(y_) for y_ in ys])
                ys = moving_average(ys, average)
                buff[buff_name] = log_buffer

                viz = self.get_viz_env(visdom_env)
                viz.line(
                    X=xs,
                    Y=ys,
                    win=win_name,
                    opts=dict(
                        title=f'{tag}: {ys[-1]:.6f}',
                        width=width,
                        height=height,
                        ytickmin=ymin,
                        ytickmax=ymax))
        except Exception as exc:
            from IPython import embed
            embed()
            exit()

    def bar(self,
            tag,
            value_dict,
            visdom_env='main',
            width=None,
            height=None,
            ymin=None,
            ymax=None,
            sort=True):

        if self.visdom:
            if len(value_dict) == 1:
                value_dict['_'] = 0.0

            labels, values = [], []

            if sort:
                items = sorted(value_dict.items())
            else:
                items = value_dict.items()

            for label, value in items:
                labels.append(label)
                values.append(value)

            viz = self.get_viz_env(visdom_env)
            viz.bar(
                X=values,
                win=tag + '_bar',
                opts=dict(
                    title=tag, rownames=labels, width=width, height=height))

    def progressbar(self,
                    tag,
                    labels,
                    values,
                    max_values,
                    visdom_env='main',
                    width=None,
                    height=None):

        if self.visdom:
            if type(labels) == str:
                labels = [labels]

            if type(values) == str:
                values = [values]

            assert len(labels) == len(values)

            labels_ = list()
            values_ = list()
            for label, value, max_value in zip(labels, values, max_values):
                labels_.append(f'{label}<br>{value:,}')
                values_.append(max(0.0, min(1.0, value / max_value)))

            viz = self.get_viz_env(visdom_env)
            viz.bar(
                X=values_,
                win=f'{tag}_progress_bar',
                opts=dict(
                    title=tag,
                    rownames=labels_,
                    width=width,
                    height=height,
                    ytickmin=0.0,
                    ytickmax=1.0))

    def text(self,
             tag,
             x,
             text,
             visdom_env='main',
             width=None,
             height=None,
             buffer_size=500):

        self.writer.add_text(tag, text, x)

        if self.visdom:
            win_name = f'{tag}_log'
            log_buffer = self.viz_buff.get(win_name, deque(maxlen=buffer_size))
            log_buffer.append(text)
            log_text = '</br>'.join(log_buffer)
            self.viz_buff[win_name] = log_buffer

            viz = self.get_viz_env(visdom_env)
            viz.text(
                log_text,
                win=win_name + 'text',
                opts=dict(title=win_name, width=width, height=height))


def save_log_to_csv(log_path, session_id):
    # csv path
    tb_log_path = os.path.join(log_path, session_id)
    log_data_path = os.path.join(log_path, session_id, 'data')
    os.makedirs(log_data_path, exist_ok=True)
    csv_file_path = os.path.join(log_data_path, 'data.csv')

    # tabulate sclars
    summary_iterators = list()
    for dname in os.listdir(tb_log_path):
        it = EventAccumulator(os.path.join(tb_log_path, dname)).Reload()
        if len(it.Tags()['scalars']) > 0:
            summary_iterators.append(it)
            break

    assert len(summary_iterators) == 1
    tags = summary_iterators[0].Tags()['scalars']

    dfx = list()
    for tag in tags:
        scalars = summary_iterators[0].Scalars(tag)
        df = pd.DataFrame(scalars)
        df = df.loc[:, ('step', 'value')]
        df = df.groupby('step').mean()
        dfx.append(df)

    dfx = pd.concat(dfx, axis=1)
    dfx.columns = tags
    dfx.to_csv(csv_file_path)


if __name__ == '__main__':

    save_log_to_csv('../logdir', '19-01-04-19-21-44')
