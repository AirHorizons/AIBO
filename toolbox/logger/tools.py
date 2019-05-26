"""
log 결과를 일반 logger, tensorboard, visdom에 모두 출력/저장/시각화 하는 모듈
"""

__author__ = 'Hyunsoo Park (hspark8312@ncsoft.com), Game AI Lab, NCSOFT'

import logging
import os
import pathlib
from collections import OrderedDict, deque
from time import gmtime, localtime, strftime

import numpy as np
import pandas as pd
import torch
import visdom
from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from ..logger.colorize import Color as C
from ..utils import backup_dir


class Tools(object):
    def __init__(self, out_path, log_level, enable_visdom):

        self.log_level = log_level
        self.out_path = out_path

        # tensorboardX
        self.writer = SummaryWriter(self.out_path)
        self.tb_text_buff = dict()

        # visdom
        self.enable_visdom = enable_visdom
        self.viz_envs = dict()
        self.viz_buff = dict()
        self.viz_commands = list()

    def get_data_path(self, tag, file_name):
        path = os.path.join(self.out_path, tag, file_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def get_viz_env(self, visdom_env):
        if visdom_env not in self.viz_envs:
            self.viz_envs[visdom_env] = visdom.Visdom(env=visdom_env)
        return self.viz_envs[visdom_env]

    @staticmethod
    def tabularize(tag, dict_obj):
        """
        dict를 html table 코드로 변환
        """
        table = ''
        # table += '<table style="width:100%;">\n'
        table += '<table>\n'
        table += f'{tag}\n'
        table += '<tr><th><strong>Key</strong></th><th><strong>Value</strong></th></tr>\n'
        for k in sorted(dict_obj.keys()):
            v = dict_obj[k]
            table += f'<tr><td>{k}</td><td>{v}</td></tr>\n'
        table += '</table>'
        return table

    @staticmethod
    def styled_table(html_text):
        style = """
        <style>
            table {
                width: 100%;
                border: 1px solid #444444;
                border-collapse: collapse;
            }
            th, td {
                border: 1px solid #444444;
                padding: 3px;
            }
        </style>
        """
        return style + html_text

    def table(self,
              tag,
              step,
              dict_obj,
              visdom_env=None,
              width=None,
              height=None):
        """
        dict나 OrderdDict를 html table로 바꿔서, 저장하거나 visdom에 시각화

        visdom 시각화에서 제외하려면 visdom_env=None 으로 설정
        """
        assert type(dict_obj) in (dict, OrderedDict)

        tag = tag.replace(' ', '_')
        table = self.tabularize(tag, dict_obj)

        # tensorboad
        if self.tb_text_buff.get(tag) != table:
            self.tb_text_buff[tag] = table
            self.writer.add_text(tag, table, step)

        # visdom
        if self.enable_visdom and visdom_env:
            viz = self.get_viz_env(visdom_env)
            viz.text(
                self.styled_table(table),
                win=f'{tag}_table',
                opts=dict(width=width, height=height))

        # to html file
        try:
            html_path = self.get_data_path('tables', f'{tag}_table.html')
            hp = pathlib.Path(html_path)
            with open(hp.absolute(), 'wt') as f:
                f.write(table)
        except OSError as exc:
            print(C.warning(f'파일 쓰기 실패: {hp.absolute()}, 예외: {exc}'))

    def line(self,
             title,
             step,
             values,
             visdom_env=None,
             width=None,
             height=None,
             ymin=None,
             ymax=None,
             buffer_size=100,
             average=1):
        """
        입력 받은 x, y 데이터를 텐서보드에 기록하고, visdom에 시각화

        visdom 시각화에서 제외하려면 visdom_env=None 으로 설정
        """

        title = title.replace(' ', '_')

        self._tb_line(title, step, values)

        if self.enable_visdom and visdom_env:
            viz = self.get_viz_env(visdom_env)
            self._viz_line(viz, visdom_env, title, step, values, width, height,
                           ymin, ymax, buffer_size, average)

    def _tb_line(self, tag, step, values):

        if self.writer:
            if type(values) is dict:
                for k, v in values.items():
                    tag_key = f'{tag}/{k}'
                    self.writer.add_scalar(tag_key, v, step)
                # self.writer.add_scalars(tag, values, step)
            elif type(values) is OrderedDict:
                n = 0
                for k, v in values.items():
                    tag_key = f'{tag}/{n:03d}-{k}'
                    self.writer.add_scalar(tag_key, v, step)
                    n += 1
                # self.writer.add_scalars(tag, values, step)
            else:
                self.writer.add_scalar(tag, values, step)

    def _viz_line(self, viz, visdom_env, tag, x, y, width, height, ymin, ymax,
                  buffer_size, average):
        buff = self.viz_buff
        win_name = f'{tag}_line'

        def moving_average(ys, window_size=3):

            if len(ys) > 1:
                buffer = deque(maxlen=window_size)
                ys_ = np.array(ys).squeeze()
                ys = np.zeros_like(ys_)

                for i, y in enumerate(ys_):
                    buffer.append(y)
                    ys[i] = np.mean(buffer)

            return ys

        if type(y) in (dict, OrderedDict):
            legend, xs, ys = [], [], []
            for k, y_ in sorted(y.items()):
                buff_name = f'{tag}_line/{k}'
                log_buffer = buff.get(buff_name, deque(maxlen=buffer_size))
                # log_buffer.append((x, y_))
                if len(log_buffer) > 0 and log_buffer[-1][0] == x:
                    log_buffer[-1][1].append(y_)
                else:
                    log_buffer.append((x, [y_]))
                xs_, ys_ = zip(*log_buffer)
                xs_, ys_ = np.array(xs_), np.array([np.mean(y_) for y_ in ys_])
                if average > 1:
                    ys_ = moving_average(ys_, min(average, len(log_buffer)))
                    if len(ys_) >= buffer_size and len(ys_) > average:
                        xs_ = xs_[average:]
                        ys_ = ys_[average:]
                legend.append(k)
                xs.append(xs_)
                ys.append(ys_)
                buff[buff_name] = log_buffer

            if viz is not None:
                viz.line(
                    X=np.column_stack(xs),
                    Y=np.column_stack(ys),
                    win=win_name,
                    opts=dict(
                        title=tag,
                        width=width,
                        height=height,
                        ytickmin=ymin,
                        ytickmax=ymax,
                        legend=legend))
        else:
            buff_name = f'{tag}_line'
            log_buffer = buff.get(buff_name, deque(maxlen=buffer_size))
            if len(log_buffer) > 0 and log_buffer[-1][0] == x:
                log_buffer[-1][1].append(y)
            else:
                log_buffer.append((x, [y]))
            xs, ys = zip(*log_buffer)
            xs, ys = np.array(xs), np.array([np.mean(y_) for y_ in ys])
            if average > 1:
                ys = moving_average(ys, min(average, len(log_buffer)))
                if len(ys) >= buffer_size and len(ys) > average:
                    xs = xs[average:]
                    ys = ys[average:]

            if viz is not None:
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
            buff[buff_name] = log_buffer

    def scatter(self,
                title,
                step,
                values,
                visdom_env=None,
                width=None,
                height=None,
                ymin=None,
                ymax=None,
                buffer_size=100,
                average=1,
                *args,
                **kwargs):

        title = title.replace(' ', '_')

        self._tb_line(title, step, values)

        if self.enable_visdom and visdom_env:
            viz = self.get_viz_env(visdom_env)
            self._viz_scatter(viz, visdom_env, title, step, values, width,
                              height, ymin, ymax, buffer_size, average)

    def _viz_scatter(self, viz, visdom_env, tag, x, y, width, height, ymin,
                     ymax, buffer_size, average):
        buff = self.viz_buff
        win_name = f'{tag}_scatter'

        if type(y) in (dict, OrderedDict):
            legend, xs, ys = [], [], []
            y_id = 1
            for k, yv in sorted(y.items()):
                buff_name = f'{tag}_scatter/{k}'
                log_buffer = buff.get(buff_name, deque(maxlen=buffer_size))
                log_buffer.append((x, yv))

                legend.append(k)
                xs.append(log_buffer)
                ys.append([y_id] * len(log_buffer))
                y_id += 1
                buff[buff_name] = log_buffer

            if viz is not None:
                viz.scatter(
                    X=np.vstack(xs),
                    Y=np.vstack(ys),
                    win=win_name,
                    opts=dict(
                        title=tag,
                        legend=legend,
                        width=width,
                        height=height,
                        ytickmin=ymin,
                        ytickmax=ymax,
                        markersymbol='cross-thin-open'))
        else:
            buff_name = f'{tag}_scatter'
            log_buffer = buff.get(buff_name, deque(maxlen=buffer_size))
            log_buffer.append((x, y))
            _, values = zip(*log_buffer)
            mean_value = np.mean(values[-average:])

            if viz is not None:
                viz.scatter(
                    X=np.array(log_buffer),
                    win=win_name,
                    opts=dict(
                        title=f'{tag}: {mean_value:.6f} ({average})',
                        width=width,
                        height=height,
                        ytickmin=ymin,
                        ytickmax=ymax,
                        markersymbol='cross-thin-open'))
            buff[buff_name] = log_buffer

    def bar(self,
            title,
            step,
            ys,
            visdom_env=None,
            width=None,
            height=None,
            ymin=None,
            ymax=None,
            sort=True):
        """
        visdom 시각화에서 제외하려면 visdom_env=None 으로 설정
        """
        title = title.replace(' ', '_')

        self._tb_line(title, step, ys)

        if self.enable_visdom and visdom_env:
            if visdom_env not in self.viz_envs:
                self.viz_envs[visdom_env] = visdom.Visdom(env=visdom_env)
            viz = self.viz_envs[visdom_env]

            labels = []
            values = []

            if sort:
                keys = sorted(ys.keys())
            else:
                keys = ys.keys()

            for k in keys:
                labels.append('_' + str(k) + '_')
                values.append(ys[k])

            if len(ys) == 1:
                labels.append('_')
                values.append(0)

            viz.bar(
                X=values,
                win=title,
                opts=dict(
                    title=title, rownames=labels, width=width, height=height))

    def progressbar(self,
                    labels,
                    step,
                    values,
                    max_values,
                    visdom_env=None,
                    width=None,
                    height=None):
        """
        visdom 시각화에서 제외하려면 visdom_env=None 으로 설정
        """
        title = labels.replace(' ', '_')

        self._tb_line(title, step, values)

        if self.enable_visdom and visdom_env:
            if visdom_env not in self.viz_envs:
                self.viz_envs[visdom_env] = visdom.Visdom(env=visdom_env)
            viz = self.viz_envs[visdom_env]

            if type(labels) == str:
                labels = [labels, '_']
                values = [values, 0]

            assert len(labels) == len(values)

            labels_ = list()
            values_ = list()
            for label, value, max_value in zip(labels, values, max_values):
                labels_.append(f'{label}<br>{value:,}')
                values_.append(max(0.0, min(1.0, value / max_value)))

            viz.bar(
                X=values_,
                win='progress bar',
                opts=dict(
                    title='Progress bar',
                    rownames=labels_,
                    width=width,
                    height=height,
                    ytickmin=0.0,
                    ytickmax=1.0))

    def text(self,
             tag,
             step,
             text,
             visdom_env=None,
             width=None,
             height=None,
             buffer_size=1000):
        """
        visdom 시각화에서 제외하려면 visdom_env=None 으로 설정
        """

        if self.tb_text_buff.get(tag) != text:
            self.tb_text_buff[tag] = text
            self.writer.add_text(tag, text, step)

        if self.enable_visdom and visdom_env:
            if visdom_env not in self.viz_envs:
                self.viz_envs[visdom_env] = visdom.Visdom(env=visdom_env)
            viz = self.viz_envs[visdom_env]

            win_name = f'{tag}_log'
            log_buffer = self.viz_buff.get(win_name, deque(maxlen=buffer_size))
            current_time = strftime('%y-%m-%d-%H-%M-%S', localtime())
            text = f'{current_time}: {text}'
            log_buffer.append(text)
            log_text = '</br>'.join(log_buffer)
            viz.text(
                log_text,
                win=win_name,
                opts=dict(title=win_name, width=width, height=height))
            self.viz_buff[win_name] = log_buffer

    def save_csv(self, file_name='result'):
        import glob

        # tensorboard log 파일 경로
        event_files = glob.glob(self.out_path + '/events.out.tfevents.*')
        assert len(list(event_files)) == 1, 'tensorboard eventfile이 여러 개 있음'
        tb_log_path = list(event_files)[0]

        # tabulate sclars
        try:
            summary_iterators = list()
            it = EventAccumulator(tb_log_path).Reload()
            if len(it.Tags()['scalars']) > 0:
                summary_iterators.append(it)
            else:
                return None
        except ValueError as exc:
            # ValueError: Unknown field metadata
            print(
                C.warning(
                    f'tensorboard log file 읽기 실패 {tb_log_path}, 예외: {exc}'))
            return None

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

        try:
            # csv 파일 경로
            # time_str = strftime('%y-%m-%d-%H-%M-%S', gmtime())
            csv_path = self.get_data_path('csv', f'{file_name}.csv')
            dfx.to_csv(csv_path)
        except OSError as exc:
            print(C.warning(f'파일 쓰기 실패: {csv_path}, 예외: {exc}'))
        return csv_path

    def backup_project(self):
        """ 현재 프로젝트를 백업 """
        # 프로젝트 경로
        proj = os.path.abspath('.')
        # 백업 경로
        time_str = strftime('%y-%m-%d-%H-%M-%S', gmtime())
        backup_path = self.get_data_path('backups', f'{time_str}.zip')

        backup_dir(proj, backup_path)

        print(C.header('프로젝트 백업 완료'))
        print(C.blue(f'프로젝트 폴더: {proj}'))
        print(C.blue(f'백업 폴어: {backup_path}'))

    def save_model(self, tag, step, model):
        model_path = self.get_data_path('models', f'{tag}-{step}.pt')
        torch.save(model.state_dict(), model_path)
        return model_path