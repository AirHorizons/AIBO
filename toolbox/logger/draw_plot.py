__author__ = 'Hyunsoo Park (hspark8312@ncsoft.com), Game AI Lab, NCSOFT'

import argparse
import functools
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
from IPython import embed
from tensorboard.backend.event_processing import event_accumulator


def get_files(args):
    event_files = glob.glob(args.logdir + '/*/events.out.tfevents.*')
    event_file_dict = dict()
    for event_file in event_files:
        cond, *_ = os.path.basename(os.path.dirname(event_file)).split('-')
        event_file_dict.setdefault(cond, list())
        event_file_dict[cond].append(os.path.abspath(event_file))

    # seed 값으로 재정렬, 기본 정렬 순서는 1 다음 2 대신 10 순서로 정렬되는 문제 해결
    for cond in event_file_dict:
        event_file_dict[cond].sort(key=lambda f: int(f.split('-')[1]))

    return event_file_dict


def read_event_file(path, tag, mtime):
    ea = read_event_file_cached(path, mtime)
    return ea.Scalars(tag)


@functools.lru_cache(maxsize=1024)
def read_event_file_cached(path, mtime):
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={  # see below regarding this argument
            event_accumulator.COMPRESSED_HISTOGRAMS: 500,
            event_accumulator.IMAGES: 4,
            event_accumulator.AUDIO: 4,
            event_accumulator.SCALARS: 0,
            event_accumulator.HISTOGRAMS: 1
        })

    ea.Reload()  # 다시 로드
    ea.Tags()  # 태그
    return ea


def aggregate_scalars(event_files, cond, tag):
    dfx = list()
    for event_file in event_files:
        print(f'Read: {cond}, Path: {event_file}')
        update_time = os.path.getmtime(event_file)
        scalars = read_event_file(event_file, tag, update_time)
        df = pd.DataFrame(scalars)
        df = df.loc[:, ['step', 'value']]
        df = df.groupby('step').mean()
        dfx.append(df)
    dfx.append(df)
    dfx = pd.concat(dfx, axis=1)
    return dfx.T.describe().T


def main(args, tag, minus_one=False):
    event_files = get_files(args)
    count = min([len(files) for files in event_files.values()])
    count = count - 1 if minus_one else count

    dfs = dict()
    for cond in event_files:
        dfs[cond] = aggregate_scalars(event_files[cond][:count], cond, tag)

    # plt.rcParams.update({'font.family': "NanumMyeongjo"})
    # # or "Noto Sans CJK KR", "Noto Sans Mono CJK KR"
    # plt.rcParams.update({'font.size': 22})
    # plt.rcParams.update({'xtick.labelsize': 22, 'ytick.labelsize': 22})

    fig, ax = plt.subplots(figsize=(8, 4), clear=True)

    ax.set_xlabel('Generation')
    ax.set_ylabel(f'{tag}')

    for cond in dfs:
        line, = ax.plot(dfs[cond].index, dfs[cond]['50%'], '-', label=cond)
        ax.fill_between(
            dfs[cond].index,
            dfs[cond]['25%'],
            dfs[cond]['75%'],
            color=line.get_color(),
            alpha=0.3)

    ax.set_title(f'{tag} ({count})')
    ax.legend()
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='../logdir')
    parser.add_argument('--tag', type=str, default='score/max_score')
    args = parser.parse_args()

    print(args)
    main(args, args.tag)
    # main(args, 'score/max_score', minus_one=True)
    # main(args, 'hyperparams/lr', minus_one=True)

    embed()
    exit()
