__author__ = "박현수 (hspark8312@ncsoft.com), NCSOFT Game AI Lab"

# sc2 패치; 반드시 sc2 보다 먼저 import 되어야 함
import sc2_patch
# 프로세스마다 쓰레드를 하나씩 사용하도록 설정
# 반드시 numpy, torch 보다 먼저 설정되어야 함
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import logging
import multiprocessing as mp
import random
import shutil
import sys
import time
import zlib
from collections import OrderedDict, deque
from pathlib import Path
from time import localtime, strftime

import numpy as np
import sc2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython import embed
from toolbox.init.argparse import parse_bool
from sc2 import Difficulty, Race, maps, run_game
from sc2.player import Bot, Computer
from torch.autograd import Variable

import sc2_utils
from toolbox.init.logging import get_logger
from toolbox.init.argparse import parse_bool
from toolbox.utils import keyboard, kill_children_processes

#
# 학습 초기화 코드
#


def get_arguments():
    parser = argparse.ArgumentParser("sc2minigame: QBot train")
    # 기본 옵션
    basic_options = parser.add_argument_group('기본 옵션')
    basic_options.add_argument(
        "--seed", type=int, default=0, help='random seed')
    basic_options.add_argument(
        "--out_path", type=str, default="../outs_sc2mini")
    basic_options.add_argument(
        "--session_id",
        type=str,
        default=strftime("%y-%m-%d-%H-%M-%S", localtime()))
    basic_options.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help='verbose level, 이 값에 따라 기본 log level이 변경,'
        '기본값: warning, -v: info, -vv: debug, -vvv: debug + visdom')
    # 게임 실행 옵션
    actor_options = parser.add_argument_group('actor 옵션(게임 플레이 옵션)')
    actor_options.add_argument(
        "--n_actors", type=int, default=1, help='한번에 실행하는 actor의 개수')
    actor_options.add_argument(
        "--game_map",
        type=str,
        default="sc2_data/maps/NCFellowship-2019_m1_v2",
        help="학습에 사용할 맵, 확장자를 제외한 경로 또는 맵 이름, "
        "맵 이름만 쓸 때는 ${SC2PATH}/Maps에 해당 파일이 있어야 함",
    )
    actor_options.add_argument(
        "--bot1",
        type=str,
        default="bots.nc_example_v7.q_bot.QBot",
        help='학습하려는 봇 경로')
    actor_options.add_argument(
        "--bot2",
        type=str,
        default="bots.nc_example_v6.drop_bot.DropBot",
        help='상대 봇 경로')
    actor_options.add_argument(
        "--difficulty",
        type=int,
        default=4,
        help="bot2가 기본 AI일때, AI의 난이도,"
        "very easy: 1, easy: 2, medium: 3, medium hard: 4, hard: 5, "
        "harder: 6, very hard: 7, cheat vision: 8, "
        "cheat money:  9, cheat insane: 10",
    )
    # 학습 옵션
    train_options = parser.add_argument_group('학습 옵션')
    train_options.add_argument(
        "--replay_memory_capacity", type=int, default=30000)
    train_options.add_argument("--min_train_games", type=int, default=100)
    train_options.add_argument(
        '--batch_size', type=int, default=64, help='미니배치 크기')
    train_options.add_argument(
        "--n_batches", type=int, default=32, help='게임로그 하나가 생성될 때마다 학습하는 횟수')
    train_options.add_argument(
        "--optimizer", choices=["sgd", "adam"], default="adam")
    train_options.add_argument("--lr", type=float, default=0.0001)
    train_options.add_argument("--gamma", type=float, default=0.9)
    train_options.add_argument("--momentum", type=float, default=0.5)
    train_options.add_argument("--max_grad_norm", type=float, default=1.0)
    train_options.add_argument("--win_pred_coef", type=float, default=0.01)
    train_options.add_argument("--ddqn", type=parse_bool, default=True)
    train_options.add_argument('--soft_tau', type=float, default=0.2)

    args = parser.parse_args()

    assert 1 <= args.difficulty <= 10

    # verbose level이 3이상이면 visdom 옵션 설정
    args.visdom = True if args.verbose >= 3 else False
    # verbose level에 따라 log level 설정
    log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    args.log_level = log_levels[min(2, args.verbose)]

    # 옵션을 포함한 원래 command line 명령어를 args에 추가(로그파일에 기록)
    args.command_line = " ".join(sys.argv)
    print(f"## log level: {args.log_level}")
    print(f"## use visdom: {args.visdom}")
    return args


#
# 주 학습 코드
#


def train():

    args = get_arguments()
    # logger 초기화
    logger = get_logger(args, tools=True)
    # 프로젝트 백업:
    # 현재 sc2minigame 폴더 전체를 압축해서 {args.outs}/{session_id}/backups에 저장
    logger.backup_project()

    # train 모델
    model = Model()
    optimizer = set_optimizer(args, model)
    # target 모델
    _model = Model() if args.ddqn else None

    # actors (게임을 실행하는 프로세스)와 공유하는 모델
    # train 모델을 주기적으로 공유메모리에 있는 shared model과 동기화 시키고,
    # AI는 shared model과 자신의 모델을 동기화 한다
    shared_model = Model()
    shared_model.load_state_dict(model.state_dict())
    shared_model.share_memory()

    # actor (child process)와 공유하는 데이터
    # 모델정보 이외에 공유가 필요한 정보를 global_values 객체를 통해 공유한다.
    # global values 객체는 프로세스간에 공유 가능한 객체
    manager = mp.Manager()
    global_values = manager.Namespace()
    global_values.seed = args.seed
    global_values.n_games = 0
    global_values.n_invalid_games = 0
    global_values.n_steps = 0
    global_values.epsilon = 1.0
    global_values.epsilon_delta = 0.01
    global_values.min_epsilon = 0.05
    global_values.game_map = args.game_map
    global_values.eval_period = 10
    global_values.debug_on = False
    global_values.n_batches = args.n_batches
    global_values.n_samples = args.batch_size
    global_values.min_train_samples = global_values.n_samples
    global_values.bot1 = args.bot1
    global_values.bot2 = args.bot2
    global_values.min_train_games = args.min_train_games
    global_values._stop_train = False

    def stop_train():
        global_values._stop_train = True

    # actors가 게임 결과를 반환할때 사용하는 큐
    # outputs 큐 객체는 actors와 train 프로세스가 공유하고 있음
    outputs = mp.Queue()

    # actors 실행
    # args.n_actors 개의 actor가 비동기적으로 게임을 플레이함
    # 최신 신경망과 게임플레이에 필요한 인자는 shared_model과 global_values를 참조하고,
    # 게임 플레이 결과는 outputs큐를 통해 반환함
    actors = list()
    for rank in range(args.n_actors):
        p = mp.Process(
            target=play_game,
            args=(rank, shared_model, global_values, outputs),
            daemon=False,
        )
        actors.append(p)

    [p.start() for p in actors]

    # replay buffer 준비
    win_memory = DequeReplayMemory(capacity=args.replay_memory_capacity)
    lose_memory = DequeReplayMemory(capacity=args.replay_memory_capacity)

    while True:
        try:
            if keyboard.event("esc"):
                # esc를 누르면 디버그 모드 설정함
                print("==== start debuge mode when ready ====")
                global_values.debug_on = True

            if outputs.qsize() > 0:
                # 게임 로그 수집
                # 한 게임이 종료될 때마다 outputs 큐에 게임 플레이 데이터가 추가됨
                logger.info(f"game end: {global_values.n_games}")
                result = outputs.get()
                if result["success"]:
                    # 게임이 성공적으로 종료된 경우
                    global_values.n_games += 1
                    global_values.n_steps += len(result["data"])
                    # 게임에 이겼을 때와 졌을 때, 별도의 리플레이 버퍼에 데이터를 저장함
                    if result["win"] > 0:
                        for sample in result["data"]:
                            win_memory.put(sample)
                    else:
                        for sample in result["data"]:
                            lose_memory.put(sample)

                    # 현재 메모리 상태 기록
                    global_values.memory_capacity_win = win_memory.capacity
                    global_values.memory_size_win = win_memory.size
                    global_values.memory_capacity_lose = lose_memory.capacity
                    global_values.memory_size_lose = lose_memory.size

                    # 게임 조건 및 결과 로깅 관련 작업들
                    game_result_log(logger, args, global_values, result)

                else:
                    # 게임이 비정상적으로 종료된 경우 (게임에 에러가 있는 경우)
                    global_values.n_invalid_games += 1
                    logger.warning(f"something worng in actor: {result}")

                if global_values.n_games > global_values.min_train_games:
                    # 현재까지 플레이한 게임이 min_tarin_games를 넘으면 모델 학습
                    logger.info(f"training: {global_values.n_games}")
                    # 모델 학습
                    loss_dict = OrderedDict()
                    grad_dict = OrderedDict()

                    for _ in range(global_values.n_batches):
                        # n_batches 횟수만큼 미니배치를 학습
                        # 이긴 게임에서 절반, 진 게임에서 절반 학습 데이터를 샘플링 함
                        n_win_samples = min(win_memory.size,
                                            global_values.n_samples // 2)
                        if n_win_samples > 0:
                            win_samples = win_memory.sample(n_win_samples)
                            lose_samples = lose_memory.sample(
                                global_values.n_samples - n_win_samples)
                            samples = [
                                np.concatenate([ws, ls], axis=0)
                                for ws, ls in zip(win_samples, lose_samples)
                            ]
                        else:
                            samples = lose_memory.sample(
                                global_values.n_samples - n_win_samples)

                        # 학습 데이터로 모델 최적화
                        model, loss_dict_, grad_dict_ = fit(
                            args, global_values, model, _model, optimizer,
                            samples, args.ddqn)

                        # loss 값 누적 (로깅용)
                        for k in loss_dict_:
                            loss_dict.setdefault(k, 0)
                            loss_dict[k] += loss_dict_[k] / \
                                global_values.n_batches
                        for k in grad_dict_:
                            grad_dict.setdefault(k, 0)
                            grad_dict[k] += grad_dict_[k] / \
                                global_values.n_batches

                    # target 모델 업데이트 (DDQN)
                    if _model is not None:
                        # polyak average 사용 train 모델 업데이트
                        # tau 만큼 가중치로 train 모델과 target 모델 가중치들을 평균내서
                        # 새로운 target 모델 가중치 설정
                        soft_tau = args.soft_tau
                        for _param, param in zip(_model.parameters(),
                                                 model.parameters()):
                            _param.data.copy_(_param.data * (1.0 - soft_tau) +
                                              param.data * soft_tau)

                    # 최신 모델 동기화
                    shared_model.load_state_dict(model.state_dict())

                    # 1 게임마다, epsilon을 조금씩 낮춤
                    # 최소값: global_values.min_epsilon
                    epsilon_delta = global_values.epsilon_delta / args.n_actors
                    global_values.epsilon = max(
                        global_values.min_epsilon,
                        global_values.epsilon - epsilon_delta,
                    )

                    # 학습 결과 로깅
                    fit_log(logger, args, global_values, grad_dict, loss_dict)

                if global_values.n_games % 10 == 0:
                    # 10 게임마다, 모델 저장
                    logger.info(f"save model: {global_values.n_games}")
                    model.save(args)

                if global_values.debug_on is True:
                    # 디버그 모드 시작
                    print("==== start debuge mode ====")
                    embed()
                    global_values.debug_on = False

        except Exception as e:
            import traceback

            global_values.debug_on = True
            print("==== something is worng in train() main loop ====")
            traceback.print_exc()
            embed()
            if global_values._stop_train:
                exit()
            global_values.debug_on = False


#
# 리플레이 버퍼 코드
#


class DequeReplayMemory(deque):
    def __init__(self, capacity):
        super().__init__(maxlen=capacity)
        self.buffer = None

    @property
    def size(self):
        return len(self)

    @property
    def capacity(self):
        return self.maxlen

    def put(self, data):
        self.append(data)

    def sample(self, n_samples):
        samples = random.sample(self, n_samples)

        if self.buffer is None or self.buffer[0].shape[0] != n_samples:
            # 버퍼 초기화(메모리 할당)
            self.buffer = list()
            for item in samples[0]:
                if type(item) is np.ndarray:
                    buff = np.zeros((n_samples, *item.shape), dtype=item.dtype)
                elif type(item) is int:
                    buff = np.zeros((n_samples, 1), dtype=int)
                elif type(item) is float:
                    buff = np.zeros((n_samples, 1), dtype=float)
                else:
                    raise NotImplementedError
                self.buffer.append(buff)

        for i, sample in enumerate(samples):
            for j, item in enumerate(sample):
                if self.buffer[j][i].ndim is 1:
                    self.buffer[j][i] = item
                else:
                    self.buffer[j][i, :] = item[:]

        return self.buffer


#
# 신경망 관련 코드
#


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

    elif classname.find("Linear") != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class Model(torch.nn.Module):
    def __init__(self, bot=None):
        super(Model, self).__init__()
        self.bot = bot

        # 입력 채널 12
        self.conv1 = nn.Conv2d(12, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        # cnn 출력 64, state 3, 이전 전략 3 차원
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32 + 4 + 5, 32)
        self.q_values = nn.Linear(32, 5)
        self.win_pred = nn.Linear(32, 1)

        self.apply(weights_init)

        linears = (
            (self.fc1, 1.0),
            (self.fc2, 1.0),
            (self.q_values, 1.0),
            (self.win_pred, 1.0),
        )
        for layer, weight in linears:
            layer.weight.data = normalized_columns_initializer(
                layer.weight.data, weight)
            layer.bias.data.fill_(0)

        self.state_value = None
        self.action_values = None

    def preprocess(self, obs, state):
        # numpy array를 torch tensor로 변환
        obs = torch.Tensor(obs) if type(obs) is np.ndarray else obs
        state = torch.Tensor(state) if type(state) is np.ndarray else state

        # 배치 차원이 없으면 추가
        obs = obs.unsqueeze_(0) if obs.dim() == 3 else obs
        state = state.unsqueeze_(0) if state.dim() == 1 else state
        return obs, state

    def visual_embedding(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return F.relu(self.fc1(x))

    def forward(self, inputs):
        obs, state = inputs
        obs, state = self.preprocess(obs, state)
        x = self.visual_embedding(obs)
        x = torch.cat([x, state], dim=1)
        x = F.relu(self.fc2(x))
        return self.q_values(x), self.win_pred(x)

    def act(self, inputs, epsilon=0.0, device="cpu"):
        with torch.no_grad():
            obs, state = inputs
            obs = torch.from_numpy(obs).unsqueeze(0).to(torch.float).to(device)
            state = torch.from_numpy(state).unsqueeze(0).to(
                torch.float).to(device)
            inputs = (obs, state)

            q_values, win_pred = self(inputs)

            # epsilon greedy
            if random.random() < epsilon:
                action = random.randint(0, q_values.shape[1] - 1)
            else:
                action = q_values.argmax(dim=1).item()

        self.state_value = q_values.max().item()
        self.action_values = q_values.cpu().numpy()
        if self.bot is not None and self.bot.rank == 0:
            from .q_bot import Strategy

            q_max, q_max_a = q_values.max(dim=1)
            text = [
                f"a: {Strategy(action)}",
                f"q_a: {q_values[0, action]:2.3f}",
                f"q_max_a: {Strategy(q_max_a.item())}",
                f"q_max: {q_max.item():2.3f}",
                f"q_mean: {q_values.mean().item():2.3f}",
                f"e: {epsilon:2.3f}",
                f"win: {(win_pred.item() + 1.) / 2.:2.3f}",
            ]

            print(">>> " + ", ".join(text))
        return action, win_pred

    def save(self, args):
        log_data_model_path = os.path.join(args.out_path, args.session_id,
                                           "data", "model.pt")
        os.makedirs(os.path.dirname(log_data_model_path), exist_ok=True)
        torch.save(self.state_dict(), log_data_model_path)

    def load(self, args):
        log_data_model_path = os.path.join(args.out_path, args.session_id,
                                           "data", "model.pt")
        if os.path.exists(log_data_model_path):
            self.load_state_dict(torch.load(log_data_model_path))
        return args


#
#  최적화 관련 코드
#


def set_optimizer(args, model):
    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    return optimizer


def fit(args, global_values, model1, model2, optimizer, samples, ddqn):
    ob1, s1, a, r, ob2, s2, done, win = samples
    ob1 = torch.tensor(ob1).to(torch.float)
    s1 = torch.tensor(s1).to(torch.float)
    a = torch.tensor(a).to(torch.long)
    r = torch.tensor(r).to(torch.float)
    ob2 = torch.tensor(ob2).to(torch.float)
    s2 = torch.tensor(s2).to(torch.float)
    done = torch.tensor(done).to(torch.float)
    win = torch.tensor(win).to(torch.float)

    with torch.no_grad():
        q2, _ = model1((ob2, s2))
        idx = torch.arange(q2.shape[0]).to(torch.long)

        if ddqn:
            # ddqn
            _, q2a = q2.max(dim=1)
            q2v, _ = model2((ob2, s2))
            q2_max = q2v[idx, q2a].unsqueeze(dim=1)
        else:
            # dqn
            q2_max = q2.max(dim=1, keepdim=True)

    # target Q 값 계산
    q1, win1 = model1((ob1, s1))
    target_q = q1.clone().data
    target_q[idx, a.squeeze()] = (
        r + args.gamma * (1.0 - done) * q2_max).squeeze()

    # Q 값 오차 계산
    q_loss = (target_q.data - q1)**2
    q_loss_mean = q_loss.mean()

    # 승률 오차 계산
    win_loss = (win.data - win1)**2
    win_loss_mean = win_loss.mean()

    # 최종 오차
    loss = q_loss_mean + args.win_pred_coef * win_loss_mean

    # loss 기록
    loss_dict = OrderedDict()
    loss_dict["q"] = q_loss_mean.item()
    loss_dict["win"] = win_loss_mean.item()
    loss_dict["total"] = loss.item()

    # optimizer 실행
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model1.parameters(), args.max_grad_norm)
    # grad 기록
    grad_dict = OrderedDict()
    for name, param in model1.named_parameters():
        grad_dict.setdefault(name, 0.0)
        if param.grad is not None:
            grad_dict[name] = param.grad.data.mean().item()
    optimizer.step()

    return model1, loss_dict, grad_dict


#
# 게임 플레이 관련 코드
#


def play_game(rank, shared_model, global_values, outputs):

    from toolbox.utils import kill_children_processes

    n_games = 0
    import torch

    torch.manual_seed(rank + global_values.seed)

    while True:
        local_values = argparse.Namespace()
        if (rank + n_games) % global_values.eval_period == 0:
            local_values.epsilon = 0.0
        else:
            local_values.epsilon = global_values.epsilon

        while True:
            # debug 모드일때 log 출력을 막기 위해 잠시 worker를 대기
            if global_values.debug_on is False:
                break
            else:
                time.sleep(1)

        play_game_process = mp.Process(
            target=_play_game,
            args=(rank, shared_model, global_values, local_values, outputs),
            daemon=True,
        )
        play_game_process.start()
        play_game_process.join(15 * 60)  # 게임 종료까지 15분 대기
        if play_game_process.is_alive():
            # 너무 오래 동안 게임이 실행중이면 강제 종료
            kill_children_processes(play_game_process.pid)
            outputs.put(
                dict(
                    success=False,
                    rank=rank,
                    exception=TimeoutError("too long game")))

        n_games += 1


def _import_bot(race, bot_desc, *args, **kwargs):
    import importlib

    if len(bot_desc) == 4 and bot_desc.lower().startswith("com"):
        # [('VeryEasy', 1), ('Easy', 2), ('Medium', 3),
        # ('MediumHard', 4), ('Hard', 5), ('Harder', 6),
        # ('VeryHard', 7),
        # ('CheatVision', 8), ('CheatMoney', 9), ('CheatInsane', 10)]
        level = int(bot_desc[3])
        bot = Computer(race, Difficulty(level))
    else:
        module, name = bot_desc.rsplit(".", 1)
        bot_cls = getattr(importlib.import_module(module), name)
        bot_ai = bot_cls(*args, **kwargs)
        bot = Bot(race, bot_ai)
    return bot


def _play_game(rank, shared_model, global_values, local_values, outputs):

    try:
        print(f"rank: {rank}, global: {global_values}, local: {local_values}")
        game_map = global_values.game_map
        bot1 = _import_bot(
            Race.Terran,
            global_values.bot1,
            train=True,
            rank=rank,
            debug=True,
            shared_model=shared_model,
            out_queue=outputs,
            epsilon=local_values.epsilon,
        )
        bot2 = _import_bot(Race.Terran, global_values.bot2, debug=True)
        bots = [bot1, bot2]

        # sc2.run_game(
        #     game_map,
        #     bots,
        #     realtime=False,
        #     rgb_render_config=dict(
        #         window_size=(800, 480), minimap_size=(128, 128)),
        # )
        sc2.run_game(game_map, bots, realtime=False)

    except Exception as e:
        outputs.put(dict(success=False, rank=rank, exception=e))
    finally:
        time.sleep(1)


#
# 로깅 관련 코드
#


def game_result_log(logger, args, global_values, result):
    n_games = global_values.n_games
    win_memory_size = global_values.memory_size_win
    lose_memory_size = global_values.memory_size_lose

    # 기본 정보 출력
    logger.table("args", n_games, args.__dict__)
    logger.table("global_values", n_games, global_values._getvalue().__dict__)

    # 게임 승/패 기록
    if result["epsilon"] > 0.0:
        title = "score/train_wins"
    else:
        title = "score/eval_wins"
    logger.line(title, n_games, (result["win"] + 1.0) / 2.0)
    logger.line("score/average_win", n_games, (result["win"] + 1.0) / 2.0)
    logger.line("hyperparams/epsilon", n_games, global_values.epsilon)
    logger.line("hyperparams/lr", n_games, args.lr)
    logger.line("hyperparams/win_pred_coef", n_games, args.win_pred_coef)
    logger.line("memory", n_games,
                dict(win=win_memory_size, lose=lose_memory_size))


def fit_log(logger, args, global_values, grad_dict, loss_dict):
    n_games = global_values.n_games

    # gradient
    logger.line("gradients", n_games, grad_dict)
    # loss
    logger.line("loss", n_games, loss_dict)


if __name__ == "__main__":

    try:
        train()

    except Exception as e:
        import traceback
        traceback.print_exc()

    finally:
        # 부모와 자식 프로세스 모두 종료
        pid = os.getpid()
        kill_children_processes(pid, including_parent=True)
