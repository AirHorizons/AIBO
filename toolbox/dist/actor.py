__author__ = 'Hyunsoo Park (hspark8312@ncsoft.com), Game AI Lab, NCSOFT'

import inspect
import multiprocessing as mp
import pickle
import pprint
import random
import time
from collections import OrderedDict, namedtuple

import cloudpickle
import torch
from IPython import embed

import toolbox
from toolbox.utils import import_component, kill_children_processes

from ..logger.colorize import Color as C
from . import proxy_worker


class Params(object):
    def __init__(self, version, model=None, parameters=None):
        self.version = version
        if model is not None:
            self.params = [p.data.cpu().numpy() for p in model.parameters()]
        elif parameters is not None:
            self.params = [p.data.cpu().numpy() for p in parameters]
        else:
            raise NotImplementedError

    @staticmethod
    def from_model(version, model):
        return Params(version, model=model, parameters=None)

    @staticmethod
    def from_params(version, parameters):
        return Params(version, model=None, parameters=parameters)

    def __repr__(self):
        return f'model version: {self.version}'

    def assign(self, model):
        for model_param, self_param in zip(model.parameters(), self.params):
            model_param.data = torch.from_numpy(self_param)
        return model


class Result(object):
    def __init__(self, success, rank, data):
        self.success = success
        self.rank = rank
        self._encoded_data = self._encode(data)
        self._decoded_data = None

    @property
    def data(self):
        if self._decoded_data is None:
            self._decoded_data = self._decode(self._encoded_data)
        return self._decoded_data

    @staticmethod
    def _encode(x):
        return pickle.dumps(x)

    @staticmethod
    def _decode(x):
        return pickle.loads(x)

    def __repr__(self):
        buff = f'rank: {self.rank}, succ: {self.success}, result: \n'
        buff += f'{self.data}'
        return buff


class ActorGroup(object):
    def __init__(self, name, base_rank, n_actors, timelimit, seed,
                 max_pending_tasks):

        self.name = name
        self.n_actors = n_actors

        self._manager = mp.Manager()

        self.GV = self._manager.Namespace()
        self.GV.remote_idle = 0  # Remote actor가 요청한 task 횟수, proxy_worker.py 참조
        self.GV.seed = seed
        self.GV.debug = False
        self.GV.timelimit = timelimit
        self.GV.max_pending_tasks = max_pending_tasks

        self.param_dict = self._manager.dict()

        self.inputs = mp.Queue()
        self.outputs = mp.Queue()

        self.last_resp = dict()  # actor가 마지막으로 응답한 시간 기록

        # 일반 actor 시작
        self.actors = list()
        print(C.notice(f'{self.name}(ActorGroup): {n_actors} actors 시작'))
        for rank in range(n_actors):
            actor_process = mp.Process(
                target=ActorGroup.actor,
                args=(base_rank + rank, self.GV, self.param_dict, self.inputs,
                      self.outputs),
                daemon=False)
            self.actors.append(actor_process)

    def add_proxy_actor(self, frontend_port, backend_port):
        try:
            print(C.notice(f'{self.name}(ActorGroup): Proxy Actor device 시작'))
            actor_process = mp.Process(
                target=proxy_worker.proxy_actor_device,
                args=(frontend_port, backend_port))
            self.actors.append(actor_process)

            print(C.notice(f'{self.name}(ActorGroup): Proxy Actor 시작'))
            actor_process = mp.Process(
                target=proxy_worker.proxy_actor,
                args=(frontend_port, backend_port, self.GV, self.param_dict,
                      self.inputs, self.outputs))
            self.actors.append(actor_process)

        except Exception as exc:
            print(C.error(f'{self.name}(ActorGroup): Proxy actor 시작 실패: {exc}'))
            import traceback
            traceback.print_exc()

    @staticmethod
    def actor(rank, GV, param_dict, inputs, outputs):
        """
        task_func가 계속 실행할 수 있도록 관리함
        """
        
        seed = GV.seed + rank
        random.seed(seed)
        
        while True:
            try:
                task_func_path, *args = inputs.get()
                # task 함수 import
                task_func = import_component(task_func_path)

                while True:
                    # debug 모드일때 log 출력을 막기 위해 잠시 worker를 대기
                    if GV.debug is False:
                        break
                    else:
                        time.sleep(1)

                process = mp.Process(
                    target=task_func,
                    args=(rank, seed, GV, param_dict, outputs, *args),
                    daemon=False)
                process.start()
                # 게임 종료까지 {self.GV.timelimit}초 대기
                process.join(GV.timelimit)
                if process.is_alive():
                    # 너무 오래 동안 task_func가 실행중이면 강제 종료
                    toolbox.utils.kill_children_processes(process.pid)
                    raise TimeoutError("too long game")
                    
            except Exception as exc:
                outputs.put(Result(success=False, rank=rank, data=exc))
                
            finally:
                seed += random.randint(0, 1000)

    def start(self):
        """
        모든 actors + proxy actor 시작
        """
        [act.start() for act in self.actors]

    def stop(self):
        """
        모든 actors와 actor의 자식 프로세스 모두 종료
        """
        kill_children_processes()

    def input_ready(self):
        """
        새로운 task를 입력 받을 준비가 되었는지 검사

        너무 많은 task가 생성되는 것을 막는 목적
        """
        if self.inputs.qsize() < self.GV.max_pending_tasks:
            return True
        else:
            return False

    def put(self, task):
        """
        task 입력
        """
        self.inputs.put(task)

    def output_ready(self):
        """
        완료된 task가 있는지 검사
        """
        return True if self.outputs.qsize() > 0 else False

    def get(self):
        """
        task 결과 데이터 반환
        """
        return self.outputs.get()

    def GV_dict(self):
        """
        현재 vars의 값을 dict 형태로 반환
        """
        return self.GV._getvalue().__dict__

    def load_GV_dict(self, GV_dict):
        """
        dict로 받은 값으로 vars의 값을 덮어씀
        """
        for k, v in GV_dict.items():
            setattr(self.GV, k, v)

    def load_param_dict(self, param_dict):
        for k, v in param_dict.items():
            self.param_dict[k] = v

    def __repr__(self):
        def to_string(n, max_size=10):
            state = '*' * min(max_size, n)
            state += ' ' * (max_size - min(max_size, n))
            state += ('+' if n > max_size else ' ')
            return state

        # local에 있는 inputs queue에 대기중이 작업들
        inputs_state = to_string(self.inputs.qsize())

        # remote에서 요청하는 작업들
        # learner가 충분히 빠르게 작업을 생성하지 못할 때 증가
        remote_state = to_string(self.GV.remote_idle)

        # outputs에서 대기 중인 결과들
        # 이 결과가 계속 많이 보인다면
        # actor의 결과를 learner가 충분히 빨리 처리하지 못한다는 의미
        outputs_state = to_string(self.outputs.qsize())
        qstate = f'inq: {inputs_state} rq: {remote_state} outq: {outputs_state}'
        GV_dict = self.GV_dict()
        last_resp = {k: v for k, v in self.last_resp.items()}
        param_dict = {k: v for k, v in self.param_dict.items()}

        buff = list()
        buff += [C.blue(f'** {self.name}(ActorGroup): {self.n_actors} actors')]
        buff += [C.blue('=== Queue State ===================')]
        buff += [qstate]
        buff += [C.blue('=== Global Variables ==============')]
        for k, v in GV_dict.items():
            buff += [f'{k:20}: {v}']
        buff += [C.blue('=== Last response time ============')]
        for k, v in last_resp.items():
            buff += [f'{k:10}: {v}']
        buff += [C.blue('=== Param dict ====================')]
        for k, v in param_dict.items():
            buff += [f'{k:5}: {v}']
        buff += [C.blue('===================================')]
        return '\n'.join(buff)


def actor_func_example(rank, seed, GV, param_dict, outputs, *args):
    """
    actor 모듈 테스트용 함수
    """

    import torch
    import numpy as np
    import time
    
    np.random.seed(seed)
    torch.manual_seed(seed)

    for _ in range(2):
        # task 하나를 처리해서 출력 두 개 반환
        try:
            # 인자 가져오기
            h, w, error = args
            # GV에서 모델 업데이트 가능?

            # 작업 실행
            time.sleep(5)
            if error:
                raise Exception(f'something worng, {error}')
            outs = np.random.random((h, w)) + np.random.random((h, w))

            # 결과 출력하기
            outputs.put(Result(success=True, rank=rank, data=outs))

        except Exception as exc:
            outputs.put(Result(success=False, rank=rank, data=exc))


def learner_example(args, logger):

    import torch
    from torch.functional import F

    # 신경망 정의
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(3, 2)
            self.fc2 = torch.nn.Linear(2, 3)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return x

        def inc(self):
            pass

    model = Model()

    # actor 그룹 생성
    actors = ActorGroup(
        name='Actors',
        base_rank=args.base_rank,
        n_actors=args.n_actors,
        timelimit=args.timelimit,
        seed=args.seed,
        max_pending_tasks=args.max_pending_tasks)
    # proxy actor 추가
    actors.add_proxy_actor(args.frontend_port, args.backend_port)
    actors.start()

    # 공유변수 및 모델 정의
    n_tasks = 50
    n_submits = 0
    actors.GV.n_outs = 0
    actors.param_dict[0] = Params(model, 0)

    start_time = time.time()
    logger.info('시작')
    while True:
        if actors.input_ready():
            # 작업 입력, n_submits % 5 == 0 일때는 반드시 예외 발생
            actors.put((args.task_func, 2, 3,
                        True if n_submits % 5 == 0 else False))
            logger.info(C.blue(f'>> task {n_submits} submit'))
            n_submits += 1

        if actors.output_ready():
            # 결과 출력
            out = actors.get()
            logger.info(C.green(f'<< result {actors.GV.n_outs}'))
            logger.debug(f'{out}')
            actors.GV.n_outs += 1

            # train
            for pid, params in actors.param_dict.items():
                actors.param_dict[pid] = Params(model, params.version + 1)
            logger.debug(f'actor state dict\n{actors}')

        if actors.GV.n_outs >= n_tasks:
            # 목표 작업량을 채우면 종료
            break

    logger.info(f'elapsed time {time.time() - start_time}')
    logger.info(f'speedups: {(n_tasks * 5) / (time.time() - start_time)}')

    # 모든 프로세스 종료
    actors.stop()


def remote_actor_example(args):
    proxy_worker.proxy_learner(
        args, args.frontend_port, args.backend_port, verbose=True)


if __name__ == '__main__':

    print('actor 모듈 테스트 코드 시작')
    # actor 기능 테스트
    # 여기 부분이 learner 역할

    # windows에서 컬러 콘솔 출력 문제가 있으면 False로 설정
    C.enable = True

    from toolbox.init.argparse import argument_parser
    from toolbox.init.logging import get_logger

    # 명령행 인자 파서 생성 + 기본 인자 추가
    parser = argument_parser()
    # Task 정의
    parser.add_argument(
        '--task_func',
        type=str,
        default='toolbox.dist.actor.actor_func_example')
    parser.add_argument('--timelimit', type=int, default=15 * 60)
    parser.add_argument('--max_pending_tasks', type=int, default=4)
    args = parser.parse_args()

    if args.LEARNER:
        print(C.notice(f'Learner 시작 {args.session_id}'))
        # logger 초기화
        logger = get_logger(args, tools=True)
        # 프로젝트 백업
        logger.backup_project()
        # learner 예제 실행
        learner_example(args, logger)
    else:
        print(C.notice('Remote actor 시작'))
        remote_actor_example(args)
