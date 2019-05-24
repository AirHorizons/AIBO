__author__ = 'Hyunsoo Park (hspark8312@ncsoft.com), Game AI Lab, NCSOFT'

import logging
import multiprocessing as mp
import time
from enum import Enum

import zmq

import toolbox

from ..logger.colorize import Color as C
from ..logger.colorize import ColoredFormatter
from . import actor


def proxy_actor_device(frontend_port: int, backend_port: int):
    """
    ZMQ Queue의 device process
    Queue 실행할 때 필요함

    :param int frontend_port: ZMQ Queue의 frontend port
    :param int backend_port: ZMQ Queue의 backend port

    **참고**: https://learning-0mq-with-pyzmq.readthedocs.io/en/latest/pyzmq/devices/queue.html
    """
    try:
        context = zmq.Context(1)
        # Socket facing clients
        frontend = context.socket(zmq.XREP)
        frontend.bind("tcp://*:{}".format(frontend_port))
        # Socket facing services
        backend = context.socket(zmq.XREQ)
        backend.bind("tcp://*:{}".format(backend_port))

        zmq.device(zmq.QUEUE, frontend, backend)
    except Exception as exc:
        import traceback
        print(C.error(exc))
        traceback.print_stack()
        print(C.error("bringing down zmq device"))
    finally:
        frontend.close()
        backend.close()
        context.term()


class ProxyMessage:
    NONE = 0
    REQ_TASK = 1
    REQ_SYNC = 2
    RESULT = 3
    TASK = 4
    SYNC = 5


def proxy_actor(frontend_port: int, backend_port: int, GV, param_dict,
                inputs: mp.Queue, outputs: mp.Queue):
    """
    Proxy Actor

    Learner에서 일반 actor의 역할을 대신하여 task를 소비하고, result를 반환하는 역할을 함
    Proxy actor가 가져간 task는 연결된 proxy learner에게 전달하고, proxy learner에게 
    대신 전달 받은 result를 learner에게 반환함

    Proxy actor는 ZMQ Queue로 구현되었음

    :param int frontend_port: ZMQ Queue의 frontend port
    :param int backend_port: ZMQ Queue의 backend port
    :param mp.Queue inputs: task가 저장된 Queue
    :param mp.Queue outputs: result를 반환할 Queue

    **참고**: https://learning-0mq-with-pyzmq.readthedocs.io/en/latest/pyzmq/devices/queue.html
    """
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.connect("tcp://localhost:{}".format(backend_port))

    GV.remote_idle = 0  # remote actor가 요청한 task 횟수

    while True:
        cmd, payload = socket.recv_pyobj()
        # 여기서 받은 명령을 반드시 아래에서 처리해야 함
        # 처리하지 않을 경우 데드락 발생

        if cmd in (ProxyMessage.REQ_TASK, ProxyMessage.RESULT):
            if cmd is ProxyMessage.RESULT:
                # RESURLT는 outputs로 포워드
                outputs.put(payload)
            elif cmd is ProxyMessage.REQ_TASK:
                # 작업 요청할 경우 remote에서 요청 했다는 기록을 함
                GV.remote_idle += payload.data

            if inputs.qsize() > 0 and GV.remote_idle > 0:
                # inputs에 task가 준비되어 있고, remote가 task를 요청하면,
                # 작업을 전송함
                socket.send_pyobj((ProxyMessage.TASK, inputs.get()))
                # 하나 뿐이라도, task를 보내고 나면 한동안 문제는 없다고 가정하고
                # GV.remote_idle을 0으로 설정함
                GV.remote_idle = 0
                # remote idle 개수만큼 작업을 보내기 시작하면 너무 많은 작업을
                # remote에게 할당하게 됨
                # remote가 필요한 task가 많으면, remote_idle이 높아지겠지만,
                # 절대적인 의미로 해석하면 안됨

            else:
                # 그 외에 경우에는 NONE을 반환
                # 데드락 방지용
                resp = actor.Result(success=True, rank=999, data=0)
                socket.send_pyobj((ProxyMessage.NONE, resp))

        elif cmd is ProxyMessage.REQ_SYNC:
            # 요청한 host의 현재 모델 버전
            remote_param_version = payload.data
            # 현재 param 중에서 원격 host에 전달할 param 선택
            # remote에 없거나, remote의 버전이 낮은 경우
            param_sync = dict()
            for pid, param in param_dict.items():
                if param.version > remote_param_version.get(pid, -1):
                    param_sync[pid] = param

            # GV 동기화
            resp = actor.Result(
                success=True,
                rank=999,
                data=(GV._getvalue().__dict__, param_sync))
            socket.send_pyobj((ProxyMessage.SYNC, resp))

        else:
            raise NotImplementedError


def host_id(ip: str, max_actors_per_host: int = 1000) -> int:
    """
    host의 id를 생성한다.
    ip 주소의 마지막에 * {max_actors_per_host}을 한다.

    actor rank = host id * {max_actors_per_host} + 로컬 pc에서 actor 생성 순서
    """
    return int(ip.rsplit('.', 1)[-1]) * (max_actors_per_host)


def proxy_learner(args, frontend_port: int, backend_port: int, verbose=False):
    """
    Proxy Learner

    원격 머신에서 실행되는 learner의 대체 객체
    Learner가 실행되는 머신의 GV, task 들을 원격 머신과 동기화 시킴 

    :param argsparse.Namespace args: 명령행 인자 옵션
    :param int frontend_port: zmq queue frontend port
    :param int backend_port: zmq queue backend port
    :param bool verbose: true일 경우 logging 메시지 화면에 출력
    """

    # IP의 마지막 숫자 * 100을 현재 PC의 ID로 함
    # 개별 actor의 ID는 HOST_ID + rank
    # (PC 하나에 consts.Actor.max_remote_actors_per_host개 이상의 actor가 있으면 id 중복 발생)
    my_ip = toolbox.utils.get_ip(args.server)

    logger = logging.getLogger(f'proxy_learner-{my_ip}')
    logger.propagate = False  # 상위 로거로 전파 금지
    logger.setLevel(logging.DEBUG if verbose is True else logging.INFO)
    # 스트림 핸들러(화면에 출력) 추가
    FORMAT = '%(asctime)-15s [%(levelname)-7s] %(message)s'
    formatter = ColoredFormatter(fmt=FORMAT)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


    proxy_learner_rank = host_id(my_ip) + 999
    # actor group 시작
    actors = actor.ActorGroup(
        name='Remote',
        base_rank=host_id(my_ip),
        n_actors=args.n_actors,
        timelimit=args.timelimit,
        seed=args.seed,
        max_pending_tasks=args.max_pending_tasks)
    actors.start()

    # zmq client 실행
    context = zmq.Context()

    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{args.server}:{frontend_port}")
    print(C.notice(f"Proxy Actor에 연결: tcp://{args.server}:{frontend_port}"))

    # 마지막으로 동기화를 수행한 시간을 기록
    last_sync_time = time.time()

    while True:
        # 한 interation에서 결과전달, 작업요청, 싱크요청
        # 세 가지 작업을 반드시 한번씩 실행

        if actors.output_ready():
            # 전달할 게임 로그가 있으면 learner에 전달
            socket.send_pyobj((ProxyMessage.RESULT, actors.get()))
            logger.debug(C.blue(f'결과 전송'))

            cmd, payload = socket.recv_pyobj()
            if cmd is ProxyMessage.TASK:
                # task를 받은 경우 작업 추가
                actors.put(payload)
                logger.debug(C.green(f'작업 수신'))
                # logger.debug(f':: state: {actors}')
            elif cmd is ProxyMessage.NONE:
                pass
            else:
                raise NotImplementedError

        if actors.input_ready():
            # 현재 대기중인 작업이 부족하면 proxy_actor에 Sync 1 전달
            socket.send_pyobj((ProxyMessage.REQ_TASK,
                               actor.Result(
                                   success=True,
                                   rank=proxy_learner_rank,
                                   data=1)))
            logger.debug(C.blue(f'작업 요청'))

            cmd, payload = socket.recv_pyobj()
            if cmd is ProxyMessage.TASK:
                # task를 받은 경우 작업 추가
                actors.put(payload)
                logger.debug(C.green(f'작업 수신'))
                # logger.debug(f':: state: {actors}')
            elif cmd is ProxyMessage.NONE:
                pass
            else:
                raise NotImplementedError

        param_version = {
            pid: param.version
            for pid, param in actors.param_dict.items()
        }
        # 불필요한 모델 동기화를 막기위해 현재 모델 버전 전달
        socket.send_pyobj((ProxyMessage.REQ_SYNC,
                            actor.Result(
                                success=True,
                                rank=proxy_learner_rank,
                                data=param_version)))
        logger.debug(C.blue(f'동기화 요청'))

        cmd, payload = socket.recv_pyobj()
        if cmd is ProxyMessage.SYNC:
            # vars 동기화
            gv_dict, param_dict = payload.data
            actors.load_GV_dict(gv_dict)
            logger.debug(C.green(f'sync variables'))
            actors.load_param_dict(param_dict)
            logger.debug(C.green(f'sync params: {param_dict}'))
            logger.debug(f'actor state dict\n{actors}')
            last_sync_time = time.time()
        elif cmd is ProxyMessage.NONE:
            pass
        else:
            raise NotImplementedError

        # 가능하다면 args.sync_interval (초) 마다 sync 를 수행
        # 다른 작업에서 시간이 오래 걸리면 실제 sync_interval은 길어질 수 있음
        elapsed_time = time.time() - last_sync_time
        sleep_time = args.sync_interval - elapsed_time
        if sleep_time > 0:
            logger.info(f'동기화 대기: {sleep_time:.3f} 초')
            time.sleep(sleep_time)
        else:
            logger.warning(f'동기화 시간 초과: {sleep_time:.3f} 초')
        last_sync_time = time.time()
