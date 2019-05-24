"""
실험 환경 초기화에 필요한 코드 조각 모음
"""

__author__ = 'Hyunsoo Park (hspark8312@ncsoft.com), Game AI Lab, NCSOFT'

import argparse
import logging
import pprint
import random
import sys
from functools import wraps
from time import gmtime, strftime

import torch

from ..logger.colorize import Color as C


def argument_parser(*args, **kwargs):
    """
    공통적으로 사용하는 arguments를 설정을 여기에 추가함
    """
    parser = argparse.ArgumentParser(*args, **kwargs)
    basic_options = parser.add_argument_group('기본 옵션')
    basic_options.add_argument(
        '--out_path',
        type=str,
        default='../outs',
        help='로그 및 산출물이 기록될 root 경로')
    basic_options.add_argument(
        '--session_id',
        type=str,
        default='NONE',
        help=
        '이 실험의 id, 실험 결과는 {out_path}/{session_id} 경로에 저장, '
        '주의) "_" 문자는 사용 불가능, "-"로 자동으로 교체됨'
        )
    basic_options.add_argument(
        '-U', '--unique_session_id', 
        action='store_true', 
        default=False, 
        help=
        'True일 경우 session_id를 {session_id}_{y-m-d-H-M-S}_{seed} 형태로 변경'
        '(unique_session_id 함수 참조')
    basic_options.add_argument(
        '-v',
        '--verbose',
        action='count',
        default=0,
        help=
        'v의 개수에 따라 로그 수준을 결정함, '
        '-v: INFO, -vv: DEBUG, -vvv: DEBUG + visdom, 기본값은 WARNING')
    basic_options.add_argument(
        '--seed', type=int, default=-1, help='랜덤 시드값 [0, 10,000] 값에서 무작위로 결정')
    basic_options.add_argument('--cuda', type=parse_bool, default=False, help='CUDA 사용여부')
    #
    # 분산처리 관련 인자
    # 
    basic_options.add_argument('--base_rank', type=int, default=0, help='actor의 rank (id) 시작 번호')
    basic_options.add_argument('--n_actors', type=int, default=1, help='이 PC 에서 실행할 actor의 수')
    basic_options.add_argument('--server', type=str, default='', 
        help='Learner 역할을 하는 server의 IP, 이값을 설정하면 actor 모드로 작동하며 서버에 접속한다.')
    basic_options.add_argument('--frontend_port', type=int, default=5559, 
        help='ZMQ Queue frontend 사용 포트, 기본값: 5559')
    basic_options.add_argument('--backend_port', type=int, default=5560, 
        help='ZMQ Queue backend 사용 포트, 기본값: 5560')
    basic_options.add_argument('--sync_interval', type=float, default=1.0, 
        help='Proxy learner가 proxy actor와 GV와 param_dict를 동기화 하는 주기')

    # 명령행으로 지정하지 않는 부분
    basic_options.add_argument('--LEARNER', default=True, 
        help='기본값 True 지만, --server에 서버 주소를 지정하면 False (actor) 모드로 변경됨, '
             '후처리에서 자동으로 설정함')
    basic_options.add_argument('--COMMAND', default='', 
        help='실행 명령어를 기록, 후처리에서 자동으로 설정함')

    # parsing 후에 추가로 처리하고 싶은 작업을 정의
    def parse_args_wrapper(parse_args):

        def wrapped():
            args = parse_args()

            # log level 설정
            log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
            args.log_level = log_levels[min(args.verbose, 2)]
            # visdom 사용여부 설정
            args.visdom = args.verbose > 2

            # 랜덤 시드가 0 이하면 무작위로 설정
            if args.seed < 0:
                args.seed = random.randint(1, 10000)

            if '_' in args.session_id:
                print(C.warning('session_id 안에는 _ 문자를 사용할 수 없음, - 으로 교체'))
                old_session_id, args.session_id = args.session_id, args.session_id.replace('_', '-')
                print(C.warning(f'old session_id: {old_session_id}'))
                print(C.warning(f'new session_id: {args.session_id}'))

            if args.unique_session_id:
                # session_id에 seed 값과 실행 시간 추가
                args.session_id = unique_session_id(args)

            # cuda 사용여부 최종 확인
            if args.cuda is True:
                if torch.cuda.is_available() is False:
                    print(C.error('** CUDA 사용불가능 **'))
                    args.cuda = False

            # LEARNER 모드 설정
            args.LEARNER = True if args.server.strip() == '' else False
            # 옵션을 포함한 실행 명령어 기록
            args.COMMAND = ' '.join(sys.argv)

            # 현재 args 출력
            print(C.header('Args'))
            print(C.blue(pprint.pformat(args.__dict__)))
            return args

        return wrapped

    # parse_args 래핑
    parser.parse_args = parse_args_wrapper(parser.parse_args)
    return parser


# session ID에 seed 값과 시간정보 추가
def unique_session_id(args):
    session_id_tokens = [
        str(args.session_id),
        strftime('%y-%m-%d-%H-%M-%S', gmtime()),
        str(args.seed)
    ]
    return '_'.join(session_id_tokens)


# type parser
def parse_bool(string: str):
    """
    입력받은 문자열을 bool 타입으로 변환
    """
    string = string.lower().strip()
    if string in ('true', 't', '1'):
        return True
    elif string in ('false', 'f', '0'):
        return False
    else:
        raise ValueError("can't parse bool string")


if __name__ == '__main__':

    parser = argument_parser()
    args = parser.parse_args()
