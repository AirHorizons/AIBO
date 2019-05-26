
__author__ = 'Hyunsoo Park (hspark8312@ncsoft.com), Game AI Lab, NCSOFT'


import logging
import os
from functools import wraps

from ..logger.colorize import Color as C
from ..logger.colorize import ColoredFormatter  
from ..logger.tools import Tools
from ..utils import non_block_func


def get_logger(args, tools=True, colored_msg=False):
    """
    logger를 초기화 하고, 필요하다면 logger_tools도 설정함

    :param args (argparse.Namesapce): 파싱된 args
    """

    logger = logging.getLogger(args.session_id)
    logger.propagate = False  # 상위 로거로 전파 금지
    logger.setLevel(args.log_level)

    # 모든 핸들러 제거
    for handler in logger.handlers:
        logger.removeHandler(handler)

    # 스트림 핸들러(화면에 출력) 추가
    FORMAT = '%(asctime)-15s [%(levelname)-7s] %(message)s'
    formatter = ColoredFormatter(fmt=FORMAT)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if tools:
        # log를 저장할 경로 설정
        out_path = os.path.join(args.out_path, args.session_id)
        if os.path.exists(out_path):
            print(C.error('"이미 경로가 존재함, 출력 결과가 덮어씌워질 수 있으므로 실행 중단함"'))
            exit(1)

        # 디렉토리 생성
        os.makedirs(out_path, exist_ok=True)
        print(C.blue(f'폴더 생성: {out_path}'))

        # 기본 logger에 Tools 객체 추가
        logger._tools = Tools(out_path, args.log_level, args.visdom)
        logger.out_path = out_path
        logger.backup_project = logger._tools.backup_project
        logger.text = non_block_func(logger._tools.text)
        logger.table = non_block_func(logger._tools.table)
        logger.line = non_block_func(logger._tools.line)
        logger.scatter = non_block_func(logger._tools.scatter)
        logger.bar = non_block_func(logger._tools.bar)
        logger.progressbar = non_block_func(logger._tools.progressbar)
        logger.get_data_path = non_block_func(logger._tools.get_data_path)
        logger.save_csv = non_block_func(logger._tools.save_csv)
        logger.save_model = non_block_func(logger._tools.save_model)

    if colored_msg:

        def colored(color_func, original_log_func):
            @wraps(original_log_func)
            def colored_log_func(msg):
                original_log_func(color_func(msg))
            return colored_log_func

        logger.error = colored(C.error, logger.error)
        logger.warning = colored(C.warning, logger.warning)
        logger.info = colored(C.green, logger.info)
        logger.debug = colored(C.blue, logger.debug)

    return logger
