__author__ = 'Hyunsoo Park (hspark8312@ncsoft.com), Game AI Lab, NCSOFT'


if __name__ == '__main__':

    from ..logger.colorize import Color

    Color.enable = True

    # 학습 환경 초기화 예
    from toolbox.init.argparse import argument_parser
    from toolbox.init.logging import get_logger

    # 명령행 인자 파싱
    parser = argument_parser()
    # parser.add_argument ... 추가 명령행 인자 추가
    args = parser.parse_args()

    # logger 초기화
    logger = get_logger(args, tools=True)

    # 프로젝트 백업
    logger.backup_project()

    # ... 프로젝트 메인 함수 ..

    logger.info('INFO')
    logger.warning('WARNING')
    logger.error('ERROR')
