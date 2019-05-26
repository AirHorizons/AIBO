

__author__ = '박현수 (hspark8312@ncsoft.com), NCSOFT Game AI Lab'


def kill_starcraft_ii_processes():
    """
    실행되고 있는 모든 스타크래프트 게임 종료
    """
    import platform
    import os

    if platform.platform().lower().startswith('windows'):
        os.system('taskkill /f /im SC2_x64.exe')
    else:
        os.system('pkill -f SC2_x64')


def parse_race(race):
    """
    종족을 나타내는 문자열을 
    python-sc2 Race enum 타입으로 변경
    """
    from sc2 import Race

    if race.lower() == 'terran':
        return Race.Terran
    elif race.lower() == 'protoss':
        return Race.Protoss
    elif race.lower() == 'zerg':
        return Race.Zerg
    else:
        return Race.Random
