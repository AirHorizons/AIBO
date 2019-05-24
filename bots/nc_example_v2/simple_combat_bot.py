
"""
매니저 단위로 작업을 분담하여 처리하는 예제
"""

__author__ = '박현수 (hspark8312@ncsoft.com), NCSOFT Game AI Lab'

import sc2_patch

import random

import sc2
from sc2.position import Point2, Point3
from sc2.ids.unit_typeid import UnitTypeId
from toolbox.logger.colorize import Color as C

from IPython import embed


class StepManager(object):
    """
    스텝 레이트 유지를 담당하는 매니저
    """
    def __init__(self, bot_ai):
        self.bot = bot_ai
        self.seconds_per_step = 0.35714  # on_step이 호출되는 주기
        self.reset()
    
    def reset(self):
        self.step = 0
        # 마지막으로 on_step이 호출된 게임 시간
        self.last_game_time_step_evoked = 0.0 

    def invalid_step(self):
        """
        너무 빠르게 on_step이 호출되지 않았는지 검사
        """
        elapsed_time = self.bot.time - self.last_game_time_step_evoked
        if elapsed_time < self.seconds_per_step:
            return True
        else:
            print(C.blue(f'man_step_time: {elapsed_time}'))
            self.step += 1
            self.last_game_time_step_evoked = self.bot.time
            return False


class TerreinManager(object):
    """
    간단한 지형정보를 다루는 매니저
    """
    def __init__(self, bot_ai):
        self.bot = bot_ai
        # 자원이 생산되는 주요 전략적 요충지 좌표
        #  A     B
        #     C
        #  D     E
        self.strategic_point = [
            #        x   y
            Point3((28, 60)),  # A 
            Point3((63, 65)),  # B
            Point3((44, 44)),  # C
            Point3((24, 22)),  # D
            Point3((59, 27)),  # E
        ]

    def reset(self):
        # 나와 적의 시작위치
        self.start_location = self.bot.start_location
        self.enemy_start_location = self.bot.enemy_start_locations[0]

    def frontline(self):
        """
        다음 공격지점 결정
        - 보다 상위의 전략적인 판단에 의해 결정될 필요가 있음
        - 일단 무작위로 다섯 지점 중 하나를 반환하도록 구현
        """
        return random.choice(self.strategic_point)


class CombatManager(object):
    """
    개별 유닛에게 직접 명령을 내리는 매니저
    """
    def __init__(self, bot_ai):
        self.bot = bot_ai
        self.target = None
    
    def reset(self):
        self.target = self.bot.terrein_manager.frontline()

    def step(self):
        unit_types = (UnitTypeId.MARINE, UnitTypeId.MARAUDER, 
            UnitTypeId.SIEGETANK, UnitTypeId.MEDIVAC, UnitTypeId.REAPER)

        actions = list()
        for unit in self.bot.units.of_type(unit_types):
            actions.append(unit.attack(self.target.to2))
            if self.bot.debug:
                p1 = unit.position3d
                p2 = Point2((self.target.x, self.target.y))
                height = self.bot.get_terrain_height(p2) // 10
                p2 = Point3((*p2, height))
                self.bot._client.debug_line_out(p1, p2)

        return actions


class SimpleCombatBot(sc2.BotAI):
    """
    30초마다 모든 유닛을 상대 본진으로 공격하는 예제
    """
    def __init__(self, debug=False, *args, **kwargs):
        super().__init__()
        self.debug = debug
        self.step_manager = StepManager(self)
        self.terrein_manager = TerreinManager(self)
        self.combat_manager = CombatManager(self)

    def on_start(self):
        self.last_time = self.time
        self.step_manager.reset()
        self.terrein_manager.reset()
        self.combat_manager.reset()

    async def on_step(self, iteration: int):
        """
        매니저 단위로 작업을 분리하여 보다 간단하게 on_step을 구현
        """
        elapsed_time = self.time - self.last_time
        self.last_time = self.time
        print(C.green(f'raw_step_time: {elapsed_time}'))

        if self.step_manager.invalid_step():
            return list()

        if self.step_manager.step % 60 == 0:
            # 60 스텝마다 새로운 공격지점 결정
            self.terrein_manager.reset()
            self.combat_manager.reset()

        actions = self.combat_manager.step()
        await self.do_actions(actions)

        if self.debug:
            await self._client.send_debug()


if __name__ == '__main__':

    import os
    import argparse

    import sc2
    from sc2 import Difficulty, Race, maps, run_game
    from sc2.data import Result
    from sc2.player import Bot, Computer

    C.enable = True

    # 봇 테스트용
    parser = argparse.ArgumentParser('Dev')
    # 게임 정보
    parser.add_argument(
        '--map_name',
        type=str,
        default='sc2_data/maps/NCFellowship-2019_m1_v2',
        help='경진대회 기본 맵')
    # 옵션
    parser.add_argument(
        '--realtime',
        action='store_true',
        default=False,
        help='false일 때는 빠르게 게임이 실행됨')
    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help='bot을 생성할 때, debug 인자로 전달함')
    args = parser.parse_args()

    # map 정보 읽기
    try:
        game_map = maps.get(args.map_name)
    except (KeyError, FileNotFoundError):
        assert os.path.exists(
            args.map_name +'.SC2Map'), f"지도 파일을 찾을 수 없음!: {args.map_name}"
        game_map = args.map_name

    bots = [Bot(Race.Terran, SimpleCombatBot(debug=args.debug)), Computer(Race.Terran, Difficulty(3))]
    result = sc2.run_game(game_map, bots, realtime=args.realtime)
    print(result)
