
"""
학습 예제

# $ python -m bots.examples.v7_q_bot train --n_actors=2 --visdom
"""

__author__ = '박현수 (hspark8312@ncsoft.com), NCSOFT Game AI Lab'


import sc2_patch
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from . import train

import numpy as np
import sc2
from sc2.position import Point2, Point3
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.data import Result
from IPython import embed

import random
from enum import Enum

# 주력부대에 속한 유닛 타입
ARMY_TYPES = (UnitTypeId.MARINE, UnitTypeId.MARAUDER, 
    UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED,
    UnitTypeId.MEDIVAC)


class StepManager(object):
    """
    스텝 레이트 유지를 담당하는 매니저
    """
    def __init__(self, bot_ai):
        self.bot = bot_ai
        self.seconds_per_step = 0.35714  # on_step이 호출되는 주기
        self.reset()
    
    def reset(self):
        self.step = -1
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
            # print(C.blue(f'man_step_time: {elapsed_time}'))
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
        self.strategic_points = [
            #        x   y   z
            Point3((28, 60, 12)),  # A 
            Point3((63, 65, 10)),  # B
            Point3((44, 44, 10)),  # C
            Point3((24, 22, 10)),  # D
            Point3((59, 27, 12)),  # E
        ]

        self.region_radius = 10
        self.current_point_idx = 1

    def reset(self):
        # 나와 적의 시작위치 설정
        self.start_location = self.bot.start_location
        self.enemy_start_location = self.bot.enemy_start_locations[0]

    def step(self):
        # 나와 적의 시작위치 설정
        if self.start_location is None:
            self.reset()

    def occupied_points(self):
        """
        지점를 점령하고 있는지 여부를 검사
        """
        units = self.bot.units.of_type(ARMY_TYPES).owned
        enemy = self.bot.known_enemy_units

        occupied = list()
        for point in self.strategic_points:
            n_units = units.closer_than(self.region_radius, point).amount
            n_enemy = enemy.closer_than(self.region_radius, point).amount
            # 해당 위치 근처에 내 유닛이 더 많으면 그 지점을 점령하고 있다고 가정함
            #   (내 유닛 개수 - 적 유닛 개수) > 0 이면 점령 중 
            occupied.append(n_units - n_enemy)
                
        return occupied

    def _map_abstraction(self):
        # 내 시작위치를 기준으로 가까운 지역부터 먼 지역까지 정렬함
        if self.start_location.distance2_to(self.strategic_points[0]) < 3:
            points = self.strategic_points
            occupancy = self.occupied_points()
        else:
            points = list(reversed(self.strategic_points))
            occupancy = list(reversed(self.occupied_points()))
        return points, occupancy

    def frontline(self):
        """
        주력 병력을 투입할 전선을 결정
        """
        points, _ = self._map_abstraction()
        self.current_point_idx = self.bot.strategic_manager.strategy.value
        return points[self.current_point_idx]

    def weak_point(self):
        """
        적 점령지역 중에 방어가 취약한 부분
        """
        points, _ = self._map_abstraction()

        if self.current_point_idx == 4:
            return points[4]
        else:
            return points[3]

    def drop_point(self):
        """
        드랍이 유효한 부분
        """
        points, _ = self._map_abstraction()
        return points[1]
    
    def debug(self):
        """
        지형정보를 게임에서 시각화
        """
        # 각 지역마다, 내가 점령하고 있는지 아닌지 구의 색상으로 시각화
        for occ, point in zip(self.occupied_points(), self.strategic_points):
            color = Point3((255, 0, 0)) if occ > 0 else Point3((0, 0, 255))
            self.bot._client.debug_sphere_out(point, self.region_radius, color)


class Tactics(Enum):
    NORMAL = 0
    REAPER = 1
    DROP = 2


class CombatGroupManager(object):
    """
    개별 유닛에게 직접 명령을 내리는 매니저
    """
    def __init__(self, bot_ai, tactics):
        self.bot = bot_ai
        self.target = None
        self.unit_tags = None
        # 그룹의 경계 범위
        self.perimeter_radious = 10
        self.tactics = tactics
        self.state = ''
    
    def reset(self):
        self.target = self.bot.terrein_manager.strategic_points[2]
        self.unit_tags = (self.bot.units & list()).tags

    def units(self):
        return self.bot.units.filter(lambda unit: unit.tag in self.unit_tags)

    async def step(self):
        actions = list()

        # 이 전투그룹에 속한 아군 유닛들
        units = self.units()

        if units.amount == 0 or self.target is None:
            return actions
        
        # 이 전투그룹 근처의 적군 유닛들
        enemy = self.bot.known_enemy_units.closer_than(
            self.perimeter_radious, units.center)

        for unit in units:
            if self.tactics == Tactics.NORMAL:
                actions += await self.normal_step(unit, units, enemy)
            elif self.tactics == Tactics.REAPER:
                actions += await self.reaper_step(unit, units, enemy)
            elif self.tactics == Tactics.DROP:
                actions += await self.drop_step(unit, units, enemy)

            if self.bot.debug:
                # 모든 유닛의 공격목표롤 시각화
                if len(unit.orders) > 0:
                    skill = unit.orders[0].ability.id.name
                    target = unit.orders[0].target
                    
                    if type(target) is Point3:
                        self.bot._client.debug_line_out(unit.position3d, target)
                        self.bot._client.debug_text_world(skill, target, size=16)

                    elif type(target) is int:
                        all_units = self.bot.units | self.bot.known_enemy_units
                        target_unit = all_units.find_by_tag(target)
                        if target_unit is not None:
                            target = target_unit.position3d
                            self.bot._client.debug_line_out(unit.position3d, target)
                            self.bot._client.debug_text_world(skill, target, size=16)

        if self.bot.debug:
            # 전투그룹의 중심점과 목표지점 시각화
            p1 = units.center
            p2 = self.target.to3
            self.bot._client.debug_sphere_out(
                Point3((p1.x, p1.y, 12)), 5, Point3((0, 255, 0)))
            self.bot._client.debug_line_out(
                Point3((p1.x, p1.y, 12)), p2, color=Point3((0, 255, 0)))
            self.bot._client.debug_sphere_out(
                self.target.to3, 5, Point3((0, 255, 0)))

            # 모든 적 유닛 시각화
            for nmy in self.bot.known_enemy_units:
                if nmy.is_structure:
                    self.bot._client.debug_sphere_out(nmy.position3d, 2, Point3((0, 0, 255)))
                else:
                    self.bot._client.debug_sphere_out(nmy.position3d, 1, Point3((0, 0, 255)))

        return actions

    async def normal_step(self, unit, friends, foes):
        actions = list()

        if unit.type_id == UnitTypeId.MARINE:
            if foes.amount > 0:
                if unit.health_percentage > 0.8 and \
                    not unit.has_buff(BuffId.STIMPACK):
                    # 스팀팩 사용
                    order = unit(AbilityId.EFFECT_STIM)
                else:
                    # 가장 가까운 목표 공격
                    order = unit.attack(foes.closest_to(unit.position))
                actions.append(order)
            else:
                if unit.distance_to(self.target) > 5:
                    # 어택땅으로 집결지로 이동
                    actions.append(unit.attack(self.target.to2))

        elif unit.type_id == UnitTypeId.MARAUDER:
            if foes.amount > 0:
                if unit.health_percentage > 0.8 and \
                    not unit.has_buff(BuffId.STIMPACKMARAUDER):
                    # 스팀팩 사용
                    order = unit(AbilityId.EFFECT_STIM_MARAUDER)
                else:
                    # 가장 가까운 목표 공격
                    order = unit.attack(foes.closest_to(unit.position))
                actions.append(order)
            else:
                if unit.distance_to(self.target) > 5:
                    # 어택땅으로 집결지로 이동
                    actions.append(unit.attack(self.target.to2))

        elif unit.type_id == UnitTypeId.SIEGETANK:
            if foes.amount > 0:
                # 근처에 적이 3이상 있으면 시즈모드
                targets = self.bot.known_enemy_units.closer_than(7, friends.center)
                if targets.amount > 3:
                    if len(unit.orders) == 0 or \
                        len(unit.orders) > 0 and unit.orders[0].ability.id not in (AbilityId.SIEGEMODE_SIEGEMODE, AbilityId.UNSIEGE_UNSIEGE):
                        order = unit(AbilityId.SIEGEMODE_SIEGEMODE)
                        actions.append(order)
                else:
                    order = unit.attack(foes.closest_to(unit.position))
                    actions.append(order)
            else:
                if unit.distance_to(self.target) > 5:
                    # 어택땅으로 집결지로 이동
                    order = unit.attack(self.target.to2)
                    actions.append(order)
                else:
                    # 대기할 때는 시즈모드로
                    if len(unit.orders) == 0 or \
                        len(unit.orders) > 0 and unit.orders[0].ability.id not in (AbilityId.SIEGEMODE_SIEGEMODE, AbilityId.UNSIEGE_UNSIEGE):
                        order = unit(AbilityId.SIEGEMODE_SIEGEMODE)
                        actions.append(order)

        elif unit.type_id == UnitTypeId.SIEGETANKSIEGED:
            # 목표지점에서 너무 멀리 떨어져 있으면 시즈모드 해제
            if unit.distance_to(self.target.to2) > 10:
                if len(unit.orders) == 0 or \
                    len(unit.orders) > 0 and unit.orders[0].ability.id not in (AbilityId.SIEGEMODE_SIEGEMODE, AbilityId.UNSIEGE_UNSIEGE):
                    order = unit(AbilityId.UNSIEGE_UNSIEGE)
                    actions.append(order)

        elif unit.type_id == UnitTypeId.MEDIVAC:
            if unit.distance_to(friends.center) > 5:
                actions.append(unit.attack(friends.center))

        else:
            raise NotImplementedError

        return actions

    async def reaper_step(self, unit, friends, foes):
        actions = list()

        if unit.type_id == UnitTypeId.REAPER:

            threaten = self.bot.known_enemy_units.closer_than(
                    self.perimeter_radious, unit.position)

            if unit.health_percentage > 0.8 and unit.energy >= 50:

                if threaten.amount > 0:
                    if unit.orders and unit.orders[0].ability.id != AbilityId.BUILDAUTOTURRET_AUTOTURRET:
                        closest_threat = threaten.closest_to(unit.position)
                        pos = unit.position.towards(closest_threat.position, 5)
                        pos = await self.bot.find_placement(
                            UnitTypeId.AUTOTURRET, pos)
                        order = unit(AbilityId.BUILDAUTOTURRET_AUTOTURRET, pos)
                        actions.append(order)
                else:
                    if unit.distance_to(self.target) > 5:
                        order = unit.move(self.target)
                        actions.append(order)

            else:
                if unit.distance_to(self.bot.terrein_manager.start_location) > 5:
                    order = unit.move(self.bot.terrein_manager.start_location)
                    actions.append(order)

        return actions

    async def drop_step(self, unit, friends, foes):

        actions = list()

        medivac = friends.of_type(UnitTypeId.MEDIVAC)

        if unit.type_id == UnitTypeId.MARAUDER:
            if foes.amount > 0:
                if unit.health_percentage > 0.8 and \
                    not unit.has_buff(BuffId.STIMPACK):
                    # 스팀팩 사용
                    order = unit(AbilityId.EFFECT_STIM)
                else:
                    # 가장 가까운 목표 공격
                    order = unit.attack(foes.closest_to(unit.position))
                actions.append(order)
            else:
                if medivac.exists:
                    # 어택땅으로 집결지로 이동
                    if unit.distance_to(medivac.center) > 5:
                        actions.append(unit.attack(medivac.center))
                else:
                    if unit.distance_to(self.target) > 5:
                        actions.append(unit.attack(self.target))

        elif unit.type_id == UnitTypeId.MEDIVAC:

            if self.state == 'ready':
                if unit.distance_to(self.target) > 5:
                    actions.append(unit.move(self.target))
                elif len(unit.passengers) < 1:
                    if len(unit.orders) == 0:
                        assult_units = self.bot.units.filter(lambda u: u.tag in self.unit_tags).of_type(UnitTypeId.MARAUDER)
                        if assult_units.amount > 0:
                            order = unit(AbilityId.LOAD, assult_units.first)
                            actions.append(order)
                else:
                    self.state = 'go'

            elif self.state == 'go':
                if unit.distance_to(self.bot.terrein_manager.enemy_start_location) > 11:
                    actions.append(unit.move(self.bot.terrein_manager.enemy_start_location))
                elif foes.amount > 3:
                    self.state = 'fallback'
                else:
                    actions.append(unit.stop())
                    self.state = 'combat'

            elif self.state == 'combat':
                if len(unit.passengers) > 0:
                    order = unit(AbilityId.UNLOADALLAT, unit.position)
                    actions.append(order)

                if foes.amount > 3 and len(friends.filter(lambda u: u.health_percentage < 0.7)) > 0:
                    self.state = 'fallback'

                if friends.filter(lambda u: u.distance_to(unit.position) < 5).of_type(UnitTypeId.MARAUDER) == 0:
                    self.state = 'fallback'

                if unit.distance_to(self.bot.terrein_manager.enemy_start_location) > 11:
                    self.state = 'fallback'

            elif self.state == 'fallback':
                assult_units = self.bot.units.filter(lambda u: u.tag in self.unit_tags).of_type(UnitTypeId.MARAUDER)
                assult_units = assult_units.filter(lambda u: u.distance_to(unit.position) < 5)
                if len(assult_units) > 0:
                    order = unit(AbilityId.LOAD, assult_units.first)
                    actions.append(order)
                else:
                    if unit.distance_to(self.target) > 5:
                        actions.append(unit.move(self.target))
                    else:
                        self.state = 'ready'

            else:
                self.state = 'ready'

        return actions

    def debug(self):
        text = [

            f'Tactics: {self.tactics}, state: {self.state}',
        ]
        self.bot._client.debug_text_screen(
            '\n\n'.join(text), pos=(0.02, 0.14), size=10)


class Strategy(Enum):
    """
    Bot이 선택할 수 있는 전략
    Strategy Manager는 언제나 이 중에 한가지 상태를 유지하고 있어야 함
    """
    P0 = 0
    P1 = 1
    P2 = 2
    P3 = 3
    P4 = 4


class StrategicManager(object):
    """
    Bot의 전략을 결정하는 매니저
    """
    def __init__(self, bot_ai):
        self.bot = bot_ai
        self.strategy = Strategy.P1
        self.win_pred = 0.5
        self.prev_strategy = np.zeros(5, dtype=np.float32)
        self.buffer = list()
        self.score = 0.0

    def reset(self):
        self.strategy = Strategy.P1
        self.buffer.clear()
        self.score = 0.0

    def step(self):
        # 입력 데이터 준비
        observation = self.bot.feature_manager.observation.copy()
        game_state = self.bot.feature_manager.state.copy()
        self.prev_strategy = np.zeros(5, dtype=np.float32)
        self.prev_strategy[self.strategy.value] = 1.
        prev_strategy_value = self.strategy.value
        state = np.concatenate((game_state, self.prev_strategy))

        # 목표지점 결정
        inputs = (observation, state)
        action, self.win_pred = self.bot.model.act(inputs, self.bot.epsilon)
        # action = int(input('?'))
        self.strategy = Strategy(action)

        # 학습 데이터 저장
        reward = 0.0
        reward += 0.1 * self.bot.vespene 
        # reward += 0.001 * self.bot.units.not_structure.amount
        # if self.bot.step_manager.step < 10:
        #     if self.bot.terrein_manager.current_point_idx >= 3:
        #         reward -= 0.05
        if prev_strategy_value == self.strategy.value:
            reward += 0.05

        if self.bot.step_manager.step < 200:
            if self.strategy.value < 3:
                reward += 0.1
            else:
                reward -= 0.1

        self.buffer.append([
            observation, 
            state, 
            action, 
            reward])

        self.score += reward

    def debug(self):
        text = [
            f'Strategy: {self.strategy}',
            f'Supply used: {self.bot.supply_used / (self.bot.supply_cap + 0.01):1.3f}'
        ]
        self.bot._client.debug_text_screen(
            '\n\n'.join(text), pos=(0.02, 0.02), size=10)


class AssignManager(object):
    """
    유닛을 부대에 배치하는 매니저
    """
    def __init__(self, bot_ai, *args, **kwargs):
        self.bot = bot_ai

    def reset(self):
        pass

    def assign(self, manager):

        units = self.bot.units

        if manager.tactics is Tactics.NORMAL:
            units = self.bot.units.of_type(ARMY_TYPES).owned
            unit_tags = units.tags
            # drop이나 reaper에서 사용중인 유닛들은 제외
            unit_tags = unit_tags - self.bot.reaper_manager.unit_tags
            unit_tags = unit_tags - self.bot.drop_manager.unit_tags
            manager.unit_tags = unit_tags

        elif manager.tactics is Tactics.REAPER:
            units = self.bot.units(UnitTypeId.REAPER).owned
            unit_tags = units.tags
            # drop이나 combat에서 사용중인 유닛들은 제외
            unit_tags = unit_tags - self.bot.combat_manager.unit_tags
            unit_tags = unit_tags - self.bot.drop_manager.unit_tags
            manager.unit_tags = unit_tags

        elif manager.tactics is Tactics.DROP:
            # 현재 지도에 존재하는 그룹 유닛
            group_units_tags = self.bot.units.tags & manager.unit_tags
            group_units = self.bot.units.filter(lambda u: u.tag in group_units_tags)
            passengers = list()
            for unit in group_units:
                for passenger in unit.passengers:
                    passengers.append(passenger)
            # 탑승 유닛까지 포함
            group_units = group_units | passengers

            # 새 유닛 요청            
            new_units = self.bot.units & list()

            medivacs = self.bot.units(UnitTypeId.MEDIVAC).owned
            if group_units.of_type(UnitTypeId.MEDIVAC).amount == 0 and medivacs.amount > 1:
                new_units = new_units | medivacs.tags_not_in(group_units.tags).take(1)

            marauders = self.bot.units(UnitTypeId.MARAUDER).owned
            if group_units.of_type(UnitTypeId.MARAUDER).amount == 0 and marauders.amount > 1:
                new_units = new_units | marauders.tags_not_in(group_units.tags).take(1)
            
            unit_tags = (group_units | new_units).tags
            
            # 다른 매니저가 사용 중인 유닛 제외
            # unit_tags = unit_tags - self.bot.combat_manager.unit_tags
            unit_tags = unit_tags - self.bot.reaper_manager.unit_tags
            manager.unit_tags = unit_tags

        else:
            raise NotImplementedError


class FeatureManager(object):
    """
    인공신경망에 입력으로 사용하는 특징을 추출하는 매니저
    """
    def __init__(self, bot_ai):
        self.bot = bot_ai

        # simple 64 맵을 바둑판으로..
        pos = (70.4, 57.2, 44, 30.8, 17.6)
        self.positions = list()
        for y in pos:
            row = list()
            for x in reversed(pos):
                row.append(Point2((x, y)))
            self.positions.append(row)

        # 타일 간격
        self.radious = 13.2 

        self.observation = np.zeros((12, 5, 5), dtype=np.float32)
        self._debug_observation = np.zeros((12, 5, 5), dtype=np.float32)
        # ch0: 건물: 0.1 * #units
        # ch1: 해병: 0.1 * #units
        # ch2: 불곰: 0.1 * #units
        # ch3: 의료선: 0.1 * #units
        # ch4: 공성전차: 0.1 * #units
        # ch5: 사신: 0.1 * #units
        # 나머지 채널은 적 유닛

        self.state = np.zeros(4, dtype=np.float32)
        # c1: 시간
        # c2: 시작지점 1
        # c3: 시작지점 2 
        # c4: 적 유닛 파괴 점수

    def reset(self):
        self.observation.fill(0)
        self._debug_observation.fill(0)
        self.state.fill(0)

    def step(self):
        self.observation.fill(0)
        self._debug_observation.fill(0)
        self.state.fill(0)
       
        # observation
        for unit in self.bot.units:
            for y in range(5):
                for x in range(5):
                    if unit.distance_to(self.positions[y][x]) < self.radious:
                        if unit.is_structure:
                            self.observation[0, y, x] += 0.1
                        elif unit.type_id == UnitTypeId.MARINE:
                            self.observation[1, y, x] += 0.1
                        elif unit.type_id == UnitTypeId.MARAUDER:
                            self.observation[2, y, x] += 0.1
                        elif unit.type_id == UnitTypeId.MEDIVAC:
                            self.observation[3, y, x] += 0.1
                        elif unit.type_id == UnitTypeId.SIEGETANK:
                            self.observation[4, y, x] += 0.1
                        elif unit.type_id == UnitTypeId.REAPER:
                            self.observation[5, y, x] += 0.1

        for unit in self.bot.known_enemy_units:
            for y in range(5):
                for x in range(5):
                    if unit.distance_to(self.positions[y][x]) < self.radious:
                        if unit.is_structure:
                            self.observation[6, y, x] += 0.1
                        elif unit.type_id == UnitTypeId.MARINE:
                            self.observation[7, y, x] += 0.1
                        elif unit.type_id == UnitTypeId.MARAUDER:
                            self.observation[8, y, x] += 0.1
                        elif unit.type_id == UnitTypeId.MEDIVAC:
                            self.observation[9, y, x] += 0.1
                        elif unit.type_id == UnitTypeId.SIEGETANK:
                            self.observation[10, y, x] += 0.1
                        elif unit.type_id == UnitTypeId.REAPER:
                            self.observation[11, y, x] += 0.1

        self._debug_observation = self.observation[:]

        # state
        self.state[0] = min(1.0, self.bot.step_manager.step / 1000.)
        self.state[1] = min(1.0, self.bot.supply_used / 250.)
        self.state[2] = min(1.0, self.bot.state.score.killed_value_units / 4500.)

        # self.state[0] = self.bot.step_manager.step / 1000
        
        if self.bot.start_location.x < self.bot.enemy_start_locations[0].x:
            self.observation = self.observation[:, ::-1, ::-1].copy()
        #     self.state[1] = 1
        #     self.state[2] = 0
        else:
            self.observation = self.observation.copy()
        #     self.state[1] = 0
        #     self.state[2] = 1
            
        # self.state[3] = self.bot.state.score.killed_value_units / 4500.

    def debug(self):

        values = list()
        for y in range(5):
            for x in range(5):
                value = self.observation[:6, y, x].sum() - self.observation[6:, y, x].sum()
                values.append((y, x, value))

        min_value = min([v for _, _, v in values])
        max_value = max([v for _, _, v in values])

        for y, x, value in values:
            pos = self.positions[y][x]
            pos = Point3((pos.x, pos.y, 12))

            norm_value = (value - min_value) / (max_value - min_value)
            color = Point3((int(255 * norm_value), 0, int(255 * (1 - norm_value))))
            self.bot._client.debug_sphere_out(pos, 0.5, color)


            observation = self._debug_observation
            text = [
                f'st: {observation[0, y, x]:0.2f} / {observation[6, y, x]:0.2f}',
                f'm1: {observation[1, y, x]:0.2f} / {observation[7, y, x]:0.2f}',
                f'm2: {observation[2, y, x]:0.2f} / {observation[8, y, x]:0.2f}',
                f'mv: {observation[3, y, x]:0.2f} / {observation[9, y, x]:0.2f}',
                f'st: {observation[4, y, x]:0.2f} / {observation[10, y, x]:0.2f}',
                f'rp: {observation[5, y, x]:0.2f} / {observation[11, y, x]:0.2f}',
                f'time: {self.state[0]:0.2f}',
                f'start_pos_1: {self.state[1]:0.2f}',
                f'start_pos_2: {self.state[2]:0.2f}',
                f'unit_score: {self.state[3]:0.2f}',
            ]
            text = '\n'.join(text)
            self.bot._client.debug_text_world(text, pos, size=10)


class QBot(sc2.BotAI):
    """
    다섯 개의 중요지점 중 어느 곳으로 병력을 보내야 하는지를 학습하는 
    간단한 강화학습 봇
    """
    def __init__(self, debug=False, *args, **kwargs):
        super().__init__()
        self.debug = debug
        self.rank = kwargs.get('rank', -1)
        self.epsilon = kwargs.get('epsilon', 0.0)
        self.shared_model = kwargs.get('shared_model', None)
        self.out_queue = kwargs.get('out_queue', None)

        self.model = train.Model(self)
        self.train_mode = kwargs.get('train', False)

        self.policy_optimizer = None
        self.input_queue = None
        self.output_queue = None

        self.step_manager = StepManager(self)
        self.terrein_manager = TerreinManager(self)
        self.combat_manager = CombatGroupManager(self, Tactics.NORMAL)
        self.reaper_manager = CombatGroupManager(self, Tactics.REAPER)
        self.drop_manager = CombatGroupManager(self, Tactics.DROP)
        self.assign_manager = AssignManager(self)
        self.feature_manager = FeatureManager(self)
        self.strategic_manager = StrategicManager(self)

    def on_start(self):
        if self.shared_model:
            self.model.load_state_dict(self.shared_model.state_dict())
            print('**** actor model synced *****')

        if not self.train_mode:
            import torch
            model_path = 'bots/nc_example_v7/models/model-89.pt'
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path))
                print(f'load model: {model_path}')
            else:
                print(f'model file does not exists: {model_path}')
        
        self.step_manager.reset()
        self.feature_manager.reset()
        self.strategic_manager.reset()
        self.assign_manager.reset()
        self.terrein_manager.reset()
        self.combat_manager.reset()
        self.reaper_manager.reset()
        self.drop_manager.reset()

    async def on_step(self, iteration: int):
        """
        매니저 단위로 작업을 분리하여 보다 간단하게 on_step을 구현
        """
        if self.step_manager.invalid_step():
            return list()

        if self.step_manager.step % 40 == 0:
            # 특징 추출
            self.feature_manager.step()

            # 전략 변경
            self.strategic_manager.step()

            # 지형정보 분석
            self.terrein_manager.step()

            # 부대 구성 변경
            self.assign_manager.assign(self.reaper_manager)
            self.assign_manager.assign(self.drop_manager)
            self.assign_manager.assign(self.combat_manager)

            # 새로운 공격지점 결정
            self.combat_manager.target = self.terrein_manager.frontline()
            self.reaper_manager.target = self.terrein_manager.weak_point()
            self.drop_manager.target = self.terrein_manager.drop_point()

        actions = list()
        actions += await self.combat_manager.step()
        actions += await self.reaper_manager.step()
        actions += await self.drop_manager.step()
        await self.do_actions(actions)

        if self.debug:
            # 현재 추출한 특징 시각화
            self.feature_manager.debug()
            # 현재 전략 게임화면에 시각화
            self.strategic_manager.debug()
            # 지형정보를 게임 화면에 시각화
            self.terrein_manager.debug()

            self.drop_manager.debug()
            await self._client.send_debug()

    def on_end(self, game_result: Result):
        
        if game_result == Result.Victory:
            win_reward = 1.0
        elif game_result == Result.Defeat:
            win_reward = -1.0
        else:
            win_reward = 0.0

        buffer = list()
        for idx in range(len(self.strategic_manager.buffer) - 1):
            ob1 = self.strategic_manager.buffer[idx][0]
            s1 = self.strategic_manager.buffer[idx][1]
            a = self.strategic_manager.buffer[idx][2]
            r = self.strategic_manager.buffer[idx][3]
            ob2 = self.strategic_manager.buffer[idx + 1][0]
            s2 = self.strategic_manager.buffer[idx + 1][1]
            done = 0.0
            buffer.append([ob1, s1, a, r, ob2, s2, done, win_reward])
        
        done = 1.0
        buffer.append([ob1, s1, a, win_reward, ob2, s2, done, win_reward])

        result = dict(success=True, win=win_reward, epsilon=self.epsilon, data=buffer)

        if self.out_queue:
            self.out_queue.put(result)
