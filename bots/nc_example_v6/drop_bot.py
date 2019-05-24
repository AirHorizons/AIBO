
"""
보조부대 컨트롤 예제
"""

__author__ = '박현수 (hspark8312@ncsoft.com), NCSOFT Game AI Lab'


import random
from enum import Enum

import sc2
from sc2.position import Point2, Point3
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId

from IPython import embed


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
        self.current_point_idx = 2

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
        points, occupancy = self._map_abstraction()

        if self.bot.strategic_manager.strategy == Strategy.ATTACK:
            # 공격 전략일 때는 전선을 전진
            if occupancy[self.current_point_idx] > 0:
                self.current_point_idx += 1
                self.current_point_idx = min(4, self.current_point_idx)

        elif self.bot.strategic_manager.strategy == Strategy.HOLD:
            # 방어 전략일 때는 전선을 후퇴
            if occupancy[self.current_point_idx] < 0:
                self.current_point_idx -= 1
                self.current_point_idx = max(1, self.current_point_idx)

        # 어떤 조건도 만족시키지 않으면, 중앙에 유닛 배치
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
        self.strategy = None
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
                    # if unit.type_id == UnitTypeId.MARINE:
                    #     embed(); exit()
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

                if friends.filter(lambda u: u.distance_to(unit) < 5).of_type(UnitTypeId.MARAUDER) == 0:
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
    NONE = 0
    ATTACK = 1
    HOLD = 2


class StrategicManager(object):
    """
    Bot의 전략을 결정하는 매니저
    """
    def __init__(self, bot_ai):
        self.bot = bot_ai
        self.strategy = Strategy.NONE

    def reset(self):
        self.strategy = Strategy.HOLD

    def step(self):
        if self.bot.supply_cap > 25:
            if self.bot.supply_used / (self.bot.supply_cap + 0.01) > 0.5:
                # 최대 보급량이 25이상이고, 
                # 최대 보급의 50% 이상을 사용했다면 병력이 준비된 것으로 판단
                # 공격전략 선택
                self.strategy = Strategy.ATTACK

            else:
            # else self.bot.supply_used / self.bot.supply_cap < 0.3:
            #     # 최대 보급의 0.3 밑이면 전선유지 전략 선택
                self.strategy = Strategy.HOLD
        else:
            self.strategy = Strategy.HOLD

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


class DropBot(sc2.BotAI):
    """
    병력이 준비되면 적 본직으로 조금씩 접근하는 가장 간단한 전략을 사용하는 봇
    """
    def __init__(self, debug=False, *args, **kwargs):
        super().__init__()
        self.debug = debug
        self.step_manager = StepManager(self)
        self.terrein_manager = TerreinManager(self)
        self.combat_manager = CombatGroupManager(self, Tactics.NORMAL)
        self.reaper_manager = CombatGroupManager(self, Tactics.REAPER)
        self.drop_manager = CombatGroupManager(self, Tactics.DROP)
        self.assign_manager = AssignManager(self)
        self.strategic_manager = StrategicManager(self)

    def on_start(self):
        self.step_manager.reset()
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

        if self.step_manager.step % 30 == 0:
            # 전략 변경
            self.strategic_manager.step()

            # 지형정보 분석
            self.terrein_manager.step()

            # 부대 구성 변경
            # self.assign_manager.step()
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
            # 현재 전략 게임화면에 시각화
            self.strategic_manager.debug()
            # 지형정보를 게임 화면에 시각화
            self.terrein_manager.debug()

            self.drop_manager.debug()
            await self._client.send_debug()
