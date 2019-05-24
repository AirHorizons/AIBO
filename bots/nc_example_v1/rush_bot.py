
__author__ = '박현수 (hspark8312@ncsoft.com), NCSOFT Game AI Lab'

import sc2
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId

from IPython import embed


SEC_PER_STEP = 0.5  # on_step이 호출되는 주기


class RushBot(sc2.BotAI):
    """
    30초마다 모든 유닛을 상대 본진으로 공격하는 예제
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.last_on_step_evoked = 0.0  # 마지막으로 on_step이 호출된 게임 시간
        self.step = 0
        self.actions = list()

    def on_start(self):
        """
        새로운 게임마다 초기화
        """
        self.last_on_step_evoked = 0.0
        self.step = 0
        self.actions = list()

    async def on_step(self, iteration: int):
        """
        iteration은 이번이 몇 번째 step인지를 알려주는 값이지만,
        게임이 realtime으로 실행될 때와 non-realtime으로 실행될 때,
        on_step이 동일한 주기로 실행되지 않기 때문에, 
        의사결정에 직접 사용하기 어렵다.

        non-realtime에서는 게임시간으로 1초마다 on-step이 두 번씩 호출되지만,
        realtime에서는 1초마다 약 110~120번 호출된다.
        따라서, 의사결정에 iteration을 직접 사용할 경우,
        realtime과 non-realtime에서 완전히 다른 행동을 보일 가능성이 높다.

        따라서, iteration을 의사결정에 사용하지 말고, 
        게임시간이나, 게임시간에서 유도된 다른 시간 척도를 사용해야 한다.

        이 봇은 30초마다 모든 유닛을 적 본진으로 보내는 예제이다.
        realtime과 non-realtime에서 똑같은 행동을 보이도록 
        스텝을 의사결정에 사용하고, 스텝을 게임 시간에 맞추기 위해 
        정해진 시간보다 빠르게 on_step이 호출되면, 해당 스텝을 스킵한다. 
        """
        if self.valid_step():
            return list()
        
        actions = list()

        if self.time > 30:  # 30초 이후에 ..
            # 적 본진 위치
            target = self.enemy_start_locations[0]

            # 내 모든 유닛
            army_units = self.units.of_type(
                (UnitTypeId.MARINE, UnitTypeId.MARAUDER, UnitTypeId.REAPER, 
                 UnitTypeId.SIEGETANK, UnitTypeId.MEDIVAC))

            # 모든 유닛들에게 적 본진위치로 공격 명령
            for unit in army_units:
                if unit.type_id == UnitTypeId.MEDIVAC:  # 의료선은 부상당한 유닛 치료
                    wounded_units = army_units.filter(
                        lambda u: u.is_biological and u.health_percentage < 1.0)
                    if wounded_units.amount > 0:
                        actions.append(
                            unit(AbilityId.MEDIVACHEAL_HEAL, wounded_units.closest_to(unit.position)))
                    else:
                        actions.append(unit.attack(army_units.center))
                else:
                    actions.append(unit.attack(target))


        await self.do_actions(actions)

    def valid_step(self):
        """
        너무 빠르게 on_step이 호출되지 않았는지 검사
        """
        if self.time - self.last_on_step_evoked < SEC_PER_STEP:
            return True
        else:
            self.step += 1
            self.last_on_step_evoked = self.time
            return False
