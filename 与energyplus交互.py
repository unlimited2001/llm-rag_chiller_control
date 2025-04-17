from controllables.energyplus import (
    System,
    Actuator,
    OutputVariable,
)
from controllables.energyplus.events import Event
from controllables.core import TemporaryUnavailableError
from controllables.core.tools.gymnasium import (
    DictSpace,
    BoxSpace,
    Agent,
)

import gymnasium as _gymnasium_
import numpy as _numpy_
import os
import pandas as pd

# 用于文件路径解析
resolve = lambda *xs: os.path.join(
    os.path.dirname(__file__),
    *xs,
)

class my_env():
    def __init__(self) -> None:
        self.world = world = System(
            building='Large office - 1AV232.idf',
            weather='USA_FL_Miami.722020_TMY2.epw',
            repeat=False,
        ).add('logging:progress')

        # 加载 thermostat 设置（每小时一个值）
        self.thermostat_schedule = pd.read_csv(resolve('rag-chiller.csv'))['extracted_values'].tolist()
        self.step_count = 0
        self.hour_index = 0
        self.current_thermostat = self.thermostat_schedule[0] if self.thermostat_schedule else 6

        # 设置控制环境
        env = Agent(dict(
            action_space=DictSpace({
                'Thermostat': BoxSpace(
                    low=5., high=15.,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(world[Actuator.Ref(
                    type='Schedule:Compact',
                    control_type='Schedule Value',
                    key='chiller-tem',
                )])
            }),
            observation_space=DictSpace({
                'temperature:drybulb': BoxSpace(
                    low=-_numpy_.inf, high=+_numpy_.inf,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(world[OutputVariable.Ref(
                    type='Zone Mean Air Temperature',
                    key='Basement ZN',
                )]),
                't_out': BoxSpace(
                    low=-_numpy_.inf, high=+_numpy_.inf,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(world[OutputVariable.Ref(
                    type='Site Outdoor Air Drybulb Temperature',
                    key='Environment',
                )]),
                'occ': BoxSpace(
                    low=-_numpy_.inf, high=+_numpy_.inf,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(world[OutputVariable.Ref(
                    type='Schedule Value',
                    key='Large Office Bldg Occ',
                )]),
                'light': BoxSpace(
                    low=-_numpy_.inf, high=+_numpy_.inf,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(world[OutputVariable.Ref(
                    type='Schedule Value',
                    key='LARGE OFFICE BLDG LIGHT',
                )]),
                'Equip': BoxSpace(
                    low=-_numpy_.inf, high=+_numpy_.inf,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(world[OutputVariable.Ref(
                    type='Schedule Value',
                    key='Large Office Bldg Equip',
                )]),
                'Energy_1': BoxSpace(
                    low=-_numpy_.inf, high=+_numpy_.inf,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(world[OutputVariable.Ref(
                    type='Cooling Coil Total Cooling Rate',
                    key='VAV_1 Clg Coil',
                )]),
                'Energy_2': BoxSpace(
                    low=-_numpy_.inf, high=+_numpy_.inf,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(world[OutputVariable.Ref(
                    type='Cooling Coil Total Cooling Rate',
                    key='VAV_2 CLG COIL',
                )]),
                'Energy_3': BoxSpace(
                    low=-_numpy_.inf, high=+_numpy_.inf,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(world[OutputVariable.Ref(
                    type='Cooling Coil Total Cooling Rate',
                    key='VAV_3 CLG COIL',
                )]),
                'Energy_4': BoxSpace(
                    low=-_numpy_.inf, high=+_numpy_.inf,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(world[OutputVariable.Ref(
                    type='Cooling Coil Total Cooling Rate',
                    key='VAV_5 CLG COIL',
                )]),
                't_chiller': BoxSpace(
                    low=-_numpy_.inf, high=+_numpy_.inf,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(world[OutputVariable.Ref(
                    type='Chiller Evaporator Outlet Temperature',
                    key='DOE REF 1980-2004 WATERCOOLED  CENTRIFUGAL CHILLER 0 1100TONS 0.7KW/TON',
                )]),
                'flow_chiller': BoxSpace(
                    low=-_numpy_.inf, high=+_numpy_.inf,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(world[OutputVariable.Ref(
                    type='Chiller Evaporator Mass Flow Rate',
                    key='DOE REF 1980-2004 WATERCOOLED  CENTRIFUGAL CHILLER 0 1100TONS 0.7KW/TON',
                )]),
                'power_chiller': BoxSpace(
                    low=-_numpy_.inf, high=+_numpy_.inf,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(world[OutputVariable.Ref(
                    type='Chiller Electricity Rate',
                    key='DOE REF 1980-2004 WATERCOOLED  CENTRIFUGAL CHILLER 0 1100TONS 0.7KW/TON',
                )]),
            }),
        ))

        self.data = []
        self.value = []

        @self.world.on(Event.Ref('end_zone_timestep_after_zone_reporting', include_warmup=False))
        def _(_):
            try:
                # 更新 thermostat：每6步更新一次
                self.step_count += 1
                if self.step_count % 6 == 1:
                    if self.hour_index < len(self.thermostat_schedule):
                        self.current_thermostat = float(self.thermostat_schedule[self.hour_index])
                        self.hour_index += 1

                # 应用当前 thermostat 动作
                env.action.value = {
                    'Thermostat': self.current_thermostat
                }

                # 记录当前时间和观测值
                self.data.append(self.world['wallclock:calendar'].value)
                self.value.append(env.observe())

            except TemporaryUnavailableError:
                pass

    def p(self, b):
        print(b)

# 启动环境并执行模拟
a = my_env()
a.world.start().wait()

# 输出数据为 CSV 文件
out = pd.DataFrame(a.data, columns=['Time'])
out.to_csv(resolve('result-2.csv'), index=False)

out = pd.DataFrame(a.value)
out.to_csv(resolve('rag-result.csv'), index=False)