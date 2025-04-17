import sys
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

resolve = lambda *xs: os.path.join(
    os.path.dirname(__file__),
    *xs,
)


class my_env():
    def __init__(self, thermostat_value) -> None:
        self.world = world = System(
            building='Large office - 1AV232.idf',
            weather='USA_FL_Miami.722020_TMY2.epw',
            repeat=False,
        ).add('logging:progress')

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
                't_chiller': BoxSpace(
                    low=-_numpy_.inf, high=+_numpy_.inf,
                    dtype=_numpy_.float32,
                    shape=(),
                ).bind(world[OutputVariable.Ref(
                    type='Chiller Evaporator Outlet Temperature',
                    key='DOE REF 1980-2004 WATERCOOLED  CENTRIFUGAL CHILLER 0 1100TONS 0.7KW/TON',
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
        self.thermostat_value = thermostat_value

        @self.world.on(Event.Ref('end_zone_timestep_after_zone_reporting', include_warmup=False))
        def _(_):
            try:
                self.data.append(self.world['wallclock:calendar'].value)
                self.value.append(env.observe())
                env.action.value = {
                    'Thermostat': self.thermostat_value,
                }
            except TemporaryUnavailableError:
                pass

    def p(self, b):
        print(b)


class ProgressFilter:
    def __init__(self, original_stdout):
        self.original_stdout = original_stdout
        self.last_percentage = -1
        self.update_interval = 5  # 每 5% 更新一次进度条

    def write(self, message):
        if '%|' in message:
            try:
                percentage = int(message.split('%|')[0])
                if percentage % self.update_interval == 0 and percentage > self.last_percentage:
                    self.last_percentage = percentage
                    self.original_stdout.write(message)
            except ValueError:
                pass
        else:
            self.original_stdout.write(message)

    def flush(self):
        self.original_stdout.flush()


# 定义 Thermostat 的不同取值
thermostat_values = [6, 7, 8, 9, 10, 11, 12]
all_dfs = []  # 用于保存最终结果的 DataFrame
energy_dfs = []  # 用于保存平均值的 DataFrame

original_stdout = sys.stdout
sys.stdout = ProgressFilter(original_stdout)

for i, value in enumerate(thermostat_values, start=1):
    a = my_env(value)
    a.world.start().wait()

    time_df = pd.DataFrame(a.data, columns=['Time'])
    observation_df = pd.DataFrame(a.value)

    # 合并时间数据和观测数据
    combined_df = pd.concat([time_df, observation_df], axis=1)

    # 计算平均值
    energy_1_ave = combined_df['Energy_1'].mean()
    energy_2_ave = combined_df['Energy_2'].mean()
    energy_3_ave = combined_df['Energy_3'].mean()
    energy_4_ave = combined_df['Energy_4'].mean()
    energy_total_ave = energy_1_ave + energy_2_ave + energy_3_ave + energy_4_ave

    # 保存平均值数据
    energy_dfs.append({
        'Thermostat': value,
        'energy_1_ave': energy_1_ave,
        'energy_2_ave': energy_2_ave,
        'energy_3_ave': energy_3_ave,
        'energy_4_ave': energy_4_ave,
        'energy_total_ave': energy_total_ave
    })

    # 添加 Thermostat 列
    combined_df['Thermostat'] = value

    # 计算 energy_total 列
    combined_df['energy_total'] = combined_df['Energy_1'] + combined_df['Energy_2'] + combined_df['Energy_3'] + combined_df['Energy_4']

    # 按照指定顺序排列列
    total_columns_order = ['Time', 'Thermostat', 't_chiller', 'temperature:drybulb', 'flow_chiller', 'power_chiller', 't_out', 'Equip', 'light', 'occ', 'Energy_1', 'Energy_2', 'Energy_3', 'Energy_4', 'energy_total']
    combined_df = combined_df[total_columns_order]

    all_dfs.append(combined_df)

sys.stdout = original_stdout

# 合并最终结果
final_df = pd.concat(all_dfs, ignore_index=True)

# 按照时间从早到晚排序
final_df['Time'] = pd.to_datetime(final_df['Time'])
final_df = final_df.sort_values(by='Time')

# 保存合并后的数据到 result_total.csv 文件
final_df.to_csv(resolve('result_1.csv'), index=False)

# 生成 energy 数据文件
energy_df = pd.DataFrame(energy_dfs)
energy_columns_order = ['Thermostat', 'energy_1_ave', 'energy_2_ave', 'energy_3_ave', 'energy_4_ave', 'energy_total_ave']
energy_df = energy_df[energy_columns_order]
energy_df.to_csv(resolve('result-energy.csv'), index=False)