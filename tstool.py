import datetime
from typing import Optional, TypeVar

import numpy as np
import pandas as pd


def window_function(features, target, seq_len: int, target_len: int, step: int = 1):
    """
    시계열에 적합한 시퀀스를 만들어줍니다.
    :param features: 독립변수
    :param target: 목표 종속변수
    :param seq_len: 타임스탬프
    :param target_len: 목료 종속 변수의 수(single: 1 mult: @)
    :param step: 단계 건너뛰기
    :return: data, label
    """
    data, label = list(), list()

    start_index = seq_len
    if target is not None:
        end_index = len(features) - target_len + 1
    elif target is None:
        end_index = len(features) - target_len + 2

    for i in range(start_index, end_index, step):
        indices = range(i - seq_len, i)
        data.append(features[indices])
        if target is not None:
            label.append(target[i:i + target_len])

    data = np.array(data)

    if target is not None:
        label = np.array(label)
        return data, label

    else:
        return data


def quarter_function(dataframe, col_name: Optional[str] = None):
    """
    :param dataframe: 데이터프레임 시간
    :param col_name: 데이터프레임이면 시간 행 이름 적기
    :return:  [0 ,0, 0, 0]
    """
    if isinstance(dataframe, pd.core.indexes.datetimes.DatetimeIndex):
        series = dataframe.quarter
    elif isinstance(dataframe, pd.core.series.Series):
        series = dataframe.dt.quarter
    elif isinstance(dataframe, pd.core.frame.DataFrame) and col_name is not None:
        series = dataframe[col_name].dt.quarter

    quarter_list = list()

    for values in series:
        values -= 1
        categories = np.zeros(4, dtype=np.int)
        categories[values] = 1

        quarter_list.append(categories)

    quarter_list = np.array(quarter_list)

    quarter_dataframe = pd.DataFrame(quarter_list, columns=['spring', 'summer', 'autumn', 'winter'])

    return quarter_dataframe


def date_function(dataframe, col_name:Optional[str] = None, min_freq: Optional[bool] = True):
    def type_ds(dataframe):
        month = dataframe.month.values
        day = dataframe.day.values
        weekday = dataframe.weekday.values
        woy = dataframe.weekofyear.values
        hour = dataframe.hour.values

        return month, day, weekday, woy, hour

    if isinstance(dataframe, pd.core.indexes.datetimes.DatetimeIndex):
        month, day, weekday, woy, hour = type_ds(dataframe)

    elif isinstance(dataframe, pd.core.series.Series):
        dataframe = dataframe.dt
        month, day, weekday, woy, hour = type_ds(dataframe)

    elif isinstance(dataframe, pd.core.frame.DataFrame) and col_name is not None:
        dataframe = dataframe[[col_name]].dt
        month, day, weekday, woy, hour = type_ds(dataframe)

    date_dataframe = pd.DataFrame({'month': month, 'day': day, 'hour': hour, 'weekday': weekday, 'weekofyear': woy})

    if min_freq:
        minute = dataframe.minute.values
        min_series = pd.Series(minute)
        date_dataframe = pd.concat([date_dataframe, min_series], axis=1)
        date_dataframe.columns = ['month', 'day', 'hour', 'weekday', 'weekofyear', 'minute']

    return date_dataframe


def signal_func(dataframe, col_name:Optional[str] = None):
    """
    Support Year, day sin cos
    """
    if isinstance(dataframe, pd.core.frame.DataFrame) and col_name is not None:
        dataframe = dataframe[col_name]
    elif dataframe.index.dtype == '<M8[ns]':
        dataframe = dataframe.index

    hour = 60 * 60
    day = 24 * hour
    year = 365.2425 * day

    timestamp_s = dataframe.map(lambda y: datetime.datetime.timestamp(y))

    year_sin = np.sin(timestamp_s * (2 * np.pi / year))
    year_cos = np.cos(timestamp_s * (2 * np.pi / year))

    day_sin = np.sin(timestamp_s * (2 * np.pi / day))
    day_cos = np.cos(timestamp_s * (2 * np.pi / day))

    return year_sin, year_cos, day_sin, day_cos
