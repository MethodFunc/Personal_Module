import re
from datetime import datetime, timedelta
from typing import Optional

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


def quarter_function(dataframe, date_col: Optional[str] = None):
    """
    :param dataframe: 데이터프레임 시간
    :param date_col: 데이터프레임이면 시간 행 이름 적기
    :return:  [0 ,0, 0, 0]
    """
    if isinstance(dataframe, pd.core.indexes.datetimes.DatetimeIndex):
        series = dataframe.quarter
    elif isinstance(dataframe, pd.core.series.Series):
        series = dataframe.dt.quarter
    elif isinstance(dataframe, pd.core.frame.DataFrame) and date_col is not None:
        series = dataframe[date_col].dt.quarter

    quarter_list = list()

    for values in series:
        values -= 1
        categories = np.zeros(4, dtype=np.int)
        categories[values] = 1

        quarter_list.append(categories)

    quarter_list = np.array(quarter_list)

    quarter_dataframe = pd.DataFrame(quarter_list, columns=['spring', 'summer', 'autumn', 'winter'])

    return quarter_dataframe


def date_function(dataframe, date_col: Optional[str] = None, min_freq: Optional[bool] = True):
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

    elif isinstance(dataframe, pd.core.frame.DataFrame) and date_col is not None:
        dataframe = dataframe[[date_col]].dt
        month, day, weekday, woy, hour = type_ds(dataframe)

    date_dataframe = pd.DataFrame({'month': month, 'day': day, 'hour': hour, 'weekday': weekday, 'weekofyear': woy})

    if min_freq:
        minute = dataframe.minute.values
        min_series = pd.Series(minute)
        date_dataframe = pd.concat([date_dataframe, min_series], axis=1)
        date_dataframe.columns = ['month', 'day', 'hour', 'weekday', 'weekofyear', 'minute']

    return date_dataframe


def signal_func(dataframe, date_col: Optional[str] = None):
    """
    Support Year, day sin cos
    """
    if isinstance(dataframe, pd.core.frame.DataFrame) and date_col is not None:
        dataframe = dataframe[date_col]
    elif dataframe.index.dtype == '<M8[ns]':
        dataframe = dataframe.index

    hour = 60 * 60
    day = 24 * hour
    year = 365.2425 * day

    timestamp_s = dataframe.map(lambda y: datetime.timestamp(y))

    year_sin = np.sin(timestamp_s * (2 * np.pi / year))
    year_cos = np.cos(timestamp_s * (2 * np.pi / year))

    day_sin = np.sin(timestamp_s * (2 * np.pi / day))
    day_cos = np.cos(timestamp_s * (2 * np.pi / day))

    return year_sin, year_cos, day_sin, day_cos


def calc_daterange(start_date: str, end_date: str) -> list:
    """
    날짜와 날짜 사이의 시간들을 뽑기 위해 만들어진 함수
    날짜 타입 입력 예시
    20221101, 2022-11-01, 2022-11-01 00:00, 2022-11-01 00:00:00, 20221101000000
    :params start_date: 2022-11-01 시작 일
    :params end_date: 2022-11-02 종료일
    :return: [2022110101, 2022110102, 2022110103, ..., 2022110221, 2022110222, 2022110223]
    """
    pattern = r'[-=+,#/\?:^.@*\"※~ㆍ!』‘|\(\)\[\]`\'…》\”\“\’·\s]'

    date_pattern = {
        8: '%Y%m%d',
        10: '%Y%m%d%H',
        12: '%Y%m%d%H%M',
        14: '%Y%m%d%H%M%S',
    }

    start_date = re.sub(pattern, '', start_date)
    start_date = datetime.strptime(start_date, date_pattern[len(start_date)])

    end_date = re.sub(pattern, '', end_date)
    end_date = datetime.strptime(end_date, date_pattern[len(end_date)])

    date_diff = end_date - start_date
    date_range = int(date_diff.days * 24 + date_diff.seconds / 3600) + 1

    date_list = [
        (start_date + timedelta(hours=i)).strftime("%Y%m%d%H%M")
        for i in range(date_range)
    ]

    return date_list


def convert_dates(dataframe, date_col: str):
    """
        가끔 00시를 오전 12시로 표기 되어 있는 날짜 타입들이 있음. 해당 날짜 타입을 정상화 시킴
        datetime 라이브러리는 한글로된 오전 오후는 인식하지 못함으로 영어로 고침
        :params dataframe: 날짜가 들어있는 데이터 프레임 (값은 모두 object형식이여야함)
        :params date_col: 날짜 데이터가 들어있는 행이름
        :return: 변환된 날짜
    """
    dataframe[date_col] = dataframe[date_col].apply(
        lambda x: x.replace('오전 12:00:00', '오전 00:00:00') if '오전 12:00:00' in x else x)
    dataframe[date_col] = dataframe[date_col].apply(lambda x: x.replace('오전', 'AM'))
    dataframe[date_col] = dataframe[date_col].apply(lambda x: x.replace('오후', 'PM'))

    convert_date_list = list()
    for date in dataframe[date_col]:
        try:
            convert_date = datetime.strptime(date, "%Y-%m-%d %p %I:%M:%S")
            convert_date_list.append(convert_date)
        except ValueError:
            # 00:00:00은 I로 컨버터 못하여 ValueError가 뜸 이는 24시간 단위로 처리하면 됨
            convert_date = datetime.strptime(date, "%Y-%m-%d %p %H:%M:%S")
            convert_date_list.append(convert_date)

    return convert_date_list
