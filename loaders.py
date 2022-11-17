from pathlib import Path

import pandas as pd


def simply_loader(path):
    """
    support only csv files
    :param path: file or directory
    :return: Dataframe
    """
    path_ = Path(path)

    if path_.is_file():
        if path_.suffix.lower() == '.csv':
            dataframe = pd.read_csv(path_)
        else:
            print(f'Do not support {path_.suffix.lower()}')

    elif path_.is_dir():
        dataframe = pd.DataFrame()
        for pathx in path_.iterdir():
            if 'csv' in pathx.suffix.lower():
                tmp = pd.read_csv(pathx)
                dataframe = dataframe.append(tmp)
            else:
                print(f'read pass. {pathx.name}')

    return dataframe


