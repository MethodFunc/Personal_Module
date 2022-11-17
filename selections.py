from typing import Optional

def split_data(data, train_ratio: float = 0.7, val_ratio: Optional[float] = 0.2):
    total_size = len(data)
    train = int(total_size * train_ratio)
    train_data = data[:train]
    test_data = data[train:]
    if val_ratio is not None:
        val_ratio = train_ratio + val_ratio
        val = int(total_size * val_ratio)
        val_data = data[train: val]
        test_data = data[val:]

        return train_data, val_data, test_data

    return train_data, test_data
