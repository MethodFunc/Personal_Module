from abc import ABCMeta

import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict
import matplotlib.pyplot as plt


class EarlyBase(metaclass=ABCMeta):
    def __init__(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_weight = model.get_weights()
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

        @abstractmethod
        def save_checkpoint(self):
            pass

    @staticmethod
    def create_path(path):
        path = Path(path)

        if not path.exists():
            path.mkdir()

        experiment = 0

        while True:
            model_path = path / f'experiment_{experiment:02}'

            if model_path.exists() and model_path.stat().st_size != 0:
                experiment += 1
            else:
                model_path.mkdir(parents=True, exist_ok=True)
                break

        return model_path


class TFEarlyStop(EarlyBase):
    def __init__(self, patience=7, verbose=True, delta=0, restore_best_weight=False):
        super(TFEarlyStop, self).__init__()
        self.patience = patience
        self.verbose = verbose
        self.val_loss_min = np.Inf
        self.delta = delta
        self.rbw = restore_best_weight
        self.best_weight = None

    def save_checkpoint(self, val_loss, model, path):
        from tensorflow.keras.models import Sequential

        if self.val_loss_min == np.Inf:
            self.model_path = self.create_path(path)

        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}. Saving model...')
        if isinstance(model, list):
            merge_model = Sequential()
            for m in model:
                merge_model.add(m)

            merge_model.save_weights(f'{self.model_path}/checkpoint.ckpt')
        else:
            model.save_weights(f'{self.model_path}/checkpoint.ckpt')

        self.val_loss_min = val_loss


class PTEarlyStop(EarlyBase):
    def __init__(self, patience=7, verbose=True, delta=0):
        super(PTEarlyStop, self).__init__()
        self.patience = patience
        self.verbose = verbose
        self.val_loss_min = np.Inf
        self.delta = delta

    def save_checkpoint(self, val_loss, model, path):
        import torch
        if self.val_loss_min == np.Inf:
            self.model_path = self.create_path(path)

        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}. Saving model...')
        torch.save(model.state_dict(), self.model_path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class CustomHistory:
    def __init__(self, figsize=(16, 9)):
        self.train_loss = []
        self.val_loss = []
        self.time_list = []
        self.get_log = OrderedDict()
        self.figsize = figsize
        self.epoch = None

    def __call__(self, epoch, logs):
        self.epoch = epoch
        if isinstance(logs, dict):
            self.train_loss.append(logs.get('loss')[epoch - 1].numpy())
            self.val_loss.append(logs.get('val_loss')[epoch - 1].numpy())

        for key, value in logs.items():
            try:
                self.get_log[key] = value[epoch - 1].numpy()
            except AttributeError:
                self.get_log[key] = value[epoch - 1]

        self.n = np.arange(0, len(self.train_loss))
        self.update_df()
        self.update_fig()

    def update_df(self):
        def update_style(s, props=''):
            if s.name == 'time':
                return np.where(s == np.max(s.values), 'background-color:rgba(255, 255, 255, 0.0);', '')
            elif 'acc' in s.name:
                return np.where(s == np.max(s.values), 'color:white;background-color:darkblue;', '')
            else:
                return np.where(s == np.min(s.values), props, '')

        if not hasattr(self, 'df'):
            self.df = pd.DataFrame()
            self.df_output = display(self.df, display_id=True)

        temp_df = pd.DataFrame(self.get_log, index=[0])
        self.df = pd.concat([self.df, temp_df], ignore_index=True)
        self.style_df = self.df.style.apply(update_style, props='color:white;background-color:brown;', axis=0)

        self.df_output.update(self.style_df)

    def update_fig(self):
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(1, figsize=self.figsize)
            self.plot_output = display(self.fig, display_id=True)

        self.ax.clear()
        self.ax.plot(self.n, self.train_loss, label='loss')
        self.ax.plot(self.n, self.val_loss, label='val_loss')
        self.ax.set_title(f"Training Loss [Epoch {self.epoch}]")
        self.ax.set_xlabel("EPOCH #")
        self.ax.set_ylabel("Loss")
        self.ax.legend(loc='upper right')

        self.plot_output.update(self.ax.figure)


