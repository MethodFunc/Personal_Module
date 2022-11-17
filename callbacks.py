from abc import ABCMeta

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt


class EarlyBase(metaclass=ABCMeta):
    def __init__(self):
        self.counter = 0
        self.val_loss_min = np.INf
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
        self.delta = delta
        self.rbw = restore_best_weight
        self.best_weight = None

    def save_checkpoint(self, val_loss, model, path):
        sequential = tf.keras.models.Sequential

        if self.val_loss_min == np.Inf:
            self.model_path = self.create_path(path)

        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}. Saving model...')
        if isinstance(model, list):
            merge_model = sequential()
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
        self.delta = delta

    def save_checkpoint(self, val_loss, model, path):
        import torch
        if self.val_loss_min == np.Inf:
            self.model_path = self.create_path(path)

        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}. Saving model...')
        torch.save(model.state_dict(), self.model_path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class CustomHistory(tf.keras.callbacks.Callback):
    """
    log loss history
    """

    def __init__(self, figsize=(16, 9)):
        super(CustomHistory, self).__init__()
        if 'ipykernel' not in sys.argv[0]:
            raise EnvironmentError("The current environment is not ipykerenl. Please check again."
                                   "Only runs in the ipykerenl environment")
        self.train_loss = []
        self.val_loss = []
        self.get_logs = {}
        self.figsize = figsize

    def on_epoch_begin(self, batch, logs=None):
        self.ts = time.time()

    def on_epoch_end(self, epoch, logs=[]):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.epoch = epoch
        self.timeit = time.time() - self.ts
        self.get_logs['time'] = self.timeit
        for key, values in logs.items():
            self.get_logs[key] = values

        self.n = np.arange(0, len(self.train_loss))

        self.update_df()
        self.update_graph()

    def on_train_end(self, logs=None):
        plt.close()

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

        temp = pd.DataFrame(self.get_logs, index=[0])
        self.df = pd.concat([self.df, temp], ignore_index=True)
        self.style_df = self.df.style.apply(update_style, props='color:white;background-color:brown;', axis=0)

        self.df_output.update(self.style_df)

    def get_df(self):
        return self.df

    def update_graph(self):
        if not hasattr(self, 'graph_fig'):
            self.graph_fig, self.graph_ax = plt.subplots(1, figsize=self.figsize)
            self.graph_out = display(self.graph_fig, display_id=True)
        self.graph_ax.clear()

        self.graph_ax.plot(self.n, self.train_loss, label="train_loss")
        self.graph_ax.plot(self.n, self.val_loss, label="val_loss")
        self.graph_ax.set_title(f"Training Loss [Epoch {self.epoch}]")
        self.graph_ax.set_xlabel("Epoch #")
        self.graph_ax.set_ylabel("Loss")
        self.graph_ax.legend(loc='upper right')

        self.graph_out.update(self.graph_ax.figure)
