import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class MNISTDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, flattened: bool = False):
        self.np_X = X
        self.np_y = y
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.flattened = flattened

    def plot_example(self):
        """
        Plot the first 5 images and their labels in a row.
        """
        if self.flattened:
            X = self.np_X.reshape(-1, 28, 28)
        else:
            X = self.np_X
        for i, (img, y) in enumerate(zip(X[:5], self.np_y[:5])):
            plt.subplot(151 + i)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            plt.title(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        if not self.flattened and x.dim() == 2:  # (28, 28) -> (1, 28, 28)
            x = x.unsqueeze(0)
        return x, self.y[idx]


class DataObject:
    def __init__(self, data_path: str, flattened: bool = False, batch_size: int = 128):
        self.data_path = data_path
        self.flattened = flattened
        self.batch_size = batch_size

    def _normalize(self, X_train, X_val, X_test):
        return X_train / 255.0, X_val / 255.0, X_test / 255.0

    def make_datasets(self) -> None:
        with open(self.data_path, "rb") as f:
            data = pickle.load(f)
        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]
        X_test, y_test = data["X_test"], data["y_test"]

        X_train, X_val, X_test = self._normalize(X_train, X_val, X_test)

        self.train_ds = MNISTDataset(X_train, y_train, flattened=self.flattened)
        self.val_ds = MNISTDataset(X_val, y_val, flattened=self.flattened)
        self.test_ds = MNISTDataset(X_test, y_test, flattened=self.flattened)

    def make_dataloaders(self) -> None:
        self.train_dl = DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True
        )
        self.val_dl = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)
        self.test_dl = DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False
        )


class WindowDataset(Dataset):
    """
    Sliding windows over a (T, n_channels) series: X is the last `lookback` steps of
    every channel, y is the next `horizon` steps of the target channel.
    """

    def __init__(self, series: np.ndarray, lookback: int, horizon: int, target_idx: int):
        self.series = torch.from_numpy(series)
        self.lookback = lookback
        self.horizon = horizon
        self.target_idx = target_idx

    def __len__(self) -> int:
        return len(self.series) - self.lookback - self.horizon + 1

    def __getitem__(self, idx: int):
        x = self.series[idx : idx + self.lookback]
        y = self.series[
            idx + self.lookback : idx + self.lookback + self.horizon, self.target_idx
        ]
        return x, y


class TSDataObject:
    def __init__(
        self,
        data_path: str,
        lookback: int,
        horizon: int,
        target: str = "OT",
        batch_size: int = 64,
    ):
        self.data_path = data_path
        self.lookback = lookback
        self.horizon = horizon
        self.target = target
        self.batch_size = batch_size

    def make_datasets(self) -> None:
        with open(self.data_path, "rb") as f:
            data = pickle.load(f)
        train, val, test = data["train"], data["val"], data["test"]
        self.columns = data["columns"]
        self.n_channels = len(self.columns)
        self.target_idx = self.columns.index(self.target)
        self.mean, self.std = data["mean"], data["std"]

        # Val/test windows borrow their lookback context from the end of the previous
        # split, so every val/test timestep is predicted whatever the lookback.
        val_series = np.concatenate([train[-self.lookback :], val])
        test_series = np.concatenate([val[-self.lookback :], test])

        self.train_ds = WindowDataset(train, self.lookback, self.horizon, self.target_idx)
        self.val_ds = WindowDataset(val_series, self.lookback, self.horizon, self.target_idx)
        self.test_ds = WindowDataset(test_series, self.lookback, self.horizon, self.target_idx)

    def make_dataloaders(self) -> None:
        self.train_dl = DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True
        )
        self.val_dl = DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)
        self.test_dl = DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False
        )
