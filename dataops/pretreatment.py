"""
This module is responsible for fetching and pre-processing the MNIST dataset.
It uses the `fetch_openml` function from the `sklearn.datasets` module to download the dataset and store it in a local directory.
As fetch_openml is an experimental feature, it can yield to some network errors.
In that case, you can manually download the dataset from https://www.openml.org/search?type=data&status=active&id=554&sort=runs and put it in the data/ directory
"""

import argparse
import os
import pickle
from urllib.request import urlretrieve

import arff
import numpy as np
import pandas as pd


def fetch_mnist_from_openml():
    os.makedirs("data", exist_ok=True)
    urlretrieve(
        url="https://www.openml.org/data/download/18238735/phpnBqZGZ",
        filename="data/fmnist_784.arff",
    )


def preprocess_mnist_data():
    with open("data/fmnist_784.arff", "r") as f:
        dataset = arff.load(f)
    print(dataset.keys())
    data = np.array(dataset["data"])
    X = data[:, :-1].astype(np.float32)
    y = data[:, -1].astype(np.int64)
    print(f"X: {X.shape}, y: {y.shape}")
    # Split the data into training and test sets based in the original MNIST split.
    X_train_val, X_test = X[:60000], X[60000:]
    y_train_val, y_test = y[:60000], y[60000:]
    X_train, X_val = X_train_val[:50000], X_train_val[50000:]
    y_train, y_val = y_train_val[:50000], y_train_val[50000:]
    print("Saving MNIST flat dataset")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    # Save the preprocessed data as pickle files.
    mnist_flat = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }
    with open("data/fmnist_flat.pkl", "wb") as f:
        pickle.dump(mnist_flat, f)

    mnist_28_28 = {
        "X_train": X_train.reshape(-1, 28, 28),
        "y_train": y_train,
        "X_val": X_val.reshape(-1, 28, 28),
        "y_val": y_val,
        "X_test": X_test.reshape(-1, 28, 28),
        "y_test": y_test,
    }
    print("MNIST 28x28 dataset")
    print(
        f"X_train: {mnist_28_28['X_train'].shape}, y_train: {mnist_28_28['y_train'].shape}"
    )
    print(f"X_val: {mnist_28_28['X_val'].shape}, y_val: {mnist_28_28['y_val'].shape}")
    print(
        f"X_test: {mnist_28_28['X_test'].shape}, y_test: {mnist_28_28['y_test'].shape}"
    )

    with open("data/fmnist_28_28.pkl", "wb") as f:
        pickle.dump(mnist_28_28, f)


def fetch_ett_from_github():
    os.makedirs("data", exist_ok=True)
    urlretrieve(
        # Repo with the data in: https://github.com/zhouhaoyi/ETDataset/tree/main
        url="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
        filename="data/ETTh1.csv",
    )


def preprocess_ett_data():
    df = pd.read_csv("data/ETTh1.csv")
    columns = [c for c in df.columns if c != "date"]
    data = df[columns].to_numpy(dtype=np.float32)
    dates = df["date"].to_numpy()
    print(f"data: {data.shape}, columns: {columns}")
    # Standard ETTh1 12 / 4 / 4 months of hourly data.
    # The splits are chronological and non-overlapping: windowing is left to the data object.
    n_train, n_val, n_test = 12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24
    train = data[:n_train]
    val = data[n_train : n_train + n_val]
    test = data[n_train + n_val : n_train + n_val + n_test]
    # Standardize with training statistics only, to avoid leaking the val/test distribution.
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    train, val, test = (train - mean) / std, (val - mean) / std, (test - mean) / std
    print("Saving ETTh1 dataset")
    print(f"train: {train.shape}, val: {val.shape}, test: {test.shape}")
    print(f"train: {dates[0]} -> {dates[n_train - 1]}")
    print(f"val:   {dates[n_train]} -> {dates[n_train + n_val - 1]}")
    print(f"test:  {dates[n_train + n_val]} -> {dates[n_train + n_val + n_test - 1]}")
    # Save the preprocessed data as a pickle file.
    ett = {
        "train": train,
        "val": val,
        "test": test,
        "columns": columns,
        "target_idx": columns.index("OT"),
        "mean": mean,
        "std": std,
    }
    with open("data/etth1.pkl", "wb") as f:
        pickle.dump(ett, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Download_n_pretreat",
        description="A program launching a routine that use helper functions for downloading and pretreating datasets",
    )
    parser.add_argument("datasetname", nargs="?", default="fmnist", choices=["fmnist", "etth1"])
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    if args.datasetname == "etth1":
        if not os.path.exists("data/ETTh1.csv"):
            fetch_ett_from_github()
        preprocess_ett_data()
    else:
        if not os.path.exists("data/fmnist_784.arff"):
            fetch_mnist_from_openml()
        preprocess_mnist_data()

