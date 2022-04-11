import json
import os
import re
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import RepeatedKFold


def evaluate_clf(
    estimator,
    X_train,
    X_test,
    y_train,
    y_test,
    cv=None,
    scaler=None,
    pos_label=["positive", "negative"],
):
    if cv is None:
        cv = RepeatedKFold(n_repeats=10, n_splits=5, random_state=0)

    scores = {
        "accuracy": [],
        "sensitivity": [],
        "specificity": [],
    }

    for train_index, test_index in cv.split(X_train):
        X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
        y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]

        # scaler
        if scaler is not None:
            X_train_cv = scaler.fit_transform(X_train_cv)
            X_test_cv = scaler.transform(X_test_cv)

        estimator.fit(X_train_cv, y_train_cv)

        # make prediction
        yhat_cv = estimator.predict(X_test_cv)

        # evaluate
        acc = accuracy_score(y_test_cv, yhat_cv)
        sensi = recall_score(y_test_cv, yhat_cv, pos_label=pos_label[0])
        spesi = recall_score(y_test_cv, yhat_cv, pos_label=pos_label[1])

        scores["accuracy"].append(acc)
        scores["sensitivity"].append(sensi)
        scores["specificity"].append(spesi)

    # print
    print("Internal validation:")
    print(
        "Accuracy   : %.2f ± %.2f"
        % (np.mean(scores["accuracy"]), np.std(scores["accuracy"]))
    )
    print(
        "Sensitivity: %.2f ± %.2f"
        % (np.mean(scores["sensitivity"]), np.std(scores["sensitivity"]))
    )
    print(
        "Specificity: %.2f ± %.2f"
        % (np.mean(scores["specificity"]), np.std(scores["specificity"]))
    )

    # training
    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    estimator.fit(X_train, y_train)

    yhat_train = estimator.predict(X_train)
    yhat_test = estimator.predict(X_test)

    def eval_(true_val, pred_val):
        acc = accuracy_score(true_val, pred_val)
        sensi = recall_score(true_val, pred_val, pos_label=pos_label[0])
        spesi = recall_score(true_val, pred_val, pos_label=pos_label[1])

        print("Accuracy   : %.2f" % acc)
        print("Sensitivity: %.2f" % sensi)
        print("Specificity: %.2f" % spesi)

    print("\nTraining score:")
    eval_(y_train, yhat_train)

    print("\nTesting score:")
    eval_(y_test, yhat_test)


@dataclass
class DataGeNose:
    path: str
    cols: List = field(default_factory=lambda: [f"S{i+1}" for i in range(10)])

    @property
    def open(self):
        try:
            ext = os.path.splitext(self.path)[1]

            if ext == ".csv":
                data = pd.read_csv(self.path)
            elif ext == ".json":
                cols = ["time(s)"] + self.cols + ["Temp", "Humid"]
                data = json.load(open(self.path, "r"))
                data = data["datasensor"]
                data = pd.DataFrame(data, columns=cols)
            else:
                raise ValueError

            return data

        except ValueError:
            print("Format not supported!")

    @property
    def transform_for_encoder(self):
        data = self.open
        select = [i for i in range(data.shape[0]) if i % 10 == 0][0:40]
        new_data = data.loc[select].reset_index(drop=True)
        new_data = new_data[self.cols]
        return new_data.transpose().to_numpy().flatten()


@dataclass
class FindData:
    path: str
    pattern: List = field(default_factory=lambda: [".csv", ".json"])

    @property
    def get_files(self):
        _files = []
        for root, dirnames, filenames in os.walk(self.path):
            for filename in filenames:
                if os.path.splitext(filename)[1] in self.pattern:
                    _files.append(os.path.join(root, filename))
        return _files

    @property
    def get_files_swap(self):
        files = self.get_files

        results = {
            "file": [],
            "swap": [],
        }

        for f in files:
            swap = (
                "positive"
                if bool(re.search("posi", f.lower()))
                else "negative"
                if bool(re.search("nega", f.lower()))
                else "unknown"
            )
            results["file"].append(f)
            results["swap"].append(swap)

        return results
