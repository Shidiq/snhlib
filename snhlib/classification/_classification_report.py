import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


class Report:
    def __init__(self, labels=["negative", "positive"]):
        self.labels = labels

    @staticmethod
    def classification_reports(
        X_train, X_test, y_train, y_test, model, report=True, plot=False, return_df=False
    ):
        if report:
            print("Train report")
            print(classification_report(y_train, model.predict(X_train)))
            print()
            print("Test report")
            print(classification_report(y_test, model.predict(X_test)))

        if return_df:
            labels = y_train.unique()
            df_train = pd.DataFrame(
                classification_report(
                    y_train, model.predict(X_train), labels=labels, output_dict=True
                )
            )

            labels = y_test.unique()
            df_test = pd.DataFrame(
                classification_report(
                    y_test, model.predict(X_test), labels=labels, output_dict=True
                )
            )

            return df_train.T, df_test.T

        if plot:
            fig = plt.figure(figsize=(11, 5))
            plt.subplots_adjust(wspace=0.4)

            plt.subplot(121)
            labels = y_train.unique()
            df_train = pd.DataFrame(
                classification_report(
                    y_train, model.predict(X_train), labels=labels, output_dict=True
                )
            )
            df_plot = df_train.iloc[:-1, : len(labels)]
            sns.heatmap(
                df_plot,
                vmin=0,
                vmax=1,
                annot=True,
                square=True,
                cmap="Blues",
                cbar=False,
                xticklabels=labels,
                yticklabels=df_plot.index,
                fmt=".2f",
                annot_kws={"fontsize": 15},
            )
            plt.yticks(rotation=0, fontsize=14)
            plt.xticks(rotation=45, horizontalalignment="right", fontsize=12)
            plt.title("Train", fontsize=14)

            plt.subplot(122)
            labels = y_test.unique()
            df_test = pd.DataFrame(
                classification_report(
                    y_test, model.predict(X_test), labels=labels, output_dict=True
                )
            )
            df_plot = df_test.iloc[:-1, : len(labels)]
            sns.heatmap(
                df_plot,
                vmin=0,
                vmax=1,
                annot=True,
                square=True,
                cmap="Greens",
                cbar=False,
                xticklabels=labels,
                yticklabels=df_plot.index,
                fmt=".2f",
                annot_kws={"fontsize": 15},
            )
            plt.yticks(rotation=0, fontsize=14)
            plt.xticks(rotation=45, horizontalalignment="right", fontsize=12)
            plt.title("Test", fontsize=14)

            return fig

    def cm(self, ytrue, ypred):
        TP = 0
        TN = 0
        FN = 0
        FP = 0
        akurasi = 0.0

        pos_spesi = self.labels[0]
        pos_sensi = self.labels[1]

        akurasi = accuracy_score(ytrue, ypred)

        try:
            cc = confusion_matrix(ytrue, ypred, labels=[pos_sensi, pos_spesi])
        except UnboundLocalError:
            cc = confusion_matrix(ytrue, ypred)

        TT = np.diag(cc)
        FF = cc.sum(axis=0) - TT

        n = np.unique(self.labels)

        if len(n) == 1:
            if n[0] == pos_sensi:
                FP = FF[1]
                FN = FF[0]
                TP = TT[1]
                TN = TT[0]
        else:
            FP = FF[0]
            FN = FF[1]
            TP = TT[0]
            TN = TT[1]

        try:
            f1 = f1_score(ytrue, ypred, pos_label=pos_sensi)
        except UnboundLocalError:
            f1 = "nan"

        try:
            if (TP + FN) == 0:
                raise ValueError

            sensi = TP / (TP + FN)
        except ValueError:
            sensi = "nan"

        try:
            if (TN + FP) == 0:
                raise ValueError

            spesi = TN / (TN + FP)
        except ValueError:
            spesi = "nan"

        PPV = TP / (TP + FP)
        NPV = TN / (TN + FN)

        res = {
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN,
            "accuracy": round(float(akurasi), 3),
            "f1-score": "nan" if f1 == "nan" else round(f1, 3),
            "sensitivity": "nan" if sensi == "nan" else round(sensi, 3),
            "specificity": "nan" if spesi == "nan" else round(spesi, 3),
            "PPV": round(PPV, 3),
            "NPV": round(NPV, 3),
        }
        return res
