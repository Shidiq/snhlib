from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.metrics import geometric_mean_score
from sklearn import metrics
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.dummy import DummyClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import make_pipeline

from snhlib.image import Style


class NaiveClassifier:
    def __init__(self, random_state=0, verbose=0, showplot=False, ref_value=0.2) -> None:
        self.random_state = random_state
        self.verbose = verbose
        self.showplot = showplot
        self.results_ = []
        self.names = []
        self.ref_value = ref_value

    def brier_skill_score(self, y_true, y_prob):
        ref_probs = [self.ref_value for _ in range(len(y_true))]
        bs_ref = brier_score_loss(y_true, ref_probs)
        bs_model = brier_score_loss(y_true, y_prob)
        return 1.0 - (bs_model / bs_ref)

    def evaluate_model(self, X, y, model, scoring):
        cv = RepeatedStratifiedKFold(n_repeats=3, n_splits=10, random_state=self.random_state)
        scores = cross_val_score(model, X, y, scoring=scoring, cv=cv, n_jobs=-1)
        return scores

    @staticmethod
    def get_models():
        models, names = list(), list()

        # uniform
        models.append(DummyClassifier(strategy="uniform"))
        names.append("Uniform")

        # prior random guess
        models.append(DummyClassifier(strategy="stratified"))
        names.append("Stratified")

        # majority class: Predict 0
        models.append(DummyClassifier(strategy="most_frequent"))
        names.append("Majority")

        # minority class: Predict 1
        models.append(DummyClassifier(strategy="constant", constant=1))
        names.append("Minority")

        # class prior
        models.append(DummyClassifier(strategy="prior"))
        names.append("Prior")
        return models, names

    @staticmethod
    def plot(results_, names):
        fig, ax = Style().paper()
        ax.boxplot(results_, labels=names, showmeans=True)
        ax.set_ylabel("Scores")
        return fig

    @staticmethod
    def pr_auc(y_true, probas_pred):
        # calculate precision-recall curve
        p, r, _ = precision_recall_curve(y_true, probas_pred)
        # calculate area under curve
        return auc(r, p)

    def fit(self, X, y, scoring="accuracy"):
        # Brier Skill Score (BBS)
        if scoring == "brier_skill_score":
            metric = make_scorer(self.brier_skill_score, needs_proba=True)
            model = DummyClassifier(strategy="prior")
            scores = self.evaluate_model(X=X, y=y, model=model, scoring=metric)

            if self.verbose != 0:
                print("Mean BSS: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))

            return scores

        # define models
        models, self.names = self.get_models()

        self.results_ = []

        if scoring == "g-mean":
            scoring = make_scorer(geometric_mean_score)
        elif scoring == "f1":
            scoring = "f1"
        elif scoring == "roc_auc":
            scoring = "roc_auc"
        elif scoring == "pr_auc":
            scoring = make_scorer(self.pr_auc, needs_proba=True)
        elif scoring == "brier_score_loss":
            scoring = make_scorer(brier_score_loss, needs_proba=True)
        else:
            scoring = scoring

        # evaluate model
        for i in range(len(models)):
            scores = self.evaluate_model(X=X, y=y, model=models[i], scoring=scoring)
            self.results_.append(scores)

            if self.verbose != 0:
                print(">%s %.3f (%.3f)" % (self.names[i], np.mean(scores), np.std(scores)))

        if self.showplot:
            fig = self.plot(self.results_, self.names)
            return fig


class ProbabilisticClassifier:
    def __init__(
        self,
        ref_value=0.2,
        random_state=0,
        verbose=1,
        showplot=True,
        preprocessing=None,
        scoring="brier_skill_score",
    ) -> None:
        self.ref_value = ref_value
        self.random_state = random_state
        self.verbose = verbose
        self.showplot = showplot
        self.preprocessing = preprocessing
        self.scoring = scoring

    def brier_skill_score(self, y_true, y_prob):
        ref_probs = [self.ref_value for _ in range(len(y_true))]
        bs_ref = brier_score_loss(y_true, ref_probs)
        bs_model = brier_score_loss(y_true, y_prob)
        return 1.0 - (bs_model / bs_ref)

    def evaluate_model(self, X, y, model):

        if self.scoring == "brier_skill_score":
            metric = make_scorer(self.brier_skill_score, needs_proba=True)
        else:
            metric = self.scoring

        cv = RepeatedStratifiedKFold(n_repeats=3, n_splits=10, random_state=self.random_state)
        scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1)
        return scores

    def get_models(self):
        models, names = list(), list()

        # LR
        if self.scoring == "brier_skill_score":
            models.append(LogisticRegression(solver="lbfgs"))
        else:
            models.append(LogisticRegression(solver="liblinear"))
        names.append("LR")

        # LDA
        models.append(LinearDiscriminantAnalysis())
        names.append("LDA")

        # QDA
        models.append(QuadraticDiscriminantAnalysis())
        names.append("QDA")

        # GNB
        models.append(GaussianNB())
        names.append("GNB")

        # MNB
        models.append(MultinomialNB())
        names.append("MNB")

        # GPC
        models.append(GaussianProcessClassifier())
        names.append("GPC")

        return models, names

    @staticmethod
    def plot(results_, names):
        fig, ax = Style().paper()
        ax.boxplot(results_, labels=names, showmeans=True)
        ax.set_ylabel("Scores")
        return fig

    def fit(self, X, y):
        models, names = self.get_models()
        results = list()

        for i in range(len(models)):

            try:

                if self.preprocessing is None:
                    pipeline = models[i]
                else:
                    pipeline = make_pipeline(self.preprocessing, models[i])

                scores = self.evaluate_model(X=X, y=y, model=pipeline)
                results.append(scores)

                if self.verbose == 1:
                    print(">%s %.3f (%.3f)" % (names[i], np.mean(scores), np.std(scores)))
            except ValueError:
                print(">%s Value Error!" % (names[i]))

        if self.showplot:
            fig = self.plot(results, names)
            return fig


class Evaluate:
    def __init__(self, estimator, n_class=2):
        self.estimator = estimator
        self.nclass = n_class
        self.pred = None
        self.pred_test = None
        self.X = None
        self.y = None
        self.X_test = None
        self.y_test = None

    def info(self):
        print(Counter(self.y))

    def validation(self, *args, extra=False, pos_label=1, neg_label=0, **kwargs):
        c = Counter(args[1])
        n = len(c.keys())
        self.nclass = n if n != 2 else 2

        scores = cross_val_score(estimator=self.estimator, X=args[0], y=args[1], **kwargs)

        if extra and self.nclass == 2:
            sensi = metrics.make_scorer(metrics.recall_score, pos_label=pos_label)
            spesi = metrics.make_scorer(metrics.recall_score, pos_label=neg_label)
            sensi_scores = cross_val_score(
                estimator=self.estimator, X=args[0], y=args[1], scoring=sensi, **kwargs
            )
            spesi_scores = cross_val_score(
                estimator=self.estimator, X=args[0], y=args[1], scoring=spesi, **kwargs
            )

            print(scores.mean())
            print(sensi_scores.mean())
            print(spesi_scores.mean())
            return scores, sensi_scores, spesi_scores

        print(scores.mean())
        return scores

    def fit(self, *args):
        self.X = args[0]
        self.y = args[1]
        if len(args) == 4:
            self.X = args[0]
            self.X_test = args[1]
            self.y = args[2]
            self.y_test = args[3]

        c = Counter(self.y)
        n = len(c.keys())
        self.nclass = n if n != 2 else 2

        self.estimator.fit(self.X, self.y)
        self.pred = self.estimator.predict(self.X)

        if self.X_test is not None:
            self.pred_test = self.estimator.predict(self.X_test)

    def report(self):
        print(metrics.accuracy_score(self.y, self.pred))
        if self.X_test is not None:
            print(metrics.accuracy_score(self.y_test, self.pred_test))


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
