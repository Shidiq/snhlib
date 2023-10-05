import numpy as np
from imblearn.metrics import geometric_mean_score
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.dummy import DummyClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, brier_score_loss, make_scorer, precision_recall_curve
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import make_pipeline

from snhlib.plot._style import custom_style


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
        fig, ax = custom_style()
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
        fig, ax = custom_style()
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
