from collections import Counter
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score
import numpy as np


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

        scores = cross_val_score(
            estimator=self.estimator, X=args[0], y=args[1], **kwargs)

        if extra and self.nclass == 2:
            sensi = metrics.make_scorer(
                metrics.recall_score, pos_label=pos_label)
            spesi = metrics.make_scorer(
                metrics.recall_score, pos_label=neg_label)
            sensi_scores = cross_val_score(
                estimator=self.estimator, X=args[0], y=args[1], scoring=sensi, **kwargs)
            spesi_scores = cross_val_score(
                estimator=self.estimator, X=args[0], y=args[1], scoring=spesi, **kwargs)

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
    @staticmethod
    def cm(ytrue, ypred, labels=['positive', 'negative']):
        pos_sensi = labels[0]
        pos_spesi = labels[1]

        akurasi = accuracy_score(ytrue, ypred)

        try:
            cc = confusion_matrix(ytrue, ypred, labels=[pos_sensi, pos_spesi])
        except UnboundLocalError:
            cc = confusion_matrix(ytrue, ypred)

        TT = np.diag(cc)
        FF = cc.sum(axis=0) - TT

        n = np.unique(labels)

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
            f1 = 'nan'

        try:
            if (TP + FN) == 0:
                raise ValueError

            sensi = TP / (TP + FN)
        except ValueError:
            sensi = 'nan'

        try:
            if (TN + FP) == 0:
                raise ValueError

            spesi = TN / (TN + FP)
        except ValueError:
            spesi = 'nan'

        PPV = TP/(TP+FP)
        NPV = TN/(TN+FN)

        res = {
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN,
            'accuracy': round(akurasi, 3),
            'f1-score': 'nan' if f1 == 'nan' else round(f1, 3),
            'sensitivity': 'nan' if sensi == 'nan' else round(sensi, 3),
            'specificity': 'nan' if spesi == 'nan' else round(spesi, 3),
            'PPV': round(PPV, 3),
            'NPV': round(NPV, 3),
        }
        return res
