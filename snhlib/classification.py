from sklearn import metrics
from sklearn.model_selection import cross_val_score
from collections import Counter


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
