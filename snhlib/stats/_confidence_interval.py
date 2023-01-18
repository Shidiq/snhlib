from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class AUC_confidence_interval:
    """_summary_

    Returns
    -------
    _type_
        _description_
    """

    n1: int
    n2: int
    auc: float
    alpha: float = 0.05

    @property
    def zcrit_(self):
        return norm.ppf(1 - self.alpha / 2)

    @property
    def se_(self):
        q0 = self.auc * (1 - self.auc)
        q1 = self.auc / (2 - self.auc) - self.auc**2
        q2 = 2 * self.auc**2 / (1 + self.auc) - self.auc**2
        return np.sqrt((q0 + (self.n1 - 1) * q1 + (self.n2 - 1) * q2) / (self.n1 * self.n2))

    @property
    def interval_(self):
        lower = self.auc - self.zcrit_ * self.se_
        upper = self.auc + self.zcrit_ * self.se_
        return lower, upper
