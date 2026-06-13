import numpy as np
from scipy.stats import entropy as scipy_entropy
from scipy.stats import ks_2samp, spearmanr
from sklearn.feature_selection import mutual_info_regression

from liquidation_task_tools.base import FeatureAnalyzer


class EntropyAnalyzer(FeatureAnalyzer):
    def __init__(self):
        super().__init__("entropy")

    def calculate(self, feature: np.ndarray) -> float:
        _, counts = np.unique(feature, return_counts=True)
        probs = counts / counts.sum()
        return float(scipy_entropy(probs, base=2))


class SpearmanCorrelationAnalyzer(FeatureAnalyzer):
    def __init__(self):
        super().__init__("spearman_correlation")

    def calculate(self, feature: np.ndarray, target: np.ndarray) -> float:
        corr, _ = spearmanr(feature, target)
        return float(corr)


class MutualInformationAnalyzer(FeatureAnalyzer):
    def __init__(self):
        super().__init__("mutual_information")

    def calculate(self, feature: np.ndarray, target: np.ndarray) -> float:
        mi = mutual_info_regression(feature.reshape(-1, 1), target, random_state=42)
        return float(mi[0])


class KSAnalyzer(FeatureAnalyzer):
    def __init__(self):
        super().__init__("ks")

    def calculate(self, expected: np.ndarray, actual: np.ndarray) -> float:
        statistic, _ = ks_2samp(expected, actual)
        return float(statistic)

