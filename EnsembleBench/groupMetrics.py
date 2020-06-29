import numpy as np

_allDiversityMetrics = set(['CC', 'QS', 'CK', 'BD', 'ENTROPY', 'KW', 'GD', 'FK'])

# pair-wise
from diversityMetrics import correlation_coefficient
def group_correlation_coefficient(predictions, label):
    pred = np.transpose(predictions, (1, 0))
    return correlation_coefficient(pred, label)
from diversityMetrics import Q_statistic
def group_Q_statistic(predictions, label):
    pred = np.transpose(predictions, (1, 0))
    return Q_statistic(pred, label)
from diversityMetrics import cohen_kappa_statistic
def group_kappa_score(predictions):
    pred = np.transpose(predictions, (1, 0))
    return cohen_kappa_statistic(pred)
from diversityMetrics import binary_disagreement
def group_binary_disagreement(predictions, label):
    pred = np.transpose(predictions, (1, 0))
    return binary_disagreement(pred, label)

# non-pair-wise
from diversityMetrics import entropy
def group_entropy(predictions, label):
    pred = np.transpose(predictions, (1, 0))
    return entropy(pred, label)
from diversityMetrics import kohavi_wolpert_variance
def group_KW_variance(predictions, label):
    pred = np.transpose(predictions, (1, 0))
    return kohavi_wolpert_variance(pred, label)
from diversityMetrics import generalized_diversity
def group_generalized_diversity(predictions, label):
    pred = np.transpose(predictions, (1, 0))
    return generalized_diversity(pred, label)
import statsmodels.stats.inter_rater
def fleiss_kappa_score(predictions):
    pred, _ = statsmodels.stats.inter_rater.aggregate_raters(predictions)
    return statsmodels.stats.inter_rater.fleiss_kappa(pred)

def calDiversityMetric(prediction, target=None, metric='CK'):
    if metric not in _allDiversityMetrics:
        raise Exception("Diversity Metric Not Found!")
    if metric == 'CK':
        return group_kappa_score(prediction)
    if metric == 'QS' and len(target) > 0:
        return group_Q_statistic(prediction, target)
    if metric == 'BD' and len(target) > 0:
        return 1.0 - group_binary_disagreement(prediction, target)
    if metric == 'FK':
        return fleiss_kappa_score(prediction)
    if metric == 'GD' and len(target) > 0:
        return 1.0 - group_generalized_diversity(prediction, target)
    if metric == 'KW' and len(target) > 0:
        return 1.0 - group_KW_variance(prediction, target)
    raise Exception("Diversity Metric Error!")

def calAllDiversityMetrics(prediction, target=None, metrics=None):
    if metrics == None:
        return
    results = list()
    for m in metrics:
        #print(m)
        results.append(calDiversityMetric(prediction, target, m))
    return results
