import numpy as np
import copy

def getThreshold(target, metric, k=1.0):
    avg = np.mean(target)
    std = np.std(target)
    for i, (m, t) in enumerate(sorted(zip(metric, target))):
        if t < avg - k * std:
            return m
    return np.max(metric)

from sklearn.cluster import KMeans
def getThresholdFromClusteringKMeans(target, metric, kmeansInit='random'):
    data = [[t, m] for t, m in zip(target, metric)]
    kmeans = KMeans(n_clusters=2, init=kmeansInit, random_state=0).fit(data)
    c0 = metric[np.ma.make_mask(kmeans.labels_)]
    c0min, c0max = min(c0), max(c0)
    c1 = metric[np.logical_not(kmeans.labels_)]
    c1min, c1max = min(c1), max(c1)
    if c0min > c1max or c1min > c0max:
        return max(c0min, c1min)
    return min(c0max, c1max)

def getThresholdFromKMeans(target, metric, kmeansInit='random'):
    data = [[t, m] for t, m in zip(target, metric)]
    kmeans = KMeans(n_clusters=2, init=kmeansInit, random_state=0).fit(data)
    c0 = metric[np.ma.make_mask(kmeans.labels_)]
    c0min, c0max = min(c0), max(c0)
    c1 = metric[np.logical_not(kmeans.labels_)]
    c1min, c1max = min(c1), max(c1)
    if c0min > c1max or c1min > c0max:
        return max(c0min, c1min)
    return min(c0max, c1max), kmeans

def getThresholdClusteringKMeans(target, metric, kmeansInit='random'):
    data = [[t, m] for t, m in zip(target, metric)]
    if kmeansInit == 'strategic':
        kmeansInit=np.array([[np.max(target), np.min(metric)],
                             [np.min(target), np.max(metric)]])
    
    kmeans = KMeans(n_clusters=2, init=kmeansInit, random_state=0).fit(data)
    c0 = metric[np.logical_not(kmeans.labels_)]
    c0min, c0max = min(c0), max(c0)
    c1 = metric[np.ma.make_mask(kmeans.labels_)]
    c1min, c1max = min(c1), max(c1)
    centers = kmeans.cluster_centers_
    if centers[0][0] > centers[1][0]:
        return c1min, kmeans
    return c0min, kmeans

def getThresholdClusteringKMeansCenter(target, metric, kmeansInit='random'):
    data = [[t, m] for t, m in zip(target, metric)]
    if kmeansInit == 'strategic':
        kmeansInit=np.array([[np.max(target), np.min(metric)],
                             [np.min(target), np.max(metric)]])
    
    kmeans = KMeans(n_clusters=2, init=kmeansInit, random_state=0).fit(data)
    c0 = metric[np.logical_not(kmeans.labels_)]
    c0min, c0max = min(c0), max(c0)
    c1 = metric[np.ma.make_mask(kmeans.labels_)]
    c1min, c1max = min(c1), max(c1)
    centers = kmeans.cluster_centers_
    if centers[0][0] > centers[1][0]:
        return centers[1][1], kmeans
    return centers[0][1], kmeans

def oneThirdThreshold(metric):
    metricSort = copy.deepcopy(metric)
    metricSort.sort()
    for i in range(len(metric)):
        if i >= len(metric)/3.0:
            return metric[i]

def normalize01(array):
    if max(array) == min(array): #TODO: to consider more cases
        return array
    return (array-min(array))/(max(array)-min(array))

def centeredMean(nums):
    if len(nums) <= 2:
        return np.mean(nums)
    else:
        return (np.sum(nums) - np.max(nums) - np.min(nums)) / (len(nums) - 2) 

def getNTeamStatistics(accuracyList, minAcc, avgAcc, maxAcc):
    nAboveMin = 0
    nAboveAvg = 0
    nAboveMax = 0
    for acc in accuracyList:
        if acc >= round(minAcc, 2):
            nAboveMin += 1
        if acc >= round(avgAcc, 2):
            nAboveAvg += 1
        if acc >= round(maxAcc, 2):
            nAboveMax += 1
    return len(accuracyList), nAboveMin, nAboveAvg, nAboveMax
