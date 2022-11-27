import numpy as np
import copy
from operator import itemgetter


def getThreshold(target, metric, k=1.0):
    avg = np.mean(target)
    std = np.std(target)
    for i, (m, t) in enumerate(sorted(zip(metric, target))):
        if t < avg - k * std:
            return m
    return np.max(metric)

from sklearn.cluster import KMeans

def getThresholdClusteringKMeans(target, metric, kmeansInit='random'):
    data = [[t, m] for t, m in zip(target, metric)]
    if kmeansInit == 'strategic':
        kmeansInit=np.array([[np.min(target), np.min(metric)],
                             [np.max(target), np.max(metric)]])
    
    kmeans = KMeans(n_clusters=2, init=kmeansInit, n_init=1, random_state=0).fit(data)
    c0 = metric[np.logical_not(kmeans.labels_)]
    c0min, c0max = min(c0), max(c0)
    c1 = metric[np.ma.make_mask(kmeans.labels_)]
    c1min, c1max = min(c1), max(c1)
    centers = kmeans.cluster_centers_
    if centers[0][0] < centers[1][0]:
        return c0max, kmeans
    return c1max, kmeans

def getThresholdClusteringKMeansCenter(target, metric, kmeansInit='random'):
    data = [[t, m] for t, m in zip(target, metric)]
    if kmeansInit == 'strategic':
        kmeansInit=np.array([[np.min(target), np.min(metric)],
                             [np.max(target), np.max(metric)]])
    
    kmeans = KMeans(n_clusters=2, init=kmeansInit, n_init=1, random_state=0).fit(data)
    c0 = metric[np.logical_not(kmeans.labels_)]
    c0min, c0max = min(c0), max(c0)
    c1 = metric[np.ma.make_mask(kmeans.labels_)]
    c1min, c1max = min(c1), max(c1)
    centers = kmeans.cluster_centers_
    if centers[0][0] < centers[1][0]:
        return centers[0][1], kmeans
    return centers[1][1], kmeans

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

def isTeamContainsAny(tA, tBs):
    setA = set(tA)
    for tB in tBs:
        assert len(tA) >= len(tB), "len(tA) >= len(tB)"
        if set(tB).issubset(setA):
            return True
    return False

def centeredMean(nums):
    if len(nums) <= 2:
        return np.mean(nums)
    else:
        return (np.sum(nums) - np.max(nums) - np.min(nums)) / (len(nums) - 2) 

def getNTeamStatistics(teamNameList, accuracyDict, minAcc, avgAcc, maxAcc, tmpAccList, targetAcc=None, calHigherMemberAvg=False):
    if targetAcc is None: # for compatibility
        targetAcc = maxAcc
    nAboveMin = 0
    nAboveAvg = 0
    nAboveMax = 0
    nHigherMember = 0
    nHigherMemberAvg = 0
    nHigherTarget = 0
    allAcc = []
    for teamName in teamNameList:
        acc = accuracyDict[teamName]
        allAcc.append(acc)
        if acc >= round(minAcc, 2):
            nAboveMin += 1
        if acc >= round(avgAcc, 2):
            nAboveAvg += 1
        if acc >= round(maxAcc, 2):
            nAboveMax += 1
            #print(teamName)
        # count whether an ensemble is higher than all its member model
        nHigherMember += 1
        if ',' in teamName:
            teamName = teamName.split(',')
        for modelName in teamName:
            if len(tmpAccList) > 1 and isinstance(tmpAccList[0], list):
                modelAcc = tmpAccList[int(modelName)][0].item()
            else:
                modelAcc = tmpAccList[int(modelName)].item()
            if acc < modelAcc:
                nHigherMember -= 1
                break
        if acc >= targetAcc:
            nHigherTarget += 1
        # count whether an ensemble is higher than member model mean
        if calHigherMemberAvg:
            tmpModelAccList = []
            for modelName in teamName:
                if len(tmpAccList) > 1 and isinstance(tmpAccList[0], list):
                    modelAcc = tmpAccList[int(modelName)][0].item()
                else:
                    modelAcc = tmpAccList[int(modelName)].item()
                tmpModelAccList.append(modelAcc)
            if acc >= np.mean(tmpModelAccList):
                nHigherMemberAvg += 1
    if calHigherMemberAvg:
        return len(teamNameList), np.min(allAcc), np.max(allAcc), np.mean(allAcc), np.std(allAcc), nHigherMember, nAboveMax, nAboveAvg, nAboveMin, nHigherTarget, nHigherMemberAvg
    else:
        return len(teamNameList), np.min(allAcc), np.max(allAcc), np.mean(allAcc), np.std(allAcc), nHigherMember, nAboveMax, nAboveAvg, nAboveMin, nHigherTarget

# random selection
def randomSelection(teamNameList, nRandomSamples = 1, nRepeat = 1, verbose = False):
    selectedTeamLists = []
    for i in range(nRepeat):
        randomIdx = np.random.choice(np.arange(len(teamNameList)), nRandomSamples)
        for idx in randomIdx:
            selectedTeamLists.append(teamNameList[idx])
    if verbose:
        print(selectedTeamLists)
    return selectedTeamLists

def printTopNTeamStatistics(teamNameList, accuracyDict, minAcc, avgAcc, maxAcc, tmpAccList, targetAcc, divScores, dm, topN=5, divFormat="teamName-dm", verbose=False):
    tmpFQTeamNameAccList = []
    for teamName in teamNameList:
        if divFormat == "dm-teamName":
            tmpFQTeamNameAccList.append([divScores[dm][teamName],
                                     teamName, accuracyDict[teamName]])
        else:
            tmpFQTeamNameAccList.append([divScores[teamName][dm],
                                     teamName, accuracyDict[teamName]])
    
    #tmpFQTeamNameAccList.sort()
    
    tmpFQTeamNameAccList = sorted(tmpFQTeamNameAccList, key=itemgetter(2), reverse=True)
    tmpFQTeamNameAccList = sorted(tmpFQTeamNameAccList, key=itemgetter(0), reverse=True)

    tmpFQTeamNameAccList = tmpFQTeamNameAccList[:topN]
    if verbose:
        tmpTeamNameList = []
        for i in range(min(topN, len(tmpFQTeamNameAccList))):
            print(tmpFQTeamNameAccList[i])
            tmpTeamNameList.append(tmpFQTeamNameAccList[i][1])
        print(tmpTeamNameList)
    print(dm, getNTeamStatistics([tmpFTA[1] for tmpFTA in tmpFQTeamNameAccList], 
                             accuracyDict, minAcc, avgAcc, maxAcc, tmpAccList, targetAcc))

