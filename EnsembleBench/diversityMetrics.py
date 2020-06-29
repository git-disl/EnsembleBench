import numpy as np


def cohen_kappa_statistic(M):
    """
     M: the multiple inputs, each input represents a model output.
    """
    from sklearn.metrics import cohen_kappa_score
    Qs = []
    for i in range(M.shape[0]):
        for j in range(i + 1, M.shape[0]):
            Qs.append(cohen_kappa_score(M[i, :], M[j, :]))
    return np.mean(Qs)

def correlation_coefficient(M, y_true):
    Qs = []
    for i in range(M.shape[0]):
        for j in range(i + 1, M.shape[0]):
            N_1_1 = np.sum(np.logical_and(y_true == M[i, :], y_true == M[j, :]))  # number of both correct
            N_0_0 = np.sum(np.logical_and(y_true != M[i, :], y_true != M[j, :]))  # number of both incorrect
            N_0_1 = np.sum(np.logical_and(y_true != M[i, :], y_true == M[j, :]))  # number of j correct but not i
            N_1_0 = np.sum(np.logical_and(y_true == M[i, :], y_true != M[j, :]))  # number of i correct but not j
            Qs.append((N_1_1*N_0_0-N_0_1*N_1_0) * 1. / (np.sqrt((N_1_1+N_1_0)*(N_0_1+N_0_0)*(N_1_1+N_0_1)*(N_1_0+N_0_0))+np.finfo(float).eps)+np.finfo(float).eps)
    return np.mean(Qs)


def Q_statistic(M, y_true):
    Qs = []
    for i in range(M.shape[0]):
        for j in range(i+1, M.shape[0]):
            N_1_1 = np.sum(np.logical_and(y_true == M[i, :], y_true == M[j, :]))  # number of both correct
            N_0_0 = np.sum(np.logical_and(y_true != M[i, :], y_true != M[j, :]))  # number of both incorrect
            N_0_1 = np.sum(np.logical_and(y_true != M[i, :], y_true == M[j, :]))  # number of j correct but not i
            N_1_0 = np.sum(np.logical_and(y_true == M[i, :], y_true != M[j, :]))  # number of i correct but not j
            Qs.append((N_1_1*N_0_0 - N_0_1*N_1_0)*1./(N_1_1*N_0_0+N_0_1*N_1_0+np.finfo(float).eps))
    return np.mean(Qs)

def binary_disagreement(M, y_true):
    Qs = []
    for i in range(M.shape[0]):
        for j in range(i + 1, M.shape[0]):
            N_1_1 = np.sum(np.logical_and(y_true == M[i, :], y_true == M[j, :]))  # number of both correct
            N_0_0 = np.sum(np.logical_and(y_true != M[i, :], y_true != M[j, :]))  # number of both incorrect
            N_0_1 = np.sum(np.logical_and(y_true != M[i, :], y_true == M[j, :]))  # number of j correct but not i
            N_1_0 = np.sum(np.logical_and(y_true == M[i, :], y_true != M[j, :]))  # number of i correct but not j
            Qs.append((N_0_1+N_1_0)*1./(N_1_1+N_1_0+N_0_1+N_0_0))
    return np.mean(Qs)


def fleiss_kappa_statistic(M, y_true, n_classes=10):
    M_ = np.zeros(shape=(M.shape[1], n_classes))
    for row in M:
        for sid in range(len(row)):
            M_[sid, row[sid]] += 1

    N, k = M_.shape  # N is # of items, k is # of categories
    n_annotators = float(np.sum(M_[0, :]))  # # of annotators

    p = np.sum(M_, axis=0) / (N * n_annotators)
    P = (np.sum(M_ * M_, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
    Pbar = np.sum(P) / N
    PbarE = np.sum(p * p)

    return (Pbar - PbarE) / (1 - PbarE)


def entropy(M, y_true):
    N = M.shape[1]
    L = M.shape[0]
    E = 0.
    for j in range(M.shape[1]):
        l_zj = list(M[:, j]).count(y_true[j])
        E += min(l_zj, L-l_zj)
    E = E * 1. / (N * (L - np.ceil(L / 2.)))
    return E

def kohavi_wolpert_variance(M, y_true):
    N = M.shape[1]
    L = M.shape[0]
    kw = 0.
    for j in range(N):
        l_zj = list(M[:, j]).count(y_true[j])
        kw += l_zj * (L-l_zj)
    kw = kw * 1. / (N * L * L)
    return kw

def generalized_diversity(M, y_true):
    N = M.shape[1]
    L = M.shape[0]
    pi = np.zeros(N)
    for i in range(N):
        pIdx = 0
        for j in range(L):
            if M[j][i] != y_true[i]:
                pIdx += 1
        pi[pIdx] += 1
    
    pi = [x*1.0/N for x in pi]
    
    P1 = 0
    P2 = 0
    for i in range(N):
        P1 += i * 1.0 * pi[i] / L
        P2 += i * (i-1) * 1.0 * pi[i] / (L * (L-1))  
    return 1.0-P2/P1 
    
    
        
