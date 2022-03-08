# for scikit-learn

import copy
import numpy as np
from sklearn.metrics import accuracy_score

# Test on scikit-learn GradientBoostingClassifier
def getEnsModelPred(
    X,
    ens_wrapper,
    model_ids,
):
    if isinstance(ens_wrapper, EnsWrapper):
        return ens_wrapper.predict(X, model_ids)
    else:
        org_estimators = ens_wrapper.estimators_
        ens_wrapper = copy.deepcopy(ens_wrapper)
        ens_wrapper.estimators_ = org_estimators[model_ids, ...]
        y_pred = ens_wrapper.predict(X)
        return y_pred


class EnsWrapper: # generally follow sklearn API
    def __init__(self, classifiers, names=None, voting='plurality', weights=None):
        self.classifiers = np.array(classifiers, dtype=object)
        self.names = names
        self.voting = voting
        self.weights = weights
    
    def fit(self, X, y):
        for name, clf in zip(names, classifiers):
            clf.fit(X_train, y_train)
            print("finish training of ", name)
    
    def predict(self, X, model_ids=None): # different from sklearn API
        if model_ids:
            classifiers = self.classifiers[model_ids, ...]
        else:
            classifiers = self.classifiers
            
        if self.voting == 'plurality':
            predictions = np.asarray([clf.predict(X) for clf in classifiers]).T.astype(np.int64)
            plural = np.apply_along_axis(lambda x:
                                      np.argmax(np.bincount(
                                          x, weights=self.weights)),
                                      axis=1,
                                      arr=predictions)
            return plural
        else:
            raise NotImplementedError("Voting methods not implemented: ", self.voting)
    
    def score(self, X, y, model_ids=None):
        return accuracy_score(y, self.predict(X, model_ids))


def calNegativeSamplesFocalModel(predictionList, target, oneTargetIdx):
    """Obtain the negative samples for the focal model oneTargetIdx"""
    sampleID = list()
    for i in range(len(target)):
        if predictionList[oneTargetIdx][i] != target[i]:
            sampleID.append(i)
    return sampleID