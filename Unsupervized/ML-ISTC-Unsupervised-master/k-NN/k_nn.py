import numpy as np
import json
from sklearn.metrics import accuracy_score
from collections import Counter


def most_common(lst):
    b = Counter(lst)
    return b.most_common(1)[0][0]

def distance(data, dp):
    distances = []
    for o_dp in data:
        distances.append(np.linalg.norm(o_dp-dp))
    return distances

class K_NN:
    def __init__(self, k):
        """
        :param k: number of nearest neighbours
        """
        self.k = k
        
    

    def fit(self, data):
        """
        :param data: 3D array, where data[i, j] is i-th classes j-th point (vector: D dimenstions)
        """
        # TODO: preprocessing
        y = []
        for i in range(data.shape[0]):
            for j in range(data[i].shape[0]):
                y.append(np.concatenate((data[i, j], [i])))
        y = np.array(y)        
        self.X = y[:,:-1]
        self.y = y[:,-1]

    def predict(self, data):
        """
        :param data: 2D array of floats N points each D dimensions
        :return: array of integers
        """
        data = np.array(data)
        shp = data.shape
        if len(data.shape) ==1:
            data = np.array([data])
		
        distances=[]
        for dp in data:
            distances.append(distance(self.X, dp))
        distances = np.array(distances)        
        y_pred = []
        for dist in distances:            
            y_pred.append(most_common(self.y[[dist.argsort()[:self.k]]]))        
        return np.array(y_pred).reshape(shp[:-1])