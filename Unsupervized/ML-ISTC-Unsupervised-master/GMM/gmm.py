import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import sys
sys.path.append("../")
from K_Means.k_means import KMeans
from scipy.stats import multivariate_normal

def calc_prob(data, mus, covs, probs):
    start = time.time()                    
    results = []                   
    for mu, cov, prob in zip(mus, covs, probs):        
        results.append(prob * multivariate_normal.pdf(data, mu, cov))
    results = np.array(results).T    
    final_results = results / np.sum(results, axis=1).reshape(len(results), 1)        
#     print('It took {0:0.4f} seconds'.format(time.time() - start))    
    return np.array(final_results)


class GMM:
    def __init__(self, k, tol=1e-6):
        self.k = k
        self.tol = tol
        self.means = []
        self.covariances = []
        self.pis = []
        self.gammas = []        

    def fit(self, data):
        """
        :params data: np.array of shape (..., dim)
                                  where dim is number of dimensions of point
        """
        data = np.array(data, np.float)
        self._initialize_params(data)
        
        old_means = self.get_means()
        self._E_step(data)
        self._M_step(data)
        
        i = 0
        while np.square(np.linalg.norm(old_means - self.means)) > self.tol:
            print(np.square(np.linalg.norm(old_means - self.means)))
            print(f"iter {i}")
            old_means = self.get_means()
            self._E_step(data)           
            self._M_step(data)
            i +=1

    def _initialize_params(self, data):
        
        kmpp = KMeans(self.k, n_iter=500, tol=1e-8)
        kmpp.fit(data)
        
        self.means = kmpp.means
        
        self.covariances = np.array([np.identity(data.shape[-1])] * self.k) 
        self.pis = np.ones(self.k, np.float)/self.k
        
    def _E_step(self, data):        

        self.gammas = calc_prob(data, self.means, self.covariances, self.pis)       

    def _M_step(self, data):
        gammaSums = self.gammas.sum(axis=0)                                
        self.covariances = np.zeros_like(self.covariances)
        for j in range(self.k):
            for i in range(data.shape[0]):
                a = (data[i] - self.means[j])
                self.covariances[j] += self.gammas[i,j] * np.outer(a, a)
            self.covariances[j] /= gammaSums[j]
            
        self.means = self.gammas.T.dot(data) / np.array([gammaSums]).T
        self.pis = gammaSums/gammaSums.sum()
    
    def predict(self, data):
        """
        :param data: np.array of shape (..., dim)
        :return: np.array of shape (...) without dims
                         each element is integer from 0 to k-1
        """
        gammas = calc_prob(data, self.means, self.covariances, self.pis)
        return np.argmax(gammas, axis=1)

    def get_means(self):
        return self.means.copy()

    def get_covariances(self):
        return self.covariances.copy()

    def get_pis(self):
        return self.pis.copy()