import numpy as np
# import matplotlib.pyplot as plt
import random
# from IPython.display import clear_output
# from collections import namedtuple
from numpy import linalg as LA
import math

def iseq(a, b):
    return abs(a - b) < 1e-5

class PocketLearningAlgorithm:
    def __init__(self, max_iter=1000):
        self.max_iter = max_iter
        self.w = None

    def error_in(self, X, y):
        """
        Calcula o erro de classificação, i.e., a quantidade de pontos 
        classificados incorretamente.
        """
        error = np.where(np.sign(self.w[0]*X[:,0] 
                                 + self.w[1]*X[:,1] 
                                 + self.w[2]*X[:,2]) != y, 1, 0)
        return np.sum(error)
        # error = 0
        # for i in range(len(y)):
        #     if np.sign(np.dot(self.w, X[i])) != y[i]:
        #         error += 1
        # return error

    def fit(self, X, y):
        # inicialização
        X = np.array(X)
        y = np.array(y)
        best_w = np.zeros(X.shape[1])
        self.w = np.zeros(X.shape[1])
        best_error = len(y) # pior caso

        for _ in range(self.max_iter):
            has_error = False
            for i in range(len(y)):
                # para o primeiro ponto classificado incorretamente
                if np.sign(np.dot(self.w, X[i])) != y[i]:
                    has_error = True
                    # atualiza o vetor de pesos ao corrigir a classificação de x
                    self.w = self.w + (y[i] * X[i])
                    error_in = self.error_in(X, y)
                    if best_error > error_in:
                        best_error = error_in
                        best_w = np.copy(self.w)
            if not has_error:
                break
        
        self.w = best_w
    
    def predict(self, X):
        """
        Classifica os dígitos em X.
        """
        return np.sign(np.dot(X, self.w))
    
    def get_w(self):
        """
        Retorna os pesos do modelo.
        """
        return self.w
    
    def get_y(self, x, shift=0):
        """
        Dado o valor de x, computa o valor de y.
        """
        return (-self.w[0]+shift - self.w[1]*x) / self.w[2]
    
class LinearRegression:
    def fit(self, X, y):
        """
        Calcula os pesos do modelo de regressão linear.
        """
        Xt = np.transpose(X)
        self.w = np.linalg.inv(Xt @ X) @ Xt @ y
     
    def predict(self, X):
        """
        Classifica os dígitos em X.
        """
        return np.sign(np.dot(X, self.w))

     
    def get_w(self):
        """
        Retorna os pesos do modelo.
        """
        return self.w
    
    def get_y(self, x, shift=0):
        """
        Dado um valor de x, computa o valor de y.
        """
        return (-self.w[0]+shift - self.w[1]*x) / self.w[2]
    
class LogisticRegression:
    def __init__(self, eta=0.1, tmax=1000, eps=1e-8, bs=1000000):
      self.eta = eta
      self.tmax = tmax
      self.eps = eps
      self.batch_size = bs
      self.w = None

    # Infere o vetor w da funçao hipotese
    # Executa a minimizao do erro de entropia cruzada pelo algoritmo gradiente 
    # de descida
    def fit(self, _X, _y):
        """
        Calcula os pesos do modelo de regressão logística.
        """
        X = np.array(_X)
        y = np.array(_y)

        N = X.shape[0]
        d = X.shape[1]

        w=np.zeros((3), float)

        for _ in range(self.tmax):
            g_t = np.full(d, 0.0)
            for n in range(N):
                num = y[n] * X[n]
                den = (1 + math.exp(y[n] * np.transpose(w) @ X[n]))
                g_t += num / den
            g_t = -(1 / N) * g_t

            # if abs(LA.norm(g_t)) < self.eps:
            #     break
            w += -self.eta * g_t
        self.w = w

    def sigmoid(self, s):
        return 1.0 / (1.0 + math.exp(-s))
        
    # funcao hipotese inferida pela regressa logistica  
    def predict_prob(self, X):
        """
        Calcula as probabilidades associadas a cada x em X.
        """
        return [self.sigmoid(np.transpose(self.w) @ x) for x in X]

    # Predicao por classificação linear
    def predict(self, X):
        """
        Classifica os dígitos em X de acordo com a probabilidade.
        """
        return [1 if self.sigmoid(np.transpose(self.w) @ x) >= 0.5 
                else -1 for x in X]
    
    def get_w(self):
        """
        Retorna os pesos do modelo.
        """
        return self.w

    def get_y(self, x, shift=0):
        """
        Dado um valor de x, computa o valor de y.
        """
        return (-self.w[0]+shift - self.w[1]*x) / self.w[2]