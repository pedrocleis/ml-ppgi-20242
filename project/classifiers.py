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
    # def PLA(X, y, f):
    def fit(self, X, y, w=np.zeros((3), float)):
        """
        Esta função corresponde ao Algoritmo de Aprendizagem do modelo Perceptron.
        
        Paramêtros:
        - X (list[]): Matriz correspondendo aos dados amostra. Cada elemento de X é uma lista que corresponde 
        às coordenadas dos pontos gerados.
        - y (list): Classificação dos pontos da amostra X.
        - f (list): Lista de dois elementos correspondendo, respectivamente, aos coeficientes angular e linear 
        da função alvo.
        
        Retorno:
        - it (int): Quantidade de iterações necessárias para corrigir todos os pontos classificados incorretamente.
        - w (list): Lista de três elementos correspondendo aos pesos do perceptron.
        """
        it = 0
        # w = np.zeros((3), float)
        listaPCI = self.constroiListaPCI(X, y, w)
        # i = self.iPrimeiroIncorreto(X, y, w)
        # while (len(listaPCI) > 0):
        N = 100 * len(X)
        best_nb_incorrects = len(listaPCI)
        self.w = w.copy()
        while it < N and len(listaPCI) > 0:
            # Escolha aleatoriamente um ponto xi pertencente à lista
            i = listaPCI[random.randint(0, len(listaPCI) - 1)]
            # i = self.iPrimeiroIncorreto(X, y, w)
            # if i < 0:
            #     break
            
            # Atualiza o vetor de pesos ao corrigir a classificação de x
            w[0] += y[i]
            w[1:3] += y[i] * X[i][1:3]

            # Atualiza o melhor classificador encontrado.
            if len(listaPCI) < best_nb_incorrects:
                best_nb_incorrects = len(listaPCI)
                self.w = w.copy()

            # Aqui você deverá contruir a lista de pontos classificados incorretamente
            listaPCI = self.constroiListaPCI(X, y, w) 
            
            # Após atualizar os pesos para correção do ponto escolhido, você irá chamar a função plotGrafico()
            # plot_grafico(X, y, w, f) 

            it += 1
        
        self.it = it
     
    def predict(self, X):
        """
        Classifica os dígitos em X.
        """
        return [1 if iseq(np.sign(self.w[0]*x[0] + self.w[1]*x[1] + self.w[2]*x[2]), 1) else -1 for x in X]
        # return [np.sign(self.w[0]*x[0] + self.w[1]*x[1] + self.w[2]*x[2]) for x in X]
     
    def get_w(self):
        return self.w
    
    def constroiListaPCI(self, X, y, w):
        """
        Esta função constrói a lista de pontos classificados incorretamente.
        
        Paramêtros:
        - X (list[]): Matriz correspondendo aos dados amostra. Cada elemento de X é uma lista que corresponde 
        às coordenadas dos pontos gerados.
        - y (list): Classificação dos pontos da amostra X.
        - w (list): Lista correspondendo aos pesos do perceptron.
    
        Retorno:
        - l (list): Lista com os pontos classificador incorretamente.
        - new_y (list): Nova classificação de tais pontos.
    
        """    
        # Substituição para melhorar a eficiência do algoritmo. Na versão anterior, 10N já era muito lento.
        return np.where(np.sign(w[0]*X[:,0] + w[1]*X[:,1] + w[2]*X[:,2]) != y)[0]
        # l = []
        # for i, x in enumerate(X):
        #     hx = np.sign(w[0]*x[0] + w[1]*x[1] + w[2]*x[2])
        #     if hx != y[i]:
        #         l.append(i)
        # print("l2:", l)
        # return l
    
    def iPrimeiroIncorreto(self, X, y, w):
        """
        Retorna o índice do primeiro ponto classificado incorretamente.
        """    
        # new_y = []
        for i, x in enumerate(X):
            hx = np.sign(w[0]*x[0] + w[1]*x[1] + w[2]*x[2])
            # print(hx, y[i], hx != y[i])
            if hx != y[i]:
                return i
                
        # return l, new_y
        return -1
    
    def get_y(self, x, shift=0):
        """
        Dado um valor de x, computa o valor de y.
        """
        return (-self.w[0]+shift - self.w[1]*x) / self.w[2]
    
class LinearRegression:
    def fit(self, X, y, w=np.zeros((3), float)):
        Xt = np.transpose(X)
        self.w = np.linalg.inv(Xt @ X) @ Xt @ y
     
    def predict(self, X):
        """
        Classifica os dígitos em X.
        """
        # return [digits[0] if iseq(np.sign(np.transpose(self.w) @ x), 1) else digits[1] for x in X]
        return [1 if iseq(np.sign(np.transpose(self.w) @ x), 1) else -1 for x in X]
     
    def get_w(self):
        return self.w
    
    def get_y(self, x, shift=0):
        """
        Dado um valor de x, computa o valor de y.
        """
        return (-self.w[0]+shift - self.w[1]*x) / self.w[2]
    
class LogisticRegression:
    def __init__(self, eta=0.1, tmax=1000, eps=0.00001, bs=1000000):
      self.eta = eta
      self.tmax = tmax
      self.eps = eps
      self.batch_size = bs
      self.w = None

    # Infere o vetor w da funçao hipotese
    #Executa a minimizao do erro de entropia cruzada pelo algoritmo gradiente de descida
    def fit(self, _X, _y, w=np.zeros((3), float)):
        X = np.array(_X)
        y = np.array(_y)

        N = X.shape[0]
        d = X.shape[1]

        for t in range(self.tmax):
            g_t = np.full(d, 0.0)
            for n in range(N):
                num = y[n] * X[n]
                den = (1 + math.exp(y[n] * np.transpose(w) @ X[n]))
                g_t += num / den
            g_t = -(1 / N) * g_t

            if abs(LA.norm(g_t)) < self.eps:
                break
            w += -self.eta * g_t
        self.w = w

    def sigmoid(self, s):
        return 1.0 / (1.0 + math.exp(-s))
        
    # funcao hipotese inferida pela regressa logistica  
    def predict_prob(self, X):
        return [self.sigmoid(np.transpose(self.w) @ x) for x in X]

    # Predicao por classificação linear
    def predict(self, X):
        """
        Classifica os dígitos em X de acordo com a probabilidade.
        """
        return [1 if self.sigmoid(np.transpose(self.w) @ x) >= 0.5 else -1 for x in X]
    
    def get_w(self):
        return self.w

    def get_y(self, x, shift=0):
        """
        Dado um valor de x, computa o valor de y.
        """
        return (-self.w[0]+shift - self.w[1]*x) / self.w[2]