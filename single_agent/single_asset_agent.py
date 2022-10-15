import numpy as np

class SingleAssetAgent:


    def __init__(self, M, lr) -> None:
        self.theta = np.random.rand(M + 2)
        self.lr = lr


    def positions(self, x):

        M = len(self.theta) - 2
        T = len(x)
        Ft = np.zeros(T)
        for t in range(M, T):
            xt = np.concatenate([[1], x[t - M:t], [Ft[t - 1]]])
            Ft[t] = np.tanh(np.dot(self.theta, xt))
        return Ft

    def update(self,etfs_returns, delta, R):
        grad,S = self.gradient(etfs_returns, delta, R)
        self.theta = self.theta + grad * self.lr
        return S

    def gradient(self, etfs_returns, delta, R):
        Ft = self.positions(etfs_returns)
        T = len(etfs_returns)
        M = len(self.theta) - 2
        
        A = np.mean(R)
        B = np.mean(np.square(R))
        S = A / np.sqrt(B - A ** 2)

        dSdA = S * (1 + S ** 2) / A
        dSdB = -S ** 3 / 2 / A ** 2
        dAdR = 1. / T
        dBdR = 2. / T * R
        
        grad = np.zeros(M + 2)  # initialize gradient
        dFpdtheta = np.zeros(M + 2)  # for storing previous dFdtheta
        
        for t in range(M, T):
            xt = np.concatenate([[1], etfs_returns[t - M:t], [Ft[t-1]]])
            dRdF = -delta * np.sign(Ft[t] - Ft[t-1])
            dRdFp = etfs_returns[t] + delta * np.sign(Ft[t] - Ft[t-1])
            dFdtheta = (1 - Ft[t] ** 2) * (xt + self.theta[-1] * dFpdtheta)
            dSdtheta = (dSdA * dAdR + dSdB * dBdR[t]) * (dRdF * dFdtheta + dRdFp * dFpdtheta)
            grad = grad + dSdtheta
            dFpdtheta = dFdtheta

        
        return grad, S

