import numpy as np

class SingleAssetAgent:


    def __init__(self, M, lr) -> None:
        self.theta = np.random.rand(M + 2)
        self.lr = lr
        self.M = M


    def positions(self, x):

        T = len(x)
        Ft = np.zeros(T)
        for t in range(self.M, T):
            xt = np.concatenate([[1], x[t - self.M:t], [Ft[t - 1]]])
            Ft[t] = np.tanh(np.dot(self.theta, xt))
        return Ft

    def update(self,etfs_returns, delta, R):
        grad,S = self._gradient(etfs_returns, delta, R)
        self.theta = self.theta + grad * self.lr
        return S

    def _gradient(self, etfs_returns, delta, R):
        Ft = self.positions(etfs_returns)
        T = len(etfs_returns)
        
        A = np.mean(R)
        B = np.mean(np.square(R))
        S = A / np.sqrt(B - A ** 2)

        dS_dA = S * (1 + S ** 2) / A
        dS_dB = -S ** 3 / 2 / A ** 2
        dA_dR = 1. / T
        dB_dR = 2. / T * R
        
        grad = np.zeros(self.M + 2)  
        dFp_dtheta = np.zeros(self.M + 2)  

        for t in range(self.M, T):
            xt = np.concatenate([[1], etfs_returns[t - self.M:t], [Ft[t-1]]])
            dR_dF = -delta * np.sign(Ft[t] - Ft[t-1])
            dR_dFp = etfs_returns[t] + delta * np.sign(Ft[t] - Ft[t-1])
            dF_dtheta = (1 - Ft[t] ** 2) * (xt + self.theta[-1] * dFp_dtheta)
            dS_dtheta = (dS_dA * dA_dR + dS_dB * dB_dR[t]) * (dR_dF * dF_dtheta + dR_dFp * dFp_dtheta)
            grad = grad + dS_dtheta
            dFp_dtheta = dF_dtheta

        
        return grad, S

