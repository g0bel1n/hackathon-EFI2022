import numpy as np
from math import sqrt
from activation import tanh, softmax_o_sigmoid


class Agent():
    def __init__(self, M, N, rho, mu, delta, T) -> None:
        """Initalizes the Agent object

        Args:
            M (int): Size of the window of past returns
            activation (function): Activation function
        """
        self.M = M
        self.N = N
        self.rho = rho
        self.mu = mu
        self.delta = delta
        self.T = T
        self.activation = tanh if N == 1 else softmax_o_sigmoid
        self.theta = np.random.normal(size=((self.M+1) * self.N + 1 + self.N, self.N))
    
    
    def forward(self, x) -> np.ndarray:
        """Forward pass of the agent.

        Args:
            x (np.ndarray): Shape (M+1) * N. Returns of the N assets over M+1 time periods
            F (np.ndarray): Shape N. Last output of the model

        Returns:
            np.ndarray: _description_
        """
        self.F = self.activation(x @ self.theta)
        return self.F
    
    def compute_A_B_R(self, r) -> None:
        """Computes A, B and R values
        """
        A = 0
        B = 0
        R = [0]
        for t in range(1, self.T+1):
            R_t = self.mu * (np.dot(self.F[t,:],r[t,:]) - self.delta * np.linalg.norm(self.F[t,:]- self.F[t-1,:], ord=1))
            R.append(R_t)
            A += R[t]
            B += R[t]**2
        self.A, self.B, self.R = A/self.T, B/self.T, R
        return self.A, self.B, self.R
        
    def compute_derivatives(self, r, x) -> None:
        """Computes all derivatives needed for the gradient ascent

        Args:
            mu (_type_): _description_
            delta (_type_): _description_
            T (_type_): _description_
            theta (_type_): _description_
            r (_type_): _description_
            x (_type_): _description_
            F (_type_): _description_
        """
        F = self.forward(x)
        A, B, R = self.compute_A_B_R(r)
        S = self.sharpe_ratio()
        
        s_d_theta = 0
        F_d_theta = np.zeros(self.N)
        for t in range(1, T+1):
            first_term = (S * (1 + S**2) * A - S**3 * R[t]) / (A**2 * self.T)
            sgn =  np.sign(F[t,:] - F[t-1,:])
            second_term = (-self.mu * self.delta * sgn) * (1 - F[t,:] @ F[t,:]) * (x[t] + self.theta[1+(self.M+1)*self.N: 1+self.N*(self.M+2), :] * F_d_theta) - (r[t]*self.mu + self.mu*self.delta*sgn) * F_d_theta
            s_d_theta += first_term * second_term
            F_d_theta = (1- F[t,:] @ F[t,:]) * (x[t] + self.theta[1+(self.M+1)*self.N: 1+self.N*(self.M+2), :]*F_d_theta)
        self.s_d_theta = s_d_theta
    
    def gradient_ascent(self) -> None:
        self.theta = self.theta + self.rho * self.s_d_theta
        
    def compute_sharpe_ratio(self) -> float:
        self.sharpe_ratio = self.A / sqrt(self.B - self.A**2)
        return self.sharpe_ratio
    