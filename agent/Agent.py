import numpy as np
from math import sqrt
from activation import tanh, softmax_o_sigmoid


class Agent():
    def __init__(self, M, N, rho) -> None:
        """Initalizes the Agent object

        Args:
            M (int): Size of the window of past returns
            activation (function): Activation function
        """
        self.M = M
        self.N = N
        self.rho = rho
        self.activation = tanh if N == 1 else softmax_o_sigmoid
        self.theta = np.random.normal(size=((self.M+1) * self.N + 1 + self.N, self.N))
    
    
    def forward(self, X) -> np.ndarray:
        """Forward pass of the agent.

        Args:
            X (np.ndarray): Shape (M+1) * N. Returns of the N assets over M+1 time periods
            F (np.ndarray): Shape N. Last output of the model

        Returns:
            np.ndarray: _description_
        """
        return self.activation(X @ self.theta)
    
    def compute_A_B_R(self, mu, delta, T, r, F) -> None:
        """Computes A, B and R values
        """
        A = 0
        B = 0
        R = [0]
        for t in range(1, T+1):
            R_t = mu * (np.dot(F[t,:],r[t,:]) - delta * np.linalg.norm(F[t,:]- F[t-1,:], ord=1))
            R.append(R_t)
            A += R[t]
            B += R[t]**2
        self.A, self.B, self.R = A/T, B/T, R
        return self.A, self.B, self.R
        
    def compute_derivatives(self, mu, delta, T, theta, r, x, F) -> None:
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
        A, B, R = self.compute_A_B_R(mu, delta, T, r, F)
        S = self.sharpe_ratio()
        s_d_theta = 0
        F_d_theta = np.zeros(self.N)
        for t in range(1, T+1):
            first_term = (S * (1 + S**2) * A - S**3 * R[t]) / (A**2 * T)
            sgn =  np.sign(F[t,:] - F[t-1,:])
            second_term = (-mu * delta * sgn) * (1 - F[t,:] @ F[t,:]) * (x[t] + theta[1+(self.M+1)*self.N: 1+self.N*(self.M+2), :] * F_d_theta) - (r[t]*mu + mu*delta*sgn) * F_d_theta
            s_d_theta += first_term * second_term
            F_d_theta = (1- F[t,:] @ F[t,:]) * (x[t] + theta[1+(self.M+1)*self.N: 1+self.N*(self.M+2), :]*F_d_theta)
        self.s_d_theta = s_d_theta
    
    def gradient_ascent(self) -> None:
        self.theta = self.theta + self.rho * self.s_d_theta
        
    def sharpe_ratio(self) -> float:
        return self.A / sqrt(self.B - self.A**2)
    