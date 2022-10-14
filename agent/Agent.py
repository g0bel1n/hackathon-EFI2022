import numpy as np
from math import sqrt
from rich import print

tanh = np.tanh

sigmoid = lambda x: 1 / (1 + np.exp(-x))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

softmax_o_sigmoid = lambda x: softmax(sigmoid(x))


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
        self.theta = np.random.normal(size=(self.M+3))
    
    
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
    
    def _compute_A_B_R(self, r, F_s):
        """Computes A, B and R values
        """
        A = 0
        B = 0
        R = [0]
<<<<<<< HEAD
=======

>>>>>>> 546c664702edb23adcb5f3f0c285a82b2ed2c1b6
       
        for t in range(self.T-self.M):

            R_t = self.mu * (F_s[t,:] @ r[t,:] - self.delta * np.linalg.norm(F_s[t,:]- F_s[t-1,:], ord=1))
            R.append(R_t)
            A += R[t]
            B += R[t]**2
        self.A, self.B, self.R = A/self.T, B/self.T, R
        return self.A, self.B, self.R
        
    def compute_derivatives(self, r, x, F_s) -> None:
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
        A, _, R = self._compute_A_B_R(r, F_s=F_s)
        S = self.compute_sharpe_ratio()
        
        s_d_theta = 0
        F_d_theta = np.zeros(size=(self.N, self.M+3))
        for t in range(self.T-self.M):
            first_term = (S * (1 + S**2) * A - S**3 * R[t]) / (A**2 * self.T)
            sgn =  np.sign(F_s[t,:] - F_s[t-1,:])
            second_term = (-self.mu * self.delta * sgn) * (1 - F_s[t,:] @ F_s[t,:]) * (x + self.theta[-1] * F_d_theta) - (r[t,:]*self.mu + self.mu*self.delta*sgn) * F_d_theta

            s_d_theta += first_term * second_term
            F_d_theta = (1- F_s[t,:] @ F_s[t,:]) * (x + self.theta[-1]*F_d_theta)
        self.s_d_theta = s_d_theta
    
    def gradient_ascent(self) -> None:

        self.theta = np.sum((self.theta , (self.rho * self.s_d_theta)))
        print(self.theta.shape)
        
    def compute_sharpe_ratio(self) -> float:
        self.sharpe_ratio = self.A / sqrt(self.B - self.A**2)
        return self.sharpe_ratio
    