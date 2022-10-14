import numpy as np
from torch import qscheme
from activation import tanh, softmax_o_sigmoid


class Agent():
    def __init__(self, M, N) -> None:
        """Initalizes the Agent object

        Args:
            M (int): Size of the window of past returns
            activation (function): Activation function
        """
        self.M = M
        self.N - N
        self.activation = tanh if N == 1 else softmax_o_sigmoid
        self.theta = np.random.normal(((self.M+1) * self.N + 1 + self.N, self.N))
    
    
    def forward(self, X, F) -> np.ndarray:
        """Forward pass of the agent.

        Args:
            X (np.ndarray): Shape (M+1) * N. Returns of the N assets over M+1 time periods
            F (np.ndarray): Shape N. Last output of the model

        Returns:
            np.ndarray: _description_
        """
        self.X = np.array([1, * self.theta.flatten(), *X])
        return self.activation(X @ self.theta)
    
    def gradient_ascent(self) -> None:
        pass
        
    