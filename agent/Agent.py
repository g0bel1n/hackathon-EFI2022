import numpy as np

class Agent():
    def __init__(self, shape, M, activation) -> None:
        """Initalizes the Agent object

        Args:
            shape (Iterable): List of the number of neurons per layers
            M (int): Size of the window of past returns
            activation (function): Activation function
        """
        self.shape = shape
        self.M = M
        self.activation = activation
    
    def init_theta_0(self):
        pass
    
    def forward(self, X) -> np.ndarray:
        pass
    
    def gradient_ascent(self) -> None:
        pass
        