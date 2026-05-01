"""
pinnflow/activations.py
────────────────────────
Scalar/array activation functions and their element-wise gradients.
All functions are pure numpy — no external dependencies.
"""
import numpy as np


def relu(x):
    return np.maximum(0, x)

def relu_g(x):
    return (x > 0).astype(float)

def tanh(x):
    return np.tanh(x)

def tanh_g(x):
    return 1 - np.tanh(x) ** 2

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

def swish(x):
    return x * sigmoid(x)

def swish_g(x):
    s = sigmoid(x)
    return s + x * s * (1 - s)
