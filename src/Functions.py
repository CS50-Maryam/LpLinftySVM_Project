import numpy as np
import src.Kernels
def compute_L_H(C, alpha1, alpha2, beta1, beta2, y1, y2):
    """
    Compute lower (L) and upper (H) bounds for alpha2.
    """
    if y1 == y2:
        L = max(-beta2, -beta1 + alpha1 + alpha2 - C, -2*C + alpha1 + alpha2 + 2*beta1)
        H = min(C - beta2, beta1 + alpha1 + alpha2, alpha1 + alpha2 + 2*beta1)
    else:
        L = max(-beta2, -beta1 + alpha2 - alpha1, -2*C + alpha1 + alpha2 + 2*beta1)
        H = min(C - beta2, C - beta1 + alpha2 - alpha1, alpha1 + alpha2 + 2*beta1)
    return L, H


def compute_b1_b2(E1, y1, a1, alpha1, k11, y2, a2, alpha2, k12, b, E2, k22):
    """
    Compute intermediate bias values b1 and b2.
    """
    b1 = b - E1 - y1 * (a1 - alpha1) * k11 - y2 * (a2 - alpha2) * k12
    b2 = b - E2 - y1 * (a1 - alpha1) * k12 - y2 * (a2 - alpha2) * k22
    return b1, b2


def compute_b(a1, a2, b1, b2, C):
    """
    Compute final bias b based on alpha values.
    """
    if 0 < a1 < C:
        return b1
    elif 0 < a2 < C:
        return b2
    else:
        return 0.5 * (b1 + b2)


def compute_delta_squared(delta_power2, w, alpha1, alpha2, a1, a2, y1, y2, b_old, b_new, x1, x2, p):
    """
    Compute updated delta squared values.
    """
    epsilon = 1e-8
    delta_power2 = np.power(np.abs(w) + epsilon, 2 - 2 * p)
    return delta_power2



import numpy as np

def compute_delta_inverse(delta_squared, epsilon=1e-6):
    """
    Compute the inverse of delta (1/sqrt(delta_squared)),
    avoiding division by zero in a vectorized way.
    """
    if delta_squared is None:
        return np.array([1/np.sqrt(epsilon)])
    
    safe_delta_sq = np.maximum(delta_squared, epsilon)
    return 1.0 / np.sqrt(safe_delta_sq)



def compute_w(delta_power2, r, y, X, eps=1e-6):
    """
    Compute weight vector w.
    """
    delta_inv2 = 1.0 / (delta_power2 + eps)   # ≈ δ⁻²
    w = (r * y) @ (delta_inv2 * X)
    return w



def decision_function(x_new, X, w=None, b=0, r=None, y=None, kernel="linear", delta=None):
    """
    Compute the decision function f(x) for a given sample.
    
    Parameters:
        x_new (np.ndarray): New data point
        X (np.ndarray): Training data
        w (np.ndarray): Weight vector (for linear kernel)
        b (float): Bias term
        kernel (function): Kernel function
        
    """
    delta_inv = compute_delta_inverse(delta)
    x_tilde = x_new * delta_inv
    return np.dot(w, x_tilde) + b
