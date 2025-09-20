def get_kernel(name, **kwargs):
    """
    Return a kernel function based on its name.

    Parameters
    ----------
    name : str
        Kernel name ('linear', 'poly', 'rbf', 'sigmoid')
    kwargs : dict
        Extra parameters for kernels (gamma, degree, coef0)

    Returns
    -------
    callable
        A function kernel(X, Z) that returns the kernel matrix
    """ 
    if name == "linear":
        def linear(X, Z, delta):
           delta_inv = Functions.compute_delta_inverse(delta)
           X_scaled = X * delta_inv
           Z_scaled = Z * delta_inv
           return X_scaled @ Z_scaled.T
        return linear

    elif name == "poly":
        degree = kwargs.get("degree", 3)
        coef0 = kwargs.get("coef0", 1.0)
        gamma = kwargs.get("gamma", 1.0)
        def poly(X, Z,delta=None):
            return (gamma * (X @ Z.T) + coef0) ** degree
        return poly

    elif name == "rbf":
        gamma = kwargs.get("gamma", 0.05)
        def rbf(X, Z,delta=None):
            n_X = X.shape[0]
            n_Z = Z.shape[0]
            K = np.empty((n_X, n_Z))
            for i in range(n_X):
                for j in range(n_Z):
                    K[i, j] = np.exp(-gamma * np.sum((X[i] - Z[j])**2))
            return K
        return rbf

    elif name == "sigmoid":
        gamma = kwargs.get("gamma", 0.1)
        coef0 = kwargs.get("coef0", 0.0)
        
        def sigmoid(X, Z,delta=None):
            return np.tanh(gamma * (X @ Z.T) + coef0)
        return sigmoid

    else:
        raise ValueError(f"Invalid kernel '{name}'. Available: linear, poly, rbf, sigmoid")
