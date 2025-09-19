import numpy as np
from src import Functions, Kernels
from src.trainer import LpLinftyTrainer
from src.FeatureSelector import FeatureSelector
from sklearn.svm import SVC  # for non-linear cases

class LpLinftySVM:
    def __init__(self, C=1.0, b=0.0, kernel=Kernels.linear_kernel, tol=1e-6, eps=1e-6, p=0.8):
        self.C = C
        self.b = b
        self.kernel = kernel
        self.tol = tol
        self.eps = eps
        self.p = p

        self.X = None
        self.y = None
        self.X_test = None
        self.alphas = None
        self.beta = None
        self.delta_power2 = None
        self.w = None
        self.errors = None
        self.svc_model = None  # only for non-linear

    def fit(self, X_train, y_train, X_test):
        self.X = X_train
        self.y = y_train
        self.X_test = X_test
        self.m, self.d = X_train.shape

        # Initialize variables
        self.alphas = np.zeros(self.m)
        self.beta = np.zeros(self.m)
        self.beta[1] = self.C / 2
        self.delta_power2 = np.ones(self.d) * 1e6
        self.w = np.ones(self.d)
        self.errors = np.empty(self.m)

        # Compute initial decision function errors
        for i in range(self.m):
            xi = self.X[i]
            self.errors[i] = Functions.decision_function(
                xi, self.X, self.w, self.b, self.alphas, self.y, self.kernel, self.delta_power2
            ) - self.y[i]

        # Always train Lp-L∞ SVM with linear kernel first
        original_kernel = self.kernel  # Save user-specified kernel
        self.kernel = Kernels.linear_kernel  # Temporarily set linear kernel
        trainer_instance = LpLinftyTrainer(self)
        trainer_instance.train(max_iter=100)

        # Feature selection and apply
        selector = FeatureSelector(self)
        selector.select_and_apply(method="linear", kernel=self.kernel)

        # Restore user's original kernel
        self.kernel = original_kernel

        # If user requested non-linear kernel, use classic SVC on selected features
        if self.kernel != Kernels.linear_kernel:
          self.svc_model = SVC(C=self.C, kernel=self.kernel)
          self.svc_model.fit(self.X, self.y)

        return self

    def predict(self):
        if self.kernel == Kernels.linear_kernel:
        # Linear kernel: vectorized computation using w
        f_x = self.X_test @ self.w + self.b
        preds = np.where(f_x > 0, 1, -1)
    else:
        # Non-linear kernel: use the trained SVC
        preds = self.svc_model.predict(self.X_test)
    return preds
