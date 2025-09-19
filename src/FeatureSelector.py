import numpy as np
from src import Functions

class FeatureSelector:
    def __init__(self, model):
        self.model = model

    def linear(self):
        """
        Select features using a simple linear thresholding rule.
        Returns:
            indices of selected features
        """
        threshold = 1e-6
        selected = np.where(np.abs(self.model.w) > threshold)[0]
        return selected

    def select_and_apply(self, method="linear", kernel=None):
        """
        Select and apply features to the model.

        Parameters
        ----------
        method : str
            Feature selection method (currently only 'linear' supported).
        kernel : callable
            Kernel function.

        Returns
        -------
        model : object
            Updated model with reduced features.
        """
        if method == "linear":
            selected_features = self.linear()
        else:
            raise ValueError(f"Unknown feature selection method: {method}")

        # Apply selected features
        self.model.X = self.model.X[:, selected_features]
        if self.model.X_test is not None:
            self.model.X_test = self.model.X_test[:, selected_features]

        self.model.w = self.model.w[selected_features]
        self.model.delta_power2 = self.model.delta_power2[selected_features]

        return self.model

