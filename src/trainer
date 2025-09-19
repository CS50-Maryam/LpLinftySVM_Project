import numpy as np
from src import Functions, Kernels

class LpLinftyTrainer:
    def __init__(self, model):
        """
        Trainer class for Lp-L∞ SVM.

        Parameters:
            model (SMOModel): Initialized model object containing data, labels, kernel, etc.
        """
        self.model = model

    def takeStep(self, i1: int, i2: int):
        """
        Perform optimization step for a pair of indices (i1, i2) in the model.
        Returns 1 if step updated the model, else 0.
        """
        model = self.model

        if i1 == i2:
            return 0

        # Extract variables
        alpha1, alpha2 = model.alphas[i1], model.alphas[i2]
        beta1, beta2   = model.beta[i1], model.beta[i2]
        y1, y2         = model.y[i1], model.y[i2]
        x1, x2         = model.X[i1], model.X[i2]
        w              = model.w
        delta_sq       = model.delta_power2
        E1, E2         = model.errors[i1], model.errors[i2]
        s              = y1 * y2

        # Compute bounds
        L, H = Functions.compute_L_H(model.C, alpha1, alpha2, beta1, beta2, y1, y2)
        if L == H:
            return 0

        delta_inv = Functions.compute_delta_inverse(delta_sq)
        k11 = model.kernel(x1, x1, delta_inv)
        k12 = model.kernel(x1, x2, delta_inv)
        k22 = model.kernel(x2, x2, delta_inv)

        eta = k11 + k22 - 2 * k12
        if eta > 0:
            a2 = alpha2 + y2 * (E1 - E2) / eta
            a2 = np.clip(a2, L, H)
        else:
            f1 = y1 * (E1 + model.b) - alpha1 * k11 - s * alpha2 * k12
            f2 = y2 * (E2 + model.b) - s * alpha1 * k12 - alpha2 * k22
            L1, H1 = alpha1 + s * (alpha2 - L), alpha1 + s * (alpha2 - H)
            Lobj = L1 * f1 + L * f2 + 0.5 * L1**2 * k11 + 0.5 * L**2 * k22 + s * L * L1 * k12
            Hobj = H1 * f1 + H * f2 + 0.5 * H1**2 * k11 + 0.5 * H**2 * k22 + s * H * H1 * k12
            a2 = L if Lobj < Hobj + model.eps else H if Lobj > Hobj - model.eps else alpha2

        # Clip a2
        a2 = 0.0 if a2 < 1e-8 else model.C if a2 > model.C - 1e-6 else a2
        a1 = alpha1 + s * (alpha2 - a2)
        be1 = 0.5 * (alpha1 + alpha2 + 2 * beta1 - a1 - a2)

        if abs(a2 - alpha2) + abs(a1 - alpha1) + abs(be1 - beta1) < model.tol:
            return 0

        # Update bias
        b_old = model.b
        b1, b2 = Functions.compute_b1_b2(E1, y1, a1, alpha1, k11,
                                         y2, a2, alpha2, k12,
                                         b_old, E2, k22)
        b_new = Functions.compute_b(a1, a2, b1, b2, model.C)

        # Update delta squared and weight for linear kernel
        if model.kernel is Kernels.linear_kernel:
            delta_sq = Functions.compute_delta_squared(
                delta_sq, w, alpha1, alpha2, a1, a2,
                y1, y2, b_old, b_new, x1, x2, model.p
            )
            model.w = Functions.compute_w(delta_sq, model.alphas, model.y, model.X)

        # Update model variables
        model.alphas[i1], model.alphas[i2] = a1, a2
        model.beta[i1] = be1
        model.delta_power2 = delta_sq
        model.b = b_new

        # Update errors
        for idx, a in zip([i1, i2], [a1, a2]):
            if 0.0 < a < model.C:
                model.errors[idx] = 0.0

        NonOpt = [n for n in range(model.m) if n not in (i1, i2)]
        model.errors[NonOpt] += (
            y1 * (a1 - alpha1) * model.kernel(x1, model.X[NonOpt], delta_inv) +
            y2 * (a2 - alpha2) * model.kernel(x2, model.X[NonOpt], delta_inv) +
            b_new - b_old
        )

        return 1

    def examineExample(self, i2):
        """
        Examine and optimize a single example index i2.
        """
        model = self.model
        y2 = model.y[i2]
        alpha2 = model.alphas[i2]
        E2 = model.errors[i2]
        r2 = E2 * y2

        if (r2 < -model.tol and alpha2 < model.C) or (r2 > model.tol and alpha2 > 0):
            non_bound_idx = np.where((model.alphas != 0) & (model.alphas != model.C))[0]
            if len(non_bound_idx) > 1:
                if model.errors[i2] > 0:
                    i1 = np.argmin(model.errors)
                else:
                    i1 = np.argmax(model.errors)

                step = self.takeStep(i1, i2)
                if step:
                    return 1

            # Randomized loop over all indices
            for i1 in np.roll(np.arange(model.m), np.random.choice(np.arange(model.m))):
                step = self.takeStep(i1, i2)
                if step:
                    return 1

        return 0

    def train(self, max_iter=100):
        """
        Train the model using the full optimization loop.
        """
        model = self.model
        num_changed = 0
        examine_all = True
        count = 0

        while count < max_iter:
            num_changed = 0
            if examine_all:
                for i in range(model.m):
                    num_changed += self.examineExample(i)
            else:
                non_bound_idx = np.where((model.alphas != 0) & (model.alphas != model.C))[0]
                for i in non_bound_idx:
                    num_changed += self.examineExample(i)

            if examine_all:
                examine_all = False
            elif num_changed == 0:
                break

            count += 1

        return model
