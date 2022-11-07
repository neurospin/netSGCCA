import numpy as np
import parsimony.functions.multiblock.losses as mb_losses
from parsimony.functions.multiblock import properties as mb_properties


class LatentVariableCovarianceCentroid(mb_losses.LatentVariableCovariance):

    def __init__(self, X, unbiased=True, scalar_multiple=1.0):
        super().__init__(X, unbiased=unbiased, scalar_multiple=scalar_multiple)

    def f(self, w):
        return - np.abs(super().f(w))

    def grad(self, w, index):
        return super().grad(w, index) * np.sign(super().f(w))


class LatentVariableCovarianceWrapper(mb_properties.MultiblockFunction,
                                      mb_properties.MultiblockGradient,
                                      mb_properties.MultiblockLipschitzContinuousGradient):

    def __init__(self, X, unbiased=True, scalar_multiple=1.0, scheme='horst'):
        if scheme == 'horst':
            self.function = mb_losses.LatentVariableCovariance(X, unbiased=unbiased, scalar_multiple=scalar_multiple)
        elif scheme == 'centroid':
            self.function = LatentVariableCovarianceCentroid(X, unbiased=unbiased, scalar_multiple=scalar_multiple)
        elif scheme == 'factorial':
            self.function = mb_losses.LatentVariableCovarianceSquared(X, unbiased=unbiased)

    def f(self, w):
        return self.function.f(w)

    def grad(self, w, index):
        return self.function.grad(w, index)

    def reset(self):
        return self.function.reset()

    def L(self, w, index):
        return self.function.L(w, index)