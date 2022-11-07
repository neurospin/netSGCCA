import numpy as np
import pandas as pd
import parsimony.algorithms.deflation as deflation
import parsimony.algorithms.multiblock as algorithms
import parsimony.functions.multiblock.losses as mb_losses
import parsimony.functions.penalties as penalties
import parsimony.utils.weights as start_vectors
from parsimony.algorithms.utils import Info
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.extmath import randomized_svd

from .latent_variable_convariance import LatentVariableCovarianceWrapper
from parsimony.algorithms.proximal import DykstrasProjectionAlgorithm

class NetSGCCA(BaseEstimator, TransformerMixin):
    def __init__(self, block_dict, downstream_blocks=None, n_comp=10, scheme='horst', l1=None, l2=None,
                 force_constraint_L2=True,
                 init='svd', eps=1e-5, info=None, max_iter=2000,
                 max_outer_iter=10, max_inner_iter=10000, stopping_criterion='iterate', algorithm='FISTA',
                 graphnet_lambda=0, graphnet_A=None, graphnet_L=None, graphnet_i=None, C = None, steps = None):
        self.graphnet_lambda = graphnet_lambda
        self.graphnet_A = graphnet_A
        self.graphnet_L = graphnet_L
        self.graphnet_i = graphnet_i
        if info is None:
            info = [Info.num_iter, Info.time, Info.converged, Info.beta, Info.gap, Info.other, Info.func_val]
        if downstream_blocks is None:
            downstream_blocks = ['deviance']
        self.info = info
        self.init = init
        self.eps = eps
        self.max_iter = max_iter
        self.max_outer_iter = max_outer_iter
        self.max_inner_iter = max_inner_iter
        self.stopping_criterion = stopping_criterion
        self.downstream_blocks = downstream_blocks
        self.block_dict = block_dict
        self.all_blocks = {**block_dict, 'extra': downstream_blocks} if len(downstream_blocks) > 0 else block_dict
        self.n_comp = n_comp
        self.scheme = scheme
        self.deflation = deflation.RankOneDeflation()
        self.force_constraint_L2 = force_constraint_L2
        if C is None:
            C = np.ones(len(self.all_blocks)) - np.eye(len(self.all_blocks))
        if self.init == 'random':
            self.random_vector = start_vectors.RandomUniformWeights(normalise=False)
        if l1 is None:
            l1 = [np.sqrt(len(c)) for g, c in self.all_blocks.items()]
        if l2 is None:
            l2 = [1] * len(self.all_blocks)
        if steps is None:
            steps = []
        self.steps = steps
        self.C = C
        self.l1 = l1
        self.l2 = l2
        self.algorithm = algorithm
        self.weights = []
        self.projections = []
        self.info_data = []

    def fit(self, X, y=None):
        self.init_params()
        internal_X = [X.loc[:, cols] for name, cols in self.all_blocks.items()]

        for k in range(self.n_comp):
            if self.init == "svd":
                w = [randomized_svd(d.values, n_components=1, random_state=0)[2].reshape(-1, 1) for d in internal_X]
            else:
                w = [self.random_vector.get_weights(X[i].shape[1]) for i in range(len(internal_X))]
            prox_combo = DykstrasProjectionAlgorithm(eps=self.eps, max_iter=self.max_inner_iter)
            w = [prox_combo.run([self.l1_constraints[i], self.l2_constraints[i]] , w[i]) for i in range(len(internal_X))]
            function = self.RGCCA_builder(internal_X)
            w_fit = self.optimizer.run(function, w)
            self.info_data.append(self.optimizer.info_get())
            t = [internal_X[i] @ w_fit[i] for i in range(len(internal_X))]
            p = [(internal_X[i].T @ t[i]) / (t[i].T @ t[i]).values[0] for i in range(len(internal_X))]
            internal_X = [self.deflation.deflate(internal_X[i], t[i], p[i]) for i in range(len(internal_X))]
            if k == 0:
                self.weights = w_fit
                self.projections = t
            else:
                self.weights = [np.concatenate([self.weights[i], w_fit[i]], axis=1) for i in range(len(w_fit))]
                self.projections = [np.concatenate([self.projections[i], t[i]], axis=1) for i in range(len(w_fit))]
        return self

    def RGCCA_builder(self, X):
        function = mb_losses.CombinedMultiblockFunction(X, norm_grad=False)
        for i in range(len(self.all_blocks)):
            function.add_constraint(self.l1_constraints[i], i)
        for i in range(len(self.all_blocks)):
            function.add_constraint(self.l2_constraints[i], i)
        if self.graphnet_lambda > 0:
            GraphNet = penalties.GraphNet(l=self.graphnet_lambda, A=self.graphnet_A, La=self.graphnet_L)
            function.add_penalty(GraphNet, self.graphnet_i)
        for i in range(len(X) - 1):
            for j in range(i + 1, len(X)):
                if(self.C[i, j] == 1):
                    cov_X1_X2 = LatentVariableCovarianceWrapper(
                    [X[i], X[j]], unbiased=True,
                    scalar_multiple=1, scheme=self.scheme)
                    function.add_loss(cov_X1_X2, i, j)
        return function

    def transform(self, X, y=None):
        internal_X = [X.loc[:, cols] for name, cols in self.block_dict.items()]
        return pd.concat([internal_X[i] @ self.weights[i] for i in range(len(internal_X))], axis=1)

    def init_params(self):
        self.l1_constraints = [penalties.L1(c=l) for l in self.l1]
        self.l2_constraints = [penalties.L2(c=l, force_constraint=self.force_constraint_L2) for l in
                               self.l2]
        if self.algorithm == "RGCCA":
            self.optimizer = algorithms.MultiblockRGCCA(eps=self.eps,
                                                        info=self.info,
                                                        max_iter=self.max_iter, max_outer_iter=self.max_outer_iter,
                                                        max_inner_iter=self.max_inner_iter,
                                                        stopping_criterion=self.stopping_criterion)
        else:
            self.optimizer = algorithms.MultiblockFISTA(eps=self.eps,
                                                        info=self.info,
                                                        max_iter=self.max_iter, max_outer_iter=self.max_outer_iter,
                                                        max_inner_iter=self.max_inner_iter,
                                                        stopping_criterion=self.stopping_criterion, steps = self.steps)
        return
