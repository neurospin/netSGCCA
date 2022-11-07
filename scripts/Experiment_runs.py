import joblib
import sys
import os
import copy
import pandas as pd
import numpy as np
import NetSGCCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from parsimony.algorithms.utils import Info
import networkx as nx
from scipy.sparse import coo_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms import community
import json
from scipy.sparse.linalg import eigs
import time
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
from matplotlib_venn import venn3, venn3_circles
from matplotlib import pyplot as plt
import logging
import multiprocessing as mp
from joblib import Parallel, delayed, parallel_backend
from functools import partial
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import argparse

def get_cmd_line_args():
    """
    Create a command line argument parser and return a dict mapping
    <argument name> -> <argument value>.
    """
    parser = argparse.ArgumentParser(
        prog="python runner for bioinformatics")

    # Required arguments
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-s", "--study", required=True, help="Study name.")

    # Create a dict of arguments to pass to the 'main' function
    args = parser.parse_args()
    kwargs = vars(args)
    return kwargs
class InternalRunner(object):
    def __init__(self,X,y,graphnet_L, graphnet_lambda,l1, ts_idx, groups = None, C = None):
        self.X=X.copy()
        self.y=y.copy()
        self.graphnet_L=graphnet_L.copy()
        self.graphnet_lambda = graphnet_lambda
        self.l1 = copy.deepcopy(l1)
        self.ts_idx = copy.deepcopy(ts_idx)
        self.groups = copy.deepcopy(groups)
        self.C = None
        
def internal_run(ii):
    t_idx, self = ii
    time0 = time.time()
    X = self.X
    y = self.y
    l1 = self.l1
    graphnet_lambda = self.graphnet_lambda
    ts_idx = self.ts_idx
    groups = self.groups
    C = self.C
    graphnet_L=self.graphnet_L
    log = logging.getLogger()

    scaler = coxgnet.PandasScaler()
    residualizer = coxgnet.CoxResiduals()
    feature_extractor = coxgnet.RGCCAFeatureExtraction(block_dict=groups, graphnet_L=graphnet_L,
                                                           graphnet_i=2,
                                                           max_inner_iter=100, info=[Info.beta], n_comp = 1,
                                                           l1 = l1, 
                                                           graphnet_lambda = graphnet_lambda, C=C)
    predictor = coxgnet.SurvivalPrediction()
    # Create pipeline
    pipe = Pipeline(steps=[('scaler', scaler), ('residualizer', residualizer), ('feature_extractor', feature_extractor),
                           ('predictor', predictor)] , memory = None)
    X_train, X_test = X.loc[t_idx,:], X.loc[ts_idx, :]
    y_train, y_test = y.loc[t_idx, :], y.loc[ts_idx, :]
    pipe.fit(X_train, y_train)
    log.warning('Fold took {0} sec'.format(time.time()-time0))
    return pipe['feature_extractor'].weights,  pipe.score(X_test, y_test)
    
def run(X, y, graphnet_L, l1, graphnet_lambda, t_idx, ts_idx, g=None, groups = None, rna_=None, C = None, runcv = False):
    if (g is None):
        A_ = pd.DataFrame(np.diag(np.diag(graphnet_L.todense())) - graphnet_L, columns = rna.columns, index= rna.columns)
        g_  = nx.from_pandas_adjacency(A_)
    else : 
        g_ = g.copy()
    print("Staring pipe creation")
    # Create nodes
    train_indexs= [resample(t_idx, replace=False, n_samples=round(0.85*len(t_idx)), random_state= i, stratify=y.loc[t_idx, 'status']) for i in range(100)]
    self = InternalRunner(X,y,graphnet_L, graphnet_lambda, l1, ts_idx, groups, C)
    print("before")
    params = zip(train_indexs, [self for _ in range(len(train_indexs))])
    print("after")
    with parallel_backend("loky", inner_max_num_threads=2):
        res = Parallel(n_jobs=10, max_nbytes="100M")(delayed(internal_run)(p) for p in params)            
    # Run boostrap
    print('Starting pipe fitting')
    n_zeros = []
    weights = []
    full_weights = []
    scores= []
    for r in res:
        selected = rna_.columns[(r[0][2] != 0).reshape(1,  -1) [0]]
        weights.append(r[0][2].reshape(1, -1))
        n_zeros.append(selected)
        scores.append(r[1])
        full_weights.append(r[0])
    #rmtree(cachedir)
    # Extract sets
    print('Starting set extraction')
    all_intersected = set(n_zeros[0])
    for i in range(len(n_zeros)):
        all_intersected = all_intersected.intersection(set(n_zeros[i]))
    union = set(n_zeros[0])
    for i in range(len(n_zeros)):
        union = union.union(set(n_zeros[i]))
    sub_g_intersected = g_.subgraph(all_intersected)
    sub_g_union = g_.subgraph(union)
    sub_g_n_zeros = [g_.subgraph(d) for d in n_zeros]
    selected_genes_data  = {}
    for gene in union:
        data = {
            'number_of_folds' : len([d for d in n_zeros if gene in d]), 
            'degree_full_graph' : g_.degree(gene),
            'degree_intersection': sub_g_intersected.degree(gene) if gene in all_intersected else 0,
            'degree_union': sub_g_union.degree(gene), 
            'degree_in_folds': [sgu.degree(gene) for sgu in sub_g_n_zeros]
        }
        selected_genes_data[gene] = data
    return {'scores' : scores, 'full_weights': full_weights,
        'union': union, 'intersected': all_intersected, 'sub_graph_intersected':sub_g_intersected, "sub_g_union": sub_g_union,
        'non_zeros_sub_graphs' : sub_g_n_zeros, 'summary':  pd.DataFrame(selected_genes_data).T.copy(),  'weights': pd.DataFrame(np.concatenate(weights, axis = 0), columns = rna.columns)}

inputs = get_cmd_line_args()

source_dir = os.getenv('ROOTCGNET')
cnv = pd.read_csv(os.path.join(source_dir, "cnv.csv"), sep=";", index_col  =0)
mirna = pd.read_csv(os.path.join(source_dir, "mirna.csv"), sep=";", index_col  =0)
rna = pd.read_csv(os.path.join(source_dir, "PathwayCommons12.All.hgnc_subrna.csv"), sep=";", index_col  =0)
y = pd.read_csv(os.path.join(source_dir, "target.csv"), sep=";", index_col  =0)

X = [cnv, mirna, rna]
X_cat = pd.concat(X, axis = 1)
groups = {
    'group_{}'.format(i) :  list(X[i].columns) for i in range(len(X))
}

L = pickle.load(open(os.path.join(source_dir, "PathwayCommons12.All.hgnc_laplacian.pkl"), "rb"))
L2 = coo_matrix(L)
A = pd.DataFrame(np.diag(np.diag(L2.todense())) - L2, columns = rna.columns, index= rna.columns)
g  = nx.from_pandas_adjacency(A)

L_normalied = pickle.load(open(os.path.join(source_dir, "PathwayCommons12.All.hgnc_normalized_laplacian.pkl"), "rb"))
L2_normalized = coo_matrix(L_normalied)

t_idx, ts_idx=  train_test_split(X_cat.index, train_size = 0.85, stratify=y['status'], random_state = 0)

if(inputs['study'] == 'OV'):
    l1 = [25.39685019840059, 120.37856952132302, 60.189, 1]
if(inputs['study'] == 'KIRP'):
    l1 = [162.2682963489788, 120.37856952132302, 6.0878, 1]
if(inputs['study'] == 'PAAD'):
    l1 = [24.73863375370596, 12.36931687685298, 6.1846, 1]
else:
    l1 = [60.18928476066151, 26, 10.07491981459117, 1]


result_dir = os.path.join(source_dir, 'runs/Results', inputs["study"])
if(not os.path.exists(result_dir)):
    os.makedirs(result_dir)

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

lambda_, _ = eigs(L2.astype('float'), k = 1)
lambda_ = 1/ lambda_[0].real
lambdas = [(10**k) * lambda_ for k in range(-3, 4)]
r = {}
for lam in lambdas :
    print(lam)
    data_pc  = results_complete = run(X_cat, y, L2, l1, lam,t_idx, ts_idx, g, groups, rna)
    r[lam] = data_pc.copy()

with open(os.path.join(result_dir, inputs['study']+"_raw.pkl"), 'wb') as f:
    pickle.dump(r, f)

lambda_, _ = eigs(L2_normalized.astype('float'), k = 1)
lambda_ = 1/ lambda_[0].real
lambdas = [(10**k) * lambda_ for k in range(-3, 4)]
r_normalized = {}
for lam in lambdas :
    print(lam)
    data_pc = run(X_cat, y, L2_normalized, l1, lam,t_idx, ts_idx, g, groups, rna)
    r_normalized[lam] = data_pc.copy()

with open(os.path.join(result_dir, inputs['study']+"_norm.pkl"), 'wb') as f:
    pickle.dump(r_normalized, f)


lambda_, _ = eigs(L2_normalized.astype('float'), k = 1)
lambda_ = 100/ lambda_[0].real
r_permutations = {}
for x in range(10) :
    rna_permuted = rna[np.random.permutation(rna.columns)]
    X_permutation = [cnv, mirna, rna_permuted]
    X_cat_permutation = pd.concat(X_permutation, axis = 1)
    groups_permuted = {
        'group_{}'.format(i) :  list(X_permutation[i].columns) for i in range(len(X_permutation))
    }
    data_pc = run(X_cat_permutation, y, L2_normalized, l1, lambda_,t_idx, ts_idx, g, groups_permuted, rna_permuted)
    r_permutations[x] = data_pc.copy()

with open(os.path.join(result_dir, inputs['study']+"_permuted2.pkl"), 'wb') as f:
    pickle.dump(r_permutations, f)

lambda_, _ = eigs(L2_normalized.astype('float'), k = 1)
lambda_ = 10/ lambda_[0].real
r_permutations = {}
for x in range(10) :
    rna_permuted = rna[np.random.permutation(rna.columns)]
    X_permutation = [cnv, mirna, rna_permuted]
    X_cat_permutation = pd.concat(X_permutation, axis = 1)
    groups_permuted = {
        'group_{}'.format(i) :  list(X_permutation[i].columns) for i in range(len(X_permutation))
    }
    data_pc = run(X_cat_permutation, y, L2_normalized, l1, lambda_,t_idx, ts_idx, g, groups_permuted, rna_permuted)
    r_permutations[x] = data_pc.copy()

with open(os.path.join(result_dir, inputs['study']+"_permuted1.pkl"), 'wb') as f:
    pickle.dump(r_permutations, f)


lambda_, _ = eigs(L2_normalized.astype('float'), k = 1)
lambda_ = 1000/ lambda_[0].real
r_permutations = {}
for x in range(10) :
    rna_permuted = rna[np.random.permutation(rna.columns)]
    X_permutation = [cnv, mirna, rna_permuted]
    X_cat_permutation = pd.concat(X_permutation, axis = 1)
    groups_permuted = {
        'group_{}'.format(i) :  list(X_permutation[i].columns) for i in range(len(X_permutation))
    }
    data_pc = run(X_cat_permutation, y, L2_normalized, l1, lambda_,t_idx, ts_idx, g, groups_permuted, rna_permuted)
    r_permutations[x] = data_pc.copy()

with open(os.path.join(result_dir, inputs['study']+"_permuted3.pkl"), 'wb') as f:
    pickle.dump(r_permutations, f)


L_msigdb = pickle.load(open(os.path.join(source_dir, "PathwayCommons10.msigdb.hgnc_normalized_laplacian.pkl"), "rb"))
L2_msigdb = coo_matrix(L_msigdb)
lambda_, _ = eigs(L2_msigdb.astype('float'), k = 1)

lambda_ = 1/ lambda_[0].real
lambdas = [(10**k) * lambda_ for k in range(-3, 4)]
r_msigdb = {}
for lam in lambdas :
    print(lam)
    data_pc = run(X_cat, y, L2_msigdb, l1, lam,t_idx, ts_idx, g, groups, rna)
    r_msigdb[lam] = data_pc.copy()

with open(os.path.join(result_dir, inputs['study']+"_msigdb.pkl"), 'wb') as f:
    pickle.dump(r_msigdb, f)


L_kegg = pickle.load(open(os.path.join(source_dir, "PathwayCommons12.kegg.hgnc_normalized_laplacian.pkl"), "rb"))
L2_kegg = coo_matrix(L_kegg)
lambda_, _ = eigs(L2_kegg.astype('float'), k = 1)
lambda_ = 1/ lambda_[0].real
lambdas = [(10**k) * lambda_ for k in range(-3, 4)]
r_kegg = {}
for lam in lambdas :
    print(lam)
    data_pc = run(X_cat, y, L2_kegg, l1, lam,t_idx, ts_idx, g, groups, rna)
    r_kegg[lam] = data_pc.copy()

with open(os.path.join(result_dir, inputs['study']+"_kegg.pkl"), 'wb') as f:
    pickle.dump(r_kegg, f)

print("finished")