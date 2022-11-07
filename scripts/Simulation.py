import time
import numpy as np
import networkx as nx
import pandas as pd
from scipy.spatial import distance_matrix

from sklearn.metrics import classification_report

import multiprocessing as mp

import NetSGCCA

D = pd.read_csv("design_index.csv", sep=";").values

def f(x):
    return -0.9*x + 0.9

def _getAplus(A):
    eigval, eigvec = np.linalg.eig(A)
    Q = np.matrix(eigvec)
    xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
    return Q*xdiag*Q.T

def _getPs(A, W=None):
    W05 = np.matrix(W**.5)
    return  W05.I * _getAplus(W05 * A * W05) * W05.I

def _getPu(A, W=None):
    Aret = np.array(A.copy())
    Aret[W > 0] = np.array(W)[W > 0]
    return np.matrix(Aret)

def nearPD(A, nit=10):
    n = A.shape[0]
    W = np.identity(n) 
# W is the matrix used for the norm (assumed to be Identity matrix here)
# the algorithm should work for any diagonal W
    deltaS = 0
    Yk = A.copy()
    for k in range(nit):
        Rk = Yk - deltaS
        Xk = _getPs(Rk, W=W)
        deltaS = Xk - Rk
        Yk = _getPu(Xk, W=W)
    return Yk
## Gen Correlation matrix
def gen_C(u, g):
    A = np.tile(u, (len(u), 1))
    B = np.abs(np.multiply((A.T- A) ,  nx.adj_matrix(g).todense()))
    myfunc_vec = np.vectorize(f)
    C = np.multiply(myfunc_vec(B) ,  nx.adj_matrix(g).todense()) + 0.1* (1- nx.adj_matrix(g).todense())
    np.fill_diagonal(C, 1)
    if(any(np.linalg.eig(C)[0] < 0)):
        C = nearPD(C)
    return C

def simulate(w, C, n, z):
    X = np.random.multivariate_normal(np.array([0]*C.shape[0]), C, size = n)
    X += np.dot(z.reshape(-1, 1), w.reshape(-1, 1).T)
    return X

def run(X1, X2, g_, graphnet_lambdas, u_):
    Results = dict()
    X = [pd.DataFrame(X1, columns = ["block1_{}".format(i) for i in range(p1)]), pd.DataFrame(X2, columns = ["block2_{}".format(i) for i in range(p2)])]
    X_train = pd.concat(X, axis=1)
    X_groups = {
        '{}'.format(i): list(c.columns) for i, c in enumerate(X)
    }
    scaler = NetSGCCA.PandasScaler()

    X_cat_scaled = scaler.fit_transform(X_train)
    for l in graphnet_lambdas:
        ll = l / np.linalg.eig(nx.laplacian_matrix(g_).todense())[0][0]
        feature_extractor_fista = NetSGCCA.NetSGCCA(graphnet_L=nx.laplacian_matrix(g_),
                                                               graphnet_i=1,
                                                             graphnet_lambda=ll,l1=[np.sqrt(25), np.sqrt(20)], force_constraint_L2=False,
                                block_dict=X_groups, downstream_blocks=[],  # data descr. are used for instanciation
                                algorithm='FISTA',                          # the optim strategy 
                                n_comp=1                                    # an example of RGCCA hypermameters
                                )
        st = time.time()
        y_fista = feature_extractor_fista.fit_transform(X_cat_scaled)
        #
        ft = time.time() - st
        print("Exec time: {} sec. , score {}".format(time.time() - st, y_fista.corr().values[0, 1]))
        w = feature_extractor_fista.weights[1]
        report = classification_report(u_ != 0 , w.flatten() != 0, output_dict = True)
        Results.update({
            'corr_fista_' + str(l) : y_fista.corr().values[0, 1],
            'precision_' +str(l): report['True']['precision'], 
            'recall_' +str(l): report['True']['recall'],
            'accuracy_' + str(l): report['accuracy'], 
            'n_selected_' +str(l): (w.flatten() != 0).sum()
        })
    return Results

p1, p2, n = 150, 100, 80
n_runs = 20

g1 = nx.path_graph(p2)
g2 = nx.Graph()
g2.add_nodes_from(range(p2))
g2.add_edges_from([(50,i) for i in range(p2) if i != 50])
g3 = nx.complete_graph(p2)
graphes = [g1, g2, g3, nx.compose(g1,g2)]

lambdas = [0] + [10**i for i in range(-4, 4,1)]

for c in [2, 0.5]:
    for i in range(3):
        for g, gt in zip(graphes, ['path', 'star','complete',"union"]):
            print(i, gt)
            async_results = []
            cnt = 0
            def runner_(cnt_):
                print(g)
                np.random.seed(cnt_)
                z = np.random.normal(0, 1, n)
                v = np.zeros(p1)
                v[25:50] = 1
                Cv = 0.1 * np.ones((p1, p1))
                np.fill_diagonal(Cv, 1)
                v = c * v 
                X1 = simulate(v, Cv, n, z)
                u = D[:, i]
                Cu = gen_C(u, g)
                u = c * u 
                X2 = simulate(u, Cu, n, z)
                return run(X1, X2, g, lambdas, u)
            with mp.Pool(processes=10) as pool:
                for _ in range(n_runs):
                    cnt+=1
                    async_results.append(pool.apply_async(runner_, args=(cnt, )))
                pool.close()
                Results = pool.join()
            results = dict()    
            for x, async_result in enumerate(async_results):
                results[x] = async_result.get()
            pd.DataFrame(results).T.to_csv("Csvs/Results_{}/{}_{}.csv".format(c, i, gt), sep=";")
