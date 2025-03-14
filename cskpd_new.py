import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from statsmodels.genmod.families.links import *
from scipy.linalg import norm, qr
from sklearn.linear_model import Lasso, Ridge, LinearRegression
import itertools
from numpy import ravel as vec
from collections import namedtuple
from joblib import Parallel,delayed
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import make_scorer, mean_squared_error, roc_auc_score, confusion_matrix, check_scoring
from convolution import convolution
import random
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score, mean_squared_error
def auc(y_true, y_pred):
    y_pred = np.array(y_pred) >= 24.0/30
    y_true = np.array(y_true) >= 24.0/30
    return roc_auc_score(y_true, y_pred)
scorers = {
    'MSE': mean_squared_error,
    'AUC': auc
}

def inv_vec(x, cols):
    return x.reshape(cols, -1).T
def orthonormalize(x):
    return qr(x, mode='economic')[0]
def b_next_iter(y,X,A,lmbda):
    R = A.shape[1]
    Xa = [vec((x.T @ A).T) for x in X]
    return inv_vec(Ridge(fit_intercept = False, alpha=lmbda).fit(Xa,y).coef_, R) # Ridge regression to stabilize B
def a_next_iter(y,X,B,lmbda):
    R = B.shape[1]
    Xb = [vec((x @ B).T) for x in X]
    return orthonormalize(inv_vec(Lasso(fit_intercept = False, alpha=lmbda).fit(Xb,y).coef_, R))
def zeta_next_iter(y,Z,lmbdz=1):
    # print(f"y: {y.shape}, Z: {Z.shape}")
    return Ridge(fit_intercept = False, alpha=lmbdz).fit(Z, y).coef_
def score_AB(A,B,zeta=None,score=None):
    stability = np.var(B)/(norm(B, 2)**2)
    sparsity = np.where(vec(A) == 0)[0].shape[0]/vec(A).shape[0]
    norms = [
        np.linalg.norm(A, 'fro'),
        np.linalg.norm(B, 'fro'),
        np.linalg.norm(zeta) if zeta is not None else 0
    ]
    if score is not None:
        dif = np.array([
            np.linalg.norm(score['B'] - B, 'fro'),#/norms[0],
            np.linalg.norm(score['A'] - A, 'fro'),#/norms[1],
            np.linalg.norm(score['zeta'] - zeta) if zeta is not None else 0 #/norms[2]
        ])
        history = score['history']
    else: 
        dif = np.array(norms)
        history = []
    history.append(dif)
    return {'stability': stability, 'sparsity': sparsity, 'dif': dif, 'zeta': zeta, 'A': A, 'B': B, 'history': history}
def Rearrange(X,p,d):
    p, d = np.array(p), np.array(d)
    assert X.ndim > 1, "The size of the tensor must be of dimensions of 2 or more."
    assert X.ndim == len(p) == len(d), f"The dimension size of X {X.ndim}, and the lengths of p ({len(p)}) and d ({len(d)}) must all be equal"
    assert (X.shape == p*d).all(), f"The dimensions of X ({X.shape}), must be equal to the product of each element of p*d ({p*d})"
    slices = []
    Rx = []
    for dim in range(X.ndim):
        slices.append([])
        for i in range(p[dim]):
            slices[dim].append(slice(d[dim]*i, d[dim]*(i+1)))
    Rx = [X[s].reshape(-1,1) for s in list(itertools.product(*slices))]
    return np.concatenate(Rx, axis=1).T
def pack_parameters(named_tuple, static_params=None, mapped_params=[], max_vals=None):
    # TODO: error handling for empty static and mapped, edge case where pack_params is empty
    # TODO: check if named_tuple response works instead of tuple
    def flatten_tuple(tup):
        return tuple(item for subtuple in tup for item in (flatten_tuple(subtuple) if isinstance(subtuple, tuple) else (subtuple,)))
    ordered_names = named_tuple._fields
    mapped_vals = zip(*[getattr(named_tuple, name) for name in mapped_params])
    static_vals = tuple([getattr(named_tuple, name) for name in static_params])
    pack_params_fields = [field for field, value in named_tuple._asdict().items() if field not in set(mapped_params + static_params)]
    name_order = pack_params_fields + mapped_params + static_params
    order = tuple(name_order.index(name) for name in ordered_names)
    pack_params = [value for field, value in named_tuple._asdict().items() if field not in set(mapped_params + static_params)]
    pack_parameters = [flatten_tuple(item) + static_vals for item in list(itertools.product(*pack_params, mapped_vals))]
    ordered_parameters = [tuple(tup[i] for i in order) for tup in pack_parameters]
    all_parameters = [type(named_tuple)(*o)._asdict() for o in ordered_parameters]
    if max_vals is not None:
        all_parameters = random.sample(all_parameters, max_vals)
    # print(all_parameters[0])
    return all_parameters
def relu(x):
    return np.maximum(0, x)

# Scorers
def mse(y_true, y_pred):
    return np.average((y_true - y_pred)**2)
def calc_mbic(S,R,p):
    def mbic(y_true, y_pred):
        # print(R,p)
        Cn, n = [np.log(np.log(R*np.prod(p))), len(y_true)]
        return np.log(mse(y_true, y_pred)) + (Cn * np.log(n)/n * S) + (2*S*np.log(0.1*np.prod(p)/4-1))
    return mbic

class CSKPD(RegressorMixin, BaseEstimator):
    def __init__(self,p=None,lama=0.0,lamb=0.0,lamz=0.0,R=1,g=Identity(),K=None,L=None,n_cores = -1,max_iter = 20,print_iter = 5,cuda=False):
        self.p = p
        # self.d = d # d is now dependent on p
        self.lama = lama if type(lama) == list else [lama]
        self.lamb = lamb if type(lamb) == list else [lamb]
        self.lamz = lamz if type(lamz) == list else [lamz]
        self.R = R if type(R) == list else [R]
        self.g = g if type(g) == list else [g]
        self.K = K if type(K) == list else [K]
        self.L = L if type(L) == list else [L]
        self.n_cores = n_cores
        self.max_iter = max_iter
        self.print_iter = print_iter
        self.cuda = cuda

        # Defined after init
        self.S, self.Cn, self.C = None, None, None

    def calc_y_hat(self,X,C,g=None):
        if g is None:
            return [(np.vdot(x,C)) for x in X]
        else:
            return [g.inverse(np.vdot(x,C)) for x in X]

    def min_SKPD(self, X=None, y=None, p=None, Z=None, lama=0.0, lamb=0.0, lamz=0.0, R=1, g=Identity(), K=None, Lambda=None, tol = 1e-6, max_iter = 20, print_iter = 5, cuda = False):
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        assert X is not None and y is not None and p is not None, "X, y, and p must be provided"
        d = [X[0].shape[i] // p[i] for i in range(len(X[0].shape))]
        Rx = [Rearrange(x,p,d) for x in X]
        A = np.linalg.svd(sum([y[i] * Rx[i] for i in range(len(Rx))]))[0][:, :R]
        z = np.zeros(len(y))
        # B = np.ones((np.prod(d), R))

        score = None
        zeta = None
        for i in range(self.max_iter+1):
            B = b_next_iter(g(y)-z,Rx,A,lamb)
            A = a_next_iter(g(y)-z,Rx,B,lama)
            # TODO: Add another stopping criterion to ensure Z is stable
            if Z is not None:
                C = sum([np.kron(A[:,r].reshape(*p),B[:,r].reshape(d)) for r in range(R)])
                # print(f"C: {self.C.shape}, g(y): {len(g(y))}, X: {X.shape}")
                zeta = zeta_next_iter(g(y)-np.array([np.vdot(x,C) for x in X]),Z,lamz)
                z = np.array(Z @ zeta)
            score = score_AB(A,B,zeta,score)
            # print(f"dif: {score['dif']}")
            if np.all(np.abs(score['dif']) < tol):
                # print(f"Completed after {i} iterations.")
                break
        C = sum([np.kron(A[:,r].reshape(*p),B[:,r].reshape(d)) for r in range(R)])
        self.A = A
        self.B = B
        self.zeta = zeta
        MSE = mse(g(y), self.calc_y_hat(X,C,g) + np.array(Z @ self.zeta) if Z is not None else self.calc_y_hat(X,C,g)) # MSE is for the generalized form
        S, Cn, n = [(1-score['sparsity']) * np.prod(d), np.log(np.log(R*np.prod(p))), len(y)]
        MBIC = np.log(MSE) + (Cn * np.log(n)/n * S) + (2*S*np.log(0.05*np.prod(C.shape)/4-1))
        print_detail = f"Iter: {i}, Instability: {np.round(score['stability']*100,1)}%, Sparsity: {np.round(score['sparsity']*100,1)}%, MSE: {np.round(MSE,2)}, MBIC: {np.round(MBIC,2)}, dif: {score['dif']}, zeta: {zeta}"

        return {"MSE":MSE, "A":A, "B":B, "C":C, "zeta": zeta, "print_detail":print_detail, "MBIC":MBIC, "S":S, "Cn":Cn, "score":score}
    
    def fit(self, X, y, **kwargs):
        Z = kwargs.get('Z', None)
        tol = kwargs.get('tol', 1e-6)
        static = kwargs.get('static', ["X", "y", "Z"])
        if self.p is None:
            p_list = []
            for n in X[0].shape:
                pairs = []
                recs = []
                for i in range(1, int(n**0.5) + 1):
                    if n % i == 0:
                        pairs.append(i)
                        recs.append(n//i)
                p_list.append(pairs + recs[::-1])
            self.p = set(itertools.product(*p_list))
        s = namedtuple('s', ['Z', 'X', 'y', 'lama', 'lamb', 'lamz', 'p', 'R', 'g'])
        s_val = s(Z, X, y, self.lama, self.lamb, self.lamz, self.p, self.R, self.g)
        parameters = pack_parameters(s_val, static, ["R"], max_vals=kwargs.get('max_vals', None))
        # print(parameters)
        if self.n_cores > 1:
            vals = Parallel(n_jobs= self.n_cores, verbose=False)(delayed(self.min_SKPD)(**parameters[i]) for i in range(len(parameters)))
        else:
            vals = [self.min_SKPD(**parameters[i]) for i in range(len(parameters))]
        
        # for i, p in enumerate(parameters):
        #     # p = {k: v for k, v in p.items() if k not in static}
        #     p['MSE'], p['A'], p['B'], p['C'], p['print_detail'], p['MBIC'], p['S'], p['Cn'] = vals[i].values()
        #     # p['test'] = 'test'
        self.parameters = [{k: v for k, v in p.items() if k not in static} for p in parameters]
        opt = np.argmin([v['MBIC'] for v in vals])
        self.C = vals[opt]['C']
        self.zeta = vals[opt]['zeta']

        self.grid_values = {
            'p': parameters[opt]['p'],
            'g': parameters[opt]['g'],
            'R': 3,
            'lama': parameters[opt]['lama'],
            'lamb': parameters[opt]['lamb'],
            'lamz': parameters[opt]['lamz'],
            'A': vals[opt]['A'],
            'B': vals[opt]['B'],
            'C': vals[opt]['C'],
            'S': vals[opt]['S'],
            'Cn': vals[opt]['Cn'],
            'print_detail': [v['print_detail'] for v in vals],
            'score': vals[opt]['score'],
            'opt_id': opt,
            'fit_mse': vals[opt]['MSE'],
            'fit_mbic': vals[opt]['MBIC'],
        }
        return self

    def predict(self, X, **kwargs):
        Z = kwargs.get('Z', None)
        # TODO: this needs to be corrected for g values
        g = self.g[0]
        res = self.calc_y_hat(X,self.C,g) + np.array(Z @ self.zeta) if Z is not None else self.calc_y_hat(X,self.C,g)
        return res
    
    def calc_score(self, X, y, scorers, **kwargs):
        k_folds = kwargs.get('k_folds', 5)
        K = kwargs.get('K', None)
        X = np.array([K.convolve(x) for x in X]) if K is not None else X
        valid_params = {'Z', 'tol', 'static'}
        fit_params = {k: v for k, v in kwargs.items() if k in valid_params}
        results = cross_validate(self, X, y, cv=k_folds, scoring=scorers, return_estimator=True, params=fit_params)
        for s in scorers:
            print(f"{k_folds}-Fold CV {s} Scores: {results['test_'+s]*scorers[s]._sign}")
            print(f"Average {k_folds}-Fold CV {s} Scores: {np.nanmean(np.array(results['test_'+s]))*scorers[s]._sign}")
        return results
    
    def custom_cross_validate(self, X, y, scorers, k_folds=5, **kwargs):
        K = kwargs.get('K', None)
        Z = kwargs.get('Z', None)
        X = np.array([K.convolve(x) for x in X]) if K is not None else X
        valid_params = {'Z', 'tol', 'static', 'max_vals'}
        fit_params = {k: v for k, v in kwargs.items() if k in valid_params}
        pred_params = fit_params.copy()

        # Initialize a KFold object
        kf = KFold(n_splits=k_folds)
        
        # Initialize lists to store scores
        models = []
        scores = {key: [] for key in scorers}
        
        for train_index, test_index in kf.split(X):
            # Split the data
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            if Z is not None:
                fit_params['Z'], pred_params['Z'] = Z[train_index], Z[test_index]

            # Clone the estimator to ensure it's a fresh model each time
            model = clone(self)

            # Fit the model
            model.fit(X_train, y_train, **fit_params)

            # Predict on the test set
            y_pred = model.predict(X_test, **pred_params)

            # Calculate and store each score
            for scorer_name, scorer in scorers.items():
                score = scorer(y_test, y_pred)
                scores[scorer_name].append(score)
            
            results = {f'test_{key}': np.array(value) for key, value in scores.items()}
            models.append({'model': model, 'results': results, 'y_hat': y_pred})

        # Calculate the mean and std of each score
        
        
        return models
    
    def model_of_models(self, model_parameters, **kwargs):
        all_models = []
        for params in model_parameters:
            models = self.custom_cross_validate(self, **params, **kwargs)
            preds = np.array([m['y_hat'] for m in models])
            auc = np.array([m['results']['test_AUC'] for m in models])
            aggregated_preds = relu(preds).sum(axis=0)
            final_preds = aggregated_preds / aggregated_preds.max()
            all_models.append(models)
        return