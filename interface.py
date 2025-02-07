    # M always means matrix
import numpy as np
import pandas as pd
import os 
import numpy.linalg as la
import copy
from cskpd import *
import time
from joblib import Parallel,delayed
from collections import namedtuple
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import make_scorer, mean_squared_error, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from statsmodels.genmod.families.links import Identity
import itertools

### update 2021.06.23
### for parallel computing
def skpdRegressor(p_list,d_list,lmbda_set,lmbda2_set,Z_train,X_train,Y_train,R_list,n_cores = -1,max_iter = 20,print_iter = 5, cuda=False, g_list = None, K_list = None, Lambda_list = None):
    na = True
    nb = True
    s = namedtuple('s', ['Z_train', 'X_train', 'y_train', 'lmbda_set', 'lmbda2_set', 'p_list', 'd_list', 'R_list', 'na', 'nb'])
    s_val = s(Z_train, X_train, Y_train, lmbda_set, lmbda2_set, p_list, d_list, R_list, na, nb)

    ## one simulation with fixed dimension and lambda
    if len(lmbda_set) == 1 and len(p_list) == 1 and len(R_list) == 1: 
        lmbda_1, lmbda_2, p, d, R = lmbda_set[0], lmbda2_set[0], p_list[0], d_list[0], R_list[0]
    ## No parallel
    else:
        if n_cores == None:
            val_values = fun_validate(p_list,d_list,lmbda_set,lmbda2_set,Z_train,X_train,Y_train,R_list,na,nb,max_iter,print_iter)
        # we take Modified BIC to select lambda in SKPD
        elif cuda == False:
            ## start parallel computing
            # print("-------start parallel computing-----------")
            Parameters = pack_parameters_new2(s_val, ["Z_train", "X_train", "y_train", "na", "nb"], ["p_list", "d_list"])
            print(Parameters)
            parallel_res = my_parallel(Parameters,n_cores)  ## TODO: combine with pack_parameters functions
            mbic = list(np.array(parallel_res).squeeze())
        else:
            Parameters = pack_parameters_new2(s_val, ["Z_train", "X_train", "y_train", "na", "nb"], ["p_list", "d_list"])
            mbic = my_parallel(Parameters,n_cores)

        if print_iter != 0:
            print("MBIC values: ",mbic)
        opt_idx = np.argmin(mbic)
        # print("opt-idx: ",opt_idx)
        # fir_idx = opt_idx//(len(R_list) * len(lmbda_set))
        # tmp1 = opt_idx % (len(R_list) * len(lmbda_set))
        # sec_idx = tmp1// len(lmbda_set)
        # third_idx = tmp1 % len(lmbda_set)
        # # print("fir_idx: ",fir_idx)
        # p, d = p_list[fir_idx], d_list[fir_idx]
        # R = R_list[sec_idx]
        # lmbda_1 = lmbda_set[third_idx]
        # lmbda_2 = lmbda2_set[0]
        # # opt_solver = solver_list[opt_idx]
        print(Parameters[opt_idx])
        Z_train,RX,Y_train,lmbda_1,lmbda_2,p,d,R,na,nb = Parameters[opt_idx]

    ### output the final estimations and model
    RX = [Rearrange(xi,p,d) for xi in X_train]
    opt_solver = AltMin(Z_train,RX,Y_train,lmbda_1,lmbda_2,p,d,R,na,nb)
    a_hat,b_hat,gamma_hat,Y_hat,err_beta,fN = opt_solver.fit(max_iter = max_iter,iter_print = print_iter)
    return a_hat,b_hat,gamma_hat,lmbda_1,lmbda_2,R,p,d

def my_parallel(Parameters,n_cores):
    return Parallel(n_jobs= n_cores, verbose=False)(delayed(my_parallel_lmbda)(*Parameters[i]) for i in range(len(Parameters)))

def my_parallel_lmbda(Z_train,X_train,Y_train,lmbda1,lmbda2,p,d,R,na,nb,max_iter = 20,print_iter = 5):
    ##### start training
    RX_train = [Rearrange(xi,p,d) for xi in X_train]
    solver = AltMin(Z_train,RX_train,Y_train,lmbda1,lmbda2,p,d,R,na,nb)
    a_hat,b_hat,gamma_hat,Y_hat,err_beta,fN = solver.fit(max_iter = max_iter,iter_print = print_iter)
    if fN == np.Inf:
        return np.Inf
    else:
        val_y = solver.predict(Z_train,solver.RX,a_hat,b_hat,gamma_hat)
        fN = RMSE(val_y,solver.Y)
        s = np.where(a_hat !=0)[0].shape[0]
        p0 = a_hat.shape[0]
        sample_size = solver.N
        MBIC = sample_size * np.log(fN ** 2) + s * np.log(sample_size) * np.log(np.log(p0))
        return MBIC

def pack_parameters(Z_train,X_train,Y_train,lmbda_set,lmbda2_set,p_list,d_list,R_list,na,nb, g_list = None, K_list = None, Lambda_list = None):
    num_lmbda = len(lmbda_set)
    assert len(p_list) == len(d_list), "p_list and d_list should have the same length"
    Parameters = []
    for item in range(len(p_list)):
        p, d = p_list[item], d_list[item]
        ## the output parameters
        ## rerange X
        RX_train = X_train # TODO: Remove this when below is moved to training
        # RX_train = [Rearrange(xi,p,d) for xi in X_train] # TODO: Move this section to training
        for R in R_list:
            for lmbda1 in lmbda_set:
                for lmbda2 in lmbda2_set:
                    Parameters.append((Z_train,RX_train,Y_train,lmbda1,lmbda2,p,d,R,na,nb))
    return Parameters

def pack_parameters_new(static_params, pack_params):
    param_list = [static_params + item for item in list(itertools.product(*pack_params))]
    return param_list

def pack_parameters_new2(named_tuple, static_params=None, mapped_params=[]):
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
    return ordered_parameters

def new_fun_combined(p_list,d_list,lmbda_set,lmbda2_set,Z_train,X_train,Y_train,R_list,na,nb,max_iter = 20,print_iter = 5):
    pass

def fun_validate(p_list,d_list,lmbda_set,lmbda2_set,Z_train,X_train,Y_train,R_list,na,nb,max_iter = 20,print_iter = 5):
    # first we can try Modified BIC as select lambda in SKPD
    train_values = []
    val_values = []
    S = []
    fN_list = []
    solver_list = []
    opt_lmbda_1 = []
    opt_lmbda_2 = []
    for item in range(len(p_list)):
        p,d = p_list[item],d_list[item]
        RX_train = [Rearrange(xi,p,d) for xi in X_train]
        for R in R_list:
            for lmbda1 in lmbda_set:
                for lmbda2 in lmbda2_set:
                    ##### start training
                    solver = AltMin(Z_train,RX_train,Y_train,lmbda1,lmbda2,p,d,R,na,nb)
                    a_hat,b_hat,gamma_hat,Y_hat,err_beta,fN = solver.fit(max_iter = max_iter,iter_print = print_iter)

                    if fN != np.Inf:  
                        _,_,kron_ab = func_kron_ab(a_hat,b_hat,R,p,d)
                        val_y = solver.predict(Z_train,solver.RX,a_hat,b_hat,gamma_hat)
                        fN = RMSE(val_y,solver.Y)
                        # the number of non-zero entry
                        s = np.where(a_hat !=0)[0].shape[0]
                        p = a_hat.shape[0]
                        sample_size = solver.N
                        MBIC = sample_size * np.log(fN ** 2) + s * np.log(sample_size) * np.log(np.log(p))
                        val_values.append(MBIC)
                        opt_lmbda_1.append(lmbda1)
                        opt_lmbda_2.append(lmbda2)
                        S.append(s)
                    else:
                        val_values.append(np.Inf)
    return val_values

def RMSE(pred,truth):
    sample_size = len(truth)
    fn = la.norm(np.asarray(np.asarray(pred).squeeze() - np.asarray(truth).squeeze()), 2) / np.sqrt(sample_size)  # RMSPE from Hongtu ZHU
    return fn

def Rearrange(C,p,d):
    p, d = np.array(p), np.array(d)
    assert C.ndim > 1, "The size of the tensor must be of dimensions of 2 or more."
    assert C.ndim == len(p) == len(d), f"The dimension size of C {C.ndim}, and the lengths of p {len(p)} and d {len(d)} must all be equal"
    assert (C.shape == p*d).all(), "The dimensions of C, must be equal to the product of each element of p*d"
    slices = []
    RC = []
    for dim in range(C.ndim):
        slices.append([])
        for i in range(p[dim]):
            slices[dim].append(slice(d[dim]*i, d[dim]*(i+1)))
    RC = [C[s].reshape(-1,1) for s in list(itertools.product(*slices))]
    return np.concatenate(RC, axis=1).T

def func_kron_ab(a_hat, b_hat, R, p, d):
    """Calculate A, B, and kron_ab matrices."""
    a_hat_reshaped = a_hat.reshape(R, -1)
    b_hat_reshaped = b_hat.reshape(R, -1)
    A = []
    B = []
    kron_ab = []

    for i in range(1, R + 1):
        a_hat_i = a_hat_reshaped[i - 1, :].reshape(-1, 1)
        b_hat_i = b_hat_reshaped[i - 1, :].reshape(-1, 1)
        A.append(a_hat_i.reshape(*p))
        B.append(b_hat_i.reshape(*d))

    kron_ab = [np.kron(A[i], B[i]) for i in range(R)]
    beta_hat = sum(kron_ab)
    kron_ab.append(beta_hat)

    return A, B, kron_ab

def rmse(y_true, y_pred):
    idx = ~np.isnan(y_true)
    return -mean_squared_error(y_true[idx], np.round(y_pred[idx],0), squared=False)

def acc_new(y_true, y_pred):
    # idx = ~np.isnan(y_true)
    # print(f"y_true: {y_true[idx]}, y_pred: {y_pred[idx]}")
    return np.mean(y_true==np.round(y_pred,0))

def custom_auc(cutoff_point=0):
    def auc(y_true, y_pred):
        idx = ~np.isnan(y_true)
        y_pred_relu = y_pred * (np.array(y_pred)>cutoff_point)
        y_pred_softmax = np.exp(np.array(y_pred_relu - np.max(y_pred_relu)).astype(np.float64)) / np.sum(np.exp(np.array(y_pred_relu - np.max(y_pred_relu)).astype(np.float64)))
        return roc_auc_score(list(np.array(y_true[idx])>cutoff_point), y_pred_softmax[idx])
    return auc

def custom_accuracy(cutoff_point=0):
    def accuracy(y_true, y_pred):
        idx = ~np.isnan(y_true)
        return np.average((np.array(y_true[idx])>cutoff_point)==(np.array(y_pred[idx])>cutoff_point))
    return accuracy

# def specificity(y_true, y_pred):
#     tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
#     fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
#     return tn / (tn + fp) if (tn + fp) != 0 else 0

# def precision(y_true, y_pred):
#     tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
#     fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
#     return tp / (tp + fp) if (tp + fp) != 0 else 0

def calc_score(model, X, y, k_folds=5, y_imputer=None, cutoff_point=0):
    scorers = {
        'acc_new': make_scorer(acc_new, greater_is_better=True),
        # 'rmse': make_scorer(rmse, greater_is_better=False),
        # 'auc': make_scorer(custom_auc(0), greater_is_better=True),
        # 'accuracy': make_scorer(custom_accuracy(0), greater_is_better=True),
        # 'specificity': make_scorer(specificity, greater_is_better=True),
        # 'precision': make_scorer(precision, greater_is_better=True)
    }
    if y_imputer is not None:
        y = y_imputer.transform(y)
    results = cross_validate(model, X, y, cv=k_folds, scoring=scorers, return_estimator=True)
    print("5-Fold CV Accuracy Scores: ", results['test_acc_new'])
    print("Average 5-Fold CV RMSE Scores: ", np.nanmean(np.array(results['test_acc_new'])))
    # print("5-Fold CV Accuracy Scores: ", results['test_accuracy'])
    # print("5-Fold CV RMSE Scores: ", results['test_rmse'])
    # print("5-Fold CV AUC Scores: ", results['test_auc'])
    # print("Average 5-Fold CV Accuracy: ", np.nanmean(results['test_accuracy']))
    # print("Average 5-Fold CV RMSE: ", np.nanmean(results['test_rmse']))
    # print("Average 5-Fold CV AUC: ", np.nanmean(np.array(results['test_auc'])))
    return results