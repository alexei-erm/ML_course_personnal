import numpy as np
from helpers import *
from implementations import *
from data_processing import *
from hyperparams import *
from classification import *

def loop(yb,x,ratio,P):

    # split data
    y_tr, x_tr, y_te, x_te = split_data(yb,x,ratio)

    # case 2

    x2 = np.copy(x_tr)

    x2, means2, std_dev2 = standardize(x2)

    initial_w = np.zeros([P,1])
    max_iters = 100
    gamma = 0.7
    w_opt_2, mse = logistic_regression(y_tr, x2, initial_w, max_iters, gamma)

    tx_new = np.copy(x_te)
    mean_te_2 = np.mean(tx_new)
    std_te_2 = np.std(tx_new)
    tx_new = tx_new - mean_te_2
    x_te_2 = tx_new / std_te_2  

    temporary = sigmoid(x_te_2@w_opt_2)
    y_result = temporary
    y_result[y_result>0.5] = 1
    y_result[y_result<0.5] = 0
    accuracy2 = get_only_accuracy(y_result, y_te)

    # case 3 

    x3 = np.copy(x_tr)
    x3, means3, std_dev3 = standardize(x3)
    x3 = add_w0(x3,x3.shape[0])

    initial_w = np.zeros([P+1,1])
    max_iters = 100
    gamma = 0.7
    w_opt_3, mse = logistic_regression(y_tr, x3, initial_w, max_iters, gamma)

    tx_new = x_te - means3
    x_te_3 = tx_new / std_dev3  
    x_te_3 = add_w0(x_te, x_te_3.shape[0])

    temporary = sigmoid(x_te_3@w_opt_3)

    y_result = temporary
    y_result[y_result>0.5] = 1
    y_result[y_result<0.5] = 0
    accuracy3 = get_only_accuracy(y_result, y_te)

    # case 4

    x4 = np.copy(x_tr)
    indices_min_max =[3,11,12,22,26]
    indices_gaussian =[0,1,6,8,13,14,16,17,24,27]
    indices_angles = [15,18,20,25,28]
    indices_gaussian_log = [2,5,7,9,10,19]
    normalize(x4, indices_gaussian_log, indices_angles, indices_gaussian, indices_min_max)
    x4= add_w0(x4,x4.shape[0])

    initial_w = np.zeros([P+1,1])
    max_iters = 100
    gamma = 0.7
    w_opt_4, mse = logistic_regression(y_tr, x4, initial_w, max_iters, gamma)

    x_te_4 = np.copy(x_te)
    normalize(x_te_4, indices_gaussian_log, indices_angles, indices_gaussian, indices_min_max)
    x_te_4 = add_w0(x_te_4,x_te_4.shape[0])

    temporary = sigmoid(x_te_4@w_opt_4)
    y_result = temporary
    y_result[y_result>0.5] = 1
    y_result[y_result<0.5] = 0
    accuracy4 = get_only_accuracy(y_result, y_te)

    # case 5

    x5 = np.copy(x_tr)
    indices_min_max =[3,11,12,22,26]
    indices_gaussian =[0,1,6,8,13,14,16,17,24,27]
    indices_angles = [15,18,20,25,28]
    indices_gaussian_log = [2,5,7,9,10,19]
    normalize(x5, indices_gaussian_log, indices_angles, indices_gaussian, indices_min_max)
    x5 = np.delete(x5,29,1)
    x5 = np.delete(x5,23,1)
    x5 = np.delete(x5,21,1)
    x5 = np.delete(x5,4,1)
    x5= add_w0(x5, x5.shape[0])

    initial_w = np.zeros([P-3,1])
    max_iters = 100
    gamma = 0.7
    w_opt_5, mse = logistic_regression(y_tr, x5, initial_w, max_iters, gamma)

    x_te_5 = np.copy(x_te)
    normalize(x_te_5, indices_gaussian_log, indices_angles, indices_gaussian, indices_min_max)
    x_te_5 = np.delete(x_te_5,29,1)
    x_te_5 = np.delete(x_te_5,23,1)
    #x_te_5 = np.delete(x_te_5,22,1)
    x_te_5 = np.delete(x_te_5,21,1)
    x_te_5 = np.delete(x_te_5,4,1)
    x_te_5= add_w0(x_te_5,x_te_5.shape[0])

    temporary = sigmoid(x_te_5@w_opt_5)
    y_result = temporary
    y_result[y_result>0.5] = 1
    y_result[y_result<0.5] = 0
    accuracy5 = get_only_accuracy(y_result, y_te)

    return accuracy2,accuracy3,accuracy4,accuracy5


def get_only_accuracy(y_result, y_te):
    """computes the accuracy of a given model output y_result compared to y_te
    
    Args: 
        y_results: shape=(N,)
        y_te: shape=(N,). Known predictions of test samples
    
    Returns:
        accuracy: scalar = (TP+TN)/(TP+TN+FP+FN)
    """ 
    difference = (y_result-y_te)
    good_guess = difference[difference==0]
    bad_guess = difference[difference!=0]
    accuracy = len(good_guess)/(len(good_guess)+len(bad_guess))
    return accuracy

def standardize(x):
    """standardize every feature of x with the mean and standard deviation of the respective feature 
    Args:
        x: shape=(N,P) N is the number of samples, D is the number of features
    Returns:
        modified tx 
    """
    X = np.copy(x)
    means = np.mean(X, axis=0)
    tx_new = X - means * np.ones(np.shape(X))
    std_dev = np.std(tx_new, axis=0)
    X = tx_new / (std_dev * np.ones(np.shape(X)))
    return X, means, std_dev
