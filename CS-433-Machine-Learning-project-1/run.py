import numpy as np
from helpers import *
from implementations import *
from data_processing import *
from hyperparams import *
# from classification import *


# loading train data 


yb, input_data, ids = load_csv_data("train.csv")
dimensions = np.shape(input_data)
N = dimensions[0]
P = dimensions[1]
yb = np.reshape(yb,[N,1])
yb[yb==-1] = 0

#Replace missing data by median if more than half of the data is missing, or mean if less
tx = data_replace(input_data)

#Create polynoms
# Construct the matrix with the polynomial expansion for each column
size_x_tr = tx.shape[0]
gamma = 0.4 #Quite low to avoid NaN
lambdas = np.logspace(-4, -1.8, 20) #the lambda tested for the logistic regression, for each lambda we test each degree
initial_w = np.ones([size_x_tr,1])
max_iters = 20
degrees_tested = [1,2,3,4,5] #the degrees tested for the polynoms
columns_to_expand = [2,3,5,6,7,8,9,10,11,12,13,14,16,17,19,22,24,25,26,27,28]#[1,2,3,5,6,7,8,9,10,11,12,13,14,16,17,19,22,24,25,26,27,28] #enlever : 1
phi, degrees_poly = phi_optimized(yb,tx,degrees_tested,P, 4, initial_w, lambdas, gamma,max_iters,columns_to_expand)


#Standardize each feature according to its type of distribution

indices_min_max =[3,11,12,22,26]
indices_gaussian =[0,1,6,8,13,14,16,17,24,27]
indices_angles = [15,18,20,25,28]
indices_gaussian_log = [2,5,7,9,10,19]

normalize (phi, indices_gaussian_log, indices_angles, indices_gaussian, indices_min_max)

##Normalize the polynoms according to their min and max (rescaling)

for x in range(29,phi.shape[1]) :
     c = phi[:,x]
     min = np.min(c, axis=0)
     max = np.max(c, axis=0)
     phi[:,x] = (c-min) / (max-min)

#Add dummy variables instead of the categorical (feature PRI jet num)

nb_indice = phi.shape[1]
new_col = np.zeros([phi.shape[0],4])
phi = np.c_[phi, new_col]
print(phi.shape)
for i in range(phi.shape[0]) : 
    if (phi[i,22] == 0) : 
        phi[i,nb_indice] = 1
    if (phi[i,22] == 1) : 
        phi[i,nb_indice +1] = 1
    if (phi[i,22] == 2) : 
        phi[i,nb_indice +2] = 1
    if (phi[i,22] == 3) : 
        phi[i,nb_indice +3] = 1


#Deleted features correlated more than 85% with another feature, and PRI jet num

phi = np.delete(phi,29,1)
phi = np.delete(phi,23,1)
phi = np.delete(phi,22,1)
phi = np.delete(phi,21,1)
phi = np.delete(phi,4,1)

phi = add_w0(phi,phi.shape[0])   

#Choose a subset to train

"""tx_reduced = (phi[range(200000),:])  # 100x30 data for faster testing of regression
y_reduced = (y_new[range(200000)])
ratio = 0.8"""

#Split into train and test data
#y_tr, x_tr, y_te, x_te = split_data(y_reduced,tx_reduced,ratio)

#Train on the whole data 
x_tr = phi
y_tr = yb

#Logistic regression
initial_w = np.zeros([x_tr.shape[1],1])   
max_iters = 1000
gamma = 0.7
w_opt,loss = logistic_regression(y_tr,x_tr,initial_w,max_iters,gamma)

#If want to do reg logistic regression

#Calculate best lambda with cross validation

"""initial_w = np.ones([x_tr.shape[1],1])
max_iters = 200
best_lambda, best_rmse = cross_validation_demo(y_tr, x_tr, 7, 4,initial_w,  np.logspace(-8, 0, 30), 3 ,gamma, max_iters )
"""
#Calculate reg logistic regression
#w_opt2,loss2 = reg_logistic_regression(y_tr, x_tr, best_lambda, initial_w, max_iters, gamma)


#Processing test data

y_test, x_test, ids = load_csv_data("test.csv")

dimensions_te = np.shape(x_test)
N = dimensions_te[0]
P = dimensions_te[1]
y_test = np.reshape(y_test,[N,1])
y_test[y_test==-1] = 0

# Replace by mean/median

x_te = data_replace(x_test)
y_te = y_test

#Build same polynoms as train data
columns_to_expand = [2,3,5,6,7,8,9,10,11,12,13,14,16,17,19,22,24,25,26,27,28]
i = 0
for column in columns_to_expand : 
    if (degrees_poly[i] > 1) : 
        poly_x = build_poly(x_te[:,column], degrees_poly[i])
        x_te = np.c_[x_te, poly_x ]
    i = i+1

#Standardize each feature according to its type of distribution

indices_min_max =[3,11,12,22,26]
indices_gaussian =[0,1,6,8,13,14,16,17,24,27]
indices_angles = [15,18,20,25,28]
indices_gaussian_log = [2,5,7,9,10,19]

normalize (x_te, indices_gaussian_log, indices_angles, indices_gaussian, indices_min_max)

#Standardize polynoms
for x in range(29,x_te.shape[1]) :
     c = x_te[:,x]
     min = np.min(c, axis=0)
     max = np.max(c, axis=0)
     x_te[:,x] = (c-min) / (max-min)


#Add dummy variables instead of the categorical (feature PRI jet num)

nb_indice = x_te.shape[1]
new_col = np.zeros([x_te.shape[0],4])
x_te = np.c_[x_te, new_col]
print(phi.shape)
for i in range(x_te.shape[0]) : 
    if (x_te[i,22] == 0) : 
        x_te[i,nb_indice] = 1
    if (x_te[i,22] == 1) : 
        x_te[i,nb_indice +1] = 1
    if (x_te[i,22] == 2) : 
        x_te[i,nb_indice +2] = 1
    if (x_te[i,22] == 3) : 
        x_te[i,nb_indice +3] = 1

#Deleted features correlated more than 85% with another feature, and PRI jet num

x_te = np.delete(x_te,29,1)
x_te = np.delete(x_te,23,1)
x_te = np.delete(x_te,22,1)
x_te = np.delete(x_te,21,1)
x_te = np.delete(x_te,4,1)

x_te= add_w0(x_te,x_te.shape[0])   

y_pred = sigmoid(x_te@w_opt)
y_pred[y_pred>0.5] = 1
y_pred[y_pred<0.5] = -1

create_csv_submission(ids, y_pred, "predictions3")
