# EPFL CS-433: PROJECT 1 "Higgs Boson Challenge"

**NOTE: This is a (almost unmodified) copy of the repository https://github.com/CS-433/ml-project-1-no_cs_members.git**
Refer to `project1_description.pdf` for the project's description, and to `CS-433-ML-Project-1-report.pdf` for the final written report.

The Higgs boson is an elementary particle which explains why other particles have a mass. Measurements during high-speed collisions of protons at CERN were made public with the aim of predicting whether the collision by-products are an actual boson or background noise.

The work was mainly done in 2 ways: data pre-processing and then applying logistic regression.  

Preprocessing can include different combinations of the following methods: (1) replacing undefined datapoints by the median/mean, (2) performing a polynomial expansion, (3) standardizing.

Logistic Regressions are subsequently implemented and legitimized by means of a 7-fold cross validation.

The entire project only uses python libraries Numpy and Matplolib (for visualisation). 

### Please add the files train.csv and test.csv directly in same folder as run.py and all the other files.

## Code description 

### `run.py`

This file produces the predictions same file used to obtain the team's ("no_CS_members") the best score on the aicrowd platform. It is self-contained and only requires access to the data and files described below.

---

### `implementations.py`

This file contains the required functions as stated in the project outline pdf file.

* *mean_squared_error_gd, mean_squared_error_sgd, least_squares, ridge_regression*
* *logistic_regression, reg_logistic_regression*

As well as auxiliary functions supporting the ones cited above.

* *compute_mse_loss, compute_mse_gradient, batch_iter, compute_stoch_mse_gradient, sigmoid, calculate_logistic_loss, calculate_logistic_gradient*
* *calculate_stoch_logistic_gradient, stoch_reg_logistic_regression*

---

### `data_processing.py`

This file contains functions used to pre-process data.

* *data_removed, data_replaced, split_data, add_w0*
* *normalize_log_gaussian, normalize_angles, normalize_gaussian, normalize_min_max, normalize*

--- 

### `hyperparams.py`

This file contains functions to optimize the hyperparameter lambda.
* *build_k_indices, cross_validation, cross_validation_demo*

And to calculate the best degree for the polynomial expansion of each feature, and build the corresponding polynom.
* *build_poly, best_degree_selection, phi_optimized*

--- 

### `classification.py`

This file contains functions used to classify the data, aswell as some for computation of evaluating metrics.
*  *simple_class, get_accuracy, get_only_accuracy, get_auc, roc_visualization*
*  *get_Kneigbors, getKpredictions*

--- 

### `our_progress_run.ipynb`

A notebook outlining the step-by-step progress of the model (each stage adds something on top of the previous version):

1. logistic regression 
2. logistic regression + normalized 
3. logistic regression + normalized + w0
4. logistic regression + normalized smart + w0
5. logistic regression + normalized smart + w0 + high correlation features removed

---
### `our_progress_loop.py`
This file allowed to run mutiple repetitions of each method described in "our progress", in order to compare their mean and standard deviation.

---
### `seven_methods.py`

This file allows to calculate the accuracy for seven methods of regression and classification coded for this project.

* A. Linear Regression with GD
* B. Linear Regression with SGD
* C. Closed form Linear Regression (least squares)
* D. Regularized Linear Regression with hyper-parameter search using k-fold cross-validation
* E. Logistic Regression 
* F. Regularized Logistic Regression with hyper-parameter search using k-fold cross-validation
* G. K-nearest neighbors classification

---
### `boxplotloop.py`

This file allows to calculate the accuracy for each method on random train sets, in order to build their box plot.

---
### `helpers.py`

Helper functions used to load the data and create the csv submission.

---
### `predictions.csv`

The exact file that was used for the group's best submission (#204331)


---
## Author (cleaned and updated current version)

* Alexei Ermochkine

*Contributions for base version by*
* Iris Toye
* Mathilde Morelli
