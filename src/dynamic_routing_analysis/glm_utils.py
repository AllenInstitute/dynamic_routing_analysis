import copy
import logging
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import xarray as xr
from numpy.linalg import LinAlgError
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV, KFold
from tqdm import tqdm

import dynamic_routing_analysis.models as mu

logger = logging.getLogger(__name__) # debug < info < warning < error

np.random.seed(0)

model_mapping = {
    'ridge_regression': mu.Ridge,
    'lasso_regression': mu.Lasso,
    'elastic_net_regression': mu.ElasticNet,
    'reduced_rank_regression': mu.ReducedRankRegression
}

class RunParams:
    def __init__(self, session_id):
        self.run_params = {
            "session_id": session_id,
            "method": 'ridge_regression',  # ['ridge_regression', 'lasso_regression','reduced_rank_regression']

            "no_nested_CV": False,
            "optimize_on": 0.3,
            "n_outer_folds": 5,
            "n_inner_folds": 5,
            "optimize_penalty_by_cell": False,
            "optimize_penalty_by_area": False,
            "optimize_penalty_by_firing_rate": False,
            "optimize_penalty_by_best_units": False, # TO DO
            "best_unit_cvr2_cut_off": 0.05, # TO DO
            "use_fixed_penalty": False,
            "num_rate_clusters": 5,

            # RIDGE + ELASTIC NET
            "L2_grid_type": 'log',
            "L2_grid_range": [1, 2**12],
            "L2_grid_num": 13,
            "L2_fixed_lambda": None,

            # LASSO
            "L1_grid_type": 'log',
            "L1_grid_range": [10**-6, 10**-2],
            "L1_grid_num": 13,
            "L1_fixed_lambda": None,

            "cell_regularization": None,
            "cell_regularization_nested": None,

            # ELASTIC NET
            "L1_ratio_grid_type": 'log',
            "L1_ratio_grid_range": [10**-6, 10**-1],
            "L1_ratio_grid_num": 9,
            "L1_ratio_fixed": None,
            "cell_L1_ratio": None,
            "cell_L1_ratio_nested": None,

            # RRR
            "rank_grid_num": 10,
            "rank_fixed": None,
            "cell_rank": None,
            "cell_rank_nested": None,

            "fullmodel_fitted": False,
        }

    def update_metric(self, key, value):
        """Update or add a parameter in the run_params dictionary."""
        if key not in self.run_params:
            logger.warning(f"{key} is not a valid key. Adding new parameter '{key}' with value {value}")
        self.run_params[key] = value

    def update_multiple_metrics(self, updates: dict):
        """Update multiple parameters at once."""
        for key, value in updates.items():
            self.update_metric(key, value)

    def get_params(self):
        """Retrieve the run_params dictionary."""
        return self.run_params

    def validate_params(self):
        """Validation logic to ensure parameters are consistent."""
        if self.run_params["method"] not in ['ridge_regression', 'lasso_regression', 'elastic_net', 'reduced_rank_regression']:
            raise ValueError("Invalid method: must be one of ['ridge_regression', 'lasso_regression', 'elastic_net_regression', 'reduced_rank_regression']")

        if self.run_params["rank_fixed"] or self.run_params["L1_ratio_fixed"] or self.run_params["L1_fixed_lambda"] or self.run_params["L2_fixed_lambda"]:
            if not self.run_params["use_fixed_penalty"]:
                logger.warning('Fixed penalty parameter passed, setting fixed penalty to True')
                self.run_params["use_fixed_penalty"] = True


def nested_train_and_test(design_mat, spike_counts, param_grid, param2_grid = None, folds_outer=10, folds_inner=6, method = 'ridge_regression'):

    """
    Performs nested cross-validation for model selection and evaluation.

    Args:
        design_mat: Design matrix (X) containing predictor variables.
        spike_counts: Response variable (y).
        param_grid: Regularizer parameter grid for ridge, lasso and elastic net, and rank grid for RRR(e.g., lambda values).
        param2_grid: Mixing parameter grid (e.g., l1_ratio for elastic net), if applicable.
        folds_outer: Number of outer cross-validation folds.
        folds_inner: Number of inner cross-validation folds.
        method: Regression method ('ridge_regression', 'lasso_regression', 'elastic_net_regression', 'reduced_rank_regression').

    Returns:
        train_r2: Cleaned R^2 values for training data across outer CV folds.
        test_r2: Cleaned R^2 values for test data across outer CV folds.
        weights: Fitted model weights.
        y_pred: Predictions on the entire dataset using the final model.
        optimal_params: Optimal parameters chosen for each fold during outer CV.
    """

    def get_param_grid(method, param_grid, param2_grid = None):
        """Get the parameter grid for the specified method."""
        param_dict = {}
        if method in ['ridge_regression','lasso_regression', 'elastic_net_regression']:
            param_dict['lam'] = param_grid
            if method == 'elastic_net_regression':
                param_dict['l1_ratio'] = param2_grid
        elif method == 'reduced_rank_regression':
            param_dict['rank'] = param_grid
        else:
            raise ValueError(f"Unknown method: {method}")
        return param_dict

    X = design_mat.data
    y = spike_counts

    kf = KFold(n_splits=folds_outer, shuffle=True, random_state=0)

    train_r2 = np.zeros((y.shape[-1], folds_outer))
    test_r2 = np.zeros((y.shape[-1], folds_outer))
    param_dict = get_param_grid(method, param_grid, param2_grid = param2_grid)
    optimal_params = {key: np.zeros(folds_outer) + np.nan for key in param_dict.keys()}

    # outer CV
    for k, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        # inner CV
        cv_inner = KFold(n_splits=folds_inner, shuffle=True, random_state=1)
        model = model_mapping.get(method)()


        try:
            search = GridSearchCV(model, param_dict, cv = cv_inner, refit=True, n_jobs=1)
        except LinAlgError:
            logger.info(f"Fold {k} failed due to a linear algebra error. Skipping fold.")
            continue

        result = search.fit(X_train, y_train)
        best_model = result.best_estimator_

        for key in param_dict.keys():
            optimal_params[key][k] = search.best_params_[key]

        # needs to be calculated because it updates
        # the r2 of the best model with the test-r2
        train_mean_score = best_model.score(X_train, y_train)
        train_r2[:, k] = best_model.r2
        test_mean_score = best_model.score(X_test, y_test)
        test_r2[:, k] = best_model.r2


    if np.sum(test_r2) == 0:
        raise LinAlgError("Test cv$R^2$ values were never updated")

    # Train the model on the entire dataset with the median parameter value
    median_param = {key: np.nanmedian(optimal_params[key]) for key in optimal_params.keys()}
    model = model_mapping[method](**median_param).fit(X, y)

    weights = model.W
    y_pred = model.predict(X)

    return clean_r2_vals(train_r2), clean_r2_vals(test_r2), weights, y_pred, optimal_params


def simple_train_and_test(design_mat, spike_counts, param, param2 = None, folds_outer=10, method = 'ridge_regression'):
    """
    Train and test a Ridge regression model using cross-validation with specified lambda values.

    Args:
        design_mat: Input design matrix containing data.
        spike_counts: Target variable (spike counts).
        param: Regularizer parameter for ridge, lasso and elastic net, and rank for RRR (single value or a list of values for each fold).
        param2: Mixing parameter (e.g., l1_ratio for elastic net), if applicable.
        folds_outer: Number of folds for outer cross-validation.

    Returns:
        train_r2: Mean R2 score on training data across folds.
        test_r2: Mean R2 score on testing data across folds.
        weights: Model weights after training on the entire dataset.
        y_pred: Predictions on the entire dataset.
    """

    def ensure_param(param, folds_outer):
        if param is None:
            return param
        if not isinstance(param, list) and not isinstance(param, np.ndarray):
            return [param] * folds_outer
        if len(param) != folds_outer:
            raise ValueError(f"Length of parameter, ({len(param)}) must match number of folds ({folds_outer}).")
        return param

    X = design_mat.data
    y = spike_counts

    kf = KFold(n_splits=folds_outer, shuffle=True, random_state=0)
    test_r2 = np.zeros((y.shape[-1], folds_outer))
    train_r2 = np.zeros((y.shape[-1], folds_outer))
    param = ensure_param(param, folds_outer)
    param2 = ensure_param(param2, folds_outer)

    for k, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        if method in ['ridge_regression', 'lasso_regression']:
            model = model_mapping[method](lam=param[k])  # Use the k-th lambda value for this fold
        elif method == 'elastic_net_regression':
            model = model_mapping[method](lam=param[k], l1_ratio=param2[k])
        elif method == 'reduced_rank_regression':
            model = model_mapping[method](rank=param[k])

        try:
            model.fit(X_train, y_train)
        except LinAlgError:
            logger.info(f"Fold {k} failed due to a linear algebra error. Skipping fold.")
            continue

        train_mean_score = model.score(X_train, y_train)
        train_r2[:, k] = model.r2
        test_mean_score = model.score(X_test, y_test)
        test_r2[:, k] = model.r2

    # check if test are empty. raise LinAlgError
    if np.sum(test_r2) == 0:
        raise LinAlgError("Test cv$R^2$ values were never updated")

    # Train the model on the entire dataset with the median parameter value
    if method in ['ridge_regression', 'lasso_regression']:
        model = model_mapping[method](lam=np.nanmedian(param)).fit(X, y)
    elif method == 'elastic_net_regression':
        model = model_mapping[method](lam=np.nanmedian(param), l1_ratio=np.nanmedian(param2))
    elif method == 'reduced_rank_regression':
        model = model_mapping[method](rank=np.nanmedian(param)).fit(X, y)

    model.fit(X, y)
    weights = model.W
    y_pred = model.predict(X)

    return clean_r2_vals(train_r2), clean_r2_vals(test_r2), weights, y_pred

def get_parameter_grid(fit,method):
    param_grid = fit['L2_grid'] if method in ['ridge_regression',  'elastic_net_regression'] else None
    param_grid = fit['rank_grid'] if method == 'reduced_rank_regression' else param_grid
    param_grid = fit['L1_grid'] if method in ['lasso_regression'] else param_grid
    param2_grid = fit['L1_ratio_grid'] if method == 'elastic_net_regression' else None
    return param_grid, param2_grid

def get_parameters(fit, method, suffix = ''):
    param = fit[f'cell_regularization{suffix}'] if method in ['ridge_regression', 'lasso_regression', 'elastic_net_regression'] else fit[f'cell_rank{suffix}']
    param2 = fit[f'cell_L1_ratio{suffix}'] if method == 'elastic_net_regression' else None
    return param, param2

def set_parameter_grids(fit, run_params, X = None):
    method = run_params['method']

    if method in ['ridge_regression', 'elastic_net_regression']:
            fit['L2_grid'] = np.geomspace(
                run_params['L2_grid_range'][0],
                run_params['L2_grid_range'][1],
                num=run_params['L2_grid_num']
            ) if run_params['L2_grid_type'] == 'log' else np.linspace(
                run_params['L2_grid_range'][0],
                run_params['L2_grid_range'][1],
                num=run_params['L2_grid_num']
            )

    if method == 'lasso_regression':
        fit['L1_grid'] = np.geomspace(
            run_params['L1_grid_range'][0],
            run_params['L1_grid_range'][1],
            num=run_params['L1_grid_num']
        ) if run_params['L1_grid_type'] == 'log' else np.linspace(
            run_params['L1_grid_range'][0],
            run_params['L1_grid_range'][1],
            num=run_params['L1_grid_num']
        )

    if method == 'elastic_net_regression':
        fit['L1_ratio_grid'] = np.linspace(run_params['L1_ratio_grid_range'][0], run_params['L1_ratio_grid_range'][1], num=run_params['L1_ratio_grid_num'])

    if method == 'reduced_rank_regression':
        fit['rank_grid'] = np.linspace(1, np.min(X.shape), num=run_params['rank_grid_num']).astype(int)

    return fit

def set_parameters_nested_CV(fit, unit_ids, method, optimal_params):
    if method in ['ridge_regression', 'lasso_regression', 'elastic_net_regression']:
        fit['cell_regularization_nested'][unit_ids] = optimal_params['lam']
        if method == 'elastic_net_regression':
            fit['cell_L1_ratio_nested'][unit_ids] = optimal_params['l1_ratio']
    elif method == 'reduced_rank_regression':
        fit['cell_rank_nested'][unit_ids] = optimal_params['rank']
    return fit

# Define function to process a single unit
def process_unit(unit_no, design_mat, fit, run_params):
    """
    Process a single unit for optimization or fitting.

    Parameters:
    - unit_no: int, the unit index to process.
    - design_mat: ndarray, the design matrix for the model.
    - fit: dict, contains fitting parameters (e.g., spike counts, L2 grid, regularization).
    - run_params: dict, contains runtime parameters (e.g., number of folds, no_nested_CV).

    Returns:
    - Tuple with results depending on the specified function.
    """
    fit_cell = fit['spike_count_arr']['spike_counts'][:, unit_no].reshape(-1, 1)
    method = run_params['method']

    # Determine regularization and other parameters based on method and run_params
    if run_params['no_nested_CV']:
        param, param2 = get_parameters(fit, method)
    elif run_params["fullmodel_fitted"]:
        param, param2 = get_parameters(fit, method, '_nested')
    else:
        param_grid, param2_grid = get_parameter_grid(fit, method)

        # If nested CV is enabled, use nested_train_and_test
        cv_train, cv_test, weights, prediction, optimal_parameters = nested_train_and_test(
            design_mat, fit_cell, param_grid=param_grid, param2_grid=param2_grid,
            folds_outer=run_params['n_outer_folds'], folds_inner=run_params['n_inner_folds'], method=method
        )
        return unit_no, cv_train, cv_test, weights, prediction, optimal_parameters

    # Perform simple training and testing for regular cases
    cv_train, cv_test, weights, prediction = simple_train_and_test(
        design_mat, fit_cell, param=param[unit_no], param2=param2[unit_no],
        folds_outer=run_params['n_outer_folds'], method=method
    )
    return unit_no, cv_train, cv_test, weights, prediction


def optimize_model(fit, design_mat, run_params):
    '''
    Optimize model parameters by performing cross-validation and selecting the best regularization or rank values 
    based on the provided parameters and data.

    Parameters:
    - fit: dict, contains model fitting information, including spike counts,and model parameters.
    - design_mat: xarray.DataArray, the design matrix with data to train the model.
    - run_params: dict, contains runtime parameters, including settings for model, optimization, penalty values, and cross-validation parameters.

    Returns:
    - fit: dict, updated with the optimized parameters and cross-validation results.

    '''

    spike_counts = fit['spike_count_arr']['spike_counts']

    num_units = spike_counts.shape[1]
    T = spike_counts.shape[0]
    method = run_params['method']
    param_keys = ['cell_regularization', 'cell_L1_ratio', 'cell_rank']
    param_keys += [key + '_nested' for key in param_keys]

    if run_params['use_fixed_penalty']:
        print('Using a hard-coded regularization value')

        for key in param_keys:
            fit[key] = None
        fit['cell_regularization'] = run_params['L2_fixed_lambda'] if method in ['ridge_regression', 'elastic_net_regression'] else None
        fit['cell_regularization'] = run_params['L1_fixed_lambda'] if method in ['lasso_regression'] else fit['cell_regularization']
        fit['cell_L1_ratio'] = run_params['L1_ratio_fixed'] if method == 'elastic_net_regression' else None
        fit['cell_rank'] = run_params['rank_fixed'] if method == 'reduced_rank_regression' else None

    elif run_params['no_nested_CV']:

        if run_params['fullmodel_fitted']:
            return fit

        for key in param_keys:
            fit[key] = np.full((num_units), np.nan)
        fit = set_parameter_grids(fit, run_params)

        # select a subset of the data to optimize on
        T_optimize = int(run_params['optimize_on'] * T)
        samples_optimize = np.random.choice(T, T_optimize, replace=False) if run_params['optimize_on'] < 1 else np.arange(T_optimize)
        design_mat_optimize = design_mat.copy()
        design_mat_optimize = design_mat_optimize.isel(timestamps=samples_optimize)
        design_mat_optimize['weights'] = design_mat.weights
        design_mat_optimize['timestamps'] = design_mat.timestamps[samples_optimize]
        spike_counts_optimize = spike_counts[samples_optimize, :]

        param_grid, param2_grid = get_parameter_grid(fit, method)

        grid_shape = (num_units, len(param_grid), len(param2_grid)) if param2_grid is not None else (num_units, len(param_grid))
        train_cv = np.full(grid_shape, np.nan)
        test_cv = np.full(grid_shape, np.nan)

        print(': optimizing parameters for all cells')
        for index1, param in enumerate(param_grid):
            for index2, param2 in (enumerate(param2_grid) if param2_grid is not None else [(None, None)]):
                # Run simple train and test for each parameter combination
                cv_var_train, cv_var_test, _, _, = simple_train_and_test(design_mat_optimize, spike_counts_optimize,
                                                                         param=param, param2=param2,
                                                                         folds_outer=run_params['n_outer_folds'],
                                                                         method=method)

                train_cv[:, index1, index2] = np.nanmean(cv_var_train, axis=1) if param2_grid is not None else np.nanmean(cv_var_train, axis=1).reshape(-1, 1)
                test_cv[:, index1, index2] = np.nanmean(cv_var_test, axis=1) if param2_grid is not None else np.nanmean(cv_var_test, axis=1).reshape(-1, 1)


        # Find the best parameter values for each cell
        if method in ['ridge_regression', 'lasso_regression']:
            fit['cell_regularization'] = [param_grid[x] for x in np.nanargmax(test_cv, 1)]

        elif method == 'elastic_net_regression':
            for unit_no in range(num_units):
                max_row, max_col = np.unravel_index(np.argmax(test_cv[unit_no]), test_cv[unit_no].shape)
                fit['cell_regularization'] = param_grid[max_row]
                fit['cell_L1_ratio'] = param2_grid[max_col]

        elif method == 'reduced_rank_regression':
            fit['cell_rank'] = [param_grid[x] for x in np.nanargmax(test_cv, 1)]


        fit['test_cv_grid'] = test_cv
        fit['train_cv_grid'] = train_cv

    else:

        fit = set_parameter_grids(fit, run_params, design_mat.data)

    return fit


def evaluate_model(fit, design_mat, run_params):
    X = design_mat.data
    spike_counts = fit['spike_count_arr']['spike_counts']
    method = run_params['method']

    # Initialize outputs
    num_units = spike_counts.shape[1]
    num_outer_folds = run_params['n_outer_folds']
    cv_var_train = np.full((num_units, num_outer_folds), np.nan)
    cv_var_test = np.full((num_units, num_outer_folds), np.nan)
    all_weights = np.full((X.shape[1], num_units), np.nan)
    all_prediction = np.full(spike_counts.shape, np.nan)

    param_keys = ['cell_regularization', 'cell_L1_ratio', 'cell_rank']
    param_keys += [key + '_nested' for key in param_keys]

    # fullmodel is completely fitted (simple or nested), and reduced model is to be fit
    if run_params["fullmodel_fitted"]:
        for key in param_keys:
            if isinstance(run_params[key], list) or isinstance(run_params[key], np.ndarray):
                fit[key] = np.array(run_params[key])

    if run_params['use_fixed_penalty']:
        param, param2 = get_parameters(fit, method)
        cv_var_train, cv_var_test, all_weights, all_prediction = simple_train_and_test(
            design_mat, spike_counts,
            param=param,
            param2=param2,
            folds_outer=num_outer_folds,
            method = run_params['method']
        )
    # fullmodel is completely fitted (simple or nested) or fullmodel is not fit but model parameters have beeen optimized
    elif run_params["fullmodel_fitted"] or run_params['no_nested_CV']:
        if run_params['optimize_penalty_by_cell']:
            print(': fitting each cell')
            with ProcessPoolExecutor(max_workers=10) as executor:
                futures = {
                    executor.submit(process_unit, unit_no, design_mat.copy(), fit.copy(), run_params): unit_no
                    for unit_no in range(num_units)
                }
                for future in tqdm(as_completed(futures), total=num_units, desc='progress'):
                    unit_no, cv_train, cv_test, weights, prediction = future.result()
                    cv_var_train[unit_no] = cv_train
                    cv_var_test[unit_no] = cv_test
                    all_weights[:, unit_no] = weights.reshape(-1)
                    all_prediction[:, unit_no] = prediction.reshape(-1)

        elif run_params['optimize_penalty_by_area']:
            areas = np.unique(fit['spike_count_arr']['structure'])
            if run_params['no_nested_CV']:
                param, param2 = get_parameters(fit, method)
            else:
                param, param2 = get_parameters(fit, method, '_nested')
            print(': fitting units by area')
            for area in tqdm(areas, total=len(areas), desc='progress'):
                unit_ids = np.where(fit['spike_count_arr']['structure'] == area)[0]
                fit_area = spike_counts[:, unit_ids]
                param_area = np.nanmedian(np.take(param, unit_ids), axis = 0)
                param2_area = np.nanmedian(np.take(param2, unit_ids), axis = 0) if param2 is not None else None
                cv_train, cv_test, weights, prediction = simple_train_and_test(design_mat, fit_area,
                                                                               param=param_area,
                                                                               param2=param2_area,
                                                                               folds_outer=run_params['n_outer_folds'],
                                                                               method = run_params['method'])
                cv_var_train[unit_ids] = cv_train
                cv_var_test[unit_ids] = cv_test
                all_weights[:, unit_ids] = weights
                all_prediction[:, unit_ids] = prediction

        elif run_params['optimize_penalty_by_firing_rate']:
            num_clusters =  np.min([run_params['num_rate_clusters'], num_units])
            rate_clusters = KMeans(n_clusters =  num_clusters, random_state = 0).fit(fit['spike_count_arr']['firing_rate'].reshape(-1,1)).labels_
            if run_params['no_nested_CV']:
                param, param2 = get_parameters(fit, method)
            else:
                param, param2 = get_parameters(fit, method, '_nested')
            print(': fitting units by firing rate')
            for cluster in tqdm(np.unique(rate_clusters), total=len(np.unique(rate_clusters)), desc='progress'):
                unit_ids = np.where(rate_clusters == cluster)[0]
                fit_rate = spike_counts[:, unit_ids]
                param_cluster = np.nanmedian(np.take(param, unit_ids), axis = 0)
                param2_cluster = np.nanmedian(np.take(param2, unit_ids), axis = 0) if param2 is not None else None
                cv_train, cv_test, weights, prediction = simple_train_and_test(design_mat, fit_rate,
                                                                               param=param_cluster,
                                                                               param2=param2_cluster,
                                                                               folds_outer=run_params['n_outer_folds'],
                                                                               method = run_params['method'])
                cv_var_train[unit_ids] = cv_train
                cv_var_test[unit_ids] = cv_test
                all_weights[:, unit_ids] = weights
                all_prediction[:, unit_ids] = prediction

        else:
            if run_params['no_nested_CV']:
                param, param2 = get_parameters(fit, method)
            else:
                param, param2 = get_parameters(fit, method, '_nested')
            print(': fitting all units')
            param = np.nanmedian(param, axis = 0)
            param2 = np.nanmedian(param2, axis = 0) if param2 is not None else None
            cv_var_train, cv_var_test, all_weights, all_prediction = simple_train_and_test(design_mat,
                                                                                           spike_counts,
                                                                                           param=param,
                                                                                           param2=param2,
                                                                                           folds_outer=run_params[
                                                                                               'n_outer_folds'],
                                                                                               method = run_params['method'])
    else: # fitting fullmodel using nested CV
        for key in param_keys:
            fit[key] = np.full((num_units, num_outer_folds), np.nan)

        param_grid, param2_grid = get_parameter_grid(fit, method)

        if run_params['optimize_penalty_by_cell']:
            print(': fitting each cell')
            with ProcessPoolExecutor(max_workers=10) as executor:
                futures = {
                    executor.submit(process_unit, unit_no, design_mat.copy(), fit.copy(), run_params): unit_no
                    for unit_no in range(num_units)
                }
                for future in tqdm(as_completed(futures), total=num_units, desc='progress'):
                    unit_no, cv_train, cv_test, weights, prediction, optimal_parameters = future.result()
                    cv_var_train[unit_no] = cv_train
                    cv_var_test[unit_no] = cv_test
                    all_weights[:, unit_no] = weights.reshape(-1)
                    all_prediction[:, unit_no] = prediction.reshape(-1)
                    fit = set_parameters_nested_CV(fit, unit_no, method, optimal_parameters)

        elif run_params['optimize_penalty_by_area']:
            areas = np.unique(fit['spike_count_arr']['structure'])
            for area in tqdm(areas, total=len(areas), desc='progress'):
                unit_ids = np.where(fit['spike_count_arr']['structure'] == area)[0]
                fit_area = spike_counts[:, unit_ids]
                cv_train, cv_test, weights, prediction, optimal_parameters = \
                    nested_train_and_test(design_mat, fit_area,
                                          folds_outer=run_params['n_outer_folds'],
                                          folds_inner=run_params['n_inner_folds'],
                                          param_grid=param_grid,
                                          param2_grid=param2_grid,
                                          method = run_params['method'])

                cv_var_train[unit_ids] = cv_train
                cv_var_test[unit_ids] = cv_test
                all_weights[:, unit_ids] = weights
                all_prediction[:, unit_ids] = prediction
                fit = set_parameters_nested_CV(fit, unit_ids, method, optimal_parameters)

        elif run_params['optimize_penalty_by_firing_rate']:
            num_clusters =  np.min([run_params['num_rate_clusters'], num_units])
            rate_clusters = KMeans(n_clusters = num_clusters, random_state = 0).fit(fit['spike_count_arr']['firing_rate'].reshape(-1,1)).labels_
            fit['spike_count_arr']['rate_clusters'] = rate_clusters
            for cluster in tqdm(np.unique(rate_clusters), total=len(np.unique(rate_clusters)), desc='progress'):
                unit_ids = np.where(rate_clusters == cluster)[0]
                fit_rate = spike_counts[:, unit_ids]
                cv_train, cv_test, weights, prediction, optimal_parameters = \
                    nested_train_and_test(design_mat, fit_rate,
                                          folds_outer=run_params['n_outer_folds'],
                                          folds_inner=run_params['n_inner_folds'],
                                          param_grid=param_grid,
                                          param2_grid=param2_grid,
                                          method = run_params['method'])

                cv_var_train[unit_ids] = cv_train
                cv_var_test[unit_ids] = cv_test
                all_weights[:, unit_ids] = weights
                all_prediction[:, unit_ids] = prediction
                fit = set_parameters_nested_CV(fit, unit_ids, method, optimal_parameters)


        else:
            cv_var_train, cv_var_test, all_weights, all_prediction, optimal_parameters = \
                nested_train_and_test(design_mat, spike_counts,
                                      folds_outer=run_params['n_outer_folds'],
                                      folds_inner=run_params['n_inner_folds'],
                                      param_grid=param_grid,
                                      param2_grid=param2_grid,
                                      method = run_params['method'])
            fit = set_parameters_nested_CV(fit, np.arange(num_units), method, optimal_parameters)

    model_label = run_params['model_label']
    fit[model_label] = {
        'weights': all_weights,
        # 'full_model_prediction': all_prediction,
        'weight_label': design_mat.weights.values,
        'cv_var_train': np.nanmean(cv_var_train, axis = 1),
        'cv_var_test': np.nanmean(cv_var_test, axis = 1)
    }

    if model_label == 'fullmodel':
        fit['fullmodel_fitted'] = True

    return fit


def get_shift_bins(run_params, fit, context):

    def convert_blocks(vector):
        vector = np.array(vector)
        unique_blocks = np.cumsum(np.diff(np.insert(vector, 0, vector[0])) != 0)
        return (unique_blocks + 1).tolist()


    bin_centers = fit['bin_centers']
    bin_width = fit['spike_bin_width']
    shift_by = run_params['linear_shift_by']
    blocks = convert_blocks(context)

    #find middle 4 block times
    first_block = bin_centers[blocks == np.nanmin(blocks)]
    middle_of_first=first_block[np.ceil(len(first_block)/2).astype('int')]

    last_block = bin_centers[blocks == np.nanmax(blocks)]
    middle_of_last = last_block[np.ceil(len(last_block)/2).astype('int')]


    middle_of_first_index = np.nanargmin(np.abs(bin_centers-middle_of_first))
    middle_of_last_index = np.nanargmin(np.abs(bin_centers-middle_of_last))
    middle_4_blocks = np.arange(middle_of_first_index, middle_of_last_index)

    #find the number of trials to shift by, from -1 to +1 block
    negative_shift= middle_4_blocks.min()
    positive_shift= len(bin_centers) - middle_4_blocks.max()
    shifts = np.arange(-negative_shift, positive_shift + 1, int(shift_by//bin_width))
    if 0 not in shifts:
        shifts = np.append(shifts, 0)
    shifts = np.sort(shifts)
    shifts = [shift for sh, shift in enumerate(shifts) if np.max(middle_4_blocks+shift) < len(bin_centers) and np.min(middle_4_blocks+shift) >= 0]
    return shifts, middle_4_blocks


def linear_shift(fit, design_mat, run_params):

    model_label = run_params['model_label']
    shifts, blocks = get_shift_bins(run_params, fit, design_mat.sel(weights='context_0'))
    shift_variables = [key for key in run_params['kernels'].keys() if run_params['kernels'][key]['shift']]
    shift_columns = [i for i, label in enumerate(design_mat.coords['weights'].values) if any([key in label for key in shift_variables])]
    shifted_cv_var_test = np.zeros((len(shifts), fit['spike_count_arr']['spike_counts'].shape[1])) + np.nan

    for sh, shift in enumerate(tqdm(shifts, desc="Shifting progress")):
        apply_shift_to_design_matrix(fit, design_mat, run_params, model_label, blocks, shift_columns, shifted_cv_var_test, sh, shift)

    fit[model_label]['shifted_cv_var_test'] = shifted_cv_var_test
    fit[model_label]['shifts'] = shifts

    return fit

def apply_shift_to_design_matrix(fit, design_mat, run_params, blocks, shift_columns, shift):
    times_used=blocks+shift
    shift_exists=[]
    for tt in times_used:
        if tt < len(fit['bin_centers']):
            shift_exists.append(True)
        else:
            shift_exists.append(False)
    shift_exists=np.array(shift_exists)
    times_used=times_used[shift_exists]


    design_mat_shifted = {'data': np.zeros((len(times_used), design_mat.data.shape[1])),
                              'weights': design_mat.weights.copy(),
                              'timestamps': design_mat['timestamps'][blocks].copy()}
    design_mat_shifted['data'][:, shift_columns] = design_mat.data[times_used][:, shift_columns]
    non_shift_columns = list(set(np.arange(design_mat.data.shape[1])) - set(shift_columns))
    design_mat_shifted['data'][:, non_shift_columns] = design_mat.data[blocks][:, non_shift_columns]

    design_mat_shifted = xr.Dataset(data_vars={"data": (["timestamps", "weights"], design_mat_shifted["data"])},
                                        coords={"timestamps": design_mat_shifted["timestamps"],
                                            "weights": design_mat_shifted["weights"]})

    fit_shift = copy.deepcopy(fit)
    fit_shift['spike_count_arr'].pop('spike_counts')
    fit_shift['spike_count_arr']['spike_counts'] = fit['spike_count_arr']['spike_counts'][blocks]
    fit_shift['bin_centers'] = fit_shift['bin_centers'][blocks]

    fit_shift = evaluate_model(fit_shift, design_mat_shifted, run_params)
    return fit_shift


def dropout(fit, design_mat, run_params):
    model_label = run_params['model_label']
    filtered_weights = [weight for weight in design_mat.weights.values if re.split(r'_(?=\d)', weight)[0] in run_params['input_variables']]
    design_matrix_reduced = design_mat.copy()
    design_matrix_reduced = design_matrix_reduced.sel(weights=filtered_weights)
    fit_drop = copy.deepcopy(fit)
    fit_drop = evaluate_model(fit_drop, design_matrix_reduced, run_params)
    return fit_drop


def clean_r2_vals(x):
    x[np.isinf(x) | np.isnan(x)] = 0
    return x
