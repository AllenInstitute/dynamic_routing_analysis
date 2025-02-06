import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import models as mu
import numpy as np
from numpy.linalg import LinAlgError
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV, KFold
from tqdm import tqdm

logger = logging.getLogger(__name__) # debug < info < warning < error

np.random.seed(0)

model_mapping = {
    'ridge_regression': mu.Ridge,
    'lasso_regression': mu.Lasso,
    'elastic_net_regression': mu.ElasticNet,
    'reduced_rank_regression': mu.ReducedRankRegression
}

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
        self
        '''
        try:
            # Use sklearn's Lasso for coordinate descent
            lasso = SklearnLasso(alpha=self.lam, fit_intercept=False, max_iter=1000)

            # If y has multiple targets, fit each one independently
            if y.ndim == 1:
                lasso.fit(X, y)
                self.W = lasso.coef_
            else:
                self.W = np.zeros((X.shape[1], y.shape[1]))
                for i in range(y.shape[1]):
                    lasso.fit(X, y[:, i])
                    self.W[:, i] = lasso.coef_

        except Exception as e:
            print("Unexpected error encountered:", e)
            raise  # Re-raise the exception to propagate unexpected errors

        assert ~np.isnan(np.sum(self.W)), 'Weights contain NaNs'

        self.mean_r2 = self.score(X, y)
        return self

    def get_params(self, deep=True):
        return {'lam': self.lam}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def score(self, X, y):
        '''
        Computes the fraction of variance in y explained by the linear model y = X*W.

        Parameters:
        X: np.ndarray
            Input data matrix of shape (n_samples, n_features).
        y: np.ndarray
            Target matrix of shape (n_samples, n_targets).

        Returns:
        float
            Mean R^2 across all targets.
        '''
        y_pred = self.predict(X)
        var_total = np.var(y, axis=0)  # Total variance in the target
        var_resid = np.var(y - y_pred, axis=0)  # Residual variance
        self.r2 = (var_total - var_resid) / var_total
        return np.nanmean(self.r2)

    def predict(self, X):
        '''
        Predicts outputs using the linear model.

        Parameters:
        X: np.ndarray
            Input data matrix of shape (n_samples, n_features).

        Returns:
        np.ndarray
            Predicted outputs of shape (n_samples, n_targets).
        '''
        return np.dot(X, self.W)


def nested_train_and_test(design_mat, spike_counts, L2_grid, folds_outer=10, folds_inner=6, method = 'ridge_regression'):
    X = design_mat.data
    y = spike_counts

    kf = KFold(n_splits=folds_outer, shuffle=True, random_state=0)
    lams = np.zeros(folds_outer) + np.nan
    train_r2 = np.zeros((y.shape[-1], folds_outer))
    test_r2 = np.zeros((y.shape[-1], folds_outer))

    # outer CV
    for k, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        # inner CV
        cv_inner = KFold(n_splits=folds_inner, shuffle=True, random_state=1)

        if method == 'ridge_regression':
            model = Ridge()
        elif method == 'lasso_regression':
            model = Lasso()

        try:
            search = GridSearchCV(model, {'lam': np.array(L2_grid)}, cv=cv_inner, refit=True,
                              n_jobs=1)
        except LinAlgError:
            logger.info(f"Fold {k} failed due to a linear algebra error. Skipping fold.")
            continue

        result = search.fit(X_train, y_train)
        best_model = result.best_estimator_
        lams[k] = result.best_params_['lam']

        # needs to be calculated because it updates
        # the r2 of the best model with the test-r2
        train_mean_score = best_model.score(X_train, y_train)
        train_r2[:, k] = best_model.r2
        test_mean_score = best_model.score(X_test, y_test)
        test_r2[:, k] = best_model.r2


    if np.sum(test_r2) == 0:
        raise LinAlgError("Test cv$R^2$ values were never updated")

    # Train the model on the entire dataset with the median lambda value
    lam = np.median(lams)
    if method == 'ridge_regression':
        model = Ridge(lam=lam).fit(X, y)
    elif method == 'lasso_regression':
        model = Lasso(lam=lam).fit(X, y)
    weights = model.W
    y_pred = model.predict(X)

    return clean_r2_vals(train_r2), clean_r2_vals(test_r2), weights, y_pred, lams


def simple_train_and_test(design_mat, spike_counts, lam, folds_outer=10, method = 'ridge_regression'):
    """
    Train and test a Ridge regression model using cross-validation with specified lambda values.

    Args:
        design_mat: Input design matrix containing data.
        spike_counts: Target variable (spike counts).
        lam: Regularization parameter (single value or a list of values for each fold).
        folds_outer: Number of folds for outer cross-validation.

    Returns:
        train_r2: Mean R2 score on training data across folds.
        test_r2: Mean R2 score on testing data across folds.
        weights: Model weights after training on the entire dataset.
        y_pred: Predictions on the entire dataset.
    """
    X = design_mat.data
    y = spike_counts

    kf = KFold(n_splits=folds_outer, shuffle=True, random_state=0)
    test_r2 = np.zeros((y.shape[-1], folds_outer))
    train_r2 = np.zeros((y.shape[-1], folds_outer))

    # If lam is a scalar, convert it to a list with the same value for each fold
    if not isinstance(lam, list):
        lam = [lam] * folds_outer

    if len(lam) != folds_outer:
        raise ValueError(f"Length of lam ({len(lam)}) must match number of folds ({folds_outer}).")

    for k, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        if method == 'ridge_regression':
            model = Ridge(lam=lam[k])  # Use the k-th lambda value for this fold
        elif method == 'lasso_regression':
            model = Lasso(lam=lam[k])
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

    # Train the model on the entire dataset with the median lambda value
    if method == 'ridge_regression':
        model = Ridge(lam=np.median(lam))
    elif method == 'lasso_regression':
        model = Lasso(lam=np.median(lam))
    model.fit(X, y)
    weights = model.W
    y_pred = model.predict(X)

    return clean_r2_vals(train_r2), clean_r2_vals(test_r2), weights, y_pred


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

    if run_params['no_nested_CV']:
        lam_value = fit['cell_L2_regularization'][unit_no]
    elif 'cell_L2_regularization_nested' in run_params.keys():
        lam_value = fit['cell_L2_regularization_nested'][unit_no]
    else:
        # If nested CV is enabled, use nested_train_and_test
        cv_train, cv_test, weights, prediction, lams = nested_train_and_test(
            design_mat,
            fit_cell,
            L2_grid=fit['L2_grid'],
            folds_outer=run_params['n_outer_folds'],
            folds_inner=run_params['n_inner_folds'],
            method = run_params['method']
        )
        return unit_no, cv_train, cv_test, weights, prediction, lams

    # Perform simple training and testing for regular cases
    cv_train, cv_test, weights, prediction = simple_train_and_test(
        design_mat,
        fit_cell,
        lam=lam_value,
        folds_outer=run_params['n_outer_folds'],
        method=run_params['method']
    )
    return unit_no, cv_train, cv_test, weights, prediction


def evaluate_ridge(fit, design_mat, run_params):
    '''
        fit, model dictionary
        design_mat, design matrix
        run_params, dictionary of parameters, which needs to include:
            optimize_penalty_by_cell     # If True, uses the best L2 value for each cell
            optimize_penalty_by_area  # If True, uses the best L2 value for this session
            use_fixed_penalty      # If True, uses the hard coded L2_fixed_lambda

            L2_fixed_lambda         # This value is used if L2_use_fixed_value
            L2_grid_range           # Min/Max L2 values for optimization
            L2_grid_num             # Number of L2 values for optimization
            L2_grid_type            # log or linear

        returns fit, with the values added:
            L2_grid                 # the L2 grid evaluated
            for the case of no nested CV,
                avg_L2_regularization      # the average optimal L2 value, or the fixed value
                cell_L2_regularization     # the optimal L2 value for each cell
    '''

    spike_counts = fit['spike_count_arr']['spike_counts']
    # x_is_continuous = [run_params['kernels'][kernel_name.rsplit('_', 1)[0]]['type'] == 'continuous'
    #                    for kernel_name in design_mat.weights.values]
    num_units = spike_counts.shape[1]
    T = spike_counts.shape[0]

    if run_params['use_fixed_penalty']:
        print(get_timestamp() + 'Using a hard-coded regularization value')
        fit['L2_regularization'] = run_params['L2_fixed_lambda']

    elif run_params['no_nested_CV']:
        if run_params['L2_grid_type'] == 'log':
            fit['L2_grid'] = np.geomspace(run_params['L2_grid_range'][0],
                                        run_params['L2_grid_range'][1],
                                        num=run_params['L2_grid_num'])
        else:
            fit['L2_grid'] = np.linspace(run_params['L2_grid_range'][0],
                                        run_params['L2_grid_range'][1],
                                        num=run_params['L2_grid_num'])

        if run_params['method'] != 'lasso_regression':
            fit['L2_grid'] = np.array([0] + list(fit['L2_grid']))

        T_optimize = int(run_params['optimize_on'] * T)
        samples_optimize = np.random.choice(T, T_optimize, replace=False)
        design_mat_optimize = design_mat.copy()
        design_mat_optimize = design_mat_optimize.isel(timestamps=samples_optimize)
        design_mat_optimize['weights'] = design_mat.weights
        design_mat_optimize['timestamps'] = design_mat.timestamps[samples_optimize]
        spike_counts_optimize = spike_counts[samples_optimize, :]

        train_cv = np.full((num_units, len(fit['L2_grid'])), np.nan)
        test_cv = np.full((num_units, len(fit['L2_grid'])), np.nan)

        print(get_timestamp() + ': optimizing L2 penalty for all cells')
        for L2_index, L2_value in enumerate(fit['L2_grid']):
            cv_var_train, cv_var_test, _, _, = simple_train_and_test(design_mat_optimize,
                                                                        spike_counts_optimize,
                                                                        lam=L2_value,
                                                                        folds_outer=run_params['n_outer_folds'],
                                                                        method = run_params['method'])
            train_cv[:, L2_index] = np.nanmean(cv_var_train, axis=1)
            test_cv[:, L2_index] = np.nanmean(cv_var_test, axis=1)
            test_cv[:, L2_index] = np.nanmean(cv_var_test, axis=1)

        fit['avg_L2_regularization'] = np.mean([fit['L2_grid'][x] for x in np.nanargmax(test_cv, 1)])
        fit['cell_L2_regularization'] = [fit['L2_grid'][x] for x in np.nanargmax(test_cv, 1)]
        fit['L2_test_cv'] = test_cv
        fit['L2_train_cv'] = train_cv
        fit['L2_at_grid_min'] = [x == 0 for x in np.nanargmax(test_cv, 1)]
        fit['L2_at_grid_max'] = [x == (len(fit['L2_grid']) - 1) for x in np.nanargmax(test_cv, 1)]

    else:
        if run_params['L2_grid_type'] == 'log':
            fit['L2_grid'] = np.geomspace(run_params['L2_grid_range'][0],
                                        run_params['L2_grid_range'][1],
                                        num=run_params['L2_grid_num'])
        else:
            fit['L2_grid'] = np.linspace(run_params['L2_grid_range'][0],
                                        run_params['L2_grid_range'][1],
                                        num=run_params['L2_grid_num'])

        if run_params['method'] != 'lasso_regression':
            fit['L2_grid'] = np.array([0] + list(fit['L2_grid']))

    return fit


def evaluate_models(fit, design_mat, run_params):
    X = design_mat.data
    spike_counts = fit['spike_count_arr']['spike_counts']
    # x_is_continuous = [run_params['kernels'][kernel_name.rsplit('_', 1)[0]]['type'] == 'continuous'
    #                    for kernel_name in design_mat.weights.values]

    # Initialize outputs
    num_units = spike_counts.shape[1]
    num_outer_folds = run_params['n_outer_folds']
    cv_var_train = np.full((num_units, num_outer_folds), np.nan)
    cv_var_test = np.full((num_units, num_outer_folds), np.nan)
    all_weights = np.full((X.shape[1], num_units), np.nan)
    all_prediction = np.full(spike_counts.shape, np.nan)

    if isinstance(run_params['cell_L2_regularization'], list):
        fit['cell_L2_regularization'] = run_params['cell_L2_regularization']

    if isinstance(run_params['cell_L2_regularization_nested'], list):
        fit['cell_L2_regularization_nested'] = run_params['cell_L2_regularization_nested']

    cell_L2_regularization_nested = np.full((num_units, num_outer_folds), np.nan)

    if run_params['use_fixed_penalty']:
        cv_var_train, cv_var_test, all_weights, all_prediction = simple_train_and_test(
            design_mat, spike_counts,
            lam=fit['L2_regularization'],
            folds_outer=num_outer_folds,
            method = run_params['method']
        )
    elif run_params['no_nested_CV']:
        if run_params['optimize_penalty_by_cell']:
            print(get_timestamp() + ': fitting each cell')
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
            print(get_timestamp() + ': fitting units by area')
            for area in tqdm(areas, total=len(areas), desc='progress'):
                unit_ids = np.where(fit['spike_count_arr']['structure'] == area)[0]
                fit_area = spike_counts[:, unit_ids]
                L2_value = np.nanmedian(np.take(fit['cell_L2_regularization'], unit_ids))
                cv_train, cv_test, weights, prediction = simple_train_and_test(design_mat, fit_area,
                                                                               lam=L2_value,
                                                                               folds_outer=run_params['n_outer_folds'],
                                                                               method = run_params['method'])
                cv_var_train[unit_ids] = cv_train
                cv_var_test[unit_ids] = cv_test
                all_weights[:, unit_ids] = weights
                all_prediction[:, unit_ids] = prediction

        elif run_params['optimize_penalty_by_firing_rate']:
            num_clusters =  np.min([run_params['num_rate_clusters'], num_units])
            rate_clusters = KMeans(n_clusters =  num_clusters, random_state = 0).fit(fit['spike_count_arr']['firing_rate'].reshape(-1,1)).labels_
            print(get_timestamp() + ': fitting units by firing rate')
            for cluster in tqdm(np.unique(rate_clusters), total=len(np.unique(rate_clusters)), desc='progress'):
                unit_ids = np.where(rate_clusters == cluster)[0]
                fit_rate = spike_counts[:, unit_ids]
                L2_value = np.nanmedian(np.take(fit['cell_L2_regularization'], unit_ids))
                cv_train, cv_test, weights, prediction = simple_train_and_test(design_mat, fit_rate,
                                                                               lam=L2_value,
                                                                               folds_outer=run_params['n_outer_folds'],
                                                                               method = run_params['method'])
                cv_var_train[unit_ids] = cv_train
                cv_var_test[unit_ids] = cv_test
                all_weights[:, unit_ids] = weights
                all_prediction[:, unit_ids] = prediction

        else:
            print(get_timestamp() + ': fitting all units')
            L2_value = np.nanmedian(np.array(fit['cell_L2_regularization']))
            cv_var_train, cv_var_test, all_weights, all_prediction = simple_train_and_test(design_mat,
                                                                                           spike_counts,
                                                                                           lam=L2_value,
                                                                                           folds_outer=run_params[
                                                                                               'n_outer_folds'],
                                                                                               method = run_params['method'])

    elif 'cell_L2_regularization_nested' in fit:
        if run_params['optimize_penalty_by_cell']:
            print(get_timestamp() + ': fitting each cell')
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
            for area in tqdm(areas, total=len(areas), desc='progress'):
                unit_ids = np.where(fit['spike_count_arr']['structure'] == area)[0]
                fit_area = spike_counts[:, unit_ids]
                L2_value = np.unique(np.take(fit['cell_L2_regularization_nested'], unit_ids), axis=0)
                cv_train, cv_test, weights, prediction = simple_train_and_test(design_mat, fit_area,
                                                                               lam=L2_value,
                                                                               folds_outer=run_params['n_outer_folds'],
                                                                               method = run_params['method'])
                cv_var_train[unit_ids] = cv_var_train
                cv_var_test[unit_ids] = cv_var_test
                all_weights[unit_ids] = weights
                all_prediction[unit_ids] = prediction

        elif run_params['optimize_penalty_by_firing_rate']:
            rate_clusters = fit['spike_count_arr']['rate_clusters']
            for cluster in tqdm(np.unique(rate_clusters), total=len(np.unique(rate_clusters)), desc='progress'):
                unit_ids = np.where(rate_clusters == cluster)[0]
                fit_rate = spike_counts[:, unit_ids]
                L2_value = np.unique(np.take(fit['cell_L2_regularization_nested'], unit_ids), axis=0)
                cv_train, cv_test, weights, prediction = simple_train_and_test(design_mat, fit_rate,
                                                                               lam=L2_value,
                                                                               folds_outer=run_params['n_outer_folds'],
                                                                               method = run_params['method'])
                cv_var_train[unit_ids] = cv_train
                cv_var_test[unit_ids] = cv_test
                all_weights[unit_ids] = weights
                all_prediction[unit_ids] = prediction

        else:
            L2_value = np.unique(np.array(fit['cell_L2_regularization_nested']), axis=0)
            cv_var_train, cv_var_test, all_weights, all_prediction = \
                simple_train_and_test(design_mat, fit['spike_count_arr']['spike_counts'],
                                      lam=L2_value,
                                      folds_outer=run_params['n_outer_folds'],
                                      method = run_params['method'])

    else:
        if run_params['optimize_penalty_by_cell']:
            print(get_timestamp() + ': fitting each cell')
            with ProcessPoolExecutor(max_workers=10) as executor:
                futures = {
                    executor.submit(process_unit, unit_no, design_mat.copy(), fit.copy(), run_params): unit_no
                    for unit_no in range(num_units)
                }
                for future in tqdm(as_completed(futures), total=num_units, desc='progress'):
                    unit_no, cv_train, cv_test, weights, prediction, lams = future.result()
                    cv_var_train[unit_no] = cv_train
                    cv_var_test[unit_no] = cv_test
                    all_weights[:, unit_no] = weights.reshape(-1)
                    all_prediction[:, unit_no] = prediction.reshape(-1)
                    cell_L2_regularization_nested[unit_no] = lams

        elif run_params['optimize_penalty_by_area']:
            areas = np.unique(fit['spike_count_arr']['structure'])
            for area in tqdm(areas, total=len(areas), desc='progress'):
                unit_ids = np.where(fit['spike_count_arr']['structure'] == area)[0]
                fit_area = spike_counts[:, unit_ids]
                cv_train, cv_test, weights, prediction, lams = \
                    nested_train_and_test(design_mat, fit_area, L2_grid=fit['L2_grid'],
                                          folds_outer=run_params['n_outer_folds'],
                                          folds_inner=run_params['n_inner_folds'], method = run_params['method'])

                cv_var_train[unit_ids] = cv_train
                cv_var_test[unit_ids] = cv_test
                all_weights[:, unit_ids] = weights
                all_prediction[:, unit_ids] = prediction
                cell_L2_regularization_nested[unit_ids] = lams

        elif run_params['optimize_penalty_by_firing_rate']:
            num_clusters =  np.min([run_params['num_rate_clusters'], num_units])
            rate_clusters = KMeans(n_clusters = run_params['num_rate_clusters'], random_state = 0).fit(fit['spike_count_arr']['firing_rate'].reshape(-1,1)).labels_
            fit['spike_count_arr']['rate_clusters'] = rate_clusters
            for cluster in tqdm(np.unique(rate_clusters), total=len(np.unique(rate_clusters)), desc='progress'):
                unit_ids = np.where(rate_clusters == cluster)[0]
                fit_rate = spike_counts[:, unit_ids]
                cv_train, cv_test, weights, prediction, lams = \
                    nested_train_and_test(design_mat, fit_rate, L2_grid=fit['L2_grid'],
                                          folds_outer=run_params['n_outer_folds'],
                                          folds_inner=run_params['n_inner_folds'],
                                          method = run_params['method'])

                cv_var_train[unit_ids] = cv_train
                cv_var_test[unit_ids] = cv_test
                all_weights[:, unit_ids] = weights
                all_prediction[:, unit_ids] = prediction
                cell_L2_regularization_nested[unit_ids] = lams

        else:
            cv_var_train, cv_var_test, all_weights, all_prediction, cell_L2_regularization_nested = \
                nested_train_and_test(design_mat, spike_counts, L2_grid=fit['L2_grid'],
                                      folds_outer=run_params['n_outer_folds'],
                                      folds_inner=run_params['n_inner_folds'],
                                      method = run_params['method'])
    model_label = run_params['model_label']
    fit[model_label] = {
        'weights': all_weights,
        'full_model_prediction': all_prediction,
        'cv_var_train': cv_var_train,
        'cv_var_test': cv_var_test
    }
    if not np.isnan(cell_L2_regularization_nested).all():
        fit['cell_L2_regularization_nested'] = cell_L2_regularization_nested

    return fit


def clean_r2_vals(x):
    x[np.isinf(x) | np.isnan(x)] = 0
    return x


def get_timestamp():
    t = time.localtime()
    return time.strftime('%Y-%m-%d: %H:%M:%S') + ' '
