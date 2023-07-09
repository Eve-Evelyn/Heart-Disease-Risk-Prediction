import numpy as np
import pandas as pd
import statsmodels.api as sm

data = pd.read_csv("framingham.csv")
data = data.dropna()
# print(data.isna().sum())
X = data[['age', 'totChol', 'sysBP', 'diaBP', 'male', 'diabetes']].to_numpy()
y = data['TenYearCHD'].to_numpy()


def kfold_split(X, k=5, random_state=42):
    """
    Function to split sample with validation set approach.

    Parameters
    ----------
    X : {array-like} of shape (n_sample, n_predictors)
        All predictors set.

    k : int, default = 5
        Number of folds.

    random_state : int
        Pass an int for reproducible output across multiple function calls.

    Returns
    -------
    train_ind_list : list
        Contains data index of train set.

    valid_ind_list : list
        Contains data index of validation set.
    """
    # Extract sample size
    n_samples, _ = data.shape

    # Set random state
    np.random.seed(random_state)

    # # Randomize index
    random_ind = np.random.choice(n_samples,
                                  size=n_samples,
                                  replace=False)

    # Calculate size of each fold
    fold_sizes = np.ones(k, dtype=int) * (n_samples // k)
    fold_sizes[:n_samples % k] += 1

    # Define initial list for each train and valid index
    train_ind_list = []
    valid_ind_list = []

    # Split sample
    current_ind = 0
    for size in fold_sizes:
        # Define index
        start_ind = current_ind
        end_ind = current_ind + size

        # Slice valid set
        # One fold for valid set, the remaining for train set
        valid_ind = random_ind[start_ind:end_ind]
        train_ind = np.concatenate((random_ind[:start_ind],
                                    random_ind[end_ind:]))

        # Update current index
        current_ind = end_ind

        # Append train and valid index in list
        train_ind_list.append(train_ind)
        valid_ind_list.append(valid_ind)

    return train_ind_list, valid_ind_list


def AIC(y_true, y_pred, p):
    """
    Function to split sample with validation set approach.

    Parameters
    ----------
    y_true : {array-like} of shape (n_sample, )
        Actual value of response variable.

    y_pred : {array-like} of shape (n_sample, 1)
        The success probability of X.

    p : int
        Number of parameters in model.

    Returns
    -------
    aic : float
        AIC value.
    """
    # Find the average log likelihood value
    llf = np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # AIC value is sensitive to number of parameters
    # The average log likelihood represent value for 1 unit observation
    # AIC from average llf is not comparable
    # Multiply llf by n_sample=173 to make its AIC comparable
    llf *= 173

    # Calculate AIC
    aic = -2 * (llf - p)

    return aic


def cross_validate(X, y, cv, random_state=42):
    """
    Function to evaluate AIC by cross-validation method.

    Parameters
    ----------
    X : {array-like} of shape (n_sample, n_predictors)
        The independent variable or predictors.

    y : {array-like} of shape (n_sample, )
        The dependent or response variable.

    method : cross-validation splitter
        Cross-validation method.

    cv : int
        Number of folds for k-Fold CV.

    random_state : int, default=42
        Pass an int for reproducible output across multiple function calls.

    Returns
    -------
    score : float
        The average AIC score.
    """
    # Split train and valid set based on kfold split
    train_ind_list, valid_ind_list = kfold_split(X=X, k=cv, random_state=random_state)

    # Define the number of train sets
    n_split = len(train_ind_list)

    # Initialize AIC score list for each valid set
    score_list = []

    for i in range(n_split):
        # Extract data from index
        X_train = X[train_ind_list[i]]
        y_train = y[train_ind_list[i]]
        X_valid = X[valid_ind_list[i]]
        y_valid = y[valid_ind_list[i]]

        # Add constant
        X_train = sm.add_constant(X_train, has_constant="add")
        X_valid = sm.add_constant(X_valid, has_constant="add")

        # Fitting model
        model = sm.Logit(y_train, X_train)
        results = model.fit(disp=False)

        # Calculate success probability
        y_pred_train = results.predict(X_train)
        y_pred_valid = results.predict(X_valid)

        # Calculate AIC
        aic_train = AIC(y_true=y_train,
                        y_pred=y_pred_train,
                        p=X_train.shape[1])
        aic_valid = AIC(y_true=y_valid,
                        y_pred=y_pred_valid,
                        p=X_train.shape[1])

        # Append AIC score in list
        score_list.append(aic_valid)

    # Calculate CV Score
    score = np.mean(score_list)

    return score


def forward(X, y, predictors, cv=5, random_state=42):
    """
    Function to perform best subset selection procedure.

    Parameters
    ----------
    X : {array-like} of shape (n_sample, n_predictors)
        All predictors set.

    y : {array-like} of shape (n_sample, )
        The dependent or response variable.

    predictors : {array-like} of shape (n_sample, )
        Index of predictors

    method : cross-validation splitter
        Cross-validation method.

    cv : int, default=5
        Number of folds for k-Fold CV.

    random_state : int, default=42
        Pass an int for reproducible output across multiple function calls.

    Returns
    -------
    models : {array-like} of shape (n_combinations, k)
        Summary of predictors and its AIC score for each possible combination.

    best_model : {array-like} of shape (2, )
        Best model of models with the smallest AIC score.
    """

    # Initialize list of results
    results = []

    # Define sample size and  number of all predictors
    n_samples, n_predictors = X.shape

    # Define list of all predictors
    col_list = np.arange(n_predictors)

    # Define remaining predictors for each k
    remaining_predictors = [p for p in col_list if p not in predictors]

    # Initialize list of predictors and its CV Score
    pred_list = []
    score_list = []

    # Cross validate each possible combination of remaining predictors
    for p in remaining_predictors:
        combi = predictors + [p]

        # Extract predictors combination
        X_ = X[:, combi]
        y_ = y

        # Cross validate to get CV Score
        score_ = cross_validate(X=X_,
                                y=y_,
                                cv=cv,
                                random_state=random_state)

        # Append predictors combination and its CV Score to the list
        pred_list.append(list(combi))
        score_list.append(score_)

    # Tabulate the results
    models = pd.DataFrame({"Predictors": pred_list,
                           "AIC": score_list})

    # Choose the best model
    best_model = models.loc[models["AIC"].argmin()]

    return models, best_model


# Fit null model
predictor = []
score_ = cross_validate(X=X[:, predictor],
                        y=y,
                        cv=10,
                        random_state=42)

# Create table for the best model of each k predictors
# Append the results of null model
forward_models = pd.DataFrame({"Predictors": [predictor],
                               "AIC": [score_]})

# Define list of predictors
predictors = []
n_predictors = X.shape[1]

# Perform forward selection procedure for k=1,...,6 predictors
for k in range(n_predictors):
    _, best_model = forward(X=X,
                            y=y,
                            predictors=predictors,
                            cv=10,
                            random_state=42)

    # Tabulate the best model of each k predictors
    forward_models.loc[k + 1] = best_model
    predictors = best_model["Predictors"]

# print(forward_models)

# Define X with best predictors
X_best = X[:, [0, 2, 4]]

# Fit best model
X_best = sm.add_constant(X_best)
best_model = sm.Logit(y, X_best)
best_model_result = best_model.fit()
# print(best_model_result.summary())
