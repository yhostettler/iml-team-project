# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from sklearn.ensemble import GradientBoostingRegressor



RANDOM_STATE = 42


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.price_cols_ = [c for c in X.columns if c.startswith("price_")]
        return self

    def transform(self, X):
        X = X.copy()
        price_df = X[self.price_cols_]

        # X["price_mean"] = price_df.mean(axis=1, skipna=True)
        # X["price_median"] = price_df.median(axis=1, skipna=True)
        # X["price_std"] = price_df.std(axis=1, skipna=True)
        # X["price_min"] = price_df.min(axis=1, skipna=True)
        # X["price_max"] = price_df.max(axis=1, skipna=True)
        # X["price_range"] = X["price_max"] - X["price_min"]
        # X["price_missing_count"] = price_df.isna().sum(axis=1)
        # X["price_present_count"] = price_df.notna().sum(axis=1)

        return X


def _get_feature_types(X):
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    return numeric_cols, categorical_cols


def _build_preprocessor(X):
    numeric_cols, categorical_cols = _get_feature_types(X)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            # ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(categories=[["spring","summer","autumn","winter"]]))        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor


def _build_models():
    models = {}

    extra_trees_configs = [
        # (2000, None, 1),
        # (1000, 64, 1),
        (300, 16, 1),
        (100, 32, 1),
        # (300, 16, 1),

    ]
    for n_estimators, max_depth, min_leaf in extra_trees_configs:
        name = f"et_n{n_estimators}_d{max_depth}_l{min_leaf}"
        models[name] = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        
    # Initialize model
    models['gbr'] = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.01,
        max_depth=64,
        min_samples_split=3,
        min_samples_leaf=1,
        subsample=0.8,
        max_features='sqrt',
        random_state=RANDOM_STATE
        )
    return models


def _build_pipelines(X_df):
    models = _build_models()
    engineered_X = FeatureEngineer().fit_transform(X_df)
    preprocessor = _build_preprocessor(engineered_X)

    pipelines = {}
    for name, model in models.items():
        pipelines[name] = Pipeline(
            [
                ("features", FeatureEngineer()),
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

    return pipelines


def _fit_stacked_ensemble(X_df, y):
    pipelines = _build_pipelines(X_df)
    model_names = list(pipelines.keys())
    cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    oof_matrix = np.zeros((len(y), len(model_names)))

    print("\nTraining with 5-fold CV...", flush=True)
    for col_idx, name in enumerate(model_names):
        for train_idx, val_idx in cv.split(X_df):
            X_tr = X_df.iloc[train_idx]
            X_val = X_df.iloc[val_idx]
            y_tr = y[train_idx]

            model = clone(pipelines[name])
            model.fit(X_tr, y_tr)
            oof_matrix[val_idx, col_idx] = model.predict(X_val)

    per_model_r2 = {}
    for col_idx, name in enumerate(model_names):
        per_model_r2[name] = r2_score(y, oof_matrix[:, col_idx])

    print("\nBase model OOF R2:", flush=True)
    for name, score in sorted(per_model_r2.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {score:.4f}", flush=True)

    blender = Ridge(alpha=50, fit_intercept=False, positive=True)
    blender.fit(oof_matrix, y)

    raw_weights = blender.coef_.copy()
    if raw_weights.sum() <= 0:
        raw_weights = np.ones(len(model_names))
    normalized_weights = raw_weights / raw_weights.sum()
    blender.coef_ = normalized_weights

    oof_blend_pred = blender.predict(oof_matrix)
    blend_r2 = r2_score(y, oof_blend_pred)

    print(f"\nBlended OOF R2: {blend_r2:.4f}", flush=True)
    print("Learned blend weights:", flush=True)
    for name, weight in zip(model_names, normalized_weights):
        print(f"  {name}: {weight:.4f}", flush=True)

    with open("cv_metrics.txt", "w", encoding="utf-8") as f:
        f.write("Base model OOF R2:\n")
        for name, score in sorted(per_model_r2.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{name}: {score:.6f}\n")
        f.write(f"\nBlended OOF R2: {blend_r2:.6f}\n")
        f.write("\nBlend weights:\n")
        for name, weight in zip(model_names, normalized_weights):
            f.write(f"{name}: {weight:.6f}\n")

    fitted_base_models = {}
    for name in model_names:
        base_model = clone(pipelines[name])
        base_model.fit(X_df, y)
        fitted_base_models[name] = base_model

    return fitted_base_models, blender, model_names, blend_r2


def load_data():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    train_df = pd.read_csv("train.csv")

    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print("\n")

    test_df = pd.read_csv("test.csv")

    print("Test data:")
    print(test_df.shape)
    print(test_df.head(2))

    train_df = train_df.dropna(subset=["price_CHF"]).reset_index(drop=True)

    X_train = train_df.drop(["price_CHF"], axis=1)
    y_train = train_df["price_CHF"].to_numpy(dtype=float)
    X_test = test_df.copy()

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test


class Model(object):
    def __init__(self):
        super().__init__()
        self._x_train = None
        self._y_train = None
        self._base_models = None
        self._blender = None
        self._model_names = None
        self._oof_r2 = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self._x_train = X_train
        self._y_train = y_train
        self._base_models, self._blender, self._model_names, self._oof_r2 = _fit_stacked_ensemble(X_train, y_train)
        print(f"Final OOF R2 (tracked): {self._oof_r2:.4f}", flush=True)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        test_matrix = np.column_stack([self._base_models[name].predict(X_test) for name in self._model_names])
        y_pred = self._blender.predict(test_matrix)
        assert y_pred.shape == (X_test.shape[0],), "Invalid data shape"
        return y_pred


# Main function. You don't have to change this
if __name__ == "__main__":
    X_train, y_train, X_test = load_data()
    model = Model()
    model.fit(X_train=X_train, y_train=y_train)
    print(f"R2 ready for this run: {model._oof_r2:.4f}", flush=True)

    y_pred = model.predict(X_test)

    dt = pd.DataFrame(y_pred)
    dt.columns = ["price_CHF"]
    dt.to_csv("results.csv", index=False)
    print("\nResults file successfully generated!")
