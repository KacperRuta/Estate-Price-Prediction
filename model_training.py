from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass(frozen=True)
class ModelEvaluationResult:
    """Store cross-validation metrics for a single model."""

    model_name: str
    mean_rmse: float
    std_rmse: float


@dataclass(frozen=True)
class FeatureImportanceResult:
    """Store feature importance details for a trained model."""

    feature_name: str
    importance: float


@dataclass(frozen=True)
class PricePredictionResult:
    """Store a predicted price for a single model."""

    model_name: str
    predicted_price: float


class HousingModelTrainer:
    """Train and compare regression models for housing price prediction."""

    def __init__(
        self,
        target_column: str = "price",
        n_splits: int = 5,
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        self._target_column = target_column
        self._cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self._random_state = random_state
        self._n_jobs = n_jobs

    def evaluate_models(self, dataframe: pd.DataFrame) -> list[ModelEvaluationResult]:
        """Evaluate multiple regressors with cross-validation using RMSE."""
        features, target = self._split_features_and_target(dataframe)
        preprocessing = self._build_preprocessing_pipeline(features)
        models = self._build_models()
        results: list[ModelEvaluationResult] = []

        for model_name, estimator in models.items():
            pipeline = self._build_model_pipeline(preprocessing, estimator)
            rmse_scores = -cross_val_score(
                pipeline,
                features,
                target,
                cv=self._cv,
                scoring="neg_root_mean_squared_error",
                n_jobs=self._n_jobs,
            )
            results.append(
                ModelEvaluationResult(
                    model_name=model_name,
                    mean_rmse=rmse_scores.mean(),
                    std_rmse=rmse_scores.std(),
                )
            )

        return sorted(results, key=lambda result: result.mean_rmse)

    def get_feature_importance(
        self,
        dataframe: pd.DataFrame,
        model_name: str = "RandomForest",
        top_n: int = 10,
    ) -> list[FeatureImportanceResult]:
        """Fit a tree-based model and return the most important features."""
        features, target = self._split_features_and_target(dataframe)
        preprocessing = self._build_preprocessing_pipeline(features)
        models = self._build_models()

        if model_name not in models:
            raise KeyError(f"Model '{model_name}' is not available.")

        estimator = models[model_name]
        pipeline = self._build_model_pipeline(preprocessing, estimator)
        pipeline.fit(features, target)

        trained_model = pipeline.named_steps["model"]
        if not hasattr(trained_model, "feature_importances_"):
            raise ValueError(f"Model '{model_name}' does not provide feature_importances_.")

        transformed_feature_names = pipeline.named_steps["preprocessing"].get_feature_names_out()
        feature_importances = trained_model.feature_importances_
        importance_frame = pd.DataFrame(
            {
                "feature_name": transformed_feature_names,
                "importance": feature_importances,
            }
        ).sort_values("importance", ascending=False)

        top_features = importance_frame.head(top_n)
        return [
            FeatureImportanceResult(
                feature_name=row.feature_name,
                importance=row.importance,
            )
            for row in top_features.itertuples(index=False)
        ]

    def train_models(self, dataframe: pd.DataFrame) -> dict[str, Pipeline]:
        """Fit all configured models on the full dataset."""
        features, target = self._split_features_and_target(dataframe)
        models = self._build_models()
        trained_models: dict[str, Pipeline] = {}

        for model_name, estimator in models.items():
            preprocessing = self._build_preprocessing_pipeline(features)
            pipeline = self._build_model_pipeline(preprocessing, estimator)
            pipeline.fit(features, target)
            trained_models[model_name] = pipeline

        return trained_models

    def predict_prices(
        self,
        trained_models: Mapping[str, Pipeline],
        input_data: Mapping[str, Any],
    ) -> list[PricePredictionResult]:
        """Predict prices using each trained model."""
        input_frame = pd.DataFrame([input_data])
        return [
            PricePredictionResult(
                model_name=model_name,
                predicted_price=float(model.predict(input_frame)[0]),
            )
            for model_name, model in trained_models.items()
        ]

    def _split_features_and_target(self, dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        if self._target_column not in dataframe.columns:
            raise KeyError(f"Target column '{self._target_column}' was not found in the dataframe.")

        features = dataframe.drop(columns=self._target_column)
        target = dataframe[self._target_column]
        return features, target

    def _build_preprocessing_pipeline(self, features: pd.DataFrame) -> ColumnTransformer:
        numeric_columns = features.select_dtypes(include="number").columns.tolist()
        categorical_columns = features.select_dtypes(exclude="number").columns.tolist()

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, numeric_columns),
                ("categorical", categorical_pipeline, categorical_columns),
            ]
        )

    def _build_model_pipeline(self, preprocessing: ColumnTransformer, estimator: object) -> Pipeline:
        return Pipeline(
            steps=[
                ("preprocessing", preprocessing),
                ("model", estimator),
            ]
        )

    def _build_models(self) -> dict[str, object]:
        return {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=self._random_state,
                n_jobs=self._n_jobs,
            ),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                random_state=self._random_state,
            ),
        }
