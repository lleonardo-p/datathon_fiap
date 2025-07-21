import pandas as pd
import pickle
import mlflow
import mlflow.sklearn
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE
from typing import Tuple


def evaluate_model_cv(model, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> dict:
    """
    Realiza validação cruzada e retorna as métricas médias.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    return {
        "accuracy": cross_val_score(model, X, y, cv=skf, scoring="accuracy").mean(),
        "precision": cross_val_score(model, X, y, cv=skf, scoring="precision").mean(),
        "recall": cross_val_score(model, X, y, cv=skf, scoring="recall").mean(),
        "f1": cross_val_score(model, X, y, cv=skf, scoring="f1").mean(),
        "roc_auc": cross_val_score(model, X, y, cv=skf, scoring="roc_auc").mean(),
    }


def train_lgbm_with_oversampling(
    df: pd.DataFrame,
    model_output_path: str = "./models/lgbm_oversample_model.pkl",
    experiment_name: str = "lgbm_candidate_prediction_oversampled"
) -> Tuple[LGBMClassifier, dict]:
    """
    Treina modelo LightGBM com oversampling e validação cruzada.
    """
    # Preparar dados
    df = df.dropna(subset=["target"])
    X = df.drop(columns=["ID", "prospect_codigo", "target"])
    y = df["target"].astype(int)

    # Aplicar SMOTE (oversampling apenas no treino)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print(f" Oversampling aplicado: {y.value_counts().to_dict()} ➡ {pd.Series(y_resampled).value_counts().to_dict()}")

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        model = LGBMClassifier(
            n_estimators=300,
            random_state=42
        )

        metrics = evaluate_model_cv(model, X_resampled, y_resampled)

        # Treina modelo final
        model.fit(X_resampled, y_resampled)

        # Loga métricas no MLflow
        for name, val in metrics.items():
            mlflow.log_metric(name, val)
        mlflow.sklearn.log_model(model, artifact_path="lgbm_model_oversampled")

        # Salvar local para API
        with open(model_output_path, "wb") as f:
            pickle.dump(model, f)

        print("Modelo treinado com SMOTE e salvo com sucesso!")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    return model, metrics


if __name__ == "__main__":
    df = pd.read_parquet("./data/processed/master_table.parquet")
    train_lgbm_with_oversampling(df)
