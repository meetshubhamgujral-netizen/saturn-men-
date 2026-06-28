"""Machine-learning engine: task detection, model training and evaluation.

Supports classification, regression and clustering. XGBoost is optional and is
only offered when the package is importable.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:  # optional dependency, degrades gracefully
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGBOOST = True
except Exception:  # pragma: no cover - depends on environment
    HAS_XGBOOST = False

from .analytics import PALETTE


def detect_task(df: pd.DataFrame, target: str | None) -> str:
    """Return 'classification', 'regression' or 'clustering'."""
    if target is None or target == "(none / clustering)":
        return "clustering"
    s = df[target].dropna()
    if not pd.api.types.is_numeric_dtype(s):
        return "classification"
    return "classification" if s.nunique() <= 15 else "regression"


def available_models(task: str) -> List[str]:
    if task == "classification":
        models = [
            "Logistic Regression", "KNN", "Decision Tree", "Random Forest",
            "Gradient Boosting", "Extra Trees", "Naive Bayes", "SVM",
        ]
        if HAS_XGBOOST:
            models.append("XGBoost")
        return models
    if task == "regression":
        models = [
            "Linear Regression", "Ridge", "Lasso", "Elastic Net",
            "Decision Tree Regressor", "Random Forest Regressor",
            "Gradient Boosting Regressor",
        ]
        if HAS_XGBOOST:
            models.append("XGBoost Regressor")
        return models
    return ["KMeans", "DBSCAN", "Hierarchical Clustering"]


def _build_model(name: str):
    factory = {
        "Logistic Regression": lambda: LogisticRegression(max_iter=1000),
        "KNN": lambda: KNeighborsClassifier(),
        "Decision Tree": lambda: DecisionTreeClassifier(random_state=42),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": lambda: GradientBoostingClassifier(random_state=42),
        "Extra Trees": lambda: ExtraTreesClassifier(n_estimators=200, random_state=42),
        "Naive Bayes": lambda: GaussianNB(),
        "SVM": lambda: SVC(probability=True, random_state=42),
        "Linear Regression": lambda: LinearRegression(),
        "Ridge": lambda: Ridge(),
        "Lasso": lambda: Lasso(),
        "Elastic Net": lambda: ElasticNet(),
        "Decision Tree Regressor": lambda: DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": lambda: RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting Regressor": lambda: GradientBoostingRegressor(random_state=42),
    }
    if name == "XGBoost" and HAS_XGBOOST:
        return XGBClassifier(eval_metric="logloss", random_state=42)
    if name == "XGBoost Regressor" and HAS_XGBOOST:
        return XGBRegressor(random_state=42)
    return factory[name]()


def _prepare_xy(df: pd.DataFrame, target: str, features: List[str]):
    data = df[features + [target]].dropna()
    X = pd.get_dummies(data[features], drop_first=True)
    y = data[target]
    return X, y


def train_supervised(
    df: pd.DataFrame,
    target: str,
    features: List[str],
    model_name: str,
    task: str,
    test_size: float = 0.25,
) -> Dict:
    """Train a single supervised model and compute evaluation artefacts."""
    X, y = _prepare_xy(df, target, features)
    if len(X) < 20:
        return {"error": "Not enough complete rows to train (need >= 20)."}

    label_encoder = None
    if task == "classification" and not pd.api.types.is_numeric_dtype(y):
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y.astype(str)), index=y.index)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42,
        stratify=y if task == "classification" and y.nunique() > 1 else None,
    )
    model = _build_model(model_name)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    result: Dict = {
        "model": model,
        "model_name": model_name,
        "task": task,
        "feature_names": list(X.columns),
        "X_test": X_test,
        "y_test": y_test,
        "predictions": preds,
        "label_encoder": label_encoder,
    }

    if task == "classification":
        avg = "binary" if y.nunique() == 2 else "weighted"
        result["metrics"] = {
            "Accuracy": round(accuracy_score(y_test, preds), 4),
            "Precision": round(precision_score(y_test, preds, average=avg, zero_division=0), 4),
            "Recall": round(recall_score(y_test, preds, average=avg, zero_division=0), 4),
            "F1 Score": round(f1_score(y_test, preds, average=avg, zero_division=0), 4),
        }
        result["confusion"] = confusion_matrix(y_test, preds)
        # ROC/AUC for binary problems with probability support.
        if y.nunique() == 2 and hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, proba)
            result["roc"] = {"fpr": fpr, "tpr": tpr}
            result["metrics"]["AUC"] = round(roc_auc_score(y_test, proba), 4)
    else:
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        result["metrics"] = {
            "MAE": round(mean_absolute_error(y_test, preds), 4),
            "MSE": round(mean_squared_error(y_test, preds), 4),
            "RMSE": round(rmse, 4),
            "R2 Score": round(r2_score(y_test, preds), 4),
        }
        result["residuals"] = (np.asarray(y_test) - np.asarray(preds))
    return result


def compare_models(
    df: pd.DataFrame, target: str, features: List[str], model_names: List[str], task: str
) -> pd.DataFrame:
    rows = []
    for name in model_names:
        res = train_supervised(df, target, features, name, task)
        if "metrics" in res:
            rows.append({"model": name, **res["metrics"]})
    return pd.DataFrame(rows)


def run_clustering(
    df: pd.DataFrame, features: List[str], algorithm: str, n_clusters: int = 3
) -> Dict:
    data = df[features].select_dtypes(include=np.number).dropna()
    if data.shape[0] < 5 or data.shape[1] < 1:
        return {"error": "Need at least 5 rows and 1 numeric feature for clustering."}
    X = StandardScaler().fit_transform(data)

    if algorithm == "KMeans":
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    elif algorithm == "DBSCAN":
        model = DBSCAN()
    else:
        model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X)

    result: Dict = {"labels": labels, "n_found": len(set(labels)) - (1 if -1 in labels else 0)}
    if len(set(labels)) > 1 and -1 not in set(labels[:1]) or len(set(labels)) > 1:
        try:
            result["silhouette"] = round(silhouette_score(X, labels), 4)
            result["davies_bouldin"] = round(davies_bouldin_score(X, labels), 4)
        except Exception:
            pass
    # 2-D PCA projection for visualisation.
    from sklearn.decomposition import PCA

    coords = PCA(n_components=2).fit_transform(X) if X.shape[1] >= 2 else np.c_[X, np.zeros(len(X))]
    proj = pd.DataFrame(coords[:, :2], columns=["PC1", "PC2"])
    proj["cluster"] = labels.astype(str)
    fig = px.scatter(
        proj, x="PC1", y="PC2", color="cluster",
        color_discrete_sequence=PALETTE, title=f"{algorithm} clusters (PCA view)",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    result["figure"] = fig
    return result


# ----- Evaluation figures -------------------------------------------------

def confusion_figure(cm: np.ndarray) -> go.Figure:
    fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Purples",
        labels=dict(x="Predicted", y="Actual"), title="Confusion Matrix",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def roc_figure(roc: Dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines",
                             name="ROC", line=dict(color=PALETTE[0], width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             name="Chance", line=dict(dash="dash", color="gray")))
    fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate",
                      yaxis_title="True Positive Rate", margin=dict(l=10, r=10, t=50, b=10))
    return fig


def residual_figure(residuals: np.ndarray, preds: np.ndarray) -> go.Figure:
    fig = px.scatter(
        x=preds, y=residuals, opacity=0.6,
        labels={"x": "Predicted", "y": "Residual"},
        color_discrete_sequence=[PALETTE[3]], title="Residual Plot",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig
