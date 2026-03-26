from curses.ascii import EOT
import os
import numpy as np
import json
import logging
import joblib
import yaml
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import matplotlib
matplotlib.use('Agg')  # Sans interface graphique
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

import tensorflow as tf
import keras
layers = keras.layers

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FASHION_LABELS = {
    0: "T-shirt/top", 1: "Trouser",  2: "Pullover",
    3: "Dress",       4: "Coat",     5: "Sandal",
    6: "Shirt",       7: "Sneaker",  8: "Bag",
    9: "Ankle boot"
}

def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def load_fashion_data(data_path):
    logger.info(f"Chargement : {data_path}")
    df = pd.read_csv(data_path)
    if "label_name" in df.columns:
        df = df.drop(columns=["label_name"])
    X = df.drop(columns=["target"]).values.astype(np.float32)
    y = df["target"].values.astype(int)
    logger.info(f"   Shape X : {X.shape}  |  Classes : {np.unique(y)}")
    return X, y

# ── GRAPHES ────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_test, y_pred, model_name):
    """Matrice de confusion avec heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    labels = [FASHION_LABELS[i] for i in range(10)]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Prédit", fontsize=12)
    ax.set_ylabel("Réel", fontsize=12)
    ax.set_title(f"Matrice de Confusion — {model_name}", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    path = f"reports/confusion_matrix_{model_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"   Matrice de confusion sauvegardée : {path}")
    return path

def plot_roc_curves(y_test, y_proba, model_name):
    """Courbes ROC pour chaque classe (One-vs-Rest)."""
    y_test_bin = label_binarize(y_test, classes=list(range(10)))
    labels = [FASHION_LABELS[i] for i in range(10)]

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    auc_scores = []
    for i, (label, color) in enumerate(zip(labels, colors)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores.append(roc_auc)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{label} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aléatoire (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Taux de Faux Positifs", fontsize=12)
    ax.set_ylabel("Taux de Vrais Positifs", fontsize=12)
    ax.set_title(f"Courbes ROC — {model_name}\nAUC moyen = {np.mean(auc_scores):.3f}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = f"reports/roc_curves_{model_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"   Courbes ROC sauvegardées : {path}")
    return path

def plot_class_distribution(y_train, y_test):
    """Distribution des classes train vs test."""
    labels = [FASHION_LABELS[i] for i in range(10)]
    train_counts = [np.sum(y_train == i) for i in range(10)]
    test_counts  = [np.sum(y_test  == i) for i in range(10)]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, train_counts, width, label="Train", color="#1f77b4", alpha=0.8)
    bars2 = ax.bar(x + width/2, test_counts,  width, label="Test",  color="#ff7f0e", alpha=0.8)

    ax.set_xlabel("Classe", fontsize=12)
    ax.set_ylabel("Nombre d'images", fontsize=12)
    ax.set_title("Distribution des classes — Train vs Test", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = "reports/class_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"   Distribution des classes sauvegardée : {path}")
    return path

def plot_sample_images(X, y, n=5):
    """Affiche des exemples d'images par classe."""
    fig, axes = plt.subplots(10, n, figsize=(n * 2, 10 * 2))
    for cls in range(10):
        indices = np.where(y == cls)[0][:n]
        for j, idx in enumerate(indices):
            img = X[idx].reshape(28, 28)
            axes[cls, j].imshow(img, cmap="gray")
            axes[cls, j].axis("off")
            if j == 0:
                axes[cls, j].set_ylabel(FASHION_LABELS[cls], fontsize=10, rotation=0,
                        labelpad=60, va="center")
    plt.suptitle("Exemples d'images par classe", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = "reports/sample_images.png"
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info(f"   Exemples d'images sauvegardés : {path}")
    return path

def plot_cnn_history(history):
    """Courbes d'entraînement CNN : accuracy et loss par epoch."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history["accuracy"],     label="Train",      color="#1f77b4", lw=2)
    axes[0].plot(history.history["val_accuracy"], label="Validation", color="#ff7f0e", lw=2)
    axes[0].set_title("Accuracy par Epoch", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"],     label="Train",      color="#1f77b4", lw=2)
    axes[1].plot(history.history["val_loss"], label="Validation", color="#ff7f0e", lw=2)
    axes[1].set_title("Loss par Epoch", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("Courbes d'entraînement — CNN Keras", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = "reports/cnn_training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"   Courbes CNN sauvegardées : {path}")
    return path

def plot_models_comparison(results):
    """Graphe comparatif des 3 modèles."""
    names    = list(results.keys())
    accuracy = [results[n]["metrics"]["accuracy"]    for n in names]
    f1       = [results[n]["metrics"]["f1_weighted"] for n in names]

    x = np.arange(len(names))
    width = 0.35
    colors_acc = ["#2196F3", "#4CAF50", "#FF5722"]
    colors_f1  = ["#1565C0", "#2E7D32", "#BF360C"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, accuracy, width, label="Accuracy",    color=colors_acc, alpha=0.85)
    bars2 = ax.bar(x + width/2, f1,       width, label="F1 Weighted", color=colors_f1,  alpha=0.85)

    # Valeurs sur les barres
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylim(0.7, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Comparaison des 3 Modèles", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = "reports/models_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"   Comparaison des modèles sauvegardée : {path}")
    return path

# ── EXPÉRIENCES ────────────────────────────────────────────────────────────

def run_sklearn_experiment(model_name, pipeline, X_train, X_test, y_train, y_test, extra_params):
    with mlflow.start_run(run_name=model_name) as run:
        logger.info(f"\n{'='*50}\nExperience : {model_name}")
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("n_train", X_train.shape[0])
        for k, v in extra_params.items():
            mlflow.log_param(k, v)

        logger.info("Entrainement en cours...")
        pipeline.fit(X_train, y_train)
        y_pred  = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)

        metrics = {
            "accuracy":    round(accuracy_score(y_test, y_pred), 4),
            "f1_macro":    round(f1_score(y_test, y_pred, average="macro"), 4),
            "f1_weighted": round(f1_score(y_test, y_pred, average="weighted"), 4),
        }
        for name, val in metrics.items():
            mlflow.log_metric(name, val)

        logger.info(f"   accuracy    : {metrics['accuracy']}")
        logger.info(f"   f1_weighted : {metrics['f1_weighted']}")

        os.makedirs("reports", exist_ok=True)

        # Rapport texte
        report_path = f"reports/report_{model_name}.txt"
        target_names = [FASHION_LABELS[i] for i in range(10)]
        with open(report_path, "w") as f:
            f.write(classification_report(y_test, y_pred, target_names=target_names))
        mlflow.log_artifact(report_path)

        # Graphes
        cm_path  = plot_confusion_matrix(y_test, y_pred, model_name)
        roc_path = plot_roc_curves(y_test, y_proba, model_name)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(roc_path)

        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        return {"run_id": run.info.run_id, "metrics": metrics, "pipeline": pipeline, "type": "sklearn"}

def run_cnn_experiment(X_train, X_test, y_train, y_test):
    model_name = "CNN_Keras"
    with mlflow.start_run(run_name=model_name) as run:
        logger.info(f"\n{'='*50}\nExperience : {model_name}")
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("epochs", 8)
        mlflow.log_param("batch_size", 128)

        # Reshape (N, 784) → (N, 28, 28, 1)
        X_train_cnn = X_train.reshape(-1, 28, 28, 1)
        X_test_cnn  = X_test.reshape(-1, 28, 28, 1)
        y_train_cat = keras.utils.to_categorical(y_train, 10)
        y_test_cat  = keras.utils.to_categorical(y_test,  10)

        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(10, activation="softmax")
        ])

        model.compile(optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"])

        logger.info("Entrainement CNN (8 epochs)...")
        history = model.fit(
            X_train_cnn, y_train_cat,
            validation_split=0.1,
            epochs=8,
            batch_size=128,
            verbose=1
        )

        y_proba = model.predict(X_test_cnn, verbose=0)
        y_pred  = np.argmax(y_proba, axis=1)

        metrics = {
            "accuracy":    round(accuracy_score(y_test, y_pred), 4),
            "f1_macro":    round(f1_score(y_test, y_pred, average="macro"), 4),
            "f1_weighted": round(f1_score(y_test, y_pred, average="weighted"), 4),
        }
        for name, val in metrics.items():
            mlflow.log_metric(name, val)

        for epoch, (acc, val_acc, loss, val_loss) in enumerate(zip(
            history.history["accuracy"], history.history["val_accuracy"],
            history.history["loss"],     history.history["val_loss"]
        )):
            mlflow.log_metric("train_acc_epoch",  acc,     step=epoch)
            mlflow.log_metric("val_acc_epoch",    val_acc, step=epoch)
            mlflow.log_metric("train_loss_epoch", loss,    step=epoch)
            mlflow.log_metric("val_loss_epoch",   val_loss,step=epoch)

        logger.info(f"   accuracy    : {metrics['accuracy']}")
        logger.info(f"   f1_weighted : {metrics['f1_weighted']}")

        os.makedirs("reports", exist_ok=True)

        # Rapport texte
        report_path = "reports/report_CNN_Keras.txt"
        target_names = [FASHION_LABELS[i] for i in range(10)]
        with open(report_path, "w") as f:
            f.write(classification_report(y_test, y_pred, target_names=target_names))
        mlflow.log_artifact(report_path)

        # Graphes
        cm_path      = plot_confusion_matrix(y_test, y_pred, model_name)
        roc_path     = plot_roc_curves(y_test, y_proba, model_name)
        history_path = plot_cnn_history(history)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(roc_path)
        mlflow.log_artifact(history_path)

        os.makedirs("models", exist_ok=True)
        model.save("models/cnn_model.keras")
        mlflow.tensorflow.log_model(model, artifact_path="model")

        return {"run_id": run.info.run_id, "metrics": metrics, "model": model, "type": "keras"}

def train_all(config):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("fashion-mnist-comparison")

    X, y = load_fashion_data(config["data"]["raw_path"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"], stratify=y
    )

    os.makedirs("data/processed", exist_ok=True)
    ref_df = pd.DataFrame(X_train[:2000], columns=[f"pixel_{i}" for i in range(784)])
    ref_df["target"] = y_train[:2000]
    ref_df.to_csv("data/processed/reference.csv", index=False)

    # Graphes généraux
    os.makedirs("reports", exist_ok=True)
    plot_class_distribution(y_train, y_test)
    plot_sample_images(X_train, y_train)

    results = {}

    # ── 1. Random Forest ───────────────────────────────────────────
    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=200, max_depth=30, n_jobs=-1, random_state=42
        ))
    ])
    results["RandomForest"] = run_sklearn_experiment(
        "RandomForest", rf, X_train, X_test, y_train, y_test,
        {"n_estimators": 200, "max_depth": 30}
    )

    # ── 2. MLP Neural Network ──────────────────────────────────────
    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=(512, 256, 128), activation="relu",
            solver="adam", max_iter=30, learning_rate_init=0.001,
            early_stopping=True, random_state=42, verbose=False
        ))
    ])
    results["MLP_NeuralNet"] = run_sklearn_experiment(
        "MLP_NeuralNet", mlp, X_train, X_test, y_train, y_test,
        {"layers": "512-256-128", "activation": "relu", "max_iter": 30}
    )

    # ── 3. CNN Keras ───────────────────────────────────────────────
    results["CNN_Keras"] = run_cnn_experiment(X_train, X_test, y_train, y_test)

    # ── Comparaison finale ─────────────────────────────────────────
    plot_models_comparison(results)

    best_name = max(results, key=lambda k: results[k]["metrics"]["f1_weighted"])
    best = results[best_name]

    logger.info(f"\n{'='*50}\nCOMPARAISON DES 3 MODELES")
    for name, res in results.items():
        marker = " <- MEILLEUR" if name == best_name else ""
        logger.info(f"   {name:<20} accuracy={res['metrics']['accuracy']}  f1={res['metrics']['f1_weighted']}{marker}")

    os.makedirs("models", exist_ok=True)
    joblib.dump([f"pixel_{i}" for i in range(784)], "models/feature_names.pkl")
    joblib.dump(FASHION_LABELS, "models/label_names.pkl")

    if best["type"] == "sklearn":
        joblib.dump(best["pipeline"], config["api"]["model_path"])
    else:
        best["model"].save("models/best_model.keras")
        joblib.dump(results["MLP_NeuralNet"]["pipeline"], config["api"]["model_path"])
        logger.info("API utilise MLP comme fallback sklearn (models/model.pkl)")

    with open("reports/metrics.json", "w") as f:
        json.dump({"best_model": best_name, **best["metrics"]}, f, indent=2)

    logger.info(f"\n{'='*50}")
    logger.info(f"Graphes generes dans reports/ :")
    for f in os.listdir("reports"):
        if f.endswith(".png"):
            logger.info(f"   reports/{f}")

    return results

if __name__ == "__main__":
    config = load_config()
    train_all(config)