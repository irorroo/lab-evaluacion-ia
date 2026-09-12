"""
supervised_model.py
----------------------
Módulo de clasificación supervisada.

Flujo:
  1. Entrena un clasificador (por defecto Regresión Logística).
  2. Predice sobre el set de prueba (ya escalado).
  3. Calcula y reporta: Accuracy, F1-score y Matriz de Confusión.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def entrenar_modelo(X_train_scaled, y_train, random_state: int = 42):
    """
    Entrena un clasificador de Regresión Logística.

    Se eligió Regresión Logística porque es rápida, interpretable y
    funciona bien como línea base (baseline) para datasets sintéticos
    como el de este proyecto. (Se podría cambiar por DecisionTreeClassifier
    o SVC sin tocar el resto del pipeline).
    """
    modelo = LogisticRegression(max_iter=1000, random_state=random_state)
    modelo.fit(X_train_scaled, y_train)
    print("[supervised_model] Modelo (Regresión Logística) entrenado.")
    return modelo


def evaluar_modelo(modelo, X_test_scaled, y_test, output_dir: str = "outputs"):
    """
    Evalúa el modelo entrenado sobre el set de prueba y reporta métricas.

    Retorna un diccionario con accuracy, f1_score y la matriz de confusión,
    y guarda una imagen de la matriz de confusión en outputs/.
    """
    y_pred = modelo.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    # 'weighted' para que funcione igual en problemas binarios y multiclase
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)

    print("\n===== MÉTRICAS DEL MODELO SUPERVISADO =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("Matriz de Confusión:")
    print(cm)
    print("\nReporte de clasificación completo:")
    print(classification_report(y_test, y_pred))

    # --- Guardamos la matriz de confusión como imagen para el README ---
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de Confusión - Modelo Supervisado")
    plt.xlabel("Predicción")
    plt.ylabel("Valor Real")
    plt.tight_layout()
    ruta_imagen = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(ruta_imagen)
    plt.close()
    print(f"[supervised_model] Matriz de confusión guardada en: {ruta_imagen}")

    return {"accuracy": acc, "f1_score": f1, "confusion_matrix": cm}
