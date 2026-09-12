"""
main.py
---------
Orquestador principal del pipeline.

Ejecuta, en orden, TODO lo que pide el enunciado:
  1. Generación de datos               -> src/data_loader.py
  2. Preprocesamiento (split + escalado) -> src/preprocessing.py
  3. Pipeline supervisado (clasificación) -> src/supervised_model.py
  4. Pipeline no supervisado (PCA + KMeans) -> src/unsupervised_model.py

Uso:
    python main.py
"""

from src.data_loader import generar_dataset, guardar_dataset, cargar_dataset
from src.preprocessing import dividir_datos, escalar_datos
from src.supervised_model import entrenar_modelo, evaluar_modelo
from src.unsupervised_model import (
    aplicar_pca,
    aplicar_kmeans,
    calcular_silhouette,
    graficar_clusters,
)


def main():
    print("=" * 60)
    print("PASO 1: Generación de Datos")
    print("=" * 60)
    df = generar_dataset()
    guardar_dataset(df)
    df = cargar_dataset()  # se recarga desde el CSV, tal como pide el enunciado

    print("\n" + "=" * 60)
    print("PASO 2: Preprocesamiento")
    print("=" * 60)
    X_train, X_test, y_train, y_test = dividir_datos(df)
    X_train_scaled, X_test_scaled, scaler = escalar_datos(X_train, X_test)

    print("\n" + "=" * 60)
    print("PASO 3: Pipeline Supervisado")
    print("=" * 60)
    modelo = entrenar_modelo(X_train_scaled, y_train)
    metricas_supervisadas = evaluar_modelo(modelo, X_test_scaled, y_test)

    print("\n" + "=" * 60)
    print("PASO 4: Pipeline No Supervisado")
    print("=" * 60)
    # Usamos todo el dataset escalado (train + test) para el clustering,
    # ya que es un análisis exploratorio no supervisado, no una predicción.
    import numpy as np
    X_completo_scaled = np.vstack([X_train_scaled, X_test_scaled])

    X_pca, pca, varianza_acumulada = aplicar_pca(X_completo_scaled)
    labels, kmeans = aplicar_kmeans(X_pca, k=3)
    silhouette = calcular_silhouette(X_pca, labels)
    graficar_clusters(X_pca, labels)

    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)
    print(f"Accuracy (supervisado)          : {metricas_supervisadas['accuracy']:.4f}")
    print(f"F1-score (supervisado)           : {metricas_supervisadas['f1_score']:.4f}")
    print(f"Silhouette Score (no supervisado): {silhouette:.4f}")
    print(f"Varianza explicada acumulada PCA : {varianza_acumulada:.4f}")


if __name__ == "__main__":
    main()
