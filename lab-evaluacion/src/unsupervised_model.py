"""
unsupervised_model.py
------------------------
Módulo de agrupamiento no supervisado.

Flujo:
  1. PCA: reduce las características originales a 2 componentes
     principales (para poder visualizar en 2D).
  2. K-Means (k=3): agrupa los puntos en el espacio de 2 componentes.
  3. Métricas: Silhouette Score y varianza explicada acumulada por PCA.
  4. Visualización: scatter plot 2D coloreado por cluster.
"""

import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def aplicar_pca(X_scaled, n_components: int = 2, random_state: int = 42):
    """
    Reduce la dimensionalidad de X_scaled a n_components (2, por defecto)
    usando PCA. PCA busca las direcciones (componentes) que capturan la
    mayor varianza posible de los datos originales.
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    varianza_explicada = pca.explained_variance_ratio_
    varianza_acumulada = varianza_explicada.sum()

    print(f"[unsupervised_model] Varianza explicada por componente: "
          f"{varianza_explicada}")
    print(f"[unsupervised_model] Varianza explicada ACUMULADA (2 comp.): "
          f"{varianza_acumulada:.4f}")

    return X_pca, pca, varianza_acumulada


def aplicar_kmeans(X_pca, k: int = 3, random_state: int = 42):
    """
    Aplica K-Means con k clusters sobre los datos ya reducidos a 2D con PCA.
    Retorna las etiquetas de cluster asignadas a cada punto y el modelo.
    """
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    print(f"[unsupervised_model] K-Means ejecutado con k={k}.")
    return labels, kmeans


def calcular_silhouette(X_pca, labels) -> float:
    """
    Calcula el Silhouette Score: mide qué tan bien separados están los
    clusters (rango de -1 a 1; mientras más cerca de 1, mejor definidos
    están los grupos).
    """
    score = silhouette_score(X_pca, labels)
    print(f"[unsupervised_model] Silhouette Score: {score:.4f}")
    return score


def graficar_clusters(X_pca, labels, output_dir: str = "outputs"):
    """
    Genera y guarda un scatter plot 2D de los clusters encontrados
    por K-Means sobre las 2 componentes principales de PCA.
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", s=40, edgecolor="k"
    )
    plt.title("Clusters (K-Means) sobre componentes de PCA")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()

    ruta_imagen = os.path.join(output_dir, "clusters_pca.png")
    plt.savefig(ruta_imagen)
    plt.close()
    print(f"[unsupervised_model] Gráfico de clusters guardado en: {ruta_imagen}")
