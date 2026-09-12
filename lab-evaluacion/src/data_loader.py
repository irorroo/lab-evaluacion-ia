"""
data_loader.py
----------------
Módulo encargado de:
  1. Generar un dataset sintético de clasificación con scikit-learn.
  2. Guardarlo en disco como CSV (data/dataset.csv).
  3. Cargarlo nuevamente desde disco (para que el resto del pipeline
     siempre lea desde el archivo, no desde memoria).

Se usa make_classification() porque nos permite controlar:
  - el número de características continuas (>= 4, pide el enunciado)
  - el número de clases (binaria o multiclase)
  - el nivel de "dificultad" del problema (n_informative, n_redundant, etc.)
"""

import os
import pandas as pd
from sklearn.datasets import make_classification

# Ruta del CSV, relativa a la raíz del proyecto (lab-evaluacion/)
DATA_PATH = os.path.join("data", "dataset.csv")


def generar_dataset(
    n_samples: int = 500,
    n_features: int = 6,
    n_informative: int = 4,
    n_redundant: int = 0,
    n_classes: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Genera un dataset sintético de clasificación.

    Parámetros
    ----------
    n_samples : int
        Número de filas (observaciones) a generar.
    n_features : int
        Número total de columnas/características continuas.
    n_informative : int
        Cuántas de esas características son realmente útiles para
        separar las clases (el resto añade ruido/redundancia).
    n_classes : int
        Número de clases del target (2 = binaria, >2 = multiclase).
    random_state : int
        Semilla para que el resultado sea reproducible.

    Retorna
    -------
    pd.DataFrame
        DataFrame con columnas feature_1..feature_n y una columna 'target'.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        n_clusters_per_class=1,  # evita error de sklearn con varias clases
        random_state=random_state,
    )

    # Nombramos las columnas para que el CSV sea legible por cualquier persona
    columnas = [f"feature_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=columnas)
    df["target"] = y

    return df


def guardar_dataset(df: pd.DataFrame, path: str = DATA_PATH) -> None:
    """Guarda el DataFrame como CSV, creando la carpeta si no existe."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[data_loader] Dataset guardado en: {path}")


def cargar_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """Carga el dataset desde el CSV en disco."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No se encontró {path}. Ejecuta primero generar_dataset() + guardar_dataset()."
        )
    df = pd.read_csv(path)
    print(f"[data_loader] Dataset cargado desde: {path} -> shape={df.shape}")
    return df


# Permite correr este archivo solo, para probar el módulo de forma aislada
if __name__ == "__main__":
    df = generar_dataset()
    guardar_dataset(df)
    df_cargado = cargar_dataset()
    print(df_cargado.head())
