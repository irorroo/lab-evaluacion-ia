"""
preprocessing.py
------------------
Módulo encargado del preprocesamiento de los datos.

Punto CRÍTICO que pide el enunciado y que suelen revisar en la
evaluación: el StandardScaler se ajusta (fit) ÚNICAMENTE con los datos
de entrenamiento, y luego se usa (transform) tanto en train como en
test. Esto evita "data leakage" (que información del set de prueba
se filtre al momento de entrenar).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def dividir_datos(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Separa el DataFrame en X (features) e y (target), y luego en
    conjuntos de entrenamiento (80%) y prueba (20%).
    """
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # mantiene la proporción de clases en train y test
    )
    print(
        f"[preprocessing] Train: {X_train.shape[0]} filas | "
        f"Test: {X_test.shape[0]} filas"
    )
    return X_train, X_test, y_train, y_test


def escalar_datos(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Ajusta el StandardScaler SOLO con X_train y transforma X_train y X_test.

    - scaler.fit(X_train)         -> aprende media y desviación estándar
                                      únicamente del set de entrenamiento.
    - scaler.transform(X_train)   -> aplica esa transformación a train.
    - scaler.transform(X_test)    -> aplica la MISMA transformación (ya
                                      aprendida) a test, sin volver a
                                      calcular media/desviación con test.
    """
    scaler = StandardScaler()

    scaler.fit(X_train)  # <-- fit únicamente en entrenamiento

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("[preprocessing] StandardScaler ajustado solo con X_train "
          "y aplicado a train/test.")

    return X_train_scaled, X_test_scaled, scaler
