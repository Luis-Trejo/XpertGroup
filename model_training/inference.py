import os
import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline

# ----- 1. Cargar el modelo entrenado al iniciar el contenedor -----
def model_fn(model_dir):
    """ Carga el modelo desde el directorio especificado por SageMaker. """
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)
    return model

# ----- 2. Procesar entrada de datos en formato JSON o CSV -----
def input_fn(input_data, content_type):
    """ Convierte el input de una invocación a un DataFrame. """
    if content_type == "application/json":
        return pd.DataFrame(input_data)
    elif content_type == "text/csv":
        return pd.read_csv(pd.compat.StringIO(input_data))
    else:
        raise ValueError(f"[ERROR] Tipo de contenido no soportado: {content_type}")

# ----- 3. Hacer la inferencia -----
def predict_fn(input_data, model):
    """ Realiza predicciones con el modelo cargado. """
    return model.predict(input_data)

# ----- 4. Dar formato a la respuesta de salida -----
def output_fn(prediction, accept):
    """ Formatea la salida de las predicciones. """
    if accept == "application/json":
        return prediction.tolist(), "application/json"
    elif accept == "text/csv":
        out = ",".join(str(x) for x in prediction)
        return out, "text/csv"
    else:
        raise ValueError(f"[ERROR] Tipo de respuesta no soportado: {accept}")
