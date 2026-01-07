import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("predicciones.csv")

# Variables  
features = [
    "sector_num",
    "placa_enc",
    "delta_peso",
    "duracion_horas",
    "dia_semana"
]

df = df.dropna(subset=features)

X = df[features]

print("Total registros:", len(df))

scaler_km = StandardScaler()
X_scaled = scaler_km.fit_transform(X)

kmeans = KMeans(
    n_clusters=6,
    random_state=42,
    n_init=10
)

df["macro_sector"] = kmeans.fit_predict(X_scaled)

y = df["macro_sector"]

print("Macro-sectores:", y.nunique())


modelo_final = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1
    ))
])

modelo_final.fit(X, y)

modelo_empaquetado = {
    "kmeans": kmeans,
    "scaler_kmeans": scaler_km,
    "clasificador": modelo_final,
    "features": features
}

with open("modelo_sector_nombre.pkl", "wb") as f:
    pickle.dump(modelo_empaquetado, f)

print("\n modelo_sector_nombre.pkl generado correctamente")