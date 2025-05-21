import os
import joblib
import tarfile
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# --------------------------
# 1. Leer dataset
# --------------------------
df = pd.read_parquet("clean_data.parquet", sep=",", engine="python").dropna()
df.columns = df.columns.str.strip()
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df = df.dropna(subset=['Price'])

# --------------------------
# 2. Enriquecimiento de features
# --------------------------

# A) Combinación Brand + Category
df['Brand_Category'] = df['Brand'] + '_' + df['Category']

# B) Material Quality Score
material_quality = {
    'Silk': 5, 'Wool': 4, 'Denim': 3, 'Cotton': 3, 'Polyester': 2, 'Nylon': 1
}
df['Material_Quality'] = df['Material'].map(material_quality)

# C) Frecuencia del color
df['Color_Freq'] = df['Color'].map(df['Color'].value_counts())

# D) Talla ordinal
size_order = {'XS': 0, 'S': 1, 'M': 2, 'L': 3, 'XL': 4, 'XXL': 5}
df['Size_Ordinal'] = df['Size'].map(size_order)

# E) Promedio por categoría
df['Avg_Category_Price'] = df.groupby('Category')['Price'].transform('mean')

# F) Promedio por marca
df['Avg_Brand_Price'] = df.groupby('Brand')['Price'].transform('mean')

# G) Flag de premium
premium_brands = ['Nike', 'Under Armour']
df['Is_Premium_Brand'] = df['Brand'].isin(premium_brands).astype(int)

# H) Material Weight (estimado)
material_weight = {
    'Silk': 1.1, 'Wool': 1.3, 'Denim': 1.4, 'Cotton': 1.0, 'Polyester': 0.9, 'Nylon': 0.8
}
df['Material_Weight'] = df['Material'].map(material_weight)

# I) Log price
df['Log_Price'] = np.log1p(df['Price'])

# --------------------------
# 3. Definir variables
# --------------------------
X = df[['Brand', 'Category', 'Color', 'Size', 'Material',
        'Brand_Category', 'Material_Quality', 'Color_Freq',
        'Size_Ordinal', 'Avg_Category_Price', 'Avg_Brand_Price',
        'Is_Premium_Brand', 'Material_Weight']]
y = df['Price']

# --------------------------
# 4. Partir dataset
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------
# 5. Preprocesamiento
# --------------------------
categorical_features = ['Brand', 'Category', 'Color', 'Size', 'Material', 'Brand_Category']
numeric_features = list(set(X.columns) - set(categorical_features))

preprocessor = ColumnTransformer(transformers=[
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features)
], remainder='passthrough')


# --------------------------
# 6. Modelo + búsqueda
# --------------------------
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(objective='reg:squarederror', verbosity=0))
])

param_dist = {
    "regressor__n_estimators": randint(50, 300),
    "regressor__max_depth": randint(3, 10),
    "regressor__learning_rate": uniform(0.01, 0.3),
    "regressor__subsample": uniform(0.6, 0.4),
    "regressor__colsample_bytree": uniform(0.6, 0.4),
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='r2',
    verbose=2,
    n_jobs=-1,
    random_state=42
)

print("[INFO] Iniciando búsqueda de hiperparámetros...")
search.fit(X_train, y_train)

# --------------------------
# 7. Evaluación
# --------------------------
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n[MEJOR MODELO ENCONTRADO]")
print(search.best_params_)

print("\n[RESULTADOS FINALES]")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.2f}")


# --------------------------
# 8. Persistir modelo
# --------------------------

os.makedirs("model_artifacts", exist_ok=True)


joblib.dump(best_model, "model_artifacts/model.joblib")


with tarfile.open("model_artifacts/model.tar.gz", "w:gz") as tar:
    tar.add("model_artifacts/model.joblib", arcname="model.joblib")

print("Modelo empaquetado como model_artifacts/model.tar.gz")