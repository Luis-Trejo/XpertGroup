import pandas as pd

# ---------- Función para cargar el CSV ----------
def read_dataset(csv_path):
    try:
        # Lee el archivo línea por línea y limpia los ;; extras
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Filtra líneas que tengan estructura de datos válida
        clean_lines = [line.strip().rstrip(';') for line in lines if ',' in line and not line.startswith('Imagen')]

        # Une las líneas limpias como si fuera un CSV en memoria
        from io import StringIO
        cleaned_csv = "\n".join(clean_lines)
        df = pd.read_csv(StringIO(cleaned_csv))

        # Forzar conversión de columnas numéricas
        if 'Price' in df.columns:
            df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

        print(f"[INFO] Dataset cargado con éxito. Registros: {len(df)}")
        return df

    except Exception as e:
        print(f"[ERROR] No se pudo leer el archivo: {e}")
        return pd.DataFrame()

# ---------- Limpieza de nulos ----------
def remove_nulls(df):
    before = len(df)
    df_clean = df.dropna()
    removed = before - len(df_clean)
    print(f"[LIMPIEZA] Se eliminaron {removed} registros con valores nulos.")
    return df_clean

# ---------- Limpieza de duplicados ----------
def remove_duplicates(df):
    before = len(df)
    df_clean = df.drop_duplicates()
    removed = before - len(df_clean)
    print(f"[LIMPIEZA] Se eliminaron {removed} duplicados.")
    return df_clean

# ---------- Exploración general ----------
def explore_dataset(df):
    # Intentamos convertir columnas que deberían ser numéricas
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    if numeric_cols:
        print("\n[EXPLORACIÓN] Estadísticas numéricas:")
        print(df[numeric_cols].describe())
    else:
        print("\n[EXPLORACIÓN] No se encontraron columnas numéricas para describir.")

    print("\n[EXPLORACIÓN] Tipos de datos:")
    print(df.dtypes)

    print("\n[EXPLORACIÓN] Valores únicos por columna:")
    for col in df.columns:
        unique_count = df[col].nunique()
        print(f"[INFO] '{col}' tiene {unique_count} valores únicos.")

    if numeric_cols:
        print("\n[EXPLORACIÓN] Correlación entre variables numéricas:")
        print(df[numeric_cols].corr())
    else:
        print("\n[EXPLORACIÓN] No hay variables numéricas para calcular correlación.")

    print("\n[EXPLORACIÓN] Posibles llaves únicas:")
    unique_keys = [col for col in df.columns if df[col].is_unique]
    if unique_keys:
        print(f"[INFO] Columnas candidatas a llave primaria: {unique_keys}")
    else:
        print("[INFO] No se encontraron columnas con valores únicos por fila.")

# ---------- Función principal ----------
def main():
    
    csv_path = "clothes_price_prediction_dat.csv"
    df = read_dataset(csv_path)

    if df.empty:
        return

    df = remove_nulls(df)
    df = remove_duplicates(df)
    explore_dataset(df)
    df.to_parquet("clean_data.parquet")

# ---------- Ejecutar ----------
if __name__ == "__main__":
    main()


