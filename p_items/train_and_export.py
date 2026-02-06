import pandas as pd
import xgboost as xgb
import pickle
import json

# 1. Cargar datos y configuración
df = pd.read_csv('healthcare-dataset-stroke-transformed.csv')
with open('config_final_app_real.json', 'r') as f:
    config = json.load(f)

# 2. Replicar la Ingeniería de Características del Notebook
# Extraemos los puntos de corte del JSON
puntos = config['puntos_corte_biologicos']
etiquetas = config['etiquetas_riesgo']

print("Aplicando discretización por cuartiles...")
df['age_cat'] = pd.cut(df['age'], bins=puntos['edad_anos'], labels=etiquetas, include_lowest=True)
df['glucose_cat'] = pd.cut(df['avg_glucose_level'], bins=puntos['glucosa_mg_dl'], labels=etiquetas, include_lowest=True)
df['bmi_cat'] = pd.cut(df['bmi'], bins=puntos['bmi_indice'], labels=etiquetas, include_lowest=True)

# 3. Preparar Dataset (Eliminar numéricas originales y crear Dummies)
target = 'stroke'
# Eliminamos las variables originales para quedarnos solo con las binned y las categóricas
df_pre = df.drop(['age', 'avg_glucose_level', 'bmi'], axis=1)
df_final = pd.get_dummies(df_pre)

# 4. Alinear con las variables exactas del modelo (según el JSON)
variables_modelo = config['variables_modelo']
X = pd.DataFrame(0, index=df_final.index, columns=variables_modelo)

for col in variables_modelo:
    if col in df_final.columns:
        X[col] = df_final[col]

y = df[target]

# 5. Entrenamiento con hiperparámetros EXACTOS
print("Entrenando Modelo Maestro...")
model = xgb.XGBClassifier(
    n_estimators=125,
    learning_rate=0.1,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.9,
    colsample_bytree=1.0,
    scale_pos_weight=19.4, 
    random_state=42,
    eval_metric='logloss'
)

model.fit(X, y)

# 6. Exportar
with open('stroke_model_final.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Modelo 'stroke_model_final.pkl' creado con éxito.")