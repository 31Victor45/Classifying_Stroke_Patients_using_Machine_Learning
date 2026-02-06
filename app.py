import streamlit as st
import pandas as pd
import pickle
import json
import numpy as np
import shap
import matplotlib.pyplot as plt

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Stroke AI Predictor Pro + SHAP",
    page_icon="🧠",
    layout="wide"
)

# --- 2. CARGA DE RECURSOS ---
@st.cache_resource
def load_resources():
    # Asegúrate de que los archivos .pkl y .json estén en la carpeta correcta
    with open('p_items/stroke_model_final.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('p_items/config_final_app_real.json', 'r') as f:
        config = json.load(f)
    # Inicializar SHAP
    explainer = shap.TreeExplainer(model)
    return model, config, explainer

try:
    model, config, explainer = load_resources()
except Exception as e:
    st.error(f"❌ Error al cargar recursos: {e}")
    st.stop()

# --- 3. BARRA LATERAL (CON IMAGEN GRANDE) ---
with st.sidebar:
    # [MODIFICADO] width=250 hace la imagen mucho más grande en la barra lateral
    st.image("p_items/risk.png", width=250, caption="Sistema de Triaje")
    
    st.header("⚙️ Configuración")
    st.markdown("Ajuste la sensibilidad del modelo.")
    umbral_manual = st.slider("Umbral de Alerta", 0.0, 1.0, 0.40, 0.05)
    st.info(f"Probabilidades ≥ {umbral_manual:.2f} se marcarán como ALTO RIESGO.")

# --- 4. MOTOR DE PROCESAMIENTO ---
def procesar_entrada(datos, config):
    df_input = pd.DataFrame(0, index=[0], columns=config['variables_modelo'])
    
    # Variables Binarias
    df_input['hypertension'] = 1 if datos['hypertension'] == "Sí" else 0
    df_input['heart_disease'] = 1 if datos['heart_disease'] == "Sí" else 0
    
    # Discretización
    puntos = config['puntos_corte_biologicos']
    etiquetas = config['etiquetas_riesgo']
    
    def obtener_cuartil(valor, bins, labels):
        return pd.cut([valor], bins=bins, labels=labels, include_lowest=True)[0]

    q_age = obtener_cuartil(datos['age'], puntos['edad_anos'], etiquetas)
    q_glu = obtener_cuartil(datos['glucose'], puntos['glucosa_mg_dl'], etiquetas)
    q_bmi = obtener_cuartil(datos['bmi'], puntos['bmi_indice'], etiquetas)
    
    # Activar Categorías
    if f"age_cat_{q_age}" in df_input.columns: df_input[f"age_cat_{q_age}"] = 1
    if f"glucose_cat_{q_glu}" in df_input.columns: df_input[f"glucose_cat_{q_glu}"] = 1
    if f"bmi_cat_{q_bmi}" in df_input.columns: df_input[f"bmi_cat_{q_bmi}"] = 1
    
    if f"gender_{datos['gender']}" in df_input.columns: df_input[f"gender_{datos['gender']}"] = 1
    if f"ever_married_{datos['married']}" in df_input.columns: df_input[f"ever_married_{datos['married']}"] = 1
    if f"work_type_{datos['work']}" in df_input.columns: df_input[f"work_type_{datos['work']}"] = 1
    if f"Residence_type_{datos['residence']}" in df_input.columns: df_input[f"Residence_type_{datos['residence']}"] = 1
    
    # Tabaquismo
    if datos['smoke'] == "Ha fumado o fuma":
        df_input['smoking_status_ever_smoked'] = 1
    elif datos['smoke'] == "Nunca fumó":
        df_input['smoking_status_never smoked'] = 1
    else:
        df_input['smoking_status_Unknown'] = 1
        
    return df_input

# --- 5. INTERFAZ DE USUARIO ---
st.title("🏥 Predictor de Ictus (Stroke AI)")

with st.form("paciente_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Edad", 0, 110, 65)
        gender = st.selectbox("Género", ["Male", "Female"])
        residence = st.selectbox("Residencia", ["Urban", "Rural"])
    with col2:
        glucose = st.number_input("Glucosa (mg/dL)", 50.0, 300.0, 110.0)
        hypertension = st.selectbox("¿Hipertensión?", ["No", "Sí"])
        married = st.selectbox("¿Casado?", ["Yes", "No"])
    with col3:
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
        heart_disease = st.selectbox("¿Cardiopatía?", ["No", "Sí"])
        smoke = st.selectbox("Tabaco", ["Nunca fumó", "Ha fumado o fuma", "Desconocido"])
    
    work = st.selectbox("Trabajo", ["Private", "Self-employed", "Govt_job", "children"])
    enviar = st.form_submit_button("🩺 DIAGNOSTICAR")

# --- 6. LÓGICA Y RESULTADOS ---
if enviar:
    # A. VALIDACIÓN
    if age > 12 and work == "children":
        st.error(f"⛔ **Error Lógico:** Paciente de {age} años no puede tener trabajo 'Children'.")
        st.stop()
    
    # B. PROCESAMIENTO
    input_dict = {
        "age": age, "glucose": glucose, "bmi": bmi, "gender": gender,
        "hypertension": hypertension, "heart_disease": heart_disease,
        "married": married, "residence": residence, "work": work, "smoke": smoke
    }
    
    df_proc = procesar_entrada(input_dict, config)
    prob = float(model.predict_proba(df_proc)[0][1])

    st.divider()
    
    # C. VISUALIZACIÓN PRINCIPAL (3 COLUMNAS: METRICA | SHAP | IMAGEN)
    # Definimos proporciones: Texto(1.2) - Grafico(2) - Imagen(1.2)
    c1, c2, c3 = st.columns([1.2, 2, 1.2])
    
    # --- COLUMNA 1: MÉTRICAS ---
    with c1:
        st.subheader("Resultados")
        st.metric(label="Probabilidad de Ictus", value=f"{prob:.1%}")
        if prob >= umbral_manual:
            st.error(f"🚨 ALTO RIESGO\n(> {umbral_manual:.0%})")
        else:
            st.success(f"✅ BAJO RIESGO\n(< {umbral_manual:.0%})")
        st.progress(prob)

    # --- COLUMNA 2: SHAP (CENTRO) ---
    with c2:
        st.write("### 📊 Análisis de Factores")
        shap_values = explainer(df_proc)
        # Ajustamos tamaño del gráfico para que quepa bien
        fig, ax = plt.subplots(figsize=(8, 4)) 
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)

    # --- COLUMNA 3: IMAGEN GRANDE (DERECHA) ---
    with c3:
        st.write("### 🩺 Diagnóstico Visual")
        if prob >= umbral_manual:
            # use_container_width=True hace que la imagen ocupe TODO el ancho de la columna 3
            st.image("p_items/dont_save.png", caption="⚠️ ATENCIÓN REQUERIDA", use_container_width=True)
        else:
            st.image("p_items/save.png", caption="✅ HÁBITOS SALUDABLES", use_container_width=True)

    st.divider()

    # D. ANÁLISIS DETALLADO
    st.write("### 🔍 Análisis Detallado del Paciente")
    col_a, col_b = st.columns(2)
    with col_a:
        if age >= 75: st.warning(f"⚠️ **Edad Avanzada:** {age} años incrementa la vulnerabilidad vascular.")
        if hypertension == "Sí": st.error("🚨 **Hipertensión:** Factor crítico detectado.")
    with col_b:
        if smoke == "Ha fumado o fuma": st.warning("⚠️ **Tabaquismo:** Daño arterial acumulado.")
        if glucose > 150: st.warning("⚠️ **Glucosa:** Niveles fuera de rango óptimo.")