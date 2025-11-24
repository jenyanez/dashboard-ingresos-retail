import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, root_mean_squared_error

# Configuración de página
st.set_page_config(page_title="Predicción de Ingresos Retail", layout="wide")

# --- 1. GENERACIÓN Y ENTRENAMIENTO (BACKEND) ---
@st.cache_resource # Para no re-entrenar cada vez que tocas un botón
def train_model():
    # Generación de datos sintéticos (Misma lógica del Notebook)
    np.random.seed(42)
    n = 1000
    data = {
        'marketing_spend': np.random.normal(5000, 2000, n),
        'competidores_cerca': np.random.randint(0, 10, n),
        'region': np.random.choice(['Norte', 'Sur', 'Este', 'Oeste'], n),
        'calidad_servicio': np.random.uniform(1, 10, n),
        'antiguedad_tienda': np.random.normal(5, 2, n)
    }
    df = pd.DataFrame(data)
    
    # Cálculo del Target (Ingresos)
    df['ingresos'] = (20000 + 3.5 * df['marketing_spend'] + 
                      1000 * df['calidad_servicio'] - 
                      500 * df['competidores_cerca'] + 
                      np.random.normal(0, 5000, n))
    
    # Simular nulos
    idx = np.random.choice(df.index, 50, replace=False)
    df.loc[idx, 'calidad_servicio'] = np.nan
    
    X = df.drop('ingresos', axis=1)
    y = df['ingresos']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Pipeline Lasso
    numeric_features = ['marketing_spend', 'competidores_cerca', 'calidad_servicio', 'antiguedad_tienda']
    categorical_features = ['region']
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', Lasso(alpha=100.0, max_iter=10000))])
    pipeline.fit(X_train, y_train)
    
    # Métricas
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    
    return pipeline, r2, rmse, X_train

pipeline, r2, rmse, X_train_sample = train_model()

# --- 2. INTERFAZ DE USUARIO (FRONTEND) ---

st.title("📊 Simulador de Ingresos Retail con IA")
st.markdown("""
Esta herramienta utiliza un modelo de **Machine Learning (Lasso Regression)** para predecir los ingresos diarios de una tienda
basándose en factores operativos y de mercado.
""")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Configurar Escenario")
    st.info("Ajusta las palancas para simular una tienda:")
    
    marketing = st.slider("Inversión en Marketing ($)", 0, 10000, 5000)
    calidad = st.slider("Calidad de Servicio (1-10)", 1.0, 10.0, 5.0)
    competidores = st.slider("Competidores Cercanos", 0, 15, 3)
    
    with st.expander("Variables que NO afectan (Según el modelo)"):
        antiguedad = st.number_input("Antigüedad (Años)", 0, 20, 5)
        region = st.selectbox("Región", ['Norte', 'Sur', 'Este', 'Oeste'])

    # Crear DataFrame con el input del usuario
    input_data = pd.DataFrame({
        'marketing_spend': [marketing],
        'competidores_cerca': [competidores],
        'region': [region],
        'calidad_servicio': [calidad],
        'antiguedad_tienda': [antiguedad]
    })

with col2:
    st.header("2. Predicción de Ingresos")
    
    # Predicción
    prediccion = pipeline.predict(input_data)[0]
    
    # Mostrar KPI grande
    st.metric(label="Ingresos Estimados Diarios", value=f"${prediccion:,.2f}")
    
    st.markdown("---")
    st.subheader("Rendimiento del Modelo")
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Confianza (R2)", f"{r2:.2%}", help="Porcentaje de variabilidad explicada por el modelo")
    kpi2.metric("Margen de Error (RMSE)", f"${rmse:,.0f}", help="Error promedio en dólares")
    kpi3.metric("Variables Clave", "3 de 8", help="Variables seleccionadas por Lasso")

    # Gráfico de Importancia
    st.markdown("### 🧠 ¿Qué está pensando el modelo?")
    lasso_model = pipeline.named_steps['regressor']
    preprocessor = pipeline.named_steps['preprocessor']
    feature_names = preprocessor.get_feature_names_out()
    
    coefs = pd.DataFrame({'Variable': feature_names, 'Peso': lasso_model.coef_})
    coefs = coefs[coefs['Peso'] != 0].sort_values(by='Peso', ascending=True) # Solo las no-cero
    
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['red' if x < 0 else 'green' for x in coefs['Peso']]
    ax.barh(coefs['Variable'], coefs['Peso'], color=colors)
    ax.set_title("Impacto de las Variables en los Ingresos (Lasso)")
    ax.set_xlabel("Impacto Negativo <---> Impacto Positivo")
    st.pyplot(fig)