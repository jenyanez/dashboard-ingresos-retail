import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, root_mean_squared_error

# --- CONFIGURACIÓN INICIAL ---
st.set_page_config(
    page_title="Retail AI Planner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS para dar look corporativo
st.markdown("""
<style>
    .main {
        background-color: #f9f9f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. BACKEND: MODELO Y DATOS (Optimizado) ---
@st.cache_resource
def build_model():
    # Generación de datos sintéticos
    np.random.seed(42)
    n = 2000 # Aumentamos datos para estabilidad
    data = {
        'marketing_spend': np.random.normal(5000, 2000, n),
        'competidores_cerca': np.random.randint(0, 10, n),
        'region': np.random.choice(['Norte', 'Sur', 'Este', 'Oeste'], n),
        'calidad_servicio': np.random.uniform(1, 10, n),
        'antiguedad_tienda': np.random.normal(5, 2, n)
    }
    df = pd.DataFrame(data)
    
    # Lógica de negocio sintética (Ground Truth)
    # Agregamos no-linealidad leve para hacer el gráfico de sensibilidad más interesante luego
    df['ingresos'] = (
        15000 + 
        3.8 * df['marketing_spend'] + 
        1200 * df['calidad_servicio'] - 
        600 * df['competidores_cerca'] + 
        100 * df['antiguedad_tienda'] +
        np.random.normal(0, 4000, n)
    )
    
    # Limpieza y Split
    X = df.drop('ingresos', axis=1)
    y = df['ingresos']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Pipeline
    numeric_features = ['marketing_spend', 'competidores_cerca', 'calidad_servicio', 'antiguedad_tienda']
    categorical_features = ['region']
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])
    
    # Usamos Lasso para selección de características
    model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', Lasso(alpha=50.0))])
    model.fit(X_train, y_train)
    
    # Métricas
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    
    return model, r2, rmse, df

pipeline, r2_score_val, rmse_val, raw_df = build_model()

# --- 2. SIDEBAR: CONTROLES TÁCTICOS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1055/1055644.png", width=80)
    st.title("Configuración de Escenario")
    
    st.markdown("### 🎮 Palancas de Gestión")
    st.info("Variables bajo control directo de la tienda")
    
    marketing = st.slider("Inversión Marketing ($)", 0, 15000, 5000, step=500, help="Presupuesto diario asignado a campañas.")
    calidad = st.slider("Score Calidad (NPS estimado)", 1.0, 10.0, 7.5, help="Basado en encuestas de satisfacción.")
    
    st.markdown("### 🌍 Entorno y Mercado")
    st.warning("Variables externas o estructurales")
    competidores = st.slider("Competidores (Radio 1km)", 0, 15, 3)
    region = st.selectbox("Región Operativa", ['Norte', 'Sur', 'Este', 'Oeste'])
    antiguedad = st.number_input("Antigüedad Tienda (Años)", 0, 50, 5)

    st.markdown("---")
    st.caption(f"Modelo v1.2 | R²: {r2_score_val:.2%}")

# --- 3. CÁLCULOS DE NEGOCIO ---
# Crear dataframe con input
input_df = pd.DataFrame({
    'marketing_spend': [marketing],
    'competidores_cerca': [competidores],
    'region': [region],
    'calidad_servicio': [calidad],
    'antiguedad_tienda': [antiguedad]
})

# Predicción
pred_ingreso = pipeline.predict(input_df)[0]

# Lógica de Negocio (Hardcoded para simulación de ROI)
costos_fijos = 8000  # Alquiler, luz, nómina base
costos_totales = costos_fijos + marketing
utilidad_neta = pred_ingreso - costos_totales
roi = (utilidad_neta / costos_totales) * 100 if costos_totales > 0 else 0

# --- 4. INTERFAZ PRINCIPAL ---

# Header
st.title("🚀 Proyección de Rentabilidad Retail")
st.markdown(f"**Escenario:** Región {region} | {competidores} Competidores | Antigüedad {antiguedad} años")

# KPI ROW
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Ingresos Proyectados", f"${pred_ingreso:,.0f}", delta_color="normal")
with col2:
    st.metric("Costo Total Operativo", f"${costos_totales:,.0f}", delta="-Costo", delta_color="inverse")
with col3:
    color_roi = "normal" if roi > 0 else "inverse"
    st.metric("Utilidad Neta Diaria", f"${utilidad_neta:,.0f}", delta=f"{roi:.1f}% ROI", delta_color=color_roi)
with col4:
    # Calculamos un 'Efficiency Score' fake basado en calidad
    st.metric("Calidad Percibida", f"{calidad}/10", delta="Target: 8.0")

# TABS
tab_business, tab_tech = st.tabs(["📊 Análisis de Negocio & ROI", "🧠 Explicabilidad del Modelo"])

with tab_business:
    row1_col1, row1_col2 = st.columns([2, 1])
    
    with row1_col1:
        st.subheader("Sensibilidad: Impacto del Marketing en Ingresos")
        
        # Generar datos para la curva de sensibilidad
        x_range = np.linspace(0, 15000, 100)
        sensitivity_data = []
        for x in x_range:
            temp_df = input_df.copy()
            temp_df['marketing_spend'] = x
            pred = pipeline.predict(temp_df)[0]
            net = pred - (costos_fijos + x) # Utilidad proyectada
            sensitivity_data.append({'Inversión Marketing': x, 'Ingresos': pred, 'Utilidad Neta': net})
        
        sens_df = pd.DataFrame(sensitivity_data)
        
        # Gráfico combinado: Ingresos vs Utilidad
        fig_sens = go.Figure()
        
        # Línea de Ingresos
        fig_sens.add_trace(go.Scatter(
            x=sens_df['Inversión Marketing'], 
            y=sens_df['Ingresos'],
            mode='lines',
            name='Ingresos Brutos',
            line=dict(color='#3498db', width=3)
        ))
        
        # Línea de Utilidad
        fig_sens.add_trace(go.Scatter(
            x=sens_df['Inversión Marketing'], 
            y=sens_df['Utilidad Neta'],
            mode='lines',
            name='Utilidad Neta',
            fill='tozeroy', # Relleno para resaltar ganancia
            line=dict(color='#2ecc71', width=3)
        ))
        
        # Punto Actual
        fig_sens.add_trace(go.Scatter(
            x=[marketing], y=[pred_ingreso],
            mode='markers', name='Selección Actual',
            marker=dict(color='red', size=12, symbol='x')
        ))
        
        fig_sens.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=20, b=20),
            height=400
        )
        st.plotly_chart(fig_sens, use_container_width=True)
        
        st.info("💡 **Insight:** Observa el punto donde la línea verde (Utilidad) deja de crecer. Ese es tu punto de saturación de marketing.")

    with row1_col2:
        st.subheader("Desglose de Rentabilidad")
        
        # Gráfico de Donut para estructura de costos vs ganancia
        labels = ['Costos Fijos', 'Marketing', 'Utilidad (Margen)']
        values = [costos_fijos, marketing, max(0, utilidad_neta)] # Evitar negativos en donut
        colors = ['#95a5a6', '#e74c3c', '#2ecc71']
        
        fig_donut = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5, marker_colors=colors)])
        fig_donut.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=300, showlegend=True)
        
        if utilidad_neta < 0:
            st.error("⚠️ ¡Alerta! Operación con pérdidas. Revisa la estructura de costos o aumenta la eficiencia.")
        else:
            st.success("✅ Operación Rentable.")
            
        st.plotly_chart(fig_donut, use_container_width=True)

with tab_tech:
    st.markdown("### ¿Cómo toma decisiones el modelo?")
    
    # Extracción de coeficientes para interpretar Lasso
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    coefs = pipeline.named_steps['regressor'].coef_
    
    coef_df = pd.DataFrame({'Feature': feature_names, 'Impacto': coefs})
    coef_df['Abs_Impacto'] = coef_df['Impacto'].abs()
    coef_df = coef_df.sort_values('Impacto', ascending=True)
    
    # Limpieza de nombres para visualización
    coef_df['Feature'] = coef_df['Feature'].str.replace('num__', '').str.replace('cat__', '')
    
    # Gráfico de Barras Horizontal con Plotly
    fig_bar = px.bar(
        coef_df, 
        x='Impacto', 
        y='Feature', 
        orientation='h',
        color='Impacto',
        color_continuous_scale='RdBu', # Rojo negativo, Azul positivo
        title='Importancia de Variables (Pesos Lasso)'
    )
    fig_bar.update_layout(height=500)
    st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("""
    **Interpretación Técnica (Analista):**
    * Las barras **Azules** indican correlación positiva (aumentan ingresos).
    * Las barras **Rojas** indican correlación negativa (reducen ingresos).
    * Variables con coeficiente 0 han sido descartadas por la regularización Lasso L1.
    """)

