import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# --- 1. CONFIGURACIÓN DE PÁGINA Y ESTILO ---
st.set_page_config(page_title="Retail Executive Dashboard", layout="wide", page_icon="🟣")

# CSS PERSONALIZADO (Para imitar Horizon UI)
st.markdown("""
<style>
    /* Fondo General */
    .stApp {
        background-color: #F4F7FE;
    }
    
    /* Estilo de Tarjetas (Cards) */
    .css-card {
        background-color: #FFFFFF;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    
    /* Tipografía */
    h1, h2, h3 {
        color: #2B3674 !important;
        font-family: 'Sans-serif';
    }
    p, label {
        color: #A3AED0 !important;
    }
    
    /* Métricas Grandes */
    .metric-value {
        font-size: 36px;
        font-weight: bold;
        color: #2B3674;
    }
    .metric-label {
        font-size: 14px;
        color: #A3AED0;
        margin-bottom: 5px;
    }
    .metric-delta-pos {
        color: #05CD99;
        font-weight: bold;
        font-size: 14px;
    }
    .metric-delta-neg {
        color: #EE5D50;
        font-weight: bold;
        font-size: 14px;
    }
    
    /* Ajustes del Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        box-shadow: 2px 0 10px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- 2. BACKEND (Modelo IA) ---
@st.cache_resource
def get_model():
    np.random.seed(42)
    n = 3000
    data = {
        'marketing': np.random.normal(5000, 2000, n),
        'competencia': np.random.randint(0, 10, n),
        'region': np.random.choice(['Norte', 'Sur', 'Este', 'Oeste'], n),
        'calidad': np.random.uniform(1, 10, n),
        'antiguedad': np.random.normal(5, 2, n)
    }
    df = pd.DataFrame(data)
    # Ecuación generadora (Ground Truth)
    df['ingresos'] = (18000 + 4.2 * df['marketing'] + 1500 * df['calidad'] - 
                      800 * df['competencia'] + np.random.normal(0, 3000, n))
    
    X = df.drop('ingresos', axis=1)
    y = df['ingresos']
    
    numeric_features = ['marketing', 'competencia', 'calidad', 'antiguedad']
    categorical_features = ['region']
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('imputer', SimpleImputer()), ('scaler', StandardScaler())]), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    
    model = Pipeline([('prep', preprocessor), ('algo', Lasso(alpha=50))])
    model.fit(X, y)
    return model

pipeline = get_model()

# --- 3. UI: SIDEBAR (Panel de Control) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/11524/11524049.png", width=60)
    st.markdown("### ⚙️ Configuración de Escenario")
    
    st.markdown("**Variables Palanca (Controlables)**")
    marketing = st.slider("Inversión Marketing ($)", 0, 15000, 6500, step=100)
    calidad = st.slider("Nivel de Servicio (1-10)", 1.0, 10.0, 8.5)
    
    st.markdown("---")
    st.markdown("**Contexto de Mercado (Externo)**")
    competencia = st.slider("Competidores Cercanos", 0, 10, 2)
    region = st.selectbox("Región", ["Norte", "Sur", "Este", "Oeste"])
    
    st.markdown("---")
    st.caption("v2.1 Business Simulator | AI Powered")

# --- 4. LÓGICA DE NEGOCIO (Cálculos) ---
# Inputs
input_data = pd.DataFrame({
    'marketing': [marketing], 'competencia': [competencia],
    'region': [region], 'calidad': [calidad], 'antiguedad': [5] # Asumimos media
})

# Proyecciones
ingreso_estimado = pipeline.predict(input_data)[0]
costos_fijos = 9500 
costos_totales = costos_fijos + marketing
utilidad = ingreso_estimado - costos_totales
margen_pct = (utilidad / ingreso_estimado) * 100

# Delta visual (comparado con objetivos fijos para demo)
target_ingreso = 45000
delta_ingreso = ((ingreso_estimado - target_ingreso) / target_ingreso) * 100

# --- 5. UI: DASHBOARD PRINCIPAL ---

# Título y Bienvenida
st.markdown("## 📊 Main Dashboard: Proyección Financiera")
st.markdown("_Vista ejecutiva para la toma de decisiones basada en IA_")
st.markdown("<br>", unsafe_allow_html=True)

# FILA 1: KPIs (Tarjetas estilo Horizon)
col1, col2, col3, col4 = st.columns(4)

def kpi_card(title, value, delta_txt, is_pos):
    color = "#05CD99" if is_pos else "#EE5D50"
    icon = "▲" if is_pos else "▼"
    return f"""
    <div class="css-card">
        <div class="metric-label">{title}</div>
        <div class="metric-value">{value}</div>
        <div style="color: {color}; font-weight: bold; font-size: 14px;">
            {icon} {delta_txt} <span style="color: #A3AED0; font-weight: normal;">vs Target</span>
        </div>
    </div>
    """

with col1:
    st.markdown(kpi_card("Ingresos Diarios Proyectados", f"${ingreso_estimado:,.0f}", f"{delta_ingreso:.1f}%", delta_ingreso > 0), unsafe_allow_html=True)

with col2:
    roi_txt = f"{margen_pct:.1f}%"
    st.markdown(kpi_card("Margen Neto Estimado", f"{roi_txt}", "Healthy", margen_pct > 15), unsafe_allow_html=True)

with col3:
    st.markdown(kpi_card("Gasto Operativo (OPEX)", f"${costos_totales:,.0f}", "Controlado", True), unsafe_allow_html=True)

with col4:
    # KPI de Eficiencia
    ratio = ingreso_estimado / marketing if marketing > 0 else 0
    st.markdown(kpi_card("Retorno por $ de Marketing", f"${ratio:.2f}", "Eficiencia", ratio > 3), unsafe_allow_html=True)

# FILA 2: GRÁFICOS ESTRATÉGICOS
col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown("### 📈 Curva de Sensibilidad: Marketing vs. Utilidad")
    
    # Generar datos para la curva
    x_vals = np.linspace(0, 15000, 50)
    y_ingresos = []
    y_utilidad = []
    
    temp_df = input_data.copy()
    for x in x_vals:
        temp_df['marketing'] = x
        ing = pipeline.predict(temp_df)[0]
        y_ingresos.append(ing)
        y_utilidad.append(ing - (costos_fijos + x))
        
    fig = go.Figure()
    
    # Área de Ingresos (Sutil)
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_ingresos, mode='lines', name='Ingresos Brutos',
        line=dict(color='#E9EDF7', width=2),
        fill='tozeroy'
    ))
    
    # Línea de Utilidad (Destacada - Púrpura Horizon)
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_utilidad, mode='lines', name='Utilidad Neta',
        line=dict(color='#4318FF', width=4)
    ))
    
    # Punto actual
    fig.add_trace(go.Scatter(
        x=[marketing], y=[utilidad], mode='markers', name='Escenario Actual',
        marker=dict(size=12, color='#2B3674', symbol='diamond')
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(showgrid=False, title='Inversión en Marketing ($)'),
        yaxis=dict(showgrid=True, gridcolor='#E0E5F2'),
        height=350,
        showlegend=True,
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown("### 💰 Estructura de P&L")
    
    # Gráfico de Barras Stacked simple y elegante
    labels = ['Costos Fijos', 'Marketing', 'Utilidad']
    values = [costos_fijos, marketing, utilidad]
    colors = ['#EFF4FB', '#A3AED0', '#4318FF'] # Gris claro, Gris medio, Púrpura
    
    fig_pie = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values, 
        hole=.6,
        marker=dict(colors=colors),
        textinfo='percent'
    )])
    
    fig_pie.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0),
        height=350,
        showlegend=True,
        legend=dict(orientation="h", y=-0.1)
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# FILA 3: TABLA COMPARATIVA (Escenarios)
st.markdown('<div class="css-card">', unsafe_allow_html=True)
st.markdown("### 📋 Análisis de Escenarios: Actual vs. Optimizado IA")

# Generamos un escenario optimizado "Fake" para comparar
marketing_opt = 8500
ingreso_opt = pipeline.predict(pd.DataFrame({'marketing': [marketing_opt], 'competencia': [competencia], 'region': [region], 'calidad': [9.0], 'antiguedad': [5]}))[0]
utilidad_opt = ingreso_opt - (costos_fijos + marketing_opt)

comparison_data = {
    "Métrica": ["Inversión Marketing", "Calidad Servicio", "Ingresos Estimados", "Utilidad Neta (Bottom Line)"],
    "Escenario Actual": [f"${marketing:,.0f}", f"{calidad}", f"${ingreso_estimado:,.0f}", f"${utilidad:,.0f}"],
    "Sugerencia IA (Optimizado)": [f"${marketing_opt:,.0f}", "9.0", f"${ingreso_opt:,.0f}", f"${utilidad_opt:,.0f}"],
    "Delta ($)": [f"${marketing_opt - marketing:,.0f}", "+Diff", f"+${ingreso_opt - ingreso_estimado:,.0f}", f"+${utilidad_opt - utilidad:,.0f}"]
}

df_comp = pd.DataFrame(comparison_data)

# Usamos HTML simple para la tabla porque st.dataframe a veces rompe el estilo "clean"
st.table(df_comp.set_index("Métrica"))

st.markdown('</div>', unsafe_allow_html=True)