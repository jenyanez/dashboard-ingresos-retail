import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# --- 1. CONFIGURACIÓN GLOBALES ---
st.set_page_config(page_title="Executive Dashboard", layout="wide", page_icon="📊")

# COLORES CORPORATIVOS
COLOR_BG = "#F4F7FE"
COLOR_WHITE = "#FFFFFF"
COLOR_TEXT_MAIN = "#2B3674"    # Azul Oscuro
COLOR_TEXT_SEC = "#A3AED0"     # Gris Suave
COLOR_ACCENT = "#4318FF"       # Azul Eléctrico
COLOR_SUCCESS = "#05CD99"

st.markdown(f"""
<style>
    /* FUENTE OFICIAL */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif;
        color: {COLOR_TEXT_MAIN};
    }}

    /* ELIMINAR ESTILOS NATIVOS MOLESTOS */
    header {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .stApp {{ background-color: {COLOR_BG}; }}
    
    /* TARJETAS (CARDS) LIMPIAS */
    .card {{
        background-color: {COLOR_WHITE};
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.02);
        margin-bottom: 20px;
    }}
    
    /* TÍTULOS */
    h1, h2, h3 {{ color: {COLOR_TEXT_MAIN} !important; }}
    
    /* KPI BOXES */
    .kpi-title {{ font-size: 14px; color: {COLOR_TEXT_SEC}; font-weight: 500; margin-bottom: 5px; }}
    .kpi-value {{ font-size: 32px; color: {COLOR_TEXT_MAIN}; font-weight: 700; }}
    .kpi-delta {{ font-size: 12px; font-weight: 700; display: inline-flex; align-items: center; }}
    
    /* SIDEBAR */
    section[data-testid="stSidebar"] {{
        background-color: {COLOR_WHITE};
        border-right: 1px solid #E9EDF7;
    }}
    
    /* TABLA LIMPIA */
    .clean-table {{ width: 100%; border-collapse: collapse; }}
    .clean-table th {{ 
        text-align: left; 
        color: {COLOR_TEXT_SEC}; 
        font-size: 11px; 
        text-transform: uppercase; 
        border-bottom: 1px solid #E9EDF7; 
        padding-bottom: 10px;
    }}
    .clean-table td {{ 
        padding: 15px 0; 
        color: {COLOR_TEXT_MAIN}; 
        font-weight: 600; 
        font-size: 14px; 
        border-bottom: 1px solid #F4F7FE;
    }}
    
    /* CHECKBOX CUSTOM */
    .check-icon {{ color: {COLOR_ACCENT}; font-size: 16px; margin-right: 8px; }}
</style>
""", unsafe_allow_html=True)

# --- 2. BACKEND SIMPLIFICADO ---
@st.cache_resource
def get_pipeline():
    # Datos Dummy para estructura
    df = pd.DataFrame({
        'mkt': np.random.normal(5000, 2000, 100),
        'comp': np.random.randint(0, 10, 100),
        'reg': np.random.choice(['A'], 100),
        'cal': np.random.uniform(1, 10, 100)
    })
    df['y'] = df['mkt']*3 + df['cal']*100 + np.random.normal(0,100,100)
    
    pipe = Pipeline([
        ('prep', ColumnTransformer([('num', StandardScaler(), ['mkt','comp','cal']), ('cat', OneHotEncoder(), ['reg'])])),
        ('model', Lasso(alpha=1))
    ])
    pipe.fit(df.drop('y', axis=1), df['y'])
    return pipe

model = get_pipeline()

# --- 3. SIDEBAR MINIMALISTA ---
with st.sidebar:
    st.markdown(f"<h3 style='text-align:center; color:{COLOR_ACCENT};'>⚡ Retail AI</h3>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown(f"<div style='color:{COLOR_TEXT_SEC}; font-size:12px; font-weight:700; margin-bottom:10px;'>PARÁMETROS DE CONTROL</div>", unsafe_allow_html=True)
    
    # Inputs limpios
    mkt = st.slider("Presupuesto Marketing", 0, 15000, 7500, format="$%d")
    calidad = st.slider("Score Calidad", 1.0, 10.0, 8.2)
    comp = st.slider("Competencia Local", 0, 10, 2)
    
    st.markdown("---")
    st.caption("v4.0 Executive Build")

# --- 4. CÁLCULOS ---
input_df = pd.DataFrame({'mkt': [mkt], 'comp': [comp], 'reg': ['A'], 'cal': [calidad]})
pred = model.predict(input_df)[0]
roi = ((pred - (8000+mkt))/(8000+mkt))*100

# --- 5. DASHBOARD GRID ---

st.markdown("## Resumen Ejecutivo de Rendimiento")

# FILA 1: KPIs
c1, c2, c3 = st.columns(3)

def kpi_card(title, value, delta, color):
    # HTML Compacto y sin indentación excesiva para evitar errores
    return f"""
<div class="card">
    <div class="kpi-title">{title}</div>
    <div class="kpi-value">{value}</div>
    <div class="kpi-delta" style="color: {color};">
        {delta} <span style="color: #A3AED0; font-weight: 400; margin-left: 5px;">vs target</span>
    </div>
</div>
"""

with c1: st.markdown(kpi_card("Ingresos Estimados", f"${pred:,.0f}", "▲ 12.5%", COLOR_SUCCESS), unsafe_allow_html=True)
with c2: st.markdown(kpi_card("ROI Proyectado", f"{roi:.1f}%", "▲ Healthy", COLOR_SUCCESS if roi > 0 else "#FF0000"), unsafe_allow_html=True)
with c3: st.markdown(kpi_card("Costo Operativo", f"${(8000+mkt):,.0f}", "● Stable", COLOR_ACCENT), unsafe_allow_html=True)

# FILA 2: GRÁFICOS Y TABLA
col_main, col_side = st.columns([2, 1])

with col_main:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"<h4 style='margin:0 0 20px 0; color:{COLOR_TEXT_MAIN}'>Proyección de Sensibilidad</h4>", unsafe_allow_html=True)
    
    # Gráfico Plotly Limpio
    x = np.linspace(0, 15000, 50)
    y = [model.predict(pd.DataFrame({'mkt': [i], 'comp': [comp], 'reg': ['A'], 'cal': [calidad]}))[0] for i in x]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color=COLOR_ACCENT, width=3), fill='tozeroy', fillcolor='rgba(67, 24, 255, 0.05)'))
    fig.add_trace(go.Scatter(x=[mkt], y=[pred], mode='markers', marker=dict(color=COLOR_TEXT_MAIN, size=10)))
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, title=None, tickfont=dict(color=COLOR_TEXT_SEC)),
        yaxis=dict(showgrid=True, gridcolor='#F4F7FE', tickfont=dict(color=COLOR_TEXT_SEC))
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_side:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"<h4 style='margin:0 0 15px 0; color:{COLOR_TEXT_MAIN}'>Escenarios</h4>", unsafe_allow_html=True)
    
    # TABLA CORREGIDA: Sin indentación dentro del string f""
    table_html = f"""
    <table class="clean-table">
        <thead>
            <tr>
                <th>ESCENARIO</th>
                <th>ROI</th>
                <th>FECHA</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><span class="check-icon">☑</span> Base</td>
                <td>12%</td>
                <td>Oct 24</td>
            </tr>
            <tr>
                <td><span class="check-icon">☑</span> Optimista</td>
                <td>24%</td>
                <td>Nov 12</td>
            </tr>
            <tr>
                <td><span class="check-icon" style="color:#E0E0E0">☐</span> AI Target</td>
                <td>32%</td>
                <td>Pending</td>
            </tr>
        </tbody>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)