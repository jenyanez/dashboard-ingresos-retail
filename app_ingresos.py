import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- 1. CONFIGURACIÓN DE PÁGINA Y ESTILO "HORIZON UI" ---
st.set_page_config(
    page_title="Executive AI Dashboard", 
    layout="wide", 
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Constantes de Color (Horizon Palette)
COLOR_BG = "#F4F7FE"           # Fondo General (Gris muy claro)
COLOR_WHITE = "#FFFFFF"        # Fondo Tarjetas
COLOR_TEXT_MAIN = "#2B3674"    # Azul Marino (Títulos y Valores)
COLOR_TEXT_SEC = "#A3AED0"     # Gris Azulado (Etiquetas)
COLOR_ACCENT = "#4318FF"       # Púrpura Eléctrico (Principal)
COLOR_SUCCESS = "#05CD99"      # Verde Éxito
COLOR_DANGER = "#EE5D50"       # Rojo Alerta

# Inyección de CSS (Estilo Global)
st.markdown(f"""
<style>
    /* Importar fuente Google Fonts: DM Sans */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
    
    /* Reset General */
    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif;
        color: {COLOR_TEXT_MAIN};
        background-color: {COLOR_BG};
    }}

    /* Ocultar elementos nativos molestos */
    header {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .stApp {{ background-color: {COLOR_BG}; }}
    
    /* Estilo Tarjeta (Card) */
    .card {{
        background-color: {COLOR_WHITE};
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0px 3px 15px rgba(112, 144, 176, 0.08);
        margin-bottom: 20px;
    }}
    
    /* Tipografía Corporativa */
    h1, h2, h3, h4 {{ color: {COLOR_TEXT_MAIN} !important; font-weight: 700; }}
    p {{ color: {COLOR_TEXT_SEC}; }}
    
    /* Estilo para los KPIs HTML */
    .kpi-box {{ display: flex; flex-direction: column; }}
    .kpi-title {{ font-size: 14px; color: {COLOR_TEXT_SEC}; font-weight: 500; margin-bottom: 4px; }}
    .kpi-value {{ font-size: 32px; color: {COLOR_TEXT_MAIN}; font-weight: 700; line-height: 1.2; }}
    .kpi-delta {{ font-size: 13px; font-weight: 700; display: flex; align-items: center; gap: 5px; margin-top: 5px; }}
    
    /* Estilo de Tabla Limpia */
    .clean-table {{ width: 100%; border-collapse: collapse; }}
    .clean-table th {{ 
        text-align: left; 
        color: {COLOR_TEXT_SEC}; 
        font-size: 11px; 
        text-transform: uppercase; 
        letter-spacing: 0.5px;
        border-bottom: 1px solid #E9EDF7; 
        padding-bottom: 12px;
    }}
    .clean-table td {{ 
        padding: 16px 0; 
        color: {COLOR_TEXT_MAIN}; 
        font-weight: 600; 
        font-size: 14px; 
        border-bottom: 1px solid #F4F7FE;
    }}
    
    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {COLOR_WHITE};
        border-right: 1px solid #E9EDF7;
    }}
    
    /* Botón Simulado */
    .btn-apply {{
        background-color: {COLOR_ACCENT}; 
        color: white; 
        border: none; 
        padding: 10px 20px; 
        border-radius: 12px; 
        font-size: 13px; 
        font-weight: bold;
        cursor: pointer;
        width: 100%;
    }}
</style>
""", unsafe_allow_html=True)

# --- 2. BACKEND: MODELO DE IA (Simulado) ---
@st.cache_resource
def load_model():
    # Generación de datos sintéticos
    np.random.seed(42)
    df = pd.DataFrame({
        'mkt': np.random.normal(5000, 2000, 500),
        'comp': np.random.randint(0, 10, 500),
        'reg': np.random.choice(['Norte', 'Sur'], 500),
        'cal': np.random.uniform(1, 10, 500)
    })
    # Fórmula: Ingresos = Base + Mkt*Impacto + Calidad*Impacto - Comp*Impacto
    df['y'] = 12000 + df['mkt']*3.5 + df['cal']*1200 - df['comp']*600 + np.random.normal(0, 1500, 500)
    
    # Pipeline Lasso
    pipe = Pipeline([
        ('prep', ColumnTransformer([
            ('num', StandardScaler(), ['mkt','comp','cal']),
            ('cat', OneHotEncoder(), ['reg'])
        ])),
        ('model', Lasso(alpha=10))
    ])
    pipe.fit(df.drop('y', axis=1), df['y'])
    return pipe

model = load_model()

# --- 3. SIDEBAR (Controles Ejecutivos) ---
with st.sidebar:
    st.markdown(f"<h3 style='text-align:center; color:{COLOR_ACCENT}; margin-bottom:30px;'>⚡ Retail AI Planner</h3>", unsafe_allow_html=True)
    
    st.markdown(f"<p style='font-size:12px; font-weight:700; color:{COLOR_TEXT_SEC}; letter-spacing:1px;'>PALANCAS DE GESTIÓN</p>", unsafe_allow_html=True)
    
    mkt = st.slider("Presupuesto Diario Marketing", 0, 15000, 6500, step=500, format="$%d")
    calidad = st.slider("Calidad de Servicio (NPS)", 1.0, 10.0, 7.8)
    comp = st.slider("Competencia en Zona (Tiendas)", 0, 10, 3)
    
    st.markdown("---")
    st.markdown(f"<div style='background-color:{COLOR_BG}; padding:15px; border-radius:12px;'><p style='margin:0; font-size:12px; color:{COLOR_TEXT_MAIN}'><b>💡 Nota Técnica:</b><br>Modelo actualizado: Nov 2025<br>Precisión (R2): 87%</p></div>", unsafe_allow_html=True)

# --- 4. CÁLCULOS DE NEGOCIO ---
input_data = pd.DataFrame({'mkt': [mkt], 'comp': [comp], 'reg': ['Norte'], 'cal': [calidad]})
pred_ingreso = model.predict(input_data)[0]

# KPIs Derivados
target_ingreso = 45000
costos_base = 8500
costo_total = costos_base + mkt
utilidad = pred_ingreso - costo_total
roi = (utilidad / costo_total) * 100

# Colores dinámicos para KPIs
color_roi = COLOR_SUCCESS if roi > 15 else (COLOR_ACCENT if roi > 0 else COLOR_DANGER)
delta_vs_target = ((pred_ingreso - target_ingreso)/target_ingreso)*100

# --- 5. DASHBOARD PRINCIPAL (GRID LAYOUT) ---

st.markdown("## Tablero de Control Estratégico")
st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

# ---> FILA 1: TARJETAS KPI (HTML) <---
c1, c2, c3 = st.columns(3)

def kpi_html(title, value, delta_val, delta_text, color_delta):
    delta_symbol = "▲" if delta_val >= 0 else "▼"
    return f"""
    <div class="card">
        <div class="kpi-box">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-delta" style="color: {color_delta};">
                {delta_symbol} {abs(delta_val):.1f}% <span style="color: {COLOR_TEXT_SEC}; font-weight: 400;">{delta_text}</span>
            </div>
        </div>
    </div>
    """

with c1: 
    color = COLOR_SUCCESS if delta_vs_target >= -10 else COLOR_DANGER
    st.markdown(kpi_html("Ingresos Proyectados", f"${pred_ingreso:,.0f}", delta_vs_target, "vs Objetivo", color), unsafe_allow_html=True)

with c2: 
    st.markdown(kpi_html("ROI Estimado (Día)", f"{roi:.1f}%", roi-10, "vs Promedio Ind.", color_roi), unsafe_allow_html=True)

with c3: 
    efficiency = pred_ingreso / mkt if mkt > 0 else 0
    st.markdown(kpi_html("Eficiencia Marketing (ROAS)", f"{efficiency:.1f}x", 5.2, "Retorno por $ invertido", COLOR_ACCENT), unsafe_allow_html=True)


# ---> FILA 2: GRÁFICO CENTRAL Y TABLA DE ACCIÓN <---
col_main, col_side = st.columns([2, 1])

with col_main:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"<h4>📉 Análisis de Brecha & Potencial</h4>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:13px;'>Proyección de ingresos basada en la inversión publicitaria, manteniendo la calidad actual.</p>", unsafe_allow_html=True)
    
    # --- GRÁFICO PLOTLY CONTEXTUALIZADO ---
    x_axis = np.linspace(0, 15000, 50)
    y_axis = [model.predict(pd.DataFrame({'mkt': [i], 'comp': [comp], 'reg': ['Norte'], 'cal': [calidad]}))[0] for i in x_axis]
    
    fig = go.Figure()
    
    # 1. Curva de Ingresos (Area)
    fig.add_trace(go.Scatter(
        x=x_axis, y=y_axis, mode='lines', name='Potencial Ingresos',
        line=dict(color=COLOR_ACCENT, width=3),
        fill='tozeroy', fillcolor='rgba(67, 24, 255, 0.05)'
    ))
    
    # 2. Línea de Objetivo (Target)
    fig.add_trace(go.Scatter(
        x=[0, 15000], y=[target_ingreso, target_ingreso], mode='lines', name='Objetivo Corp.',
        line=dict(color=COLOR_SUCCESS, width=2, dash='dash')
    ))
    
    # 3. Punto Actual
    fig.add_trace(go.Scatter(
        x=[mkt], y=[pred_ingreso], mode='markers', name='Tu Selección',
        marker=dict(color=COLOR_TEXT_MAIN, size=14, line=dict(color='white', width=2))
    ))
    
    # Configuración limpia (Sin grids molestos, colores correctos)
    fig.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            title="Inversión Marketing ($)", 
            title_font=dict(size=12, color=COLOR_TEXT_SEC),
            showgrid=False, 
            tickfont=dict(color=COLOR_TEXT_SEC)
        ),
        yaxis=dict(
            title="Ingresos ($)", 
            showgrid=True, gridcolor='#F4F7FE', 
            tickfont=dict(color=COLOR_TEXT_SEC),
            zeroline=False
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Insight Narrativo (Caja Gris)
    gap = target_ingreso - pred_ingreso
    if gap > 0:
        msg_ia = f"⚠️ <b>Atención:</b> Estás a <b>${gap:,.0f}</b> del objetivo. Considera aumentar el marketing a ${mkt + 2000:,.0f} o subir la calidad."
    else:
        msg_ia = f"✅ <b>Excelente:</b> Superas el objetivo por <b>${abs(gap):,.0f}</b>. Es un buen momento para optimizar márgenes."
        
    st.markdown(f"""
    <div style='background-color:{COLOR_BG}; padding:12px; border-radius:10px; display:flex; align-items:center;'>
        <span style='font-size:14px; color:{COLOR_TEXT_MAIN};'>{msg_ia}</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col_side:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"<h4>📋 Plan de Acción</h4>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:13px; margin-bottom:15px;'>Estrategias sugeridas para cerrar la brecha.</p>", unsafe_allow_html=True)
    
    # --- TABLA HTML ESTÁTICA (Accionable) ---
    table_content = f"""
    <table class="clean-table">
        <thead>
            <tr>
                <th style="width:40%">ESTRATEGIA</th>
                <th>ACCIÓN</th>
                <th style="text-align:right">IMPACTO</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><span style="color:{COLOR_ACCENT}; margin-right:5px;">●</span> Actual</td>
                <td>Mantener</td>
                <td style="text-align:right; color:{COLOR_TEXT_SEC}">--</td>
            </tr>
            <tr>
                <td><span style="color:{COLOR_TEXT_SEC}; margin-right:5px;">○</span> Calidad+</td>
                <td>Subir a 9.0</td>
                <td style="text-align:right; color:{COLOR_SUCCESS}; font-weight:700">+12%</td>
            </tr>
            <tr>
                <td><span style="color:{COLOR_TEXT_SEC}; margin-right:5px;">○</span> Agresivo</td>
                <td>Mkt $10k</td>
                <td style="text-align:right; color:{COLOR_SUCCESS}; font-weight:700">+25%</td>
            </tr>
        </tbody>
    </table>
    <br>
    <button class="btn-apply">Aplicar Mejor Escenario</button>
    """
    st.markdown(table_content, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)