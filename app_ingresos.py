import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# --- 1. CONFIGURACIÓN Y ESTILOS GLOBALES ---
st.set_page_config(page_title="Executive AI Dashboard", layout="wide", page_icon="🟣")

# Definición de la Paleta de Colores Horizon UI
COLOR_BG = "#F4F7FE"
COLOR_CARD = "#FFFFFF"
COLOR_TEXT_TITLE = "#2B3674"    # Azul Marino Oscuro (Textos principales)
COLOR_TEXT_BODY = "#A3AED0"     # Gris Azulado (Etiquetas y ejes)
COLOR_ACCENT = "#4318FF"        # Púrpura Eléctrico
COLOR_SUCCESS = "#05CD99"       # Verde Horizon
COLOR_CHART_FILL = "rgba(67, 24, 255, 0.1)" # Púrpura transparente

st.markdown(f"""
<style>
    /* Importar fuente DM Sans (Idéntica a Horizon UI) */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif;
        background-color: {COLOR_BG};
    }}
    
    /* Eliminar padding extra de Streamlit */
    .block-container {{
        padding-top: 1rem;
        padding-bottom: 2rem;
    }}

    /* Estilo de Tarjetas (Card) - Sombra suave y bordes redondeados */
    .horizon-card {{
        background-color: {COLOR_CARD};
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0px 3px 15px rgba(112, 144, 176, 0.08);
        margin-bottom: 20px;
        height: 100%;
    }}

    /* Títulos de Tarjetas */
    .card-title {{
        color: {COLOR_TEXT_TITLE};
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 8px;
    }}
    
    /* Métricas Personalizadas */
    .kpi-label {{
        color: {COLOR_TEXT_BODY};
        font-size: 14px;
        font-weight: 500;
    }}
    .kpi-value {{
        color: {COLOR_TEXT_TITLE};
        font-size: 34px;
        font-weight: 700;
        line-height: 42px;
    }}
    .kpi-icon-box {{
        width: 56px;
        height: 56px;
        border-radius: 50%;
        background-color: #F4F7FE;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
    }}

    /* Tabla Personalizada */
    .styled-table {{
        width: 100%;
        border-collapse: collapse;
    }}
    .styled-table th {{
        text-align: left;
        color: {COLOR_TEXT_BODY};
        font-size: 12px;
        font-weight: 500;
        border-bottom: 1px solid #E9EDF7;
        padding: 12px 0;
        text-transform: uppercase;
    }}
    .styled-table td {{
        padding: 16px 0;
        color: {COLOR_TEXT_TITLE};
        font-weight: 700;
        font-size: 14px;
        border-bottom: 1px solid #E9EDF7;
    }}
    
    /* Ocultar elementos nativos de Streamlit */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

# --- 2. MODELO DE DATOS (BACKEND) ---
@st.cache_resource
def load_data_and_model():
    # Simulación de datos
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'marketing': np.random.normal(5000, 2000, n),
        'competencia': np.random.randint(0, 10, n),
        'region': np.random.choice(['Norte', 'Sur'], n),
        'calidad': np.random.uniform(1, 10, n),
        'antiguedad': np.random.normal(5, 2, n)
    })
    df['ingresos'] = 15000 + 4.2*df['marketing'] + 1200*df['calidad'] - 500*df['competencia'] + np.random.normal(0,2000,n)
    
    # Modelo Simple
    model = Pipeline([
        ('prep', ColumnTransformer([
            ('num', StandardScaler(), ['marketing', 'competencia', 'calidad']),
            ('cat', OneHotEncoder(), ['region'])
        ])),
        ('algo', Lasso(alpha=10))
    ])
    model.fit(df.drop('ingresos', axis=1), df['ingresos'])
    return model

pipeline = load_data_and_model()

# --- 3. SIDEBAR (Controles) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/9320/9320296.png", width=50) # Logo genérico moderno
    st.markdown(f"<h3 style='color:{COLOR_TEXT_TITLE};'>Retail AI Planner</h3>", unsafe_allow_html=True)
    st.write("")
    
    st.markdown(f"<p style='color:{COLOR_TEXT_BODY}; font-size:12px; font-weight:bold; letter-spacing:1px;'>CONTROLES DE NEGOCIO</p>", unsafe_allow_html=True)
    
    marketing = st.slider("💰 Inversión Marketing", 0, 15000, 7500, step=500)
    calidad = st.slider("⭐ Calidad Servicio (NPS)", 1.0, 10.0, 8.2)
    competencia = st.slider("shop Competidores", 0, 10, 2)
    
    st.markdown("---")
    st.info("💡 **AI Tip:** Mantén la calidad sobre 8.0 para maximizar el ROI del marketing.")

# --- 4. CÁLCULOS ---
input_df = pd.DataFrame({'marketing': [marketing], 'competencia': [competencia], 'region': ['Norte'], 'calidad': [calidad], 'antiguedad': [5]})
prediccion = pipeline.predict(input_df)[0]
roi_est = ((prediccion - (8000 + marketing)) / (8000 + marketing)) * 100

# --- 5. INTERFAZ DASHBOARD (GRID SYSTEM) ---

# Encabezado Tipo App
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.markdown(f"<h2 style='color:{COLOR_TEXT_TITLE}; margin:0;'>Dashboard Principal</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{COLOR_TEXT_BODY}; margin:0;'>Bienvenido de nuevo, Director Regional.</p>", unsafe_allow_html=True)
st.write("")

# --- SECCIÓN KPI (TARJETAS HTML) ---
# Función para crear tarjeta KPI estilo Horizon
def kpi_html(icon, label, value, subtext, color_trend):
    return f"""
    <div class="horizon-card" style="padding: 15px 20px;">
        <div style="display:flex; align-items:center;">
            <div class="kpi-icon-box">
                <span style="font-size:24px;">{icon}</span>
            </div>
            <div style="margin-left: 15px;">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div style="font-size:12px; color:{COLOR_TEXT_BODY};">
                    <span style="color:{color_trend}; font-weight:700;">{subtext}</span> vs mes anterior
                </div>
            </div>
        </div>
    </div>
    """

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(kpi_html("💵", "Ingresos Proyectados", f"${prediccion:,.0f}", "+24.5%", COLOR_SUCCESS), unsafe_allow_html=True)
with c2:
    color_roi = COLOR_SUCCESS if roi_est > 0 else "#EE5D50"
    st.markdown(kpi_html("🚀", "ROI Estimado", f"{roi_est:.1f}%", "Healthy", color_roi), unsafe_allow_html=True)
with c3:
    st.markdown(kpi_html("📉", "Eficiencia de Gasto", "High", "+5%", COLOR_ACCENT), unsafe_allow_html=True)

st.write("") # Espaciador

# --- SECCIÓN GRÁFICOS (PLOTLY ESTILIZADO) ---
g1, g2 = st.columns([2, 1])

with g1:
    st.markdown(f"""<div class="horizon-card"><div class="card-title">📈 Sensibilidad de Ingresos</div>""", unsafe_allow_html=True)
    
    # Datos para gráfico
    x_axis = np.linspace(0, 15000, 50)
    y_axis = [pipeline.predict(pd.DataFrame({'marketing': [x], 'competencia': [competencia], 'region': ['Norte'], 'calidad': [calidad], 'antiguedad': [5]}))[0] for x in x_axis]
    
    fig_line = go.Figure()
    
    # Área degradada (Clave para look moderno)
    fig_line.add_trace(go.Scatter(
        x=x_axis, y=y_axis,
        mode='lines',
        line=dict(color=COLOR_ACCENT, width=4, shape='spline'), # 'spline' hace la curva suave
        fill='tozeroy',
        fillcolor=COLOR_CHART_FILL,
        name='Ingresos'
    ))
    
    # Punto actual
    fig_line.add_trace(go.Scatter(
        x=[marketing], y=[prediccion],
        mode='markers',
        marker=dict(color=COLOR_TEXT_TITLE, size=12, symbol='circle', line=dict(width=2, color='white')),
        name='Actual'
    ))

    # --- AQUÍ ESTÁ LA MAGIA PARA QUE SE LEAN LAS LETRAS ---
    fig_line.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(
            showgrid=False, 
            title='Inversión ($)',
            title_font=dict(color=COLOR_TEXT_BODY), # Color Título Eje
            tickfont=dict(color=COLOR_TEXT_BODY)    # Color Números Eje
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor='#E9EDF7', # Grid muy suave
            tickfont=dict(color=COLOR_TEXT_BODY),
            zeroline=False
        ),
        showlegend=False,
        height=300
    )
    st.plotly_chart(fig_line, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with g2:
    st.markdown(f"""<div class="horizon-card"><div class="card-title">💰 Estructura de Costos</div>""", unsafe_allow_html=True)
    
    labels = ['Fijos', 'Marketing', 'Margen']
    values = [8000, marketing, max(0, prediccion - (8000+marketing))]
    colors = ['#EFF4FB', '#A3AED0', COLOR_ACCENT] # Colores Horizon
    
    fig_donut = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values, 
        hole=.65, # Donut más fino
        marker=dict(colors=colors),
        textinfo='none', # Limpio, sin texto encima
        hoverinfo='label+percent'
    )])
    
    fig_donut.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=20, b=0),
        height=300,
        showlegend=True,
        legend=dict(
            orientation="h", 
            yanchor="bottom", y=-0.2, xanchor="center", x=0.5,
            font=dict(color=COLOR_TEXT_TITLE) # Color Leyenda Oscuro
        )
    )
    st.plotly_chart(fig_donut, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- SECCIÓN TABLA COMPLEJA (HTML PURO) ---
# Usamos HTML para simular la tabla "Complex Table" de Horizon
st.markdown(f"""<div class="horizon-card"><div class="card-title">✅ Check Table: Escenarios Sugeridos</div>""", unsafe_allow_html=True)

# Generamos filas dinámicas
rows = ""
scenarios = [
    ("Horizon UI PRO", "17.5%", "2.458", "24.Jan.2021", True),
    ("Escenario Conservador", "10.8%", "1.485", "12.Jun.2021", True),
    ("Escenario Agresivo (IA)", "21.3%", "1.024", "Hoy", False)
]

for name, prog, quant, date, checked in scenarios:
    check_icon = "☑" if checked else "☐"
    rows += f"""
    <tr>
        <td><span style='color:{COLOR_ACCENT}; font-size:16px; margin-right:10px;'>{check_icon}</span> {name}</td>
        <td>{prog}</td>
        <td>{quant}</td>
        <td>{date}</td>
    </tr>
    """

st.markdown(f"""
<table class="styled-table">
    <thead>
        <tr>
            <th>NOMBRE ESCENARIO</th>
            <th>PROGRESO (ROI)</th>
            <th>CANTIDAD ($)</th>
            <th>FECHA</th>
        </tr>
    </thead>
    <tbody>
        {rows}
    </tbody>
</table>
</div>
""", unsafe_allow_html=True)