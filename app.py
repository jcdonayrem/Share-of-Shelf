"""
app.py
Aplicación Streamlit para análisis de Share of Shelf (SoS) en perchas de supermercado.
Ejecutar con: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO

from analyzer import (
    validate_dataframe,
    compute_sos,
    compute_summary,
    compute_facings_heatmap,
    get_template_csv,
    normalize_columns,
)

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Shelf Analyzer | Share of Shelf",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# ESTILOS CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Fuente principal */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #0d4f8c 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .main-header h1 {
        color: #f8fafc;
        font-size: 1.9rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 0.92rem;
        margin: 0.4rem 0 0 0;
    }
    .badge {
        display: inline-block;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 0.7rem;
    }

    /* Métricas */
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        text-align: center;
    }
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1;
    }
    .metric-card .metric-label {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 0.3rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card .metric-sub {
        font-size: 0.82rem;
        color: #0d4f8c;
        font-weight: 600;
        margin-top: 0.15rem;
    }

    /* Sección */
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
        margin: 1.5rem 0 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* Info box */
    .info-box {
        background: #eff6ff;
        border-left: 4px solid #2563eb;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
        color: #1e40af;
        margin: 0.8rem 0;
    }

    /* Template section */
    .template-box {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        font-size: 0.84rem;
        color: #166534;
        margin: 0.8rem 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.78rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e2e8f0;
    }

    /* Streamlit overrides */
    .stFileUploader > div {
        border: 2px dashed #cbd5e1 !important;
        border-radius: 10px !important;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'DM Sans', sans-serif !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="badge">🛒 TRADE MARKETING ANALYTICS</div>
    <h1>Shelf Analyzer — Share of Shelf</h1>
    <p>Visualiza la distribución de espacio en percha por marca y nivel del expositor</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR: CARGA DE ARCHIVO Y CONFIGURACIÓN
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Carga de Datos")

    template_csv = get_template_csv()
    st.download_button(
        label="⬇ Descargar plantilla CSV",
        data=template_csv,
        file_name="plantilla_percha.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown('<div class="template-box">La plantilla incluye: <strong>Marca</strong>, <strong>Nivel</strong>, <strong>Facings</strong>. Nivel 1 = nivel de suelo.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Sube tu archivo CSV o Excel",
        type=["csv", "xlsx", "xls"],
        help="El archivo debe tener columnas: Marca, Nivel, Facings",
    )

    st.markdown("---")
    st.markdown("### ⚙️ Opciones de visualización")

    color_scale = st.selectbox(
        "Escala de color",
        options=["Blues", "YlOrRd", "Greens", "Purples", "RdYlGn", "Viridis"],
        index=0,
    )

    show_values = st.checkbox("Mostrar valores en celdas", value=True)
    show_facings = st.checkbox("Mostrar vista de Facings absolutos", value=False)
    
    st.markdown("---")
    st.markdown("### ℹ️ Metodología")
    st.markdown("""
    **Share of Shelf** = Facings de la marca en ese nivel ÷ Total de facings en ese nivel × 100

    El SoS se calcula **por nivel**, no sobre el total de la percha. Esto permite identificar dominancia localizada.
    """)


# ─────────────────────────────────────────────
# FUNCIÓN: CARGAR DATOS
# ─────────────────────────────────────────────
@st.cache_data
def load_data(file) -> pd.DataFrame:
    """Carga CSV o Excel y retorna DataFrame crudo."""
    if file.name.endswith(".csv"):
        # Detecta separador automáticamente
        content = file.read().decode("utf-8", errors="replace")
        sniffer_sample = content[:2000]
        sep = ";"  if sniffer_sample.count(";") > sniffer_sample.count(",") else ","
        return pd.read_csv(StringIO(content), sep=sep, encoding_errors="replace")
    else:
        return pd.read_excel(file)


# ─────────────────────────────────────────────
# FUNCIÓN: HEATMAP SoS
# ─────────────────────────────────────────────
def build_sos_heatmap(pivot: pd.DataFrame, color_scale: str, show_values: bool) -> go.Figure:
    z = pivot.values
    x_labels = list(pivot.columns)
    y_labels = list(pivot.index)

    # Texto de celdas
    if show_values:
        text = [[f"{v:.1f}%" if v > 0 else "—" for v in row] for row in z]
        texttemplate = "%{text}"
    else:
        text = None
        texttemplate = None

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        colorscale=color_scale,
        zmin=0,
        zmax=100,
        text=text,
        texttemplate=texttemplate,
        textfont={"size": 12, "family": "DM Sans"},
        hoverongaps=False,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Nivel: %{x}<br>"
            "Share of Shelf: <b>%{z:.1f}%</b><extra></extra>"
        ),
        colorbar=dict(
            title=dict(text="SoS (%)", font=dict(size=13)),
            ticksuffix="%",
            thickness=16,
            len=0.85,
        ),
    ))

    fig.update_layout(
        title=dict(
            text="Share of Shelf por Marca y Nivel de Expositor",
            font=dict(size=16, family="DM Sans", color="#0f172a"),
            x=0,
        ),
        xaxis=dict(
            title="Nivel del Expositor",
            side="top",
            tickfont=dict(size=12, family="DM Sans"),
            titlefont=dict(size=13, color="#475569"),
        ),
        yaxis=dict(
            title="Marca",
            autorange="reversed",
            tickfont=dict(size=12, family="DM Sans"),
            titlefont=dict(size=13, color="#475569"),
        ),
        height=max(350, 60 * len(y_labels) + 150),
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="DM Sans"),
    )
    return fig


# ─────────────────────────────────────────────
# FUNCIÓN: HEATMAP FACINGS
# ─────────────────────────────────────────────
def build_facings_heatmap(pivot: pd.DataFrame, color_scale: str, show_values: bool) -> go.Figure:
    z = pivot.values
    x_labels = list(pivot.columns)
    y_labels = list(pivot.index)

    text = [[str(int(v)) if v > 0 else "—" for v in row] for row in z] if show_values else None

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        colorscale=color_scale,
        text=text,
        texttemplate="%{text}" if show_values else None,
        textfont={"size": 12, "family": "DM Sans"},
        hoverongaps=False,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Nivel: %{x}<br>"
            "Facings: <b>%{z}</b><extra></extra>"
        ),
        colorbar=dict(
            title=dict(text="Facings", font=dict(size=13)),
            thickness=16,
            len=0.85,
        ),
    ))

    fig.update_layout(
        title=dict(
            text="Facings Absolutos por Marca y Nivel",
            font=dict(size=16, family="DM Sans", color="#0f172a"),
            x=0,
        ),
        xaxis=dict(title="Nivel del Expositor", side="top",
                   tickfont=dict(size=12), titlefont=dict(size=13, color="#475569")),
        yaxis=dict(title="Marca", autorange="reversed",
                   tickfont=dict(size=12), titlefont=dict(size=13, color="#475569")),
        height=max(350, 60 * len(y_labels) + 150),
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="DM Sans"),
    )
    return fig


# ─────────────────────────────────────────────
# FUNCIÓN: GRÁFICO BARRAS SoS TOTAL
# ─────────────────────────────────────────────
def build_total_sos_bar(df: pd.DataFrame) -> go.Figure:
    df_norm = normalize_columns(df.copy())
    df_norm["Facings"] = pd.to_numeric(df_norm["Facings"])
    total = df_norm["Facings"].sum()
    marca_sos = (
        df_norm.groupby("Marca")["Facings"].sum()
        .reset_index()
        .assign(SoS=lambda x: (x["Facings"] / total * 100).round(1))
        .sort_values("SoS", ascending=True)
    )

    colors = ["#0d4f8c" if i == len(marca_sos) - 1 else "#93c5fd"
              for i in range(len(marca_sos))]

    fig = go.Figure(go.Bar(
        x=marca_sos["SoS"],
        y=marca_sos["Marca"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in marca_sos["SoS"]],
        textposition="outside",
        textfont=dict(size=12, family="DM Sans"),
        hovertemplate="<b>%{y}</b><br>SoS global: %{x:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Share of Shelf Global por Marca", font=dict(size=15, family="DM Sans", color="#0f172a"), x=0),
        xaxis=dict(title="SoS (%)", ticksuffix="%", range=[0, marca_sos["SoS"].max() * 1.2]),
        yaxis=dict(title=""),
        height=max(300, 45 * len(marca_sos) + 120),
        margin=dict(l=20, r=60, t=60, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="DM Sans"),
        bargap=0.35,
    )
    return fig


# ─────────────────────────────────────────────
# FUNCIÓN: SoS POR NIVEL (stacked bar)
# ─────────────────────────────────────────────
def build_nivel_stacked(pivot_sos: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    
    n_marcas = len(pivot_sos.index)
    palette = px.colors.qualitative.Set2 if n_marcas <= 8 else px.colors.qualitative.Alphabet

    for i, marca in enumerate(pivot_sos.index):
        fig.add_trace(go.Bar(
            name=marca,
            x=list(pivot_sos.columns),
            y=list(pivot_sos.loc[marca]),
            marker_color=palette[i % len(palette)],
            hovertemplate=f"<b>{marca}</b><br>%{{x}}<br>SoS: %{{y:.1f}}%<extra></extra>",
        ))

    fig.update_layout(
        barmode="stack",
        title=dict(text="Composición de SoS por Nivel", font=dict(size=15, family="DM Sans", color="#0f172a"), x=0),
        xaxis=dict(title="Nivel del Expositor"),
        yaxis=dict(title="Share of Shelf (%)", ticksuffix="%", range=[0, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380,
        margin=dict(l=20, r=20, t=80, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="DM Sans"),
    )
    return fig


# ─────────────────────────────────────────────
# MAIN: PROCESAMIENTO Y VISUALIZACIÓN
# ─────────────────────────────────────────────
if uploaded_file is None:
    # Estado inicial: instrucciones
    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.markdown('<div class="section-title">📋 ¿Cómo usar esta herramienta?</div>', unsafe_allow_html=True)
        st.markdown("""
        **Paso 1.** Descarga la plantilla CSV desde el panel lateral.
        
        **Paso 2.** Registra los datos de tu percha:
        - **Marca**: nombre de la marca (ej. Coca-Cola, Pepsi)
        - **Nivel**: número del nivel en el expositor (1 = suelo, 2, 3, 4...)
        - **Facings**: número de frentes visibles de cada producto en ese nivel
        
        **Paso 3.** Sube el archivo y obtén el análisis de Share of Shelf.
        
        **Paso 4.** Usa los controles del panel lateral para personalizar los visuales.
        """)

        st.markdown('<div class="info-box">💡 <strong>Tip de Trade:</strong> El "ojo del shopper" se ubica entre el nivel 2 y 3. Una marca que domina SoS en esos niveles tiene ventaja de visibilidad significativa sobre el precio promedio en categoría.</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">📐 Estructura del archivo</div>', unsafe_allow_html=True)
        sample_data = pd.read_csv(StringIO(get_template_csv()))
        st.dataframe(sample_data.head(10), use_container_width=True, hide_index=True)

else:
    # ── CARGA Y VALIDACIÓN ──────────────────
    try:
        df_raw = load_data(uploaded_file)
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
        st.stop()

    is_valid, msg = validate_dataframe(df_raw)
    if not is_valid:
        st.error(f"**Error en los datos:**\n\n{msg}")
        st.stop()

    df_raw = normalize_columns(df_raw)
    df_raw["Facings"] = pd.to_numeric(df_raw["Facings"])

    # ── CÓMPUTOS ────────────────────────────
    pivot_sos = compute_sos(df_raw)
    summary = compute_summary(df_raw)

    # ── MÉTRICAS RESUMEN ────────────────────
    st.markdown('<div class="section-title">📊 Resumen de Percha</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        (c1, str(summary["n_marcas"]), "Marcas en percha", ""),
        (c2, str(summary["n_niveles"]), "Niveles registrados", ""),
        (c3, str(summary["total_facings"]), "Facings totales", ""),
        (c4, summary["marca_lider"], "Marca líder", f"{summary['sos_lider']}% SoS global"),
        (c5, f"Nivel {summary['nivel_top']}", "Nivel más disputado", "Mayor # facings"),
    ]
    for col, val, label, sub in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
                {"<div class='metric-sub'>" + sub + "</div>" if sub else ""}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TABS DE VISUALIZACIÓN ───────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔥 Mapa de Calor SoS",
        "📊 Análisis por Marca",
        "📐 Composición por Nivel",
        "📋 Datos Fuente",
    ])

    with tab1:
        st.plotly_chart(
            build_sos_heatmap(pivot_sos, color_scale, show_values),
            use_container_width=True,
        )

        if show_facings:
            st.markdown("---")
            pivot_facings = compute_facings_heatmap(df_raw)
            st.plotly_chart(
                build_facings_heatmap(pivot_facings, color_scale, show_values),
                use_container_width=True,
            )

        # Exportar heatmap como CSV
        st.markdown("---")
        csv_export = pivot_sos.copy()
        csv_export.index.name = "Marca"
        st.download_button(
            label="⬇ Exportar tabla SoS como CSV",
            data=csv_export.to_csv(float_format="%.2f"),
            file_name="share_of_shelf.csv",
            mime="text/csv",
        )

    with tab2:
        col_bar, col_table = st.columns([1.4, 1])
        with col_bar:
            st.plotly_chart(build_total_sos_bar(df_raw), use_container_width=True)

        with col_table:
            st.markdown('<div class="section-title">Ranking de Marcas</div>', unsafe_allow_html=True)
            df_ranking = (
                df_raw.groupby("Marca")["Facings"].sum()
                .reset_index()
                .assign(SoS=lambda x: (x["Facings"] / x["Facings"].sum() * 100).round(1))
                .sort_values("SoS", ascending=False)
                .reset_index(drop=True)
            )
            df_ranking.index += 1
            df_ranking.columns = ["Marca", "Facings Totales", "SoS Global (%)"]
            st.dataframe(df_ranking, use_container_width=True)

    with tab3:
        st.plotly_chart(build_nivel_stacked(pivot_sos), use_container_width=True)

        st.markdown('<div class="section-title">SoS por Nivel (tabla)</div>', unsafe_allow_html=True)
        display_pivot = pivot_sos.copy()
        display_pivot.index.name = "Marca"
        display_pivot = display_pivot.style.format("{:.1f}%").background_gradient(
            cmap="Blues", axis=None, vmin=0, vmax=100
        )
        st.dataframe(display_pivot, use_container_width=True)

    with tab4:
        st.markdown(f"**{len(df_raw)} registros cargados** — Vista de datos fuente (primeros 200 registros)")
        st.dataframe(df_raw.head(200), use_container_width=True, hide_index=True)

        # Estadísticas básicas
        with st.expander("Ver estadísticas de Facings"):
            st.dataframe(
                df_raw.groupby("Marca")["Facings"].describe().round(1),
                use_container_width=True,
            )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Shelf Analyzer · Share of Shelf Analysis Tool · Trade Marketing Analytics<br>
    Sube tus datos de percha para obtener el análisis
</div>
""", unsafe_allow_html=True)
