"""
app.py — Shelf Share of Shelf Analyzer
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

import easyocr
from analyzer import (
    load_image,
    preprocess_image,
    detect_shelf_levels,
    cluster_brands,
    compute_share_of_shelf,
    build_annotated_image,
)
from charts import (
    make_heatmap,
    make_global_sos_bar,
    make_level_stacked_bar,
    make_brand_color_legend,
)

# ─────────────────────────────────────────────
# OCR READER CACHE
# ─────────────────────────────────────────────

@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(['es', 'en'], gpu=False)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Shelf Analyzer | Share of Shelf",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b3a5c 50%, #1f6096 100%);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }

    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .main-header p {
        font-size: 0.95rem;
        opacity: 0.82;
        margin: 0.4rem 0 0 0;
    }

    .metric-card {
        background: #f0f4f8;
        border-left: 4px solid #1f6096;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }

    .metric-card .label {
        font-size: 0.78rem;
        color: #555;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }

    .metric-card .value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #0d1b2a;
        line-height: 1.2;
    }

    .metric-card .subtitle {
        font-size: 0.8rem;
        color: #888;
    }

    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1a1a2e;
        margin: 1.5rem 0 0.5rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #e0e7ef;
    }

    .info-box {
        background: #e8f4fd;
        border: 1px solid #b3d7f0;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.88rem;
        color: #1a4a6e;
    }

    .warning-box {
        background: #fff8e1;
        border: 1px solid #ffe082;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.88rem;
        color: #5d4037;
    }

    div[data-testid="stTabs"] button {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🛒 Shelf Analyzer · Share of Shelf</h1>
    <p>Análisis automático de distribución de marcas en percha de supermercado</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR — CONFIGURATION
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuración del análisis")

    st.markdown("### 📁 Imagen de la percha")
    uploaded = st.file_uploader(
        "Sube una foto de la percha",
        type=["jpg", "jpeg", "png", "webp"],
        help="Recomendado: foto frontal, bien iluminada, sin ángulo.",
    )

    st.markdown("---")
    st.markdown("### 🔢 Parámetros del expositor")

    n_levels = st.slider(
        "Número de niveles (filas)",
        min_value=2, max_value=10, value=4,
        help="Cuántos estantes tiene el expositor de arriba a abajo.",
    )

    n_cols = st.slider(
        "Columnas por nivel",
        min_value=2, max_value=30, value=10,
        help="Número de 'facings' o posiciones horizontales a analizar.",
    )

    n_brands = st.slider(
        "Número estimado de marcas",
        min_value=2, max_value=15, value=5,
        help="Cuántas marcas distintas estimas que hay en la percha.",
    )

    st.markdown("---")
    st.markdown("### 🏷️ Nombra tus marcas")
    st.caption("Puedes renombrar las marcas detectadas después del análisis.")

    st.markdown("---")
    st.markdown("### 🔧 Opciones avanzadas")
    with st.expander("Mostrar opciones avanzadas"):
        alpha_overlay = st.slider(
            "Opacidad del overlay de color",
            min_value=0.1, max_value=0.7, value=0.35, step=0.05,
        )
        use_edge_detection = st.checkbox(
            "Detección automática de niveles por bordes",
            value=True,
            help="Si está activo, intenta detectar las barras del expositor. Si no, divide uniformemente.",
        )

    st.markdown("---")
    analyze_btn = st.button(
        "🔍 Analizar percha",
        type="primary",
        use_container_width=True,
        disabled=(uploaded is None),
    )

    if uploaded is None:
        st.markdown("""
        <div class="info-box">
            ℹ️ Sube una imagen para comenzar el análisis.
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────

if uploaded is None:
    # Welcome / instructions
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<p class="section-title">¿Cómo funciona?</p>', unsafe_allow_html=True)
        st.markdown("""
        1. **Sube una foto** frontal de la percha del supermercado
        2. **Configura** el número de niveles, columnas y marcas esperadas
        3. **El sistema detecta** automáticamente los niveles usando análisis de bordes
        4. **Agrupa los colores** dominantes en clusters → una marca por cluster
        5. **Calcula el Share of Shelf** por marca y por nivel
        6. **Genera visualizaciones** interactivas con Plotly
        """)

        st.markdown("""
        <div class="warning-box">
        ⚠️ <b>Limitación técnica a considerar:</b> el análisis se basa en color dominante por celda.
        Funciona mejor con perchas donde las marcas tienen colores de packaging claramente diferenciados.
        Para mayor precisión comercial, combina con etiquetado manual post-análisis.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="section-title">Métricas que obtendrás</p>', unsafe_allow_html=True)

        metrics = [
            ("Share of Shelf por marca", "% de espacio total ocupado por cada marca"),
            ("SoS por nivel", "Distribución en cada fila del expositor"),
            ("Mapa de calor interactivo", "Visualización marca × nivel con valor de SoS"),
            ("Imagen anotada", "Overlay de colores sobre la foto original"),
            ("Tabla exportable", "Datos en CSV para análisis posterior"),
        ]

        for title, desc in metrics:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{title}</div>
                <div class="subtitle">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

elif analyze_btn or st.session_state.get("analyzed"):
    # ── Run analysis
    if analyze_btn:
        with st.spinner("Analizando imagen..."):
            try:
                img_bgr_raw = load_image(uploaded)
                img_bgr = preprocess_image(img_bgr_raw)

                if use_edge_detection:
                    levels = detect_shelf_levels(img_bgr, n_levels)
                else:
                    h = img_bgr.shape[0]
                    step = h // n_levels
                    levels = [(i * step, (i + 1) * step) for i in range(n_levels)]

                ocr_reader = get_ocr_reader()
                cluster_result = cluster_brands(
                    img_bgr, levels, n_cols, n_brands, ocr_reader=ocr_reader
                )

                sos_result = compute_share_of_shelf(
                    cluster_result["brand_grid"],
                    cluster_result["brand_names"],
                )

                annotated = build_annotated_image(
                    img_bgr,
                    levels,
                    cluster_result["brand_grid"],
                    cluster_result["brand_colors"],
                    cluster_result["brand_names"],
                    alpha=alpha_overlay,
                )

                st.session_state["analyzed"] = True
                st.session_state["cluster_result"] = cluster_result
                st.session_state["sos_result"] = sos_result
                st.session_state["annotated"] = annotated
                st.session_state["levels"] = levels

            except Exception as e:
                st.error(f"❌ Error durante el análisis: {e}")
                st.stop()

    # ── Retrieve from session
    cluster_result = st.session_state.get("cluster_result")
    sos_result = st.session_state.get("sos_result")
    annotated = st.session_state.get("annotated")
    levels = st.session_state.get("levels")

    if not cluster_result:
        st.warning("Presiona 'Analizar percha' para procesar la imagen.")
        st.stop()

    brand_names = cluster_result["brand_names"]
    brand_colors = cluster_result["brand_colors"]

    # ── Brand rename interface
    st.markdown('<p class="section-title">🏷️ Renombra las marcas detectadas</p>', unsafe_allow_html=True)
    st.caption("Los nombres que asignes aquí se reflejarán en todos los gráficos.")

    rename_cols = st.columns(min(len(brand_names), 5))
    custom_names = {}
    for i, bid in enumerate(sorted(brand_names.keys())):
        col_idx = i % len(rename_cols)
        with rename_cols[col_idx]:
            rgb = brand_colors.get(bid, (128, 128, 128))
            hex_color = f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"
            st.markdown(
                f'<div style="width:18px;height:18px;background:{hex_color};'
                f'border-radius:50%;display:inline-block;margin-right:6px;'
                f'border:1px solid #ccc;vertical-align:middle;"></div>'
                f'<small style="vertical-align:middle;">Color detectado</small>',
                unsafe_allow_html=True,
            )
            new_name = st.text_input(
                f"Sugerencia: {brand_names[bid]}",
                value=brand_names[bid],
                key=f"brand_name_{bid}",
                label_visibility="collapsed",
            )
            custom_names[bid] = new_name

    # Apply custom names
    cluster_result["brand_names"] = custom_names
    sos_result = compute_share_of_shelf(
        cluster_result["brand_grid"], custom_names
    )

    st.markdown("---")

    # ── KPI Row
    st.markdown('<p class="section-title">📊 Resumen ejecutivo</p>', unsafe_allow_html=True)

    top_brand = max(sos_result["sos_total"], key=sos_result["sos_total"].get)
    top_value = sos_result["sos_total"][top_brand]
    n_brands_detected = len(custom_names)
    total_cells = cluster_result["n_levels"] * cluster_result["n_cols"]

    kpi_cols = st.columns(4)
    kpi_data = [
        ("Marca líder en percha", top_brand, f"{top_value:.1f}% del espacio"),
        ("Marcas detectadas", str(n_brands_detected), "clusters de color"),
        ("Celdas analizadas", str(total_cells), f"{cluster_result['n_levels']} niveles × {cluster_result['n_cols']} cols"),
        ("Concentración", f"Top-2: {sorted(sos_result['sos_total'].values(), reverse=True)[:2]}", ""),
    ]

    for col, (label, value, sub) in zip(kpi_cols, kpi_data):
        with col:
            # Format top-2
            if label == "Concentración":
                top2 = sorted(sos_result["sos_total"].values(), reverse=True)[:2]
                value = f"{sum(top2):.1f}%"
                sub = "espacio de las 2 marcas principales"
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value">{value}</div>
                <div class="subtitle">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── TABS
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌡️ Mapa de calor",
        "📸 Imagen anotada",
        "📊 SoS Global",
        "📈 Por nivel",
        "📋 Datos",
    ])

    with tab1:
        st.markdown('<p class="section-title">Mapa de calor — Share of Shelf por Marca y Nivel</p>',
                    unsafe_allow_html=True)
        st.caption(
            "Eje X: niveles del expositor (de arriba a abajo). "
            "Eje Y: marcas detectadas. Color: % de Share of Shelf."
        )
        fig_hm = make_heatmap(
            sos_result["matrix"],
            sos_result["brand_labels"],
            sos_result["level_labels"],
            brand_colors,
            sos_result["all_brands"],
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    with tab2:
        st.markdown('<p class="section-title">Imagen original con overlay de marcas</p>',
                    unsafe_allow_html=True)
        st.caption("Cada color representa una marca detectada. Las líneas blancas delimitan la grilla de análisis.")
        
        col_img, col_legend = st.columns([3, 1])
        with col_img:
            st.image(annotated, use_container_width=True)
        with col_legend:
            st.markdown("**Leyenda de marcas**")
            for bid in sorted(custom_names.keys()):
                rgb = brand_colors.get(bid, (128, 128, 128))
                hex_color = f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"
                sos_val = sos_result["sos_total"].get(custom_names[bid], 0)
                st.markdown(
                    f'<div style="display:flex;align-items:center;margin-bottom:8px;">'
                    f'<div style="width:22px;height:22px;background:{hex_color};'
                    f'border-radius:4px;margin-right:8px;border:1px solid #ddd;flex-shrink:0;"></div>'
                    f'<div><strong style="font-size:0.85rem;">{custom_names[bid]}</strong>'
                    f'<br><small style="color:#666;">{sos_val}%</small></div></div>',
                    unsafe_allow_html=True,
                )

    with tab3:
        st.markdown('<p class="section-title">Share of Shelf Global por Marca</p>',
                    unsafe_allow_html=True)
        fig_bar = make_global_sos_bar(
            sos_result["sos_total"],
            brand_colors,
            sos_result["all_brands"],
            custom_names,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab4:
        st.markdown('<p class="section-title">Composición de SoS por Nivel del Expositor</p>',
                    unsafe_allow_html=True)
        fig_stacked = make_level_stacked_bar(
            sos_result["matrix"],
            sos_result["brand_labels"],
            sos_result["level_labels"],
            brand_colors,
            sos_result["all_brands"],
        )
        st.plotly_chart(fig_stacked, use_container_width=True)

    with tab5:
        st.markdown('<p class="section-title">Tabla de datos — Share of Shelf por Marca y Nivel</p>',
                    unsafe_allow_html=True)

        # Build dataframe
        df_data = {}
        df_data["Marca"] = sos_result["brand_labels"]
        df_data["SoS Global (%)"] = [
            sos_result["sos_total"].get(bl, 0) for bl in sos_result["brand_labels"]
        ]
        for li, lv_label in enumerate(sos_result["level_labels"]):
            df_data[lv_label] = np.round(sos_result["matrix"][:, li], 1)

        df = pd.DataFrame(df_data).sort_values("SoS Global (%)", ascending=False)
        df = df.reset_index(drop=True)

        st.dataframe(
            df.style.format({col: "{:.1f}%" for col in df.columns if col != "Marca"}),
            use_container_width=True,
            height=min(400, (len(df) + 1) * 40 + 40),
        )

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Descargar datos en CSV",
            data=csv,
            file_name="share_of_shelf_analysis.csv",
            mime="text/csv",
        )

        st.markdown('<p class="section-title">Datos brutos — Grilla por celda</p>',
                    unsafe_allow_html=True)
        st.caption("Cada celda muestra la marca asignada por el algoritmo de clustering.")

        grid_data = {}
        brand_grid = cluster_result["brand_grid"]
        for li, level_label in enumerate(sos_result["level_labels"]):
            grid_data[level_label] = [
                custom_names.get(brand_grid[li][ci], "?")
                for ci in range(cluster_result["n_cols"])
            ]

        df_grid = pd.DataFrame(grid_data)
        df_grid.index = [f"Col {i+1}" for i in range(cluster_result["n_cols"])]
        st.dataframe(df_grid, use_container_width=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#aaa;font-size:0.78rem;'>"
    "Shelf Analyzer v1.0 · Análisis basado en clustering de color (K-Means) · "
    "Para uso en trade marketing y category management"
    "</div>",
    unsafe_allow_html=True,
)
