"""
charts.py
Generates all interactive Plotly visualizations:
  - Share of Shelf heatmap (brand × level)
  - Global SoS bar chart with brand colors
  - SoS stacked bar by level
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd


def _rgb_to_hex(rgb_tuple) -> str:
    r, g, b = [int(x) for x in rgb_tuple]
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def make_heatmap(
    matrix: np.ndarray,
    brand_labels: list,
    level_labels: list,
    brand_colors: dict,
    all_brands: list,
) -> go.Figure:
    """
    Interactive heatmap: X = Shelf Levels, Y = Brands, Color = SoS %.
    Each brand row uses its detected color palette for the colorscale.
    Uses a single diverging colorscale for the full matrix.
    """
    # Round for display
    z_rounded = np.round(matrix, 1)

    # Custom hover text
    hover = [
        [
            f"<b>{brand_labels[bi]}</b><br>Nivel: {level_labels[li]}<br>SoS: {z_rounded[bi, li]:.1f}%"
            for li in range(matrix.shape[1])
        ]
        for bi in range(matrix.shape[0])
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_rounded,
            x=level_labels,
            y=brand_labels,
            text=z_rounded,
            texttemplate="%{text:.1f}%",
            textfont={"size": 13, "color": "white"},
            hovertext=hover,
            hoverinfo="text",
            colorscale=[
                [0.0,  "#0d1b2a"],
                [0.2,  "#1b3a5c"],
                [0.4,  "#1f6096"],
                [0.6,  "#2196c4"],
                [0.8,  "#4fc3e8"],
                [1.0,  "#e8f9ff"],
            ],
            colorbar=dict(
                title=dict(text="Share of<br>Shelf (%)", font=dict(size=12)),
                ticksuffix="%",
                thickness=16,
                len=0.8,
            ),
            zmin=0,
            zmax=100,
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Share of Shelf · Mapa de Calor</b><br>"
                 "<sup>Distribución por Marca y Nivel del Expositor</sup>",
            x=0.5,
            font=dict(size=18, color="#1a1a2e"),
        ),
        xaxis=dict(
            title="Nivel del Expositor",
            tickfont=dict(size=12),
            side="bottom",
        ),
        yaxis=dict(
            title="Marca",
            tickfont=dict(size=12),
            autorange="reversed",
        ),
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="#ffffff",
        margin=dict(l=120, r=80, t=100, b=60),
        height=max(350, len(brand_labels) * 60 + 150),
    )

    return fig


def make_global_sos_bar(sos_total: dict, brand_colors: dict, all_brands: list, brand_names: dict) -> go.Figure:
    """Horizontal bar chart: global Share of Shelf per brand."""
    brands = [brand_names[b] for b in all_brands]
    values = [sos_total.get(brand_names[b], 0) for b in all_brands]
    colors = [_rgb_to_hex(brand_colors.get(b, (100, 100, 200))) for b in all_brands]

    # Sort descending
    sorted_pairs = sorted(zip(values, brands, colors), key=lambda x: x[0])
    values_s, brands_s, colors_s = zip(*sorted_pairs) if sorted_pairs else ([], [], [])

    fig = go.Figure(
        go.Bar(
            x=list(values_s),
            y=list(brands_s),
            orientation="h",
            marker=dict(
                color=list(colors_s),
                line=dict(color="rgba(0,0,0,0.2)", width=1),
            ),
            text=[f"{v:.1f}%" for v in values_s],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>SoS Global: %{x:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Share of Shelf Global</b><br><sup>Participación total en la percha</sup>",
            x=0.5,
            font=dict(size=16, color="#1a1a2e"),
        ),
        xaxis=dict(
            title="Share of Shelf (%)",
            range=[0, max(list(values_s) or [100]) * 1.25],
            ticksuffix="%",
        ),
        yaxis=dict(title="Marca"),
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="#ffffff",
        margin=dict(l=100, r=80, t=90, b=50),
        height=max(300, len(brands) * 50 + 150),
        showlegend=False,
    )

    return fig


def make_level_stacked_bar(
    matrix: np.ndarray,
    brand_labels: list,
    level_labels: list,
    brand_colors: dict,
    all_brands: list,
) -> go.Figure:
    """Stacked 100% bar chart: SoS composition per shelf level."""
    fig = go.Figure()

    for bi, (brand_label, brand_id) in enumerate(zip(brand_labels, all_brands)):
        color_hex = _rgb_to_hex(brand_colors.get(brand_id, (150, 150, 150)))
        fig.add_trace(
            go.Bar(
                name=brand_label,
                x=level_labels,
                y=matrix[bi, :],
                marker_color=color_hex,
                hovertemplate=(
                    f"<b>{brand_label}</b><br>%{{x}}<br>SoS: %{{y:.1f}}%<extra></extra>"
                ),
                text=[f"{v:.0f}%" if v > 5 else "" for v in matrix[bi, :]],
                textposition="inside",
                textfont=dict(color="white", size=11),
            )
        )

    fig.update_layout(
        barmode="stack",
        title=dict(
            text="<b>Composición de SoS por Nivel</b><br>"
                 "<sup>Participación de cada marca en cada nivel del expositor</sup>",
            x=0.5,
            font=dict(size=16, color="#1a1a2e"),
        ),
        xaxis=dict(title="Nivel del Expositor"),
        yaxis=dict(title="Share of Shelf (%)", ticksuffix="%", range=[0, 105]),
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="#ffffff",
        margin=dict(l=60, r=60, t=100, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.35,
            xanchor="center",
            x=0.5,
        ),
        height=420,
    )

    return fig


def make_brand_color_legend(brand_colors: dict, brand_names: dict) -> go.Figure:
    """Small color swatch legend showing detected brand colors."""
    n = len(brand_names)
    colors = [_rgb_to_hex(brand_colors.get(b, (128, 128, 128))) for b in sorted(brand_names)]
    labels = [brand_names[b] for b in sorted(brand_names)]

    fig = go.Figure()
    for i, (label, color) in enumerate(zip(labels, colors)):
        fig.add_trace(
            go.Bar(
                x=[1],
                y=[label],
                orientation="h",
                marker_color=color,
                showlegend=False,
                hovertemplate=f"<b>{label}</b><br>Color detectado: {color}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text="<b>Colores Detectados por Marca</b>", x=0.5, font=dict(size=14)),
        xaxis=dict(visible=False),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="#ffffff",
        margin=dict(l=80, r=20, t=50, b=20),
        height=max(200, n * 40 + 80),
        barmode="overlay",
    )

    return fig
