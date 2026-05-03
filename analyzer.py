"""
analyzer.py
Core engine: detects shelf levels, segments cells, clusters colors into brands,
and computes Share of Shelf per brand per level.
"""

import numpy as np
import cv2
from collections import Counter
from sklearn.cluster import KMeans
from PIL import Image


# ─────────────────────────────────────────────
# 1. IMAGE LOADING
# ─────────────────────────────────────────────

def load_image(uploaded_file) -> np.ndarray:
    """Convert Streamlit UploadedFile → OpenCV BGR array."""
    img = Image.open(uploaded_file).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────
# 2. SHELF LEVEL DETECTION
# ─────────────────────────────────────────────

def detect_shelf_levels(img_bgr: np.ndarray, n_levels: int) -> list:
    """
    Returns a list of (y_start, y_end) tuples, one per shelf level.
    Uses horizontal edge density to find shelf separators; falls back to
    uniform partition when the image lacks clear horizontal lines.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    sobel_h = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    row_energy = np.abs(sobel_h).mean(axis=1)

    kernel_size = max(5, h // 40)
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(row_energy, kernel, mode="same")

    sorted_rows = np.argsort(smoothed)[::-1]
    min_gap = h // (n_levels * 2)
    separators = []
    for r in sorted_rows:
        if all(abs(r - s) > min_gap for s in separators):
            separators.append(int(r))
        if len(separators) == n_levels - 1:
            break

    separators = sorted(separators)
    boundaries = [0] + separators + [h]

    levels = []
    for i in range(len(boundaries) - 1):
        levels.append((boundaries[i], boundaries[i + 1]))

    # Fallback to uniform split
    if len(levels) != n_levels:
        step = h // n_levels
        levels = [(i * step, (i + 1) * step) for i in range(n_levels)]

    return levels


# ─────────────────────────────────────────────
# 3. COLOR EXTRACTION PER CELL
# ─────────────────────────────────────────────

def extract_dominant_colors(img_bgr: np.ndarray, n_colors: int = 1) -> np.ndarray:
    """Returns the n dominant colors (RGB) in a region using K-Means."""
    data = img_bgr.reshape(-1, 3).astype(np.float32)

    brightness = data.mean(axis=1)
    mask = (brightness > 30) & (brightness < 225)
    filtered = data[mask]

    if len(filtered) < n_colors * 10:
        return np.array([[128, 128, 128]])

    k = min(n_colors, len(filtered))
    km = KMeans(n_clusters=k, n_init=5, random_state=42)
    km.fit(filtered)

    counts = np.bincount(km.labels_)
    order = np.argsort(counts)[::-1]
    centers = km.cluster_centers_[order]

    return centers[:, ::-1].astype(int)  # BGR → RGB


# ─────────────────────────────────────────────
# 4. BRAND CLUSTERING ACROSS THE SHELF
# ─────────────────────────────────────────────

def cluster_brands(
    img_bgr: np.ndarray,
    levels: list,
    n_cols: int,
    n_brands: int,
    colors_per_cell: int = 2,
) -> dict:
    """
    Divides the shelf into a grid (levels x columns), extracts dominant colors,
    globally clusters all colors into n_brands, and returns analysis data.
    """
    h, w = img_bgr.shape[:2]
    col_edges = [int(i * w / n_cols) for i in range(n_cols + 1)]

    cell_colors = []
    for y0, y1 in levels:
        row_colors = []
        for ci in range(n_cols):
            x0, x1 = col_edges[ci], col_edges[ci + 1]
            cell = img_bgr[y0:y1, x0:x1]
            dominant = extract_dominant_colors(cell, n_colors=colors_per_cell)
            row_colors.append(dominant[0])
        cell_colors.append(row_colors)

    all_colors = np.array([c for row in cell_colors for c in row], dtype=np.float32)

    n_clusters = min(n_brands, len(all_colors))
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(all_colors)

    brand_grid = []
    idx = 0
    for row in cell_colors:
        brand_row = []
        for _ in row:
            brand_row.append(int(labels[idx]))
            idx += 1
        brand_grid.append(brand_row)

    brand_colors = {}
    for bid in range(n_clusters):
        center_rgb = km.cluster_centers_[bid][::-1]
        brand_colors[bid] = tuple(center_rgb.astype(int))

    brand_names = {bid: f"Marca {bid + 1}" for bid in range(n_clusters)}

    return {
        "brand_grid": brand_grid,
        "brand_colors": brand_colors,
        "brand_names": brand_names,
        "cell_colors": cell_colors,
        "n_levels": len(levels),
        "n_cols": n_cols,
    }


# ─────────────────────────────────────────────
# 5. SHARE OF SHELF CALCULATION
# ─────────────────────────────────────────────

def compute_share_of_shelf(brand_grid: list, brand_names: dict) -> dict:
    """
    Returns matrix, labels and SoS percentages per brand and level.
    """
    n_levels = len(brand_grid)
    n_cols = len(brand_grid[0]) if brand_grid else 0
    total_cells = n_levels * n_cols

    all_brands = sorted(brand_names.keys())
    level_labels = [f"Nivel {i + 1}" for i in range(n_levels)]
    brand_labels = [brand_names[b] for b in all_brands]

    # Matrix: rows = brands, cols = levels
    matrix = np.zeros((len(all_brands), n_levels))

    for li, row in enumerate(brand_grid):
        level_counts = {b: 0 for b in all_brands}
        for cell_brand in row:
            if cell_brand in level_counts:
                level_counts[cell_brand] += 1
        for bi, brand_id in enumerate(all_brands):
            count = level_counts[brand_id]
            matrix[bi, li] = (count / n_cols * 100) if n_cols > 0 else 0

    flat = [cell for row in brand_grid for cell in row]
    counts = Counter(flat)
    sos_total = {
        brand_names[b]: round(counts.get(b, 0) / total_cells * 100, 1)
        for b in all_brands
    }

    return {
        "matrix": matrix,
        "brand_labels": brand_labels,
        "level_labels": level_labels,
        "sos_total": sos_total,
        "all_brands": all_brands,
    }


# ─────────────────────────────────────────────
# 6. ANNOTATED PREVIEW IMAGE
# ─────────────────────────────────────────────

def build_annotated_image(
    img_bgr: np.ndarray,
    levels: list,
    brand_grid: list,
    brand_colors: dict,
    brand_names: dict,
    alpha: float = 0.35,
) -> np.ndarray:
    """
    Returns an RGB image with semi-transparent brand color overlays
    and level grid lines drawn on the original shelf photo.
    """
    h, w = img_bgr.shape[:2]
    n_cols = len(brand_grid[0]) if brand_grid else 1
    col_edges = [int(i * w / n_cols) for i in range(n_cols + 1)]

    overlay = img_bgr.copy()

    for li, (y0, y1) in enumerate(levels):
        for ci in range(n_cols):
            x0, x1 = col_edges[ci], col_edges[ci + 1]
            brand_id = brand_grid[li][ci]
            rgb = brand_colors.get(brand_id, (128, 128, 128))
            bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
            cv2.rectangle(overlay, (x0, y0), (x1, y1), bgr, -1)

    blended = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)

    for y0, y1 in levels:
        cv2.line(blended, (0, y0), (w, y0), (255, 255, 255), 2)
    cv2.line(blended, (0, h - 1), (w, h - 1), (255, 255, 255), 2)
    for x in col_edges:
        cv2.line(blended, (x, 0), (x, h), (255, 255, 255), 1)

    for li, (y0, y1) in enumerate(levels):
        for ci in range(n_cols):
            x0, x1 = col_edges[ci], col_edges[ci + 1]
            brand_id = brand_grid[li][ci]
            label = brand_names.get(brand_id, "?")
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            font_scale = max(0.3, min(0.6, (x1 - x0) / 120))
            cv2.putText(
                blended, label, (cx - 20, cy),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA
            )

    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
