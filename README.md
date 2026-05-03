# 🛒 Shelf Analyzer — Share of Shelf

Herramienta de análisis automático de perchas de supermercado.
Detecta marcas por color dominante, calcula Share of Shelf por marca
y nivel del expositor, y genera visualizaciones interactivas.

---

## Instalación

```bash
# 1. Clonar o descomprimir el proyecto
cd shelf_analyzer

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate.bat     # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la aplicación
streamlit run app.py
```

La app se abrirá automáticamente en `http://localhost:8501`

---

## Cómo funciona

### Pipeline de análisis

```
Imagen → Detección de niveles → Grilla de celdas → Color dominante por celda
       → K-Means global → Asignación de marca por celda
       → Share of Shelf (%) por marca × nivel
       → Visualizaciones interactivas
```

### Pasos técnicos

1. **Detección de niveles**: Análisis de energía horizontal (Sobel Y) para
   detectar las barras del expositor. Fallback a partición uniforme.

2. **Extracción de color**: K-Means local por celda, filtrando píxeles
   muy oscuros o muy claros (estructura del mueble, no producto).

3. **Clustering global de marcas**: K-Means sobre todos los colores
   dominantes de las celdas → N clusters = N marcas estimadas.

4. **Cálculo de SoS**: Conteo de celdas por marca por nivel.
   SoS(marca, nivel) = celdas_marca_en_nivel / total_celdas_en_nivel × 100

---

## Estructura de archivos

```
shelf_analyzer/
├── app.py           — Interfaz Streamlit principal
├── analyzer.py      — Motor de análisis (CV + clustering)
├── charts.py        — Visualizaciones Plotly
├── requirements.txt — Dependencias Python
└── README.md        — Este archivo
```

---

## Limitaciones técnicas

| Limitación | Impacto | Mitigación |
|------------|---------|------------|
| Análisis por color | Marcas con colores similares se confunden | Ajustar N de marcas; renombrar post-análisis |
| No reconoce logos/texto | No puede leer nombres de marcas | Uso manual del renombrador en UI |
| Sensible a iluminación | Luz desigual afecta los colores | Foto con luz uniforme, frontal |
| Grid uniforme por nivel | No detecta facings variables | Ajustar columnas en configuración |

Para análisis de precisión comercial: combinar con etiquetado manual
o con modelos de reconocimiento de logotipos (Google Vision API, etc.).

---

## Mejoras posibles (roadmap)

- [ ] Integración con Google Cloud Vision para reconocimiento de logotipos
- [ ] Exportación a PDF con informe completo
- [ ] Comparación de dos perchas (antes/después, tienda A vs B)
- [ ] Detección automática de precio de etiquetas con OCR
- [ ] Base de datos de marcas por categoría

---

## Uso comercial sugerido

- **Trade marketing**: auditoría de ejecución en PDV
- **Category management**: análisis de distribución vs planograma
- **Shopper insights**: visibilidad de marca por nivel de eye-level
- **Negociación con retail**: evidencia de cumplimiento de acuerdos de espacio
