# Shelf Analyzer — Share of Shelf

Herramienta de análisis de perchas de supermercado para Trade Marketing.

## Instalación rápida

```bash
# 1. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar
streamlit run app.py
```

## Estructura del proyecto

```
shelf_analyzer/
├── app.py              # Aplicación Streamlit principal
├── analyzer.py         # Lógica de negocio y cálculos
├── datos_ejemplo.csv   # Dataset de prueba con marcas de bebidas
├── requirements.txt    # Dependencias
└── README.md
```

## Formato de datos

El archivo CSV/Excel debe tener exactamente estas tres columnas:

| Columna   | Tipo    | Descripción                                  |
|-----------|---------|----------------------------------------------|
| `Marca`   | Texto   | Nombre de la marca                           |
| `Nivel`   | Entero  | Nivel del expositor (1 = suelo, sube hacia arriba) |
| `Facings` | Entero  | Número de frentes visibles en ese nivel      |

**Aliases aceptados:** `brand/marca`, `level/nivel/shelf/fila/piso`, `facings/frentes/facing`

### Ejemplo válido

```csv
Marca,Nivel,Facings
Coca-Cola,1,12
Coca-Cola,2,16
Pepsi,1,8
Pepsi,2,10
```

## Funcionalidades

- **Mapa de calor SoS**: Share of Shelf por marca × nivel, con escala de color configurable
- **Vista de facings absolutos**: volumen real de frentes por celda (opcional)
- **Ranking global de marcas**: SoS total de la percha
- **Composición por nivel**: barras apiladas para ver batalla por nivel
- **Exportación CSV**: tabla de SoS descargable
- **Soporte multiidioma de columnas**: inglés y español

## Metodología de cálculo

```
SoS(marca, nivel) = Facings(marca, nivel) / Σ Facings(todas las marcas, nivel) × 100
```

El SoS se calcula **por nivel**, no sobre el total de la percha.
Esto es correcto para análisis de shelf porque el denominador relevante es
el espacio disponible en ese nivel específico, no el expositor completo.

## Requisitos

- Python 3.9+
- Ver `requirements.txt` para librerías
