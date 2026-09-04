# Detección de Fraude Transaccional

Proyecto end-to-end de detección de fraude en transacciones, que cubre desde el análisis exploratorio de datos hasta la comparación de modelos y la preparación para despliegue en producción.

## Descripción del proyecto

El objetivo es construir un modelo de clasificación capaz de identificar transacciones fraudulentas en un contexto de **fuerte desbalance de clases** (~0.44% de fraude), priorizando buenas prácticas de validación (uso de una base **Out-Of-Time (OOT)** para medir estabilidad temporal) y un criterio de selección de modelos basado en el costo de negocio, no solo en métricas técnicas.

**Etapas cubiertas:**
- Análisis exploratorio de datos (EDA) y limpieza de datos
- Ingeniería de variables
- Entrenamiento y optimización de hiperparámetros de 4 modelos: Logit, Random Forest, XGBoost y LightGBM
- Optimización de umbrales de decisión
- Validación de estabilidad temporal (Prueba vs. OOT)
- Selección de modelo final con criterio de negocio
- Despliegue básico vía API

## Estructura del repositorio

```
fraud-detection-project/
│
├── data/                   # Datos crudos y procesados (no versionados en Git, ver .gitignore)
├── data_cleaning/          # Notebooks/scripts de limpieza y EDA
├── logit/                  # Entrenamiento y evaluación del modelo de regresión logística
├── randomforest/           # Entrenamiento y evaluación de Random Forest
├── xgboost/                # Entrenamiento y evaluación de XGBoost (modelo seleccionado)
├── lightgbm/               # Entrenamiento y evaluación de LightGBM
├── requirements.txt
├── .gitignore
└── README.md
```

> Nota: los datos crudos no se incluyen en el repositorio por tamaño. Ver sección [Datos](#-datos) para más detalles.

## Resultados y selección de modelo

Se evaluaron 4 modelos, optimizando hiperparámetros y umbral de decisión para cada uno, y validando su desempeño tanto en una base de **Prueba** como en una base **OOT** (fuera de tiempo) para verificar estabilidad temporal.

### Comparación en OOT (umbral óptimo por modelo)

| Modelo         | Umbral | AUC-ROC | F2-score | Recall (fraude) | Precision (fraude) | F1 (fraude) |
|----------------|--------|---------|----------|------------------|----------------------|-------------|
| **XGBoost**    | 0.89   | 0.9932  | 0.6085   | 0.808            | 0.306                | 0.444       |
| Random Forest  | —      | —       | —        | 0.713            | 0.376                | 0.492       |
| LightGBM       | 0.90   | 0.9926  | 0.5925   | 0.799            | 0.291                | 0.427       |

*(Logit fue descartado por bajo desempeño general frente a los modelos basados en árboles.)*

### Criterio de selección

Se seleccionó **XGBoost** como modelo final porque:

1. Presenta el mejor **AUC-ROC** y **F2-score** en la base OOT.
2. Tiene el **mayor recall de fraude** (0.808), lo cual es prioritario dado que el costo de un fraude no detectado (falso negativo) suele ser considerablemente mayor que el costo de una revisión manual adicional (falso positivo).
3. Muestra buena **estabilidad temporal**: el recall se mantiene prácticamente igual entre Prueba (0.803) y OOT (0.808), y la degradación en precisión es comparativamente menor que la de LightGBM y Random Forest.

Random Forest ofrece mejor precisión y F1, por lo que sería la alternativa preferida si el volumen de revisión manual fuera una restricción operativa más crítica que el fraude no detectado.

## Instalación

```bash
git clone https://github.com/<tu-usuario>/fraud-detection-project.git
cd fraud-detection-project
python -m venv venv
source venv/bin/activate   # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

Cada carpeta contiene el notebook correspondiente a esa etapa del proyecto:

1. `data_cleaning/` → EDA y limpieza de datos.
2. `logit/`, `randomforest/`, `xgboost/`, `lightgbm/` → entrenamiento, optimización de hiperparámetros/umbral y evaluación de cada modelo.

Ejecutar en el orden: `data_cleaning` → modelos individuales.

## Datos

Por razones de tamaño, los datos crudos no están incluidos en este repositorio. Si deseas reproducir el proyecto, coloca tus archivos en la carpeta `data/` respetando el formato esperado (ver notebook de `data_cleaning`).


## Tecnologías utilizadas

- Python 3.x
- pandas, numpy
- scikit-learn
- xgboost, lightgbm
- matplotlib, seaborn
- Jupyter Notebook

## Autor

Proyecto desarrollado como parte de un portafolio de ciencia de datos enfocado en detección de fraude transaccional por David Rubio.
