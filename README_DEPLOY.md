# Streamlit ACA 1 – versión mejorada

## ¿Qué mejora esta versión?

La aplicación se organiza en cuatro secciones:

1. **Resumen**
   - Indicadores principales.
   - Comparación general.
   - Justificación del modelo seleccionado.

2. **Comparación de modelos**
   - Tabla completa de métricas.
   - Comparación gráfica.
   - F1-score.
   - Verdaderos positivos y falsos negativos.
   - Tiempo de optimización.

3. **Análisis del modelo**
   - Selector de modelo.
   - Matriz de confusión.
   - ROC-AUC.
   - Precision-Recall.
   - Importancia de variables o coeficientes.
   - Visualización PCA.

4. **Predicción individual**
   - Selección del modelo.
   - Formulario de características.
   - Predicción.
   - Probabilidad de abandono.
   - Probabilidad de permanencia.

## Generar el archivo del modelo

Ejecute todas las celdas del notebook `ACA2_Abandono` y, al final, ejecute:

`celda_exportar_modelos_v3.txt`

La celda genera:

`modelos_despliegue.joblib`

## Estructura

```text
ACA2_Streamlit/
├── app.py
├── requirements.txt
├── modelos_despliegue.joblib
└── README_DEPLOY.md
```

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue

Suba los cuatro archivos al repositorio de GitHub y configure `app.py` como archivo principal en Streamlit Community Cloud.

## Nota

La aplicación no vuelve a ejecutar GridSearchCV. Utiliza los modelos ya optimizados y los resultados calculados en el notebook.
