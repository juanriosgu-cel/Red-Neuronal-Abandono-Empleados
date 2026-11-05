import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(
    page_title="Predicción del abandono de empleados",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CARGA DE LOS MODELOS Y RESULTADOS
# ============================================================
@st.cache_resource
def cargar_datos():
    ruta_modelos = Path(__file__).resolve().parent / "modelos_despliegue.joblib"
    return joblib.load(ruta_modelos)


# La portada y el EDA pueden consultarse aunque el archivo de modelos no esté
# disponible. Las secciones de clasificación requieren el artefacto original.
try:
    data = cargar_datos()
    error_modelos = None
except FileNotFoundError:
    data = None
    error_modelos = "No se encontró modelos_despliegue.joblib. Las secciones de presentación y EDA siguen disponibles; para usar los módulos de clasificación, agregue el archivo original junto a app.py."

modelos = data["models"] if data else {}
columnas_modelo = data["columns"] if data else []
metricas = pd.DataFrame(data["metrics"] if data else [], columns=["Modelo", "Accuracy", "Precision", "Recall", "F1", "F1_Macro", "AUC", "PR_AUC"])
matrices = data.get("confusion_matrices", {}) if data else {}
roc_data = data.get("roc_data", {}) if data else {}
pr_data = data.get("pr_data", {}) if data else {}
importancias = data.get("feature_importance", {}) if data else {}
coeficientes = data.get("coefficients", {}) if data else {}
tiempos = pd.DataFrame(data.get("optimization_times", []) if data else [])
categoricas = data.get("categorical_values", {}) if data else {}
defaults = data.get("numeric_defaults", {}) if data else {}
mins = data.get("numeric_min", {}) if data else {}
maxs = data.get("numeric_max", {}) if data else {}
pca_data = pd.DataFrame(data.get("pca_data", []) if data else [])
pca_variance = data.get("pca_variance", []) if data else []


# ============================================================
# ESTILO
# ============================================================
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.02em;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.55rem;
    }

    .small-note {
        color: #6b7280;
        font-size: 0.88rem;
    }

    .section-card {
        padding: 0.9rem 1rem;
        border-radius: 0.65rem;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================
def obtener_fila_modelo(nombre):
    filas = metricas[metricas["Modelo"] == nombre]
    if filas.empty:
        return None
    return filas.iloc[0]


def preparar_entrada(datos):
    """Convierte los datos del formulario al esquema esperado por el modelo."""
    entrada = pd.DataFrame([datos])

    entrada_encoded = pd.get_dummies(
        entrada,
        drop_first=True,
        dtype=int
    )

    entrada_encoded = entrada_encoded.reindex(
        columns=columnas_modelo,
        fill_value=0
    )

    return entrada_encoded


def obtener_probabilidad_abandono(modelo, entrada_encoded):
    """Obtiene la probabilidad asociada a la clase 1/Abandonó."""
    probabilidades = modelo.predict_proba(entrada_encoded)[0]

    if hasattr(modelo, "classes_"):
        clases = list(modelo.classes_)
        if 1 in clases:
            return float(probabilidades[clases.index(1)])
        if "Abandonó" in clases:
            return float(probabilidades[clases.index("Abandonó")])

    # Compatibilidad con los modelos exportados originalmente.
    return float(probabilidades[-1])


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("Navegación")

pagina = st.sidebar.radio(
    "Seleccione una sección",
    [
        "Presentación",
        "Análisis exploratorio (EDA)",
        "Resumen",
        "Comparación de modelos",
        "Análisis del modelo",
        "Predicción individual",
        "Metodología"
    ]
)

st.sidebar.divider()

if modelos:
    modelo_seleccionado = st.sidebar.selectbox(
        "Modelo para análisis",
        list(modelos.keys()),
        index=(list(modelos.keys()).index("Random Forest") if "Random Forest" in modelos else 0)
    )
else:
    modelo_seleccionado = None
    st.sidebar.info("Modelos no cargados")

st.sidebar.caption(
    "El modelo final del estudio es Random Forest. "
    "La interfaz permite consultar también las alternativas evaluadas."
)


# ============================================================
# ENCABEZADO
# ============================================================
if pagina not in ["Presentación"]:
    st.title("Clasificación del abandono de empleados")
    st.caption(
        "Aplicación de aprendizaje automático para analizar y clasificar "
        "el abandono de empleados mediante modelos supervisados."
    )


# ============================================================
# PRESENTACIÓN
# ============================================================
if pagina == "Presentación":
    st.markdown("""
    <div style="padding:2rem 2.2rem;border-radius:18px;
                background:linear-gradient(125deg,#142d4e 0%,#176b87 55%,#23a6a1 100%);
                color:white;border:1px solid rgba(255,255,255,.18);">
      <div style="font-size:.9rem;letter-spacing:.14em;font-weight:700;opacity:.9">
        APRENDIZAJE AUTOMÁTICO
      </div>
      <h1 style="color:white;margin:.7rem 0 .2rem 0">Clasificación del abandono de empleados</h1>
      <div style="font-size:1.08rem;margin-bottom:1.2rem">
        Especialización en Inteligencia Artificial
      </div>
      <div style="height:1px;background:rgba(255,255,255,.4);margin:1.2rem 0"></div>
      <div style="font-size:1rem;font-weight:600">Dataset IBM HR Analytics — Employee Attrition</div>
      <div style="margin-top:1.2rem;font-size:.85rem;letter-spacing:.08em;font-weight:700">PRESENTADO POR</div>
      <div style="margin-top:.4rem;line-height:1.9">
        Juan Guillermo Ríos<br>Víctor Rodríguez Polo<br>Michelle Ángel Castillo Mendoza
      </div>
      <div style="margin-top:1rem;font-size:.85rem;letter-spacing:.08em;font-weight:700">DOCENTE</div>
      <div style="margin-top:.35rem">Hamilton Rivera</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### Exploración")
        st.write("Conocer la estructura, calidad y comportamiento de los datos.")
    with c2:
        st.markdown("#### Clasificación")
        st.write("Comparar algoritmos supervisados para identificar la clase de abandono.")
    with c3:
        st.markdown("#### Evaluación")
        st.write("Interpretar métricas, errores y probabilidades estimadas.")
    st.info("Esta aplicación se centra en clasificación del abandono. No incluye módulos de regresión de ingresos ni segmentación de empleados.")
    st.caption("Proyecto académico · Especialización en Inteligencia Artificial")


# ============================================================
# ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ============================================================
elif pagina == "Análisis exploratorio (EDA)":
    st.header("Análisis exploratorio de datos (EDA)")
    st.write(
        "Exploración descriptiva del dataset IBM HR Analytics antes de interpretar "
        "los resultados de los modelos de clasificación. Los patrones observados "
        "son descriptivos y no demuestran causalidad."
    )

    @st.cache_data
    def cargar_dataset_eda():
        carpeta_app = Path(__file__).resolve().parent
        nombres_csv = [
            "WA_Fn-UseC_-HR-Employee-Attrition.csv",
            "WA_Fn-UseC_-HR-Employee-Attrition(1).csv"
        ]
        for nombre_csv in nombres_csv:
            ruta_csv = carpeta_app / nombre_csv
            if ruta_csv.is_file():
                return pd.read_csv(ruta_csv)
        return None

    df_eda = cargar_dataset_eda()
    if df_eda is None:
        st.error(
            "No se encontró el CSV del dataset. Incluya "
            "WA_Fn-UseC_-HR-Employee-Attrition.csv en la misma carpeta que app.py."
        )
        st.stop()

    numeric_cols = df_eda.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df_eda.select_dtypes(exclude=np.number).columns.tolist()
    n_total = len(df_eda)
    n_abandono = int((df_eda["Attrition"] == "Yes").sum()) if "Attrition" in df_eda else 0
    tasa_abandono = n_abandono / n_total if n_total else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Registros", f"{n_total:,}")
    m2.metric("Variables", len(df_eda.columns))
    m3.metric("Abandonos", f"{n_abandono:,}")
    m4.metric("Tasa de abandono", f"{tasa_abandono:.1%}")

    tab_calidad, tab_distribucion, tab_relaciones, tab_atipicos, tab_hallazgos = st.tabs([
        "Calidad de datos", "Distribuciones", "Relaciones", "Valores atípicos", "Hallazgos"
    ])

    with tab_calidad:
        st.subheader("Estructura y calidad del conjunto")
        calidad = pd.DataFrame({
            "Indicador": ["Filas", "Columnas", "Celdas vacías", "Filas duplicadas", "Variables constantes"],
            "Resultado": [
                len(df_eda), len(df_eda.columns), int(df_eda.isna().sum().sum()),
                int(df_eda.duplicated().sum()),
                int(sum(df_eda[c].nunique(dropna=False) <= 1 for c in df_eda.columns))
            ]
        })
        st.dataframe(calidad, use_container_width=True, hide_index=True)
        faltantes = df_eda.isna().sum().rename("Faltantes").reset_index().rename(columns={"index":"Variable"})
        faltantes = faltantes[ faltantes["Faltantes"] > 0 ]
        if faltantes.empty:
            st.success("No se identificaron valores faltantes en el dataset.")
        else:
            st.dataframe(faltantes, use_container_width=True, hide_index=True)
        st.markdown("#### Tipos de variables")
        tipos = pd.DataFrame({"Variable": df_eda.columns, "Tipo": df_eda.dtypes.astype(str).values,
                              "Valores distintos": [df_eda[c].nunique(dropna=False) for c in df_eda.columns]})
        st.dataframe(tipos, use_container_width=True, hide_index=True)

    with tab_distribucion:
        st.subheader("Distribución de la variable objetivo")
        if "Attrition" in df_eda:
            conteo = df_eda["Attrition"].value_counts().rename_axis("Abandono").reset_index(name="Empleados")
            fig = px.bar(conteo, x="Abandono", y="Empleados", color="Abandono",
                         color_discrete_map={"Yes":"#E76F51", "No":"#2878B5"},
                         text="Empleados", title="Distribución de empleados por abandono")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        variable_num = st.selectbox("Variable numérica", numeric_cols,
                                    index=numeric_cols.index("MonthlyIncome") if "MonthlyIncome" in numeric_cols else 0)
        fig_hist = px.histogram(df_eda, x=variable_num, color="Attrition" if "Attrition" in df_eda else None,
                                marginal="box", barmode="overlay", opacity=.72,
                                title=f"Distribución de {variable_num}")
        st.plotly_chart(fig_hist, use_container_width=True)
        if "Attrition" in df_eda:
            st.markdown("#### Tasa de abandono por variable categórica")
            opciones_cat = [c for c in categorical_cols if c != "Attrition"]
            cat = st.selectbox("Variable categórica", opciones_cat,
                               index=opciones_cat.index("OverTime") if "OverTime" in opciones_cat else 0)
            tasas = df_eda.groupby(cat, dropna=False)["Attrition"].apply(lambda s: (s == "Yes").mean()).mul(100).reset_index(name="Tasa de abandono (%)")
            tasas[cat] = tasas[cat].astype(str)
            fig_tasa = px.bar(tasas.sort_values("Tasa de abandono (%)", ascending=False),
                              x=cat, y="Tasa de abandono (%)", color="Tasa de abandono (%)",
                              color_continuous_scale=["#8ECAE6","#219EBC","#E76F51"],
                              text=tasas.sort_values("Tasa de abandono (%)", ascending=False)["Tasa de abandono (%)"].map(lambda x:f"{x:.1f}%"),
                              title=f"Tasa de abandono según {cat}")
            st.plotly_chart(fig_tasa, use_container_width=True)

    with tab_relaciones:
        st.subheader("Relaciones entre variables")
        corr_cols = [c for c in numeric_cols if df_eda[c].nunique() > 1]
        default_corr = [c for c in ["Age", "MonthlyIncome", "YearsAtCompany", "TotalWorkingYears"] if c in corr_cols]
        corr_selected = st.multiselect("Variables para correlación", corr_cols, default=default_corr[:8])
        if len(corr_selected) >= 2:
            corr = df_eda[corr_selected].corr(numeric_only=True)
            fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                                 zmin=-1, zmax=1, title="Matriz de correlación")
            st.plotly_chart(fig_corr, use_container_width=True)
        scatter_defaults = [c for c in ["Age", "MonthlyIncome", "YearsAtCompany", "TotalWorkingYears"] if c in numeric_cols]
        if len(scatter_defaults) >= 2:
            st.markdown("#### Matriz de dispersión de variables principales")
            pair_vars = st.multiselect("Variables de la matriz", numeric_cols, default=scatter_defaults)
            if len(pair_vars) >= 2:
                fig_pair = px.scatter_matrix(df_eda, dimensions=pair_vars,
                    color="Attrition" if "Attrition" in df_eda else None,
                    color_discrete_map={"Yes":"#E76F51", "No":"#2878B5"},
                    opacity=.55, title="Relaciones multivariadas")
                fig_pair.update_layout(height=750)
                st.plotly_chart(fig_pair, use_container_width=True)
        if "Department" in df_eda and "Attrition" in df_eda:
            dept = df_eda.groupby("Department")["Attrition"].apply(lambda s:(s=="Yes").mean()*100).reset_index(name="Tasa (%)")
            fig_dept = px.bar(dept, x="Department", y="Tasa (%)", color="Tasa (%)",
                              color_continuous_scale=["#8ECAE6","#219EBC","#E76F51"],
                              title="Tasa de abandono por departamento")
            st.plotly_chart(fig_dept, use_container_width=True)

    with tab_atipicos:
        st.subheader("Detección estadística mediante rango intercuartílico (IQR)")
        variable_out = st.selectbox("Variable para detectar valores atípicos", numeric_cols,
                                    index=numeric_cols.index("MonthlyIncome") if "MonthlyIncome" in numeric_cols else 0)
        q1 = df_eda[variable_out].quantile(.25)
        q3 = df_eda[variable_out].quantile(.75)
        iqr = q3 - q1
        limite_inf, limite_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask_out = (df_eda[variable_out] < limite_inf) | (df_eda[variable_out] > limite_sup)
        st.caption(f"Límite inferior: {limite_inf:,.2f} · Límite superior: {limite_sup:,.2f}")
        a, b, c = st.columns(3)
        a.metric("Valores atípicos detectados", int(mask_out.sum()))
        b.metric("Dentro de los límites", int((~mask_out).sum()))
        c.metric("Porcentaje atípico", f"{mask_out.mean():.1%}")
        df_box = df_eda[[variable_out]].copy()
        df_box["Tipo de dato"] = np.where(mask_out, "Atípico", "Dentro del rango")
        fig_box = px.box(df_box, x="Tipo de dato", y=variable_out, color="Tipo de dato",
                         color_discrete_map={"Atípico":"#E76F51","Dentro del rango":"#2878B5"},
                         points="all", title=f"Distribución y valores atípicos: {variable_out}")
        st.plotly_chart(fig_box, use_container_width=True)
        st.info("Los valores atípicos estadísticos no son necesariamente errores. Deben interpretarse en el contexto de la variable y no eliminarse automáticamente.")

    with tab_hallazgos:
        st.subheader("Hallazgos descriptivos")
        st.write(f"La tasa global de abandono en la base es **{tasa_abandono:.2%}** ({n_abandono} de {n_total} empleados).")
        if "YearsAtCompany" in df_eda and "Attrition" in df_eda:
            med = df_eda.groupby("Attrition")["YearsAtCompany"].median()
            if "Yes" in med.index and "No" in med.index:
                st.write(f"La mediana de antigüedad es **{med['Yes']:.0f} años** entre quienes abandonaron y **{med['No']:.0f} años** entre quienes permanecieron.")
        if "MonthlyIncome" in df_eda and "Attrition" in df_eda:
            med_ing = df_eda.groupby("Attrition")["MonthlyIncome"].median()
            if "Yes" in med_ing.index and "No" in med_ing.index:
                st.write(f"La mediana de ingreso mensual es **{med_ing['Yes']:,.0f}** para quienes abandonaron y **{med_ing['No']:,.0f}** para quienes permanecieron.")
        if "OverTime" in df_eda and "Attrition" in df_eda:
            overtime = df_eda.groupby("OverTime")["Attrition"].apply(lambda s:(s=="Yes").mean()*100)
            if "Yes" in overtime.index and "No" in overtime.index:
                st.write(f"La tasa de abandono entre empleados con horas extra es **{overtime['Yes']:.1f}%**, frente a **{overtime['No']:.1f}%** para quienes no reportan horas extra.")
        st.warning("Estos hallazgos describen asociaciones observadas en los datos; no prueban que una variable cause el abandono.")


# ============================================================
# CONTROL DE DISPONIBILIDAD DE LOS MODELOS
# ============================================================
paginas_modelo = ["Resumen", "Comparación de modelos", "Análisis del modelo", "Predicción individual", "Metodología"]
if pagina in paginas_modelo and data is None:
    st.warning(error_modelos)
    st.stop()


# ============================================================
# RESUMEN
# ============================================================
if pagina == "Resumen":

    st.header("Resumen del estudio")

    mejor_recall = metricas.loc[metricas["Recall"].idxmax()]
    mejor_f1 = metricas.loc[metricas["F1"].idxmax()]
    mejor_auc = metricas.loc[metricas["AUC"].idxmax()]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Modelos evaluados",
            len(modelos)
        )

    with c2:
        st.metric(
            "Mayor Recall",
            mejor_recall["Modelo"],
            f'{mejor_recall["Recall"]:.4f}'
        )

    with c3:
        st.metric(
            "Mayor F1",
            mejor_f1["Modelo"],
            f'{mejor_f1["F1"]:.4f}'
        )

    with c4:
        st.metric(
            "Mayor ROC-AUC",
            mejor_auc["Modelo"],
            f'{mejor_auc["AUC"]:.4f}'
        )

    st.info(
        "Para este estudio, la clase de interés es «Abandonó». "
        "Por ello, Recall y F1-score de esta clase tienen especial "
        "importancia para la selección del modelo."
    )

    st.subheader("Resultados principales")

    columnas_mostrar = [
        "Modelo", "Accuracy", "Precision",
        "Recall", "F1", "F1_Macro", "AUC", "PR_AUC"
    ]

    st.dataframe(
        metricas[columnas_mostrar].style.format({
            c: "{:.4f}" for c in columnas_mostrar
            if c != "Modelo"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Descargar tabla de resultados (CSV)",
        metricas[columnas_mostrar].to_csv(index=False).encode("utf-8"),
        "resultados_modelos_ACA1.csv",
        "text/csv"
    )

    st.subheader("Modelo seleccionado")

    if "Random Forest" in modelos:
        rf = obtener_fila_modelo("Random Forest")

        st.success(
            f"Random Forest fue seleccionado en el estudio por su "
            f"desempeño en la identificación de la clase «Abandonó»: "
            f"Recall = {rf['Recall']:.4f} y F1 = {rf['F1']:.4f}."
        )

    st.warning(
        "Los resultados son de apoyo analítico. Un caso clasificado "
        "como posible abandono no constituye una determinación causal "
        "ni una decisión automática sobre un empleado."
    )


# ============================================================
# COMPARACIÓN DE MODELOS
# ============================================================
elif pagina == "Comparación de modelos":

    st.header("Comparación de modelos")

    st.write(
        "La comparación se realiza sobre el conjunto de prueba utilizando "
        "las métricas calculadas durante el desarrollo de la ACA 1 – Caso 1."
    )

    tab1, tab2, tab3 = st.tabs([
        "📊 Métricas de desempeño",
        "⚠️ Errores de clasificación",
        "⏱️ Costo computacional"
    ])

    # --------------------------------------------------------
    # MÉTRICAS
    # --------------------------------------------------------
    with tab1:

        columnas = [
            "Modelo", "Accuracy", "Precision",
            "Recall", "F1", "F1_Macro",
            "AUC", "PR_AUC"
        ]

        st.dataframe(
            metricas[columnas].style.format({
                c: "{:.4f}" for c in columnas if c != "Modelo"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("#### Comparación general")

        metricas_largas = metricas.melt(
            id_vars="Modelo",
            value_vars=[
                "Accuracy", "Precision", "Recall",
                "F1", "F1_Macro", "AUC", "PR_AUC"
            ],
            var_name="Métrica",
            value_name="Valor"
        )

        fig = px.bar(
            metricas_largas,
            x="Modelo",
            y="Valor",
            color="Métrica",
            barmode="group",
            title="Desempeño comparativo de los modelos"
        )
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        fig.update_layout(
            legend_title_text="Métrica",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Criterios principales para la clase «Abandonó»")

        metricas_interes = metricas[
            ["Modelo", "Recall", "F1"]
        ].melt(
            id_vars="Modelo",
            var_name="Métrica",
            value_name="Valor"
        )

        fig_interes = px.bar(
            metricas_interes,
            x="Modelo",
            y="Valor",
            color="Métrica",
            barmode="group",
            text="Valor",
            title="Recall y F1 de la clase «Abandonó»"
        )
        fig_interes.update_yaxes(range=[0, 1], tickformat=".0%")
        fig_interes.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside"
        )
        fig_interes.update_layout(
            legend_title_text="Métrica",
            hovermode="x unified"
        )
        st.plotly_chart(fig_interes, use_container_width=True)

        st.caption(
            "Estas dos métricas son especialmente relevantes porque el objetivo "
            "del estudio es identificar correctamente casos de abandono."
        )

        st.markdown("#### Capacidad de discriminación")

        auc_df = metricas[
            ["Modelo", "AUC", "PR_AUC"]
        ].melt(
            id_vars="Modelo",
            var_name="Métrica",
            value_name="Valor"
        )

        fig_auc = px.bar(
            auc_df,
            x="Modelo",
            y="Valor",
            color="Métrica",
            barmode="group",
            text="Valor",
            title="ROC-AUC y PR-AUC"
        )
        fig_auc.update_yaxes(range=[0, 1], tickformat=".0%")
        fig_auc.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside"
        )
        st.plotly_chart(fig_auc, use_container_width=True)

    # --------------------------------------------------------
    # ERRORES
    # --------------------------------------------------------
    with tab2:

        tabla_error = []

        for nombre in modelos:
            cm = np.array(matrices[nombre])
            tn, fp, fn, tp = cm.ravel()

            tabla_error.append({
                "Modelo": nombre,
                "Verdaderos positivos": int(tp),
                "Falsos negativos": int(fn),
                "Falsos positivos": int(fp),
                "Verdaderos negativos": int(tn)
            })

        df_error = pd.DataFrame(tabla_error)

        st.dataframe(
            df_error,
            use_container_width=True,
            hide_index=True
        )

        fig_error = px.bar(
            df_error,
            x="Modelo",
            y=[
                "Verdaderos positivos",
                "Falsos negativos"
            ],
            barmode="group",
            title="Verdaderos positivos y falsos negativos"
        )
        st.plotly_chart(fig_error, use_container_width=True)

        st.caption(
            "Los falsos negativos corresponden a casos reales de abandono "
            "que el modelo no identifica. Por esta razón son especialmente "
            "relevantes para este estudio."
        )

    # --------------------------------------------------------
    # COSTO COMPUTACIONAL
    # --------------------------------------------------------
    with tab3:

        if not tiempos.empty:

            tiempos_ordenados = tiempos.sort_values(
                "Tiempo_segundos"
            )

            fig_t = px.bar(
                tiempos_ordenados,
                x="Modelo",
                y="Tiempo_segundos",
                text="Tiempo_segundos",
                title="Tiempo de optimización de hiperparámetros"
            )

            fig_t.update_traces(
                texttemplate="%{text:.3f} s",
                textposition="outside"
            )

            st.plotly_chart(
                fig_t,
                use_container_width=True
            )

            st.dataframe(
                tiempos_ordenados.style.format(
                    {"Tiempo_segundos": "{:.3f}"}
                ),
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "El tiempo de optimización es un criterio complementario "
                "y no constituye el criterio principal de selección del modelo."
            )
        else:
            st.info(
                "No se encontró información de tiempos de optimización "
                "en el archivo de despliegue."
            )


# ============================================================
# ANÁLISIS DEL MODELO
# ============================================================
elif pagina == "Análisis del modelo":

    st.header(f"Análisis: {modelo_seleccionado}")

    fila = obtener_fila_modelo(modelo_seleccionado)

    if fila is None:
        st.error("No se encontraron métricas para el modelo seleccionado.")
        st.stop()

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric("Accuracy", f'{fila["Accuracy"]:.4f}')

    with c2:
        st.metric("Precision", f'{fila["Precision"]:.4f}')

    with c3:
        st.metric("Recall", f'{fila["Recall"]:.4f}')

    with c4:
        st.metric("F1", f'{fila["F1"]:.4f}')

    with c5:
        st.metric("ROC-AUC", f'{fila["AUC"]:.4f}')

    tab1, tab2, tab3, tab4 = st.tabs([
        "Matriz de confusión",
        "ROC y Precision-Recall",
        "Variables",
        "PCA"
    ])

    # --------------------------------------------------------
    # MATRIZ
    # --------------------------------------------------------
    with tab1:

        cm = np.array(matrices[modelo_seleccionado])

        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=["No abandonó", "Abandonó"],
            y=["No abandonó", "Abandonó"],
            labels={
                "x": "Predicción",
                "y": "Valor real",
                "color": "Cantidad"
            },
            title=f"Matriz de confusión – {modelo_seleccionado}"
        )

        fig_cm.update_layout(
            xaxis_title="Predicción del modelo",
            yaxis_title="Valor real"
        )

        st.plotly_chart(
            fig_cm,
            use_container_width=True
        )

        tn, fp, fn, tp = cm.ravel()

        a, b, c, d = st.columns(4)

        with a:
            st.metric("Verdaderos positivos", int(tp))

        with b:
            st.metric("Falsos negativos", int(fn))

        with c:
            st.metric("Falsos positivos", int(fp))

        with d:
            st.metric("Verdaderos negativos", int(tn))

        st.warning(
            f"El modelo deja {int(fn)} casos reales de abandono "
            "clasificados como «No abandonó»."
        )

    # --------------------------------------------------------
    # ROC Y PR
    # --------------------------------------------------------
    with tab2:

        col_roc, col_pr = st.columns(2)

        with col_roc:

            roc = roc_data[modelo_seleccionado]

            fig_roc = go.Figure()

            fig_roc.add_trace(
                go.Scatter(
                    x=roc["fpr"],
                    y=roc["tpr"],
                    mode="lines",
                    name=f"AUC = {fila['AUC']:.4f}"
                )
            )

            fig_roc.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Referencia",
                    line=dict(dash="dash")
                )
            )

            fig_roc.update_layout(
                title="Curva ROC",
                xaxis_title="Tasa de falsos positivos",
                yaxis_title="Tasa de verdaderos positivos",
                yaxis=dict(range=[0, 1]),
                xaxis=dict(range=[0, 1])
            )

            st.plotly_chart(
                fig_roc,
                use_container_width=True
            )

        with col_pr:

            pr = pr_data[modelo_seleccionado]

            fig_pr = go.Figure()

            fig_pr.add_trace(
                go.Scatter(
                    x=pr["recall"],
                    y=pr["precision"],
                    mode="lines",
                    name=f"PR-AUC = {fila['PR_AUC']:.4f}"
                )
            )

            fig_pr.update_layout(
                title="Curva Precision-Recall",
                xaxis_title="Recall",
                yaxis_title="Precision",
                yaxis=dict(range=[0, 1]),
                xaxis=dict(range=[0, 1])
            )

            st.plotly_chart(
                fig_pr,
                use_container_width=True
            )

        st.caption(
            "La curva ROC muestra la capacidad de discriminación del modelo. "
            "La curva Precision-Recall resulta especialmente informativa cuando "
            "la identificación de la clase de interés es prioritaria."
        )

    # --------------------------------------------------------
    # VARIABLES
    # --------------------------------------------------------
    with tab3:

        if modelo_seleccionado in importancias:

            df_imp = pd.DataFrame(
                importancias[modelo_seleccionado]
            ).sort_values("Importancia", ascending=True)

            fig_imp = px.bar(
                df_imp,
                x="Importancia",
                y="Variable",
                orientation="h",
                title="Importancia de variables"
            )

            st.plotly_chart(
                fig_imp,
                use_container_width=True
            )

            st.caption(
                "La importancia predictiva indica la contribución relativa "
                "de las variables dentro del modelo y no implica causalidad."
            )

        elif modelo_seleccionado in coeficientes:

            df_coef = pd.DataFrame(
                coeficientes[modelo_seleccionado]
            )

            df_coef["Abs"] = df_coef["Coeficiente"].abs()
            df_coef = df_coef.sort_values("Abs", ascending=True)

            fig_coef = px.bar(
                df_coef,
                x="Coeficiente",
                y="Variable",
                orientation="h",
                title="Coeficientes de Regresión Logística"
            )

            st.plotly_chart(
                fig_coef,
                use_container_width=True
            )

            st.caption(
                "El signo del coeficiente indica dirección dentro del modelo; "
                "no debe interpretarse como causalidad."
            )

        else:
            st.info(
                "Este modelo no cuenta con una medida directa de importancia "
                "de variables exportada."
            )

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------
    with tab4:

        if not pca_data.empty:

            fig_pca = px.scatter_3d(
                pca_data,
                x="PC1",
                y="PC2",
                z="PC3",
                color="Clase",
                symbol="Clase",
                title="Visualización 3D mediante PCA"
            )

            st.plotly_chart(
                fig_pca,
                use_container_width=True
            )

            if pca_variance:

                var = np.array(pca_variance)

                st.write(
                    f"Varianza explicada: PC1 = {var[0]:.4f}, "
                    f"PC2 = {var[1]:.4f}, PC3 = {var[2]:.4f}. "
                    f"Acumulada = {var.sum():.4f}."
                )

            st.caption(
                "El PCA se utiliza como recurso de visualización y no "
                "interviene en la selección del modelo."
            )

        else:
            st.info(
                "La información PCA aún no fue exportada desde el notebook."
            )


# ============================================================
# PREDICCIÓN INDIVIDUAL
# ============================================================
elif pagina == "Predicción individual":

    st.header("Predicción individual")

    st.write(
        "Seleccione un modelo y diligencie las características del empleado. "
        "La aplicación utilizará el modelo optimizado correspondiente."
    )

    st.info(
        "Esta predicción es una estimación del modelo. No representa "
        "una determinación definitiva sobre el comportamiento futuro "
        "de un empleado."
    )

    with st.form("formulario_prediccion"):

        datos = {}

        if categoricas:

            st.subheader("Información categórica")

            cols = st.columns(2)

            for i, variable in enumerate(categoricas):

                opciones = categoricas[variable]

                with cols[i % 2]:

                    datos[variable] = st.selectbox(
                        variable.replace("_", " "),
                        opciones
                    )

        numericas = [
            c for c in defaults
            if c not in categoricas
        ]

        if numericas:

            st.subheader("Información numérica")

            cols = st.columns(2)

            for i, variable in enumerate(numericas):

                minimo = float(mins[variable])
                maximo = float(maxs[variable])
                valor = float(defaults[variable])

                with cols[i % 2]:

                    if minimo == maximo:

                        datos[variable] = st.number_input(
                            variable.replace("_", " "),
                            value=valor,
                            disabled=True
                        )

                    else:

                        valor = min(
                            max(valor, minimo),
                            maximo
                        )

                        datos[variable] = st.number_input(
                            variable.replace("_", " "),
                            min_value=minimo,
                            max_value=maximo,
                            value=valor,
                            step=1.0
                        )

        ejecutar = st.form_submit_button(
            "Realizar predicción",
            type="primary"
        )

    if ejecutar:

        entrada_encoded = preparar_entrada(datos)

        modelo = modelos[modelo_seleccionado]

        prediccion = int(
            modelo.predict(entrada_encoded)[0]
        )

        probabilidad = obtener_probabilidad_abandono(
            modelo,
            entrada_encoded
        )

        st.divider()
        st.subheader("Resultado")

        if prediccion == 1:
            st.error("Resultado: posible abandono")
        else:
            st.success("Resultado: no abandono")

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Probabilidad de abandono",
                f"{probabilidad:.2%}"
            )

        with c2:
            st.metric(
                "Probabilidad de permanencia",
                f"{1 - probabilidad:.2%}"
            )

        fig_prob = px.bar(
            x=["Abandono", "Permanencia"],
            y=[probabilidad, 1 - probabilidad],
            labels={
                "x": "",
                "y": "Probabilidad"
            },
            title="Probabilidades estimadas"
        )

        fig_prob.update_yaxes(
            range=[0, 1],
            tickformat=".0%"
        )

        st.plotly_chart(
            fig_prob,
            use_container_width=True
        )


# ============================================================
# METODOLOGÍA
# ============================================================
elif pagina == "Metodología":

    st.header("Metodología del modelamiento")

    st.write(
        "La aplicación presenta los resultados del proceso de modelamiento "
        "desarrollado para la ACA 1 – Caso 1 y permite consultar los modelos "
        "evaluados, sus métricas y sus resultados de predicción."
    )

    pasos = [
        (
            "1. Datos",
            "Preparación del conjunto de datos utilizado para el estudio "
            "del abandono de empleados."
        ),
        (
            "2. Preprocesamiento",
            "Transformación y preparación de las variables para que puedan "
            "ser utilizadas por los algoritmos de clasificación."
        ),
        (
            "3. Modelamiento",
            "Evaluación de Regresión Logística, KNN, Árbol de Decisión "
            "y Random Forest."
        ),
        (
            "4. Optimización",
            "Ajuste de los modelos mediante los procedimientos definidos "
            "en el desarrollo de la ACA 1 – Caso 1."
        ),
        (
            "5. Evaluación",
            "Comparación mediante Accuracy, Precision, Recall, F1, "
            "ROC-AUC y PR-AUC, además de la matriz de confusión."
        ),
        (
            "6. Selección",
            "Random Forest fue seleccionado considerando especialmente "
            "el desempeño en la identificación de la clase «Abandonó»."
        ),
        (
            "7. Despliegue",
            "Los modelos y resultados fueron integrados en una aplicación "
            "interactiva desarrollada con Streamlit."
        )
    ]

    for titulo, descripcion in pasos:
        with st.container(border=True):
            st.subheader(titulo)
            st.write(descripcion)

    st.subheader("Criterio de selección del modelo")

    if "Random Forest" in modelos:

        rf = obtener_fila_modelo("Random Forest")

        st.success(
            f"Random Forest presenta Recall = {rf['Recall']:.4f} y "
            f"F1 = {rf['F1']:.4f} para la clase «Abandonó». "
            "Estas métricas fueron consideradas prioritarias para el "
            "objetivo del estudio."
        )

    st.subheader("Interpretación")

    st.write(
        "Las métricas y visualizaciones deben interpretarse como evidencia "
        "del desempeño predictivo de los modelos sobre los datos evaluados. "
        "La importancia de variables y las predicciones no implican "
        "relaciones causales."
    )

    st.warning(
        "La aplicación tiene finalidad académica y de apoyo analítico. "
        "Una predicción de posible abandono no constituye una decisión "
        "automática sobre un empleado."
    )


# ============================================================
# PIE DE PÁGINA
# ============================================================
st.divider()

st.caption(
    "Proyecto académico – Especialización en Inteligencia Artificial. "
    "Los modelos corresponden a la ejecución documentada en el notebook ACA 1 – Caso 1."
)
