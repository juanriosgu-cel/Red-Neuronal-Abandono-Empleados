import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="ACA 2 | Red neuronal para abandono", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
BASE = Path(__file__).resolve().parent

@st.cache_resource
def cargar_red():
    import tensorflow as tf
    model_path = BASE / "aca2_red_neuronal.keras"
    prep_path = BASE / "aca2_preprocesamiento.joblib"
    meta_candidates = [BASE / "metadatos_aca2.joblib", BASE / "aca2_metadata.joblib"]
    meta_path = next((p for p in meta_candidates if p.exists()), None)
    if not model_path.exists(): raise FileNotFoundError(f"No se encontró {model_path.name}")
    if not prep_path.exists(): raise FileNotFoundError(f"No se encontró {prep_path.name}")
    if meta_path is None: raise FileNotFoundError("No se encontró metadatos_aca2.joblib ni aca2_metadata.joblib")
    model = tf.keras.models.load_model(model_path)
    preprocessor = joblib.load(prep_path)
    metadata = joblib.load(meta_path)
    return model, preprocessor, metadata

@st.cache_data
def cargar_dataset():
    candidates = [p for p in BASE.glob("*.csv") if "HR-Employee-Attrition" in p.name or "Employee-Attrition" in p.name]
    if not candidates: return None
    return pd.read_csv(candidates[0])

# ACA 1 reference metrics; positive class is Attrition = Yes / Abandonó.
ACA1 = pd.DataFrame([
 {"Modelo":"Random Forest (ACA 1)","Accuracy":0.8231,"Precision":0.4468,"Recall":0.4468,"F1":0.4468,"F1 Macro":0.6708,"ROC-AUC":0.7482,"PR-AUC":0.3815},
 {"Modelo":"Regresión Logística (ACA 1)","Accuracy":0.8605,"Precision":0.6154,"Recall":0.3404,"F1":0.4384,"F1 Macro":0.6794,"ROC-AUC":0.8079,"PR-AUC":0.5716},
 {"Modelo":"Árbol de Decisión (ACA 1)","Accuracy":0.8163,"Precision":0.4146,"Recall":0.3617,"F1":0.3864,"F1 Macro":0.6392,"ROC-AUC":0.6510,"PR-AUC":0.3444},
 {"Modelo":"KNN (ACA 1)","Accuracy":0.8537,"Precision":0.6111,"Recall":0.2340,"F1":0.3385,"F1 Macro":0.6281,"ROC-AUC":0.6697,"PR-AUC":0.3407},
])

st.markdown("""<style>.block-container{padding-top:1.5rem;padding-bottom:2rem}h1,h2,h3{letter-spacing:-.02em}</style>""", unsafe_allow_html=True)
st.sidebar.title("Navegación")
pagina = st.sidebar.radio("Seleccione una sección", ["Presentación","Análisis exploratorio (EDA)","Resumen","Comparación de modelos","Análisis del modelo","Predicción individual","Metodología"])
st.sidebar.divider()
st.sidebar.info("Modelo principal ACA 2: red neuronal MLP. Random Forest y los demás algoritmos aparecen únicamente como referencia del ACA 1.")

# Load data/model lazily so the EDA and presentation remain accessible even if model artifacts are absent.
df = cargar_dataset()
model = preprocessor = metadata = None
model_error = None
if pagina in ["Resumen","Comparación de modelos","Análisis del modelo","Predicción individual","Metodología"]:
    try: model, preprocessor, metadata = cargar_red()
    except Exception as e: model_error = str(e)

if pagina != "Presentación":
    st.title("ACA 2 | Predicción del abandono de empleados mediante una red neuronal")
    st.caption("Especialización en Inteligencia Artificial · Dataset IBM HR Analytics")

if pagina == "Presentación":
    st.markdown("""<div style='padding:2rem;border-radius:18px;background:linear-gradient(125deg,#142d4e,#176b87,#23a6a1);color:white'><div style='font-weight:700;letter-spacing:.12em'>APRENDIZAJE AUTOMÁTICO</div><h1 style='color:white'>Predicción del abandono de empleados mediante una red neuronal</h1><p>ACA 2 · Especialización en Inteligencia Artificial</p><hr><b>Dataset:</b> IBM HR Analytics — Employee Attrition<br><br><b>Presentado por</b><br>Juan Guillermo Ríos<br>Víctor Rodríguez Polo<br>Michelle Ángel Castillo Mendoza<br><br><b>Docente</b><br>Hamilton Rivera</div>""", unsafe_allow_html=True)
    a,b,c=st.columns(3)
    a.markdown("#### Exploración"); a.write("EDA del conjunto de datos y revisión de patrones descriptivos.")
    b.markdown("#### Red neuronal"); b.write("Modelamiento MLP para estimar la probabilidad de abandono.")
    c.markdown("#### Evaluación"); c.write("Métricas, análisis de errores y comparación con ACA 1.")
    st.info("Random Forest y los demás modelos clásicos se conservan como referencia comparativa del ACA 1; las predicciones de esta aplicación se generan con la red neuronal del ACA 2.")

if pagina == "Análisis exploratorio (EDA)":
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



elif pagina in ["Resumen","Comparación de modelos","Análisis del modelo","Predicción individual","Metodología"]:
    if model_error:
        st.error("No fue posible cargar los artefactos de la red neuronal: " + model_error)
        st.stop()

    # ACA2 metrics may be exported as joblib; otherwise use the observed notebook result supplied for this deployment.
    metrics_path = BASE / "aca2_metricas.joblib"
    if metrics_path.exists():
        metrics = joblib.load(metrics_path)
    else:
        metrics = {"Accuracy":0.8061,"Precision (Abandonó)":0.4107,"Recall (Abandonó)":0.4894,"F1 (Abandonó)":0.4466,"ROC-AUC":0.7446,"PR-AUC (Average Precision)":0.4886}
    mlp_row = {"Modelo":"Red neuronal MLP (ACA 2)","Accuracy":metrics.get("Accuracy",np.nan),"Precision":metrics.get("Precision (Abandonó)",np.nan),"Recall":metrics.get("Recall (Abandonó)",np.nan),"F1":metrics.get("F1 (Abandonó)",np.nan),"F1 Macro":metrics.get("F1 Macro",0.6645),"ROC-AUC":metrics.get("ROC-AUC",np.nan),"PR-AUC":metrics.get("PR-AUC (Average Precision)",np.nan)}
    comparacion = pd.concat([ACA1,pd.DataFrame([mlp_row])],ignore_index=True)

    if pagina == "Resumen":
        st.header("Resumen del ACA 2")
        c1,c2,c3=st.columns(3)
        c1.metric("Modelo implementado","Red neuronal MLP")
        c2.metric("F1 abandono",f"{mlp_row['F1']:.4f}")
        c3.metric("Recall abandono",f"{mlp_row['Recall']:.4f}")
        st.write("La aplicación utiliza una red neuronal MLP para estimar la probabilidad de abandono. La evaluación se complementa con la comparación de los modelos clásicos del ACA 1.")
        st.dataframe(pd.DataFrame([mlp_row]).style.format({c:"{:.4f}" for c in comparacion.columns if c!="Modelo"}),use_container_width=True,hide_index=True)
        st.warning("Las predicciones son estimaciones académicas y no deben utilizarse como decisiones automáticas sobre personas.")

    elif pagina == "Comparación de modelos":
        st.header("Comparación de modelos: ACA 1 vs. ACA 2")
        st.write("Se compara la red neuronal MLP del ACA 2 con los modelos clásicos del ACA 1. La métrica central es el F1-score de la clase positiva «Abandonó».")
        st.dataframe(comparacion.style.format({c:"{:.4f}" for c in comparacion.columns if c!="Modelo"}),use_container_width=True,hide_index=True)
        fig=px.bar(comparacion,x="Modelo",y="F1",color="Modelo",title="F1-score de la clase Abandonó: ACA 1 y ACA 2",text_auto=".4f")
        fig.add_hline(y=0.4468,line_dash="dash",line_color="red",annotation_text="Random Forest ACA 1 = 0.4468")
        fig.update_layout(showlegend=False,xaxis_tickangle=-25)
        st.plotly_chart(fig,use_container_width=True)
        rf_f1=0.4468; delta=mlp_row["F1"]-rf_f1
        st.metric("Diferencia de F1: MLP vs. Random Forest",f"{delta:+.4f}",f"{delta*100:+.2f} puntos porcentuales")
        st.markdown(f"**Lectura:** la MLP obtuvo F1 = {mlp_row['F1']:.4f} y Recall = {mlp_row['Recall']:.4f}; Random Forest obtuvo F1 = 0.4468 y Recall = 0.4468. La interpretación debe considerar también Precision y PR-AUC.")
        st.caption("Nota: la comparación es descriptiva si los modelos no se evaluaron sobre exactamente la misma partición de prueba.")
        st.download_button("Descargar tabla comparativa CSV",comparacion.to_csv(index=False).encode("utf-8-sig"),"comparacion_aca1_aca2.csv","text/csv")

    elif pagina == "Análisis del modelo":
        st.header("Análisis de la red neuronal")
        st.write("Resultados de evaluación de la MLP para la clase positiva «Abandonó».")
        cols=st.columns(4)
        for col,(label,key) in zip(cols,[("Accuracy","Accuracy"),("Precision","Precision (Abandonó)"),("Recall","Recall (Abandonó)"),("F1-score","F1 (Abandonó)")]):
            val=metrics.get(key)
            col.metric(label,f"{val:.4f}" if val is not None else "N/D")
        st.write("ROC-AUC:",f"{metrics.get('ROC-AUC',np.nan):.4f}","· PR-AUC:",f"{metrics.get('PR-AUC (Average Precision)',np.nan):.4f}")
        st.info("Para mostrar matriz de confusión y curvas ROC/Precision-Recall en esta página, exporta también esos valores o vectores desde el notebook. No se reconstruyen a partir de métricas agregadas.")

    elif pagina == "Predicción individual":
        st.header("Predicción individual con red neuronal MLP")
        if df is None:
            st.error("No se encontró el CSV del dataset para construir el formulario.")
            st.stop()
        input_cols=metadata.get("input_columns",[c for c in df.columns if c!="Attrition"])
        num_features=metadata.get("numeric_features",[])
        cat_features=metadata.get("categorical_features",[])
        threshold=float(metadata.get("threshold",0.5))
        defaults={c: (df[c].median() if c in num_features else df[c].mode(dropna=True).iloc[0]) for c in input_cols if c in df.columns}
        with st.form("form_mlp_prediction"):
            entered={}
            st.subheader("Características del empleado")
            cols=st.columns(2)
            for i,c in enumerate(input_cols):
                if c not in df.columns: continue
                with cols[i%2]:
                    if c in cat_features or not pd.api.types.is_numeric_dtype(df[c]):
                        opts=sorted(df[c].dropna().astype(str).unique().tolist())
                        entered[c]=st.selectbox(c,opts,index=opts.index(str(defaults[c])) if str(defaults[c]) in opts else 0)
                    else:
                        mn=float(df[c].min()); mx=float(df[c].max()); val=float(defaults[c])
                        entered[c]=st.number_input(c,min_value=mn,max_value=mx,value=min(max(val,mn),mx),step=1.0)
            submit=st.form_submit_button("Predecir con la red neuronal",type="primary")
        if submit:
            row=pd.DataFrame([entered]).reindex(columns=input_cols)
            for c in cat_features:
                if c in row: row[c]=row[c].astype(str)
            transformed=preprocessor.transform(row)
            probability=float(np.asarray(model.predict(transformed,verbose=0)).reshape(-1)[0])
            pred=int(probability>=threshold)
            st.divider(); st.subheader("Resultado de la predicción")
            if pred: st.error("Clasificación estimada: posible abandono")
            else: st.success("Clasificación estimada: permanencia")
            a,b=st.columns(2); a.metric("Probabilidad de abandono",f"{probability:.2%}"); b.metric("Probabilidad de permanencia",f"{1-probability:.2%}")
            st.caption(f"Umbral de clasificación utilizado: {threshold:.2f}")
            fig=px.bar(x=["Abandono","Permanencia"],y=[probability,1-probability],labels={"x":"","y":"Probabilidad"},title="Probabilidades estimadas")
            fig.update_yaxes(range=[0,1],tickformat=".0%"); st.plotly_chart(fig,use_container_width=True)
            st.warning("Resultado orientativo del modelo; no constituye una decisión automática sobre el empleado.")

    elif pagina == "Metodología":
        st.header("Metodología de la red neuronal")
        steps=[("1. Dataset","Se utiliza IBM HR Analytics Employee Attrition, el mismo conjunto de datos del ACA 1."),("2. Exploración","Se revisan estructura, calidad, distribución de la variable objetivo y relaciones descriptivas."),("3. Preprocesamiento","Se imputan y transforman variables numéricas y categóricas con el preprocesador ajustado en entrenamiento."),("4. Arquitectura","Se utiliza una red neuronal MLP con capas densas para aprender patrones no lineales en datos tabulares."),("5. Entrenamiento","La red ajusta sus pesos con los datos de entrenamiento y monitorea validación para controlar el sobreajuste."),("6. Evaluación","Se calculan métricas como Accuracy, Precision, Recall, F1, ROC-AUC y PR-AUC, priorizando la clase Abandonó."),("7. Comparación","Se contrasta la MLP con los modelos clásicos reportados en ACA 1."),("8. Despliegue","Streamlit carga el modelo y artefactos exportados para generar predicciones sin reentrenar en cada sesión.")]
        for title,desc in steps:
            with st.container(border=True): st.subheader(title); st.write(desc)
        st.warning("La aplicación tiene finalidad académica y de apoyo analítico. Las métricas no implican causalidad.")

st.divider()
st.caption("ACA 2 · Especialización en Inteligencia Artificial · Red neuronal MLP para clasificación del abandono de empleados")
