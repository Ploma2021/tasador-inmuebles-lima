import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURACIÓN GENERAL DE LA APLICACIÓN (ETAPA 8 - DESPLIEGUE EN STREAMLIT)
# ==============================================================================
st.set_page_config(
    page_title="Tasador Inteligente - Lima Metropolitana",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CARPETA_MODELOS = "modelos_exportados"
RUTA_MODELO = os.path.join(CARPETA_MODELOS, "modelo_tasador_lima.pkl")
RUTA_DIM_DISTRITO = os.path.join(CARPETA_MODELOS, "dim_distrito_app.csv")
RUTA_ABLACION = os.path.join(CARPETA_MODELOS, "resultados_ablacion.csv")
RUTA_METRICAS = os.path.join(CARPETA_MODELOS, "metricas_modelo_final.csv")
RUTA_AVISOS = os.path.join(CARPETA_MODELOS, "avisos_por_distrito.csv")
MIN_AVISOS_CONFIABLE = 20
RUTA_CSV_RESPALDO = "dataset_modelo_lima.csv"

CADENA_CONEXION_SQL = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=TasadorInteligenteDB;"
    "Trusted_Connection=yes;"
)

# ==============================================================================
# FUNCIONES DE CARGA EN CACHÉ
# ==============================================================================
@st.cache_resource
def cargar_modelo():
    if not os.path.exists(RUTA_MODELO):
        return None
    return joblib.load(RUTA_MODELO)


@st.cache_data
def cargar_dim_distrito():
    # Intenta leer primero desde SQL Server; si no está disponible, usa el CSV exportado
    try:
        import pyodbc
        conn = pyodbc.connect(CADENA_CONEXION_SQL, timeout=3)
        df_dim = pd.read_sql("SELECT * FROM Dim_Distrito ORDER BY Distrito", conn)
        conn.close()
        return df_dim, "SQL Server (Dim_Distrito)"
    except Exception:
        if os.path.exists(RUTA_DIM_DISTRITO):
            df_dim = pd.read_csv(RUTA_DIM_DISTRITO).sort_values("Distrito").reset_index(drop=True)
            return df_dim, "Respaldo local (dim_distrito_app.csv)"
        return None, "No encontrado"


@st.cache_data
def cargar_referencia_mercado():
    # Carga opcional para comparar el $/m² estimado contra la mediana real del distrito
    try:
        import pyodbc
        conn = pyodbc.connect(CADENA_CONEXION_SQL, timeout=3)
        df_mkt = pd.read_sql(
            "SELECT Distrito, Precio_USD, Metraje_m2, Precio_m2_USD FROM vw_Dataset_Modelo",
            conn,
        )
        conn.close()
        return df_mkt
    except Exception:
        if os.path.exists(RUTA_CSV_RESPALDO):
            return pd.read_csv(RUTA_CSV_RESPALDO)[
                ["Distrito", "Precio_USD", "Metraje_m2", "Precio_m2_USD"]
            ]
        return None


@st.cache_data
def cargar_ablacion():
    if os.path.exists(RUTA_ABLACION):
        return pd.read_csv(RUTA_ABLACION)
    return None


@st.cache_data
def cargar_metricas():
    if os.path.exists(RUTA_METRICAS):
        return pd.read_csv(RUTA_METRICAS)
    return None


@st.cache_data
def cargar_avisos_por_distrito():
    if os.path.exists(RUTA_AVISOS):
        return pd.read_csv(RUTA_AVISOS)
    return None


def clasificar_peligrosidad(indice):
    if indice < 33.3:
        return "Bajo", "🟢", "#2e8b57"
    elif indice < 66.6:
        return "Moderado", "🟡", "#d9822b"
    else:
        return "Alto", "🔴", "#c0392b"


# ==============================================================================
# CARGA INICIAL Y VALIDACIÓN
# ==============================================================================
modelo = cargar_modelo()
df_dim, fuente_dim = cargar_dim_distrito()
df_mercado = cargar_referencia_mercado()
df_ablacion = cargar_ablacion()
df_metricas = cargar_metricas()
df_avisos = cargar_avisos_por_distrito()

st.title("🏙️ Tasador Inteligente de Viviendas — Lima Metropolitana")
st.caption(
    "Modelo predictivo basado en Machine Learning integrando atributos "
    "estructurales de Urbania Perú e indicadores de seguridad ciudadana del MININTER / INEI (43 distritos)."
)

if modelo is None or df_dim is None:
    st.error(
        "No se encontraron los artefactos exportados en la carpeta `modelos_exportados/` "
        "(`modelo_tasador_lima.pkl` y `dim_distrito_app.csv`). "
        "Ejecuta primero el notebook `04_Etapas6_7_Modelado_Ablacion_SHAP.ipynb`."
    )
    st.stop()

# Métricas del modelo operativo y rango empírico de error (percentiles 10 y 90 en el conjunto de prueba)
nombre_modelo = "Modelo exportado"
mape_modelo = None
error_p10, error_p90 = -0.20, 0.20
if df_metricas is not None and not df_metricas.empty:
    nombre_modelo = str(df_metricas.iloc[0]["Modelo"])
    mape_modelo = float(df_metricas.iloc[0]["MAPE_test_%"])
    error_p10 = float(df_metricas.iloc[0]["Error_rel_p10"])
    error_p90 = float(df_metricas.iloc[0]["Error_rel_p90"])

# ==============================================================================
# BARRA LATERAL: PARÁMETROS DEL INMUEBLE
# ==============================================================================
st.sidebar.header("📐 Características del Inmueble")

lista_distritos = df_dim["Distrito"].tolist()
idx_default = lista_distritos.index("Miraflores") if "Miraflores" in lista_distritos else 0

distrito_sel = st.sidebar.selectbox("Distrito", lista_distritos, index=idx_default)
metraje_m2 = st.sidebar.number_input("Área total (m²)", min_value=20.0, max_value=850.0, value=90.0, step=5.0)
dormitorios = st.sidebar.slider("Dormitorios", min_value=1, max_value=8, value=3)
banos = st.sidebar.slider("Baños", min_value=1, max_value=7, value=2)
estacionamientos = st.sidebar.slider("Estacionamientos (cocheras)", min_value=0, max_value=5, value=1)

st.sidebar.divider()
st.sidebar.subheader("💱 Conversión Monetaria")
tipo_cambio = st.sidebar.number_input(
    "Tipo de cambio SBS (S/ por USD)", min_value=2.500, max_value=5.000, value=3.455, step=0.005, format="%.3f"
)

st.sidebar.divider()
st.sidebar.caption(f"**Modelo activo:** {nombre_modelo}")
st.sidebar.caption(f"**Fuente distrital:** {fuente_dim}")

# Extraer indicadores del distrito seleccionado
fila_dist = df_dim[df_dim["Distrito"] == distrito_sel].iloc[0]
tasa_patrimonio = float(fila_dist["Tasa_Delitos_Patrimonio_10k"])
indice_pelig = float(fila_dist["Indice_Peligrosidad"])
pct_violentos = float(fila_dist["Pct_Delitos_Violentos"])
pob_inei = int(fila_dist["Poblacion_INEI"])
total_delitos = int(fila_dist["Total_Delitos"])

# Construir DataFrame con el esquema exacto entrenado en la Etapa 6
X_input = pd.DataFrame([{
    "Metraje_m2": float(metraje_m2),
    "Dormitorios": int(dormitorios),
    "Banos": int(banos),
    "Estacionamientos": int(estacionamientos),
    "Tasa_Delitos_Patrimonio_10k": tasa_patrimonio,
    "Pct_Delitos_Violentos": pct_violentos,
    "Distrito": distrito_sel,
}])

# Inferencia en tiempo real
precio_usd_est = float(modelo.predict(X_input)[0])
precio_pen_est = precio_usd_est * tipo_cambio
precio_m2_est = precio_usd_est / metraje_m2

rango_inf_usd = precio_usd_est * (1.0 + error_p10)
rango_sup_usd = precio_usd_est * (1.0 + error_p90)

n_avisos_distrito = None
if df_avisos is not None:
    fila_avisos = df_avisos[df_avisos["Distrito"] == distrito_sel]
    if not fila_avisos.empty:
        n_avisos_distrito = int(fila_avisos.iloc[0]["N_Avisos"])

nivel_seg, icono_seg, color_seg = clasificar_peligrosidad(indice_pelig)

# ==============================================================================
# PESTAÑAS PRINCIPALES
# ==============================================================================
tab1, tab2, tab3 = st.tabs([
    "🏠 Tasación y Seguridad",
    "🛡️ Panorama Distrital (MININTER)",
    "📊 Evaluación y Ablación (H1)",
])

# ------------------------------------------------------------------------------
# TAB 1: RESULTADO DE TASACIÓN
# ------------------------------------------------------------------------------
with tab1:
    st.subheader(f"Estimación Comercial en {distrito_sel}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Valor Estimado (USD)", f"$ {precio_usd_est:,.0f}")
    c2.metric("Valor Estimado (PEN)", f"S/ {precio_pen_est:,.0f}")
    c3.metric("Precio Unitario Estimado", f"$ {precio_m2_est:,.0f} / m²")
    c4.metric("Nivel de Peligrosidad Zonal", f"{icono_seg} {nivel_seg}", f"Índice: {indice_pelig:.1f} / 100")

    st.info(
        f"**Rango orientativo:** entre **$ {rango_inf_usd:,.0f}** y **$ {rango_sup_usd:,.0f} USD** "
        f"(S/ {rango_inf_usd * tipo_cambio:,.0f} — S/ {rango_sup_usd * tipo_cambio:,.0f} PEN). "
        f"En el conjunto de prueba, 8 de cada 10 precios reales cayeron dentro de este margen alrededor de la predicción."
    )

    if n_avisos_distrito is not None and n_avisos_distrito < MIN_AVISOS_CONFIABLE:
        st.warning(
            f"El modelo se entrenó con solo {n_avisos_distrito} avisos de {distrito_sel}. "
            f"La estimación para este distrito es menos confiable que para distritos con {MIN_AVISOS_CONFIABLE} o más avisos."
        )

    col_izq, col_der = st.columns([1.15, 1])

    with col_izq:
        st.markdown("#### 🛡️ Indicadores de Seguridad Ciudadana del Distrito")
        sec_df = pd.DataFrame({
            "Indicador": [
                "Población proyectada (INEI)",
                "Total de denuncias registradas (MININTER)",
                "Tasa de delitos contra el patrimonio (x 10,000 hab.)",
                "Índice de peligrosidad normalizado (0 - 100)",
                "Proporción de delitos violentos (%)",
            ],
            "Valor": [
                f"{pob_inei:,} hab.",
                f"{total_delitos:,}",
                f"{tasa_patrimonio:.2f}",
                f"{indice_pelig:.2f}",
                f"{pct_violentos:.2f}%",
            ],
        })
        st.dataframe(sec_df, use_container_width=True, hide_index=True)

        # Comparativa rápida contra promedio metropolitano
        prom_tasa = df_dim["Tasa_Delitos_Patrimonio_10k"].mean()
        prom_viol = df_dim["Pct_Delitos_Violentos"].mean()
        st.caption(
            f"Referencia Lima Metropolitana (promedio 43 distritos): "
            f"Tasa patrimonial = **{prom_tasa:.1f}** por 10k hab. | "
            f"% Delitos violentos = **{prom_viol:.1f}%**."
        )

    with col_der:
        st.markdown("#### 📈 Posición frente al Mercado del Distrito")
        if df_mercado is not None:
            df_dist_mkt = df_mercado[df_mercado["Distrito"] == distrito_sel]
            if len(df_dist_mkt) >= 2:
                mediana_m2 = df_dist_mkt["Precio_m2_USD"].median()
                dif_pct = ((precio_m2_est - mediana_m2) / mediana_m2) * 100

                fig, ax = plt.subplots(figsize=(6.2, 3.6))
                ax.hist(df_dist_mkt["Precio_m2_USD"], bins=18, color="#5b9bd5", alpha=0.75, edgecolor="white")
                ax.axvline(mediana_m2, color="#1f4e79", linestyle="--", linewidth=1.8, label=f"Mediana distrito: ${mediana_m2:,.0f}/m²")
                ax.axvline(precio_m2_est, color="#c0392b", linestyle="-", linewidth=2.2, label=f"Tu inmueble: ${precio_m2_est:,.0f}/m²")
                ax.set_title(f"Distribución de $/m² en {distrito_sel} (n={len(df_dist_mkt)} avisos)")
                ax.set_xlabel("USD / m²")
                ax.set_ylabel("Frecuencia")
                ax.legend(fontsize=8.5)
                plt.tight_layout()
                st.pyplot(fig)

                st.caption(
                    f"Tu tasación unitaria se ubica **{dif_pct:+.1f}%** respecto a la mediana "
                    f"de avisos publicados en **{distrito_sel}**."
                )
            else:
                st.warning(f"Hay pocos avisos publicados en {distrito_sel} para graficar la distribución histórica.")
        else:
            st.info("Conecta SQL Server o coloca `dataset_modelo_lima.csv` en la carpeta para ver la distribución histórica por distrito.")

# ------------------------------------------------------------------------------
# TAB 2: PANORAMA DISTRITAL DE SEGURIDAD
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Indicadores Delictivos por Distrito (MININTER / INEI)")

    metrica_orden = st.radio(
        "Ordenar ranking por:",
        ["Tasa_Delitos_Patrimonio_10k", "Pct_Delitos_Violentos", "Indice_Peligrosidad"],
        horizontal=True,
        format_func=lambda x: {
            "Tasa_Delitos_Patrimonio_10k": "Tasa Patrimonio (x 10k hab.)",
            "Pct_Delitos_Violentos": "% Delitos Violentos",
            "Indice_Peligrosidad": "Índice de Peligrosidad (0-100)",
        }[x],
    )

    df_rank = df_dim.sort_values(metrica_orden, ascending=False).reset_index(drop=True)
    top15 = df_rank.head(15).copy()

    # Asegurar que el distrito seleccionado aparezca en el gráfico si no está en el Top 15
    if distrito_sel not in top15["Distrito"].values:
        top15 = pd.concat([top15, df_rank[df_rank["Distrito"] == distrito_sel]], ignore_index=True)

    top15 = top15.sort_values(metrica_orden, ascending=True)
    colores = ["#c0392b" if d == distrito_sel else "#2b5c8f" for d in top15["Distrito"]]

    fig2, ax2 = plt.subplots(figsize=(10, 5.2))
    ax2.barh(top15["Distrito"], top15[metrica_orden], color=colores)
    ax2.set_title(f"Top 15 Distritos según {metrica_orden} (en rojo: {distrito_sel})")
    ax2.set_xlabel(metrica_orden)
    plt.tight_layout()
    st.pyplot(fig2)

    st.markdown("#### Tabla completa de los 43 distritos")
    st.dataframe(
        df_rank[[
            "Distrito", "Poblacion_INEI", "Total_Delitos", "Delitos_Patrimonio",
            "Tasa_Delitos_Patrimonio_10k", "Indice_Peligrosidad", "Pct_Delitos_Violentos"
        ]],
        use_container_width=True,
        hide_index=True,
    )

# ------------------------------------------------------------------------------
# TAB 3: EVALUACIÓN Y ESTUDIO DE ABLACIÓN
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Resultados del Estudio de Ablación (Validación de Hipótesis H1)")

    if df_ablacion is not None:
        st.caption(
            "Validación cruzada agrupada por distrito (GroupKFold, 5 particiones): cada modelo se evalúa en distritos "
            "que no vio al entrenar. Escenario A: solo atributos físicos. Escenario B: atributos físicos + criminalidad."
        )
        st.dataframe(
            df_ablacion[["Escenario", "Modelo", "R2_media", "R2_std", "RMSE_media", "MAE_media", "MAPE_media"]]
            .sort_values(["Escenario", "RMSE_media"]),
            use_container_width=True,
            hide_index=True,
        )

        pivot_r2 = df_ablacion.pivot(index="Modelo", columns="Escenario", values="R2_media")
        pivot_std = df_ablacion.pivot(index="Modelo", columns="Escenario", values="R2_std")

        # Los modelos con R² medio negativo (inestables entre particiones) se muestran solo en la tabla
        modelos_graf = pivot_r2[(pivot_r2 > 0).all(axis=1)].index
        modelos_omitidos = [m for m in pivot_r2.index if m not in modelos_graf]
        pivot_r2 = pivot_r2.loc[modelos_graf].sort_values(pivot_r2.columns[-1], ascending=False)
        pivot_std = pivot_std.loc[pivot_r2.index]

        fig3, ax3 = plt.subplots(figsize=(9, 4.2))
        pivot_r2.plot(kind="bar", yerr=pivot_std, ax=ax3, color=["#5b9bd5", "#1f4e79"], capsize=3)
        ax3.set_ylim(0, 1.15)
        ax3.set_title("R² medio en distritos no vistos — Solo físicas vs. Físicas + criminalidad")
        ax3.set_ylabel("R² (media ± desv. estándar)")
        ax3.set_xlabel("")
        ax3.legend(["A: solo físicas", "B: físicas + criminalidad"], loc="upper center", ncol=2, fontsize=8.5, frameon=False)
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig3)

        if modelos_omitidos:
            st.caption(
                f"Se omite del gráfico: {', '.join(modelos_omitidos)}. Su R² medio es negativo y varía mucho entre "
                f"particiones (ver tabla); incluirlo ampliaría el eje y ocultaría las diferencias entre los demás modelos."
            )
    else:
        st.info("No se encontró `resultados_ablacion.csv` en `modelos_exportados/`.")

    st.divider()
    st.markdown(
        """
        **Notas metodológicas y alcance del tasador:**
        - Los valores estimados se basan en **precios de oferta (*asking prices*)** publicados en Urbania Perú, por lo que reflejan expectativas comerciales de mercado e incluyen márgenes habituales de negociación.
        - La criminalidad tiene un solo valor por distrito. Su aporte al modelo indica cuánto ayuda a ordenar los distritos por precio; no mide un efecto causal del delito sobre el precio.
        - Los indicadores de seguridad provienen de denuncias georreferenciadas del **Mapa del Delito del MININTER** normalizadas con proyecciones poblacionales del **INEI**, sujetas al subregistro delictivo propio de estadísticas policiales.
        """
    )
