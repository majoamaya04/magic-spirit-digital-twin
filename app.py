"""
Magic Spirit - Gemelo Digital Interactivo
==========================================
Aplicación Streamlit para el Proyecto C2 de Planeación, Programación
y Control de las Operaciones - Universidad de La Sabana.

Ejecutar con:
    streamlit run app.py
"""
import warnings
warnings.filterwarnings("ignore")

import copy

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from digital_twin import (
    FORECAST_FAMILIA,
    PARAMS_PRODUCTO,
    PARAMS_SISTEMA,
    MotorSimulacion,
    Producto,
    correr_escenario,
    tabla_comparativa,
)

# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Magic Spirit - Gemelo Digital",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta de colores coherente
COLORS = {
    "primary":   "#8B2E5A",   # vino - identidad Magic Spirit
    "secondary": "#D4A574",   # dorado cálido
    "success":   "#2E8B57",
    "warning":   "#F39C12",
    "danger":    "#C0392B",
    "info":      "#3498DB",
    "muted":     "#95A5A6",
}

# CSS ligero para que se vea más profesional (nada agresivo)
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1, h2, h3 { color: #8B2E5A; }
    [data-testid="stMetricValue"] { font-size: 1.6rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding-left: 18px; padding-right: 18px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CACHING DE SIMULACIONES
# ============================================================

@st.cache_data(show_spinner="Simulando gemelo digital...")
def correr_simulacion_base(seed: int, horas_extra: bool, forecast_tuple: tuple):
    """
    Wrapper cacheable de correr_escenario. El forecast se pasa como tupla
    para ser hasheable; aquí lo convertimos de vuelta a dict.
    """
    forecast = dict(forecast_tuple)
    motor = correr_escenario(
        "Base",
        forecast_mod    = forecast,
        seed            = seed,
        con_horas_extra = horas_extra,
    )
    return _extraer_resultados(motor)


@st.cache_data(show_spinner="Corriendo escenarios what-if (puede tardar ~5-10 s)...")
def correr_todos_escenarios(seed: int, forecast_tuple: tuple):
    """Corre los 9 escenarios del notebook original."""
    forecast_base = dict(forecast_tuple)
    escenarios_cfg = {
        "E0 Base":              {},
        "E1 Demanda +20%":      {"forecast_mod": {k: int(v * 1.20) for k, v in forecast_base.items()}},
        "E2 Demanda -20%":      {"forecast_mod": {k: int(v * 0.80) for k, v in forecast_base.items()}},
        "E3 Premium x2":        {"prop_premium": 0.40},
        "E4 Carga comerc. +15%":{"fracc_comercial_add": 0.15},
        "E5 Sin horas extra":   {"con_horas_extra": False},
        "E6 Lead time +7d":     {"lead_time_add": 7},
        "E7 Stock hilo bajo":   {"stock_inicial_rollos": 2},
        "E8 Combinado crítico": {
            "forecast_mod":        {k: int(v * 1.20) for k, v in forecast_base.items()},
            "fracc_comercial_add": 0.15,
            "con_horas_extra":     False,
        },
        "E9 Combinado favor.":  {"stock_inicial_rollos": 15, "con_horas_extra": True},
    }

    resultados = {}
    for nombre, cfg in escenarios_cfg.items():
        # Llenar defaults
        cfg.setdefault("forecast_mod", forecast_base)
        motor = correr_escenario(nombre, seed=seed, **cfg)
        resultados[nombre] = _extraer_resultados(motor)
    return resultados


def _extraer_resultados(motor: MotorSimulacion) -> dict:
    """Extrae todo lo relevante del motor a estructuras serializables."""
    df_sens = motor.sensores.a_dataframe()
    return {
        "df_sensores":       df_sens,
        "kpis":              motor.sensores.resumen_kpis(),
        "n_despachados":     len(motor.pedidos_despachados),
        "n_incumplidos":     len(motor.pedidos_incumplidos),
        "alertas":           list(motor.alertas_log),
        "stock_hilo_final":  motor.inventario.stock_rollos,
    }


# ============================================================
# SIDEBAR - CONTROLES GLOBALES
# ============================================================

with st.sidebar:
    st.markdown("## 🧵 Magic Spirit")
    st.caption("Gemelo Digital Interactivo")
    st.markdown("---")

    st.markdown("### ⚙️ Parámetros de simulación")
    seed = st.number_input(
        "Semilla aleatoria", min_value=0, max_value=9999, value=42, step=1,
        help="Cambia la semilla para ver diferentes realizaciones estocásticas."
    )
    horas_extra_on = st.checkbox(
        "Activar horas extra en empaque", value=True,
        help="Permite hasta 60 min extra/día durante meses pico (Feb, May, Nov, Dic).",
    )

    st.markdown("### 📊 Ajuste de demanda")
    demand_multiplier = st.slider(
        "Multiplicador global de demanda", 0.5, 2.0, 1.0, 0.05,
        help="Escala todo el pronóstico Holt-Winters. 1.0 = base."
    )

    st.markdown("---")
    st.markdown("### 📘 Sobre este modelo")
    st.caption(
        "Gemelo digital del sistema productivo de Magic Spirit. "
        "Simula 1 año de operación (mar 2026 - feb 2027) con demanda estocástica, "
        "producción y empaque con capacidad limitada, e inventario de hilo rojo "
        "con lead time variable."
    )
    st.caption(
        "**Proyecto C2** - Planeación, Programación y Control de las Operaciones. "
        "Universidad de La Sabana, 2026."
    )

# Forecast ajustado por el slider
forecast_ajustado = {
    k: max(1, int(round(v * demand_multiplier)))
    for k, v in FORECAST_FAMILIA.items()
}
forecast_tuple = tuple(sorted(forecast_ajustado.items()))


# ============================================================
# HEADER
# ============================================================

col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("🧵 Magic Spirit — Gemelo Digital")
    st.markdown(
        "*Modelo virtual interactivo del sistema de producción y S&OP*"
    )
with col_h2:
    st.markdown("")
    st.markdown(
        f"<div style='text-align: right; padding-top: 8px;'>"
        f"<small style='color: #95A5A6;'>Horizonte simulado</small><br>"
        f"<b style='color: #8B2E5A;'>Mar 2026 → Feb 2027</b>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Estudio Preliminar",
    "📈 Pronóstico de Demanda",
    "🏭 Simulación Gemelo Digital",
    "🔍 Escenarios What-If & S&OP",
])


# ============================================================
# TAB 1 - ESTUDIO PRELIMINAR
# ============================================================

with tab1:
    st.header("Estudio Preliminar del Sistema Productivo")
    st.markdown(
        "Caracterización del sistema de manufactura **Make-to-Order** de Magic Spirit, "
        "incluyendo métricas de eficiencia, parámetros por producto y restricciones de capacidad."
    )

    # ── Métricas clave del sistema ──
    st.markdown("### Parámetros base del sistema")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Capacidad producción", "180 min/día", "~16 uds/día")
    c2.metric("Capacidad empaque",    "60 min/día",  "~25 uds/día (bruto)")
    c3.metric("Plazo estándar",       "3 días",      "1 día premium")
    c4.metric("Productos en familia", "6 SKUs",      "todos con hilo rojo")

    st.markdown("### Parámetros por producto")
    df_productos = pd.DataFrame([
        {
            "Producto": p.nombre,
            "T. producción (min/ud)": p.tiempo_produccion_min,
            "Hilo rojo (cm/ud)": p.hilo_rojo_cm,
            "Horas-hombre (h/ud)": p.horas_hombre,
            "Participación histórica (%)": round(
                PARAMS_SISTEMA.participacion_productos.get(prod, 0) * 100, 1
            ),
        }
        for prod, p in PARAMS_PRODUCTO.items()
    ])
    st.dataframe(df_productos, use_container_width=True, hide_index=True)

    # ── Métricas Little's Law ──
    st.markdown("### Métricas de eficiencia (Little's Law)")
    st.markdown(
        "Cálculos teóricos basados en capacidad nominal y demanda promedio del horizonte. "
        "Los valores simulados aparecen en la pestaña **Simulación Gemelo Digital**."
    )

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        demanda_anual    = sum(forecast_ajustado.values())
        dias_habiles_año = 260
        throughput_teorico = demanda_anual / dias_habiles_año
        takt_time = (180 * 60) / throughput_teorico if throughput_teorico > 0 else 0  # en seg

        st.markdown("**Throughput objetivo**")
        st.metric("Unidades/día promedio", f"{throughput_teorico:.2f}")
        st.metric("Takt time (s/unidad)",  f"{takt_time:.0f}")

    with col_m2:
        cycle_time_prod_ud = PARAMS_SISTEMA.tiempo_produccion_min_unidad
        cycle_time_emp_ud  = PARAMS_SISTEMA.tiempo_empaque_min_unidad

        st.markdown("**Cycle time unitario**")
        st.metric("Producción", f"{cycle_time_prod_ud:.2f} min/ud")
        st.metric("Empaque",    f"{cycle_time_emp_ud:.2f} min/ud")

    # ── Cuellos de botella teóricos ──
    st.markdown("### Análisis de capacidad vs demanda")
    horizonte_df = pd.DataFrame([
        {
            "Mes":             mes.strftime("%b-%Y"),
            "Forecast mensual": unidades,
            "Factor estac.":    PARAMS_SISTEMA.factores_estacionales[mes.month],
            "Uds/día esperadas": round(
                (unidades / 22) * PARAMS_SISTEMA.factores_estacionales[mes.month], 2
            ),
        }
        for mes, unidades in forecast_ajustado.items()
    ])

    cap_prod_dia = 180 / PARAMS_SISTEMA.tiempo_produccion_min_unidad
    cap_emp_dia  = (60 * 0.70) / PARAMS_SISTEMA.tiempo_empaque_min_unidad

    fig_cap = go.Figure()
    fig_cap.add_bar(
        x=horizonte_df["Mes"], y=horizonte_df["Uds/día esperadas"],
        name="Demanda (uds/día)",
        marker_color=COLORS["primary"],
    )
    fig_cap.add_hline(
        y=cap_prod_dia, line_dash="dash", line_color=COLORS["info"],
        annotation_text=f"Cap. producción ({cap_prod_dia:.1f})",
        annotation_position="top right",
    )
    fig_cap.add_hline(
        y=cap_emp_dia, line_dash="dash", line_color=COLORS["warning"],
        annotation_text=f"Cap. empaque efectiva ({cap_emp_dia:.1f})",
        annotation_position="bottom right",
    )
    fig_cap.update_layout(
        title="Demanda esperada vs capacidad nominal",
        yaxis_title="Unidades / día",
        xaxis_title="",
        height=400,
        template="plotly_white",
        showlegend=True,
    )
    st.plotly_chart(fig_cap, use_container_width=True)

    with st.expander("💡 Interpretación"):
        st.markdown(
            "- **Cuello de botella identificado:** Empaque (~17 uds/día efectivas) es más "
            "restrictivo que producción (~16 uds/día). En meses pico (febrero, mayo, "
            "noviembre, diciembre) la demanda supera la capacidad de empaque, lo cual "
            "justifica el uso de horas extra y apoyo cruzado desde producción.\n"
            "- **Estacionalidad:** diciembre (×1.40) y febrero (×1.35) son los picos "
            "más exigentes. Agosto y octubre (×0.80) son valles que permiten acumular "
            "inventario de producto terminado si se activa una política híbrida MTO/MTS.\n"
            "- **Restricción de materia prima:** el hilo rojo 1mm se importa con lead "
            "time de ~30 días ± 7. El punto de reorden (1 rollo) y stock inicial "
            "(2 rollos) son políticas sensibles que se evalúan en los escenarios."
        )

    # ── Estacionalidad ──
    st.markdown("### Factores estacionales (mensuales)")
    meses_labels = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
    factores = [PARAMS_SISTEMA.factores_estacionales[m] for m in range(1, 13)]
    colores_fact = [
        COLORS["danger"]  if f >= 1.30 else
        COLORS["warning"] if f >= 1.10 else
        COLORS["muted"]   if f <= 0.85 else
        COLORS["info"]
        for f in factores
    ]

    fig_est = go.Figure()
    fig_est.add_bar(
        x=meses_labels, y=factores, marker_color=colores_fact,
        text=[f"{f:.2f}" for f in factores], textposition="outside",
    )
    fig_est.add_hline(y=1.0, line_dash="dot", line_color="gray")
    fig_est.update_layout(
        title="Multiplicador estacional sobre la demanda media",
        yaxis_title="Factor", xaxis_title="",
        height=350, template="plotly_white", showlegend=False,
        yaxis=dict(range=[0, max(factores) * 1.2]),
    )
    st.plotly_chart(fig_est, use_container_width=True)


# ============================================================
# TAB 2 - PRONÓSTICO
# ============================================================

with tab2:
    st.header("Pronóstico de Demanda — Holt-Winters Multiplicativo")
    st.markdown(
        "Pronóstico obtenido del modelo **Holt-Winters multiplicativo** entrenado con "
        "3 años de historia (Mar 2023 – Feb 2026) sobre la serie mensual agregada de la "
        "familia Magic Spirit. Mejor MAPE frente al modelo aditivo en validación."
    )

    # ── Forecast mensual ──
    df_fcst = pd.DataFrame([
        {
            "Mes":              mes,
            "MesLabel":         mes.strftime("%b-%Y"),
            "Unidades":         uds,
            "Factor estac.":    PARAMS_SISTEMA.factores_estacionales[mes.month],
        }
        for mes, uds in forecast_ajustado.items()
    ])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total anual proyectado", f"{df_fcst['Unidades'].sum():,} uds")
    c2.metric("Mes pico",      f"{df_fcst.loc[df_fcst['Unidades'].idxmax(), 'MesLabel']}",
              f"{df_fcst['Unidades'].max()} uds")
    c3.metric("Mes valle",     f"{df_fcst.loc[df_fcst['Unidades'].idxmin(), 'MesLabel']}",
              f"{df_fcst['Unidades'].min()} uds")
    c4.metric("Promedio mensual", f"{df_fcst['Unidades'].mean():.0f} uds")

    # ── Gráfica principal ──
    colores_fcst = [
        COLORS["danger"]  if f >= 1.30 else
        COLORS["warning"] if f >= 1.10 else
        COLORS["muted"]   if f <= 0.85 else
        COLORS["info"]
        for f in df_fcst["Factor estac."]
    ]

    fig_fcst = go.Figure()
    fig_fcst.add_bar(
        x=df_fcst["MesLabel"], y=df_fcst["Unidades"],
        marker_color=colores_fcst,
        text=df_fcst["Unidades"], textposition="outside",
        name="Forecast",
    )
    fig_fcst.update_layout(
        title="Pronóstico mensual — Familia Magic Spirit",
        yaxis_title="Unidades", xaxis_title="",
        height=420, template="plotly_white", showlegend=False,
    )
    st.plotly_chart(fig_fcst, use_container_width=True)

    # ── Desagregación por producto ──
    st.markdown("### Desagregación por producto")
    st.markdown(
        "El pronóstico de la familia se desagrega usando la participación histórica "
        "de cada SKU durante los últimos 12 meses."
    )

    # Construir tabla desagregada
    productos_list = list(PARAMS_SISTEMA.participacion_productos.keys())
    shares = np.array(
        [PARAMS_SISTEMA.participacion_productos[p] for p in productos_list]
    )
    shares = shares / shares.sum()

    matriz = []
    for mes, uds_fam in forecast_ajustado.items():
        fila = {"Mes": mes.strftime("%b-%Y")}
        for i, prod in enumerate(productos_list):
            fila[prod.value] = int(round(uds_fam * shares[i]))
        matriz.append(fila)
    df_desagregado = pd.DataFrame(matriz)

    st.dataframe(df_desagregado, use_container_width=True, hide_index=True)

    # ── Gráfica de barras apiladas ──
    fig_stack = go.Figure()
    colores_productos = px.colors.qualitative.Set2
    for i, prod in enumerate(productos_list):
        fig_stack.add_bar(
            x=df_desagregado["Mes"],
            y=df_desagregado[prod.value],
            name=prod.value,
            marker_color=colores_productos[i % len(colores_productos)],
        )
    fig_stack.update_layout(
        barmode="stack",
        title="Pronóstico desagregado por producto (unidades)",
        yaxis_title="Unidades", xaxis_title="",
        height=450, template="plotly_white",
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_stack, use_container_width=True)

    with st.expander("📊 Métricas de desempeño del modelo (del notebook)"):
        st.markdown(
            "El modelo **Holt-Winters multiplicativo** fue seleccionado sobre el aditivo "
            "al obtener menor error en el conjunto de validación (últimos 12 meses):"
        )
        df_metricas = pd.DataFrame({
            "Modelo":   ["HW Multiplicativo", "HW Aditivo"],
            "MAPE (%)": ["menor",             "mayor"],
            "RMSE":     ["menor",             "mayor"],
            "TS":       ["sin sesgo",         "sin sesgo"],
            "Decisión": ["✅ Elegido",         "Descartado"],
        })
        st.dataframe(df_metricas, use_container_width=True, hide_index=True)
        st.caption(
            "Los valores numéricos exactos están en el notebook original. "
            "La estacionalidad marcada de la familia (diciembre ×1.40, febrero ×1.35) "
            "favorece el modelo multiplicativo."
        )


# ============================================================
# TAB 3 - SIMULACIÓN GEMELO DIGITAL
# ============================================================

with tab3:
    st.header("Simulación del Gemelo Digital")
    st.markdown(
        "Motor de simulación diaria que integra: generación estocástica de demanda "
        "(Poisson + Normal), cola de producción priorizada por pedidos premium, "
        "empaque con carga comercial variable, apoyo cruzado entre operarios, "
        "inventario de hilo con lead time variable y política de reorden, "
        "y sensores virtuales que registran 22 variables de estado por día."
    )

    col_run1, col_run2 = st.columns([1, 3])
    with col_run1:
        run_btn = st.button("▶️ Correr simulación", type="primary", use_container_width=True)
    with col_run2:
        st.caption(
            "La simulación cubre **~260 días hábiles** con ~2,000-3,500 pedidos procesados. "
            "El resultado queda en caché por semilla y parámetros."
        )

    # Correr automáticamente al entrar también
    resultados = correr_simulacion_base(seed, horas_extra_on, forecast_tuple)

    df_s           = resultados["df_sensores"]
    kpis           = resultados["kpis"]
    n_despachados  = resultados["n_despachados"]
    n_incumplidos  = resultados["n_incumplidos"]
    alertas        = resultados["alertas"]

    # ── KPIs header ──
    st.markdown("### 📊 KPIs principales")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Throughput total", f"{kpis['Throughput total (unidades)']:,} uds")
    k2.metric("% a tiempo",       f"{kpis['% despacho a tiempo (promedio)']}%",
              delta=f"{kpis['% despacho a tiempo (promedio)'] - 80:.1f} pp vs meta 80%",
              delta_color="normal")
    k3.metric("Cycle time",       f"{kpis['Cycle time promedio (días)']} días")
    k4.metric("Backlog máx",      f"{kpis['Backlog máximo']} pedidos",
              delta=f"límite {PARAMS_SISTEMA.max_backlog_pedidos}",
              delta_color="off")
    k5.metric("Util. empaque",    f"{kpis['Utilización empaque (promedio)']}%")

    k6, k7, k8, k9, k10 = st.columns(5)
    k6.metric("Pedidos despach.", f"{n_despachados:,}")
    k7.metric("Pedidos incumpl.", f"{n_incumplidos:,}",
              delta=f"{n_incumplidos / max(n_despachados + n_incumplidos, 1) * 100:.1f}%",
              delta_color="inverse")
    k8.metric("Util. producción", f"{kpis['Utilización producción (promedio)']}%")
    k9.metric("Escasez hilo",     f"{kpis['Eventos escasez hilo']} eventos",
              delta_color="inverse")
    k10.metric("Compras emerg.",  f"{kpis['Compras emergencia (rollos)']} rollos",
               delta_color="inverse")

    # ── Dashboard sensores (6 paneles) ──
    st.markdown("### 🔬 Dashboard de sensores virtuales")

    fig_dash = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "WIP y colas",              "Throughput diario",
            "Utilización de recursos",  "Nivel de servicio",
            "Inventario hilo rojo",     "Backlog acumulado",
        ),
        vertical_spacing=0.10, horizontal_spacing=0.08,
    )

    # 1. WIP y colas
    fig_dash.add_trace(go.Scatter(
        x=df_s.index, y=df_s["wip_total"],
        name="WIP total", line=dict(color=COLORS["danger"], width=2),
    ), row=1, col=1)
    fig_dash.add_trace(go.Scatter(
        x=df_s.index, y=df_s["cola_produccion"],
        name="Cola producción", line=dict(color=COLORS["info"], dash="dash"),
    ), row=1, col=1)
    fig_dash.add_trace(go.Scatter(
        x=df_s.index, y=df_s["cola_empaque"],
        name="Cola empaque", line=dict(color=COLORS["warning"], dash="dash"),
    ), row=1, col=1)

    # 2. Throughput
    fig_dash.add_trace(go.Bar(
        x=df_s.index, y=df_s["unidades_empacadas_dia"],
        name="Empacadas", marker_color=COLORS["success"],
    ), row=1, col=2)
    fig_dash.add_trace(go.Scatter(
        x=df_s.index, y=df_s["unidades_producidas_dia"],
        name="Producidas", line=dict(color=COLORS["info"], width=1),
    ), row=1, col=2)

    # 3. Utilización
    fig_dash.add_trace(go.Scatter(
        x=df_s.index, y=df_s["utilizacion_produccion"] * 100,
        name="Prod (%)", line=dict(color="#9B59B6"),
    ), row=2, col=1)
    fig_dash.add_trace(go.Scatter(
        x=df_s.index, y=df_s["utilizacion_empaque"] * 100,
        name="Empaque (%)", line=dict(color=COLORS["warning"]),
    ), row=2, col=1)

    # 4. Nivel de servicio
    fig_dash.add_trace(go.Scatter(
        x=df_s.index, y=df_s["pct_despacho_a_tiempo"] * 100,
        name="% a tiempo", line=dict(color="#1ABC9C", width=2),
        fill="tozeroy", fillcolor="rgba(26, 188, 156, 0.2)",
    ), row=2, col=2)

    # 5. Inventario hilo
    fig_dash.add_trace(go.Scatter(
        x=df_s.index, y=df_s["stock_hilo_rollos"],
        name="Stock (rollos)", line=dict(color=COLORS["danger"], width=2),
    ), row=3, col=1)

    # 6. Backlog
    fig_dash.add_trace(go.Scatter(
        x=df_s.index, y=df_s["backlog_acumulado"],
        name="Backlog", fill="tozeroy",
        line=dict(color=COLORS["danger"]),
        fillcolor="rgba(192, 57, 43, 0.3)",
    ), row=3, col=2)

    fig_dash.update_layout(
        height=850, template="plotly_white",
        showlegend=True, legend=dict(orientation="h", y=-0.05),
    )
    fig_dash.update_yaxes(title_text="Pedidos",  row=1, col=1)
    fig_dash.update_yaxes(title_text="Unidades", row=1, col=2)
    fig_dash.update_yaxes(title_text="%", range=[0, 110], row=2, col=1)
    fig_dash.update_yaxes(title_text="%", range=[0, 105], row=2, col=2)
    fig_dash.update_yaxes(title_text="Rollos",   row=3, col=1)
    fig_dash.update_yaxes(title_text="Pedidos",  row=3, col=2)

    st.plotly_chart(fig_dash, use_container_width=True)

    # ── Tabla mensual ──
    st.markdown("### 📅 Resumen mensual")
    df_s_idx = df_s.copy()
    df_s_idx.index = pd.DatetimeIndex(df_s_idx.index)
    mensual = df_s_idx.resample("ME").agg({
        "unidades_producidas_dia":  "sum",
        "unidades_empacadas_dia":   "sum",
        "pedidos_despachados_dia":  "sum",
        "backlog_acumulado":        "max",
        "utilizacion_produccion":   "mean",
        "utilizacion_empaque":      "mean",
        "horas_extra_empaque_min":  "sum",
        "pct_despacho_a_tiempo":    "mean",
        "stock_hilo_rollos":        "min",
    }).round(2)
    mensual.index = mensual.index.strftime("%b-%Y")
    mensual.columns = [
        "Uds producidas", "Uds empacadas", "Pedidos desp.",
        "Backlog máx", "Util. prod (%)", "Util. emp (%)",
        "HExtra emp (min)", "% a tiempo", "Stock hilo mín",
    ]
    mensual["Util. prod (%)"] = (mensual["Util. prod (%)"] * 100).round(1)
    mensual["Util. emp (%)"]  = (mensual["Util. emp (%)"] * 100).round(1)
    mensual["% a tiempo"]     = (mensual["% a tiempo"] * 100).round(1)
    st.dataframe(mensual, use_container_width=True)

    # ── Alertas ──
    if alertas:
        st.markdown("### 🚨 Alertas registradas")
        with st.expander(f"Ver {len(alertas)} alertas emitidas por el sistema SCADA virtual"):
            # Mostrar solo las últimas 50 para no saturar
            for alerta in alertas[-50:]:
                st.text(alerta)
            if len(alertas) > 50:
                st.caption(f"... mostrando las últimas 50 de {len(alertas)} totales.")


# ============================================================
# TAB 4 - ESCENARIOS WHAT-IF
# ============================================================

with tab4:
    st.header("Análisis de Escenarios What-If & Plan S&OP")
    st.markdown(
        "Evaluación de **10 escenarios** para informar el proceso de Sales & Operations "
        "Planning. Cada escenario corre una simulación completa del año con los "
        "mismos sensores virtuales, y se comparan sobre KPIs clave de servicio, "
        "throughput y costos operativos."
    )

    correr_wi = st.button(
        "▶️ Correr los 10 escenarios",
        type="primary",
        help="Si ya se corrieron con estos parámetros, se usan los cacheados (instantáneo).",
    )

    # Correr todos los escenarios (cacheado)
    todos = correr_todos_escenarios(seed, forecast_tuple)

    # Construir tabla comparativa
    filas = []
    for nombre, res in todos.items():
        kpis_esc = res["kpis"]
        filas.append({
            "Escenario":           nombre,
            "Uds empacadas":       kpis_esc.get("Throughput total (unidades)", 0),
            "Ped. despachados":    res["n_despachados"],
            "Ped. incumplidos":    res["n_incumplidos"],
            "Backlog máx":         int(res["df_sensores"]["backlog_acumulado"].max()),
            "Cycle time (d)":      kpis_esc.get("Cycle time promedio (días)", 0),
            "Util. prod (%)":      kpis_esc.get("Utilización producción (promedio)", 0),
            "Util. emp (%)":       kpis_esc.get("Utilización empaque (promedio)", 0),
            "% a tiempo":          kpis_esc.get("% despacho a tiempo (promedio)", 0),
            "HExtra (min)":        kpis_esc.get("Horas extra empaque (min totales)", 0),
            "Escasez hilo":        kpis_esc.get("Eventos escasez hilo", 0),
            "Compras emerg.":      kpis_esc.get("Compras emergencia (rollos)", 0),
        })
    tabla = pd.DataFrame(filas).set_index("Escenario")

    # ── KPIs comparativos vs base ──
    st.markdown("### 🎯 Comparativa global")
    base_row = tabla.loc["E0 Base"]
    mejor_ns = tabla["% a tiempo"].idxmax()
    peor_back = tabla["Backlog máx"].idxmax()
    mayor_tp = tabla["Uds empacadas"].idxmax()

    cc1, cc2, cc3 = st.columns(3)
    cc1.info(f"🏆 **Mejor servicio:** {mejor_ns}  \n{tabla.loc[mejor_ns, '% a tiempo']}% a tiempo")
    cc2.warning(f"🚨 **Mayor backlog:** {peor_back}  \n{int(tabla.loc[peor_back, 'Backlog máx'])} pedidos")
    cc3.success(f"📦 **Mayor throughput:** {mayor_tp}  \n{int(tabla.loc[mayor_tp, 'Uds empacadas']):,} uds")

    st.markdown("### 📋 Tabla comparativa completa")
    st.dataframe(
        tabla.style.format({
            "Uds empacadas":    "{:,.0f}",
            "Ped. despachados": "{:,.0f}",
            "Ped. incumplidos": "{:,.0f}",
            "Backlog máx":      "{:.0f}",
            "Cycle time (d)":   "{:.2f}",
            "Util. prod (%)":   "{:.1f}",
            "Util. emp (%)":    "{:.1f}",
            "% a tiempo":       "{:.1f}",
            "HExtra (min)":     "{:,.0f}",
            "Escasez hilo":     "{:.0f}",
            "Compras emerg.":   "{:.0f}",
        }).background_gradient(subset=["% a tiempo"], cmap="RdYlGn", vmin=50, vmax=100)
          .background_gradient(subset=["Backlog máx"], cmap="RdYlGn_r"),
        use_container_width=True,
    )

    # ── Gráficas comparativas ──
    st.markdown("### 📊 Comparativa visual por KPI")

    fig_comp = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            "Throughput (uds empacadas)", "Backlog máximo",         "Nivel de servicio",
            "Utilización empaque",        "Cycle time promedio",    "Pedidos incumplidos",
        ),
        vertical_spacing=0.18, horizontal_spacing=0.08,
    )

    nombres_short = [n.split(" ", 1)[0] for n in tabla.index]
    paleta = px.colors.qualitative.Bold[:len(tabla.index)]

    def _add_bar(fig, row, col, col_name, umbral=None):
        fig.add_trace(go.Bar(
            x=nombres_short, y=tabla[col_name],
            marker_color=paleta, showlegend=False,
            text=tabla[col_name].round(1), textposition="outside",
        ), row=row, col=col)
        if umbral is not None:
            fig.add_hline(
                y=umbral, line_dash="dash", line_color="red",
                row=row, col=col,
            )

    _add_bar(fig_comp, 1, 1, "Uds empacadas")
    _add_bar(fig_comp, 1, 2, "Backlog máx", umbral=PARAMS_SISTEMA.max_backlog_pedidos)
    _add_bar(fig_comp, 1, 3, "% a tiempo", umbral=80)
    _add_bar(fig_comp, 2, 1, "Util. emp (%)", umbral=90)
    _add_bar(fig_comp, 2, 2, "Cycle time (d)")
    _add_bar(fig_comp, 2, 3, "Ped. incumplidos")

    fig_comp.update_layout(height=700, template="plotly_white", showlegend=False)
    st.plotly_chart(fig_comp, use_container_width=True)

    # ── Evolución del backlog ──
    st.markdown("### 📉 Evolución temporal del backlog por escenario")

    fig_bt = go.Figure()
    for i, (nombre, res) in enumerate(todos.items()):
        fig_bt.add_trace(go.Scatter(
            x=res["df_sensores"].index,
            y=res["df_sensores"]["backlog_acumulado"],
            name=nombre, line=dict(color=paleta[i % len(paleta)], width=1.5),
            opacity=0.85,
        ))
    fig_bt.add_hline(
        y=PARAMS_SISTEMA.max_backlog_pedidos, line_dash="dot", line_color="black",
        annotation_text="Límite operacional", annotation_position="top right",
    )
    fig_bt.update_layout(
        title="Comparativa de evolución del backlog (pedidos en cola)",
        yaxis_title="Pedidos en backlog", xaxis_title="",
        height=500, template="plotly_white",
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    # ── Plan S&OP ──
    st.markdown("### 📑 Plan S&OP derivado del análisis")
    with st.expander("Ver propuesta de plan agregado", expanded=True):
        st.markdown(f"""
**Diagnóstico del escenario base (E0):**
- Throughput de **{int(base_row['Uds empacadas']):,}** unidades en el año con **{base_row['% a tiempo']:.1f}%** de pedidos a tiempo.
- Backlog máximo de **{int(base_row['Backlog máx'])}** pedidos — {'⚠️ por encima' if base_row['Backlog máx'] > PARAMS_SISTEMA.max_backlog_pedidos else '✅ dentro'} del límite operacional ({PARAMS_SISTEMA.max_backlog_pedidos}).
- Utilización de empaque promedio de **{base_row['Util. emp (%)']:.1f}%** — es el cuello de botella confirmado.

**Políticas recomendadas para el S&OP Mar 2026 – Feb 2027:**

1. **Producción:** mantener la política MTO con priorización de pedidos premium en cola. Activar apoyo cruzado producción→empaque en días con cola de empaque > 3 pedidos.

2. **Capacidad:** activar horas extra en empaque (+60 min/día) durante los meses declarados pico: **febrero, mayo, noviembre y diciembre**. El escenario E5 (sin horas extra) muestra cómo cae el servicio sin esta medida.

3. **Inventario de hilo rojo:** mantener stock inicial de al menos **3 rollos** y reorden de 3 rollos al caer a 1. El escenario E7 (stock inicial bajo) y E6 (lead time +7d) muestran riesgo de paros productivos. El escenario E9 con 15 rollos iniciales confirma que sobreaprovisionar es excesivo.

4. **Pricing/mix:** la propuesta de duplicar premium (E3) muestra impacto en cycle time por la priorización — revisar capacidad antes de agresivamente mover el mix comercial.

5. **Señal de alarma:** si la demanda real se acerca a E1 (+20%) con la carga comercial actual +15% (E4), entramos en el escenario crítico E8 donde el sistema colapsa. Monitorear el throughput semanal contra el forecast para activar medidas correctivas tempranas.
        """)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "🧵 Magic Spirit — Gemelo Digital | Proyecto C2 · Planeación, Programación y Control "
    "de las Operaciones · Universidad de La Sabana · 2026. Construido con Streamlit, "
    "Plotly y un motor de simulación diaria en Python."
)
