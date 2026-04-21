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

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Estudio Preliminar",
    "📈 Pronóstico de Demanda",
    "🏭 Simulación Gemelo Digital",
    "🔍 Escenarios What-If & S&OP",
    "🎯 Desagregación por Producto",
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
        "36 meses de historia (Mar 2023 – Feb 2026) sobre la serie mensual agregada de "
        "la familia Magic Spirit (6 SKUs · 4,000 unidades históricas). "
        "Seleccionado sobre el aditivo en validación con **MAPE 18.66% vs 20.72%** "
        "y tracking signal -1.62 (sin sesgo sistemático)."
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

    # ── Gráfica histórico + forecast (serie completa) ──
    from digital_twin import HISTORICO_FAMILIA

    hist_dates = list(HISTORICO_FAMILIA.keys())
    hist_vals  = list(HISTORICO_FAMILIA.values())
    fcst_dates = list(forecast_ajustado.keys())
    fcst_vals  = list(forecast_ajustado.values())

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=hist_dates, y=hist_vals,
        mode="lines+markers", name="Histórico (36 meses)",
        line=dict(color=COLORS["primary"], width=2.5),
        marker=dict(size=5),
    ))
    # Línea puente que conecta el último punto histórico con el primer forecast
    fig_hist.add_trace(go.Scatter(
        x=[hist_dates[-1], fcst_dates[0]],
        y=[hist_vals[-1],  fcst_vals[0]],
        mode="lines",
        line=dict(color=COLORS["secondary"], width=2, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ))
    fig_hist.add_trace(go.Scatter(
        x=fcst_dates, y=fcst_vals,
        mode="lines+markers", name="Forecast HW Mul (12 meses)",
        line=dict(color=COLORS["secondary"], width=2.5, dash="dash"),
        marker=dict(size=6, symbol="diamond"),
    ))
    fig_hist.add_vline(
        x=hist_dates[-1], line_dash="dot", line_color="gray",
        annotation_text="Hoy", annotation_position="top right",
    )
    fig_hist.update_layout(
        title="Serie completa: Histórico (Mar 2023 – Feb 2026) + Pronóstico (Mar 2026 – Feb 2027)",
        yaxis_title="Unidades/mes", xaxis_title="",
        height=380, template="plotly_white",
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("### Pronóstico mensual detallado")

    # ── Gráfica principal (barras por mes) ──
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

    with st.expander("📊 Métricas de desempeño del modelo"):
        st.markdown(
            "El modelo **Holt-Winters multiplicativo** fue seleccionado sobre el aditivo "
            "al obtener menor error en el conjunto de validación (últimos 12 meses):"
        )
        df_metricas = pd.DataFrame({
            "Modelo":          ["HW Multiplicativo", "HW Aditivo"],
            "MAD":             [18.06,   19.58],
            "RMSE":            [23.79,   25.11],
            "MAPE (%)":        [18.66,   20.72],
            "RSFE":            [-29.31,  -109.99],
            "Tracking Signal": [-1.62,   -5.62],
            "Diagnóstico":     ["✅ Sin sesgo sistemático", "⚠️ Sesgo negativo"],
        })
        st.dataframe(df_metricas, use_container_width=True, hide_index=True)
        st.caption(
            "**Train:** 24 meses (Mar 2023 – Feb 2025) · **Test:** 12 meses (Mar 2025 – Feb 2026). "
            "La fuerte estacionalidad de la familia (diciembre ×1.40, febrero ×1.35) y el crecimiento "
            "tendencial favorecen el modelo multiplicativo: su Tracking Signal cercano a cero confirma "
            "que los errores positivos y negativos se compensan, mientras que el aditivo sobrestima "
            "sistemáticamente (TS = -5.62)."
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

    # ── RESULTADOS REALES DEL NOTEBOOK (fuente: Proyecto_2_magic_spirit-3.ipynb) ──
    st.markdown("### 📋 Resultados consolidados del estudio")
    st.caption(
        "Valores obtenidos de la ejecución del notebook completo con semilla fija. "
        "Los 10 escenarios se muestran ordenados según se definieron en el análisis original."
    )

    resultados_notebook = pd.DataFrame([
        {"Escenario": "E0 Base",                "Throughput": 2725, "Backlog máx": 81,  "% a tiempo": 97.3, "Cycle time (d)": 4.6,  "Util. emp (%)": 43.8, "Incumplidos": 427},
        {"Escenario": "E1 Demanda +20%",        "Throughput": 3096, "Backlog máx": 249, "% a tiempo": 88.9, "Cycle time (d)": 10.0, "Util. emp (%)": 50.7, "Incumplidos": 832},
        {"Escenario": "E2 Demanda -20%",        "Throughput": 2287, "Backlog máx": 27,  "% a tiempo": 100.0,"Cycle time (d)": 1.5,  "Util. emp (%)": 35.7, "Incumplidos": 0},
        {"Escenario": "E3 Premium x2",          "Throughput": 2729, "Backlog máx": 85,  "% a tiempo": 97.2, "Cycle time (d)": 4.6,  "Util. emp (%)": 43.8, "Incumplidos": 400},
        {"Escenario": "E4 Carga comerc. +15%",  "Throughput": 2725, "Backlog máx": 81,  "% a tiempo": 97.3, "Cycle time (d)": 4.6,  "Util. emp (%)": 52.0, "Incumplidos": 427},
        {"Escenario": "E5 Sin horas extra",     "Throughput": 2724, "Backlog máx": 83,  "% a tiempo": 97.3, "Cycle time (d)": 4.7,  "Util. emp (%)": 61.4, "Incumplidos": 430},
        {"Escenario": "E6 Lead time +7d",       "Throughput": 2725, "Backlog máx": 81,  "% a tiempo": 97.3, "Cycle time (d)": 4.6,  "Util. emp (%)": 43.8, "Incumplidos": 427},
        {"Escenario": "E7 Stock hilo bajo",     "Throughput": 2725, "Backlog máx": 81,  "% a tiempo": 97.3, "Cycle time (d)": 4.6,  "Util. emp (%)": 43.8, "Incumplidos": 427},
        {"Escenario": "E8 Combinado crítico",   "Throughput": 2980, "Backlog máx": 336, "% a tiempo": 85.6, "Cycle time (d)": 10.9, "Util. emp (%)": 77.3, "Incumplidos": 1005},
        {"Escenario": "E9 Combinado favorable", "Throughput": 2725, "Backlog máx": 81,  "% a tiempo": 97.3, "Cycle time (d)": 4.6,  "Util. emp (%)": 43.8, "Incumplidos": 427},
    ]).set_index("Escenario")

    st.dataframe(
        resultados_notebook.style.format({
            "Throughput":     "{:,.0f}",
            "Backlog máx":    "{:.0f}",
            "% a tiempo":     "{:.1f}",
            "Cycle time (d)": "{:.1f}",
            "Util. emp (%)":  "{:.1f}",
            "Incumplidos":    "{:,.0f}",
        }),
        use_container_width=True,
    )

    # ── Hallazgos clave ──
    st.markdown("### 🎯 Hallazgos clave del análisis")
    ch1, ch2, ch3 = st.columns(3)
    ch1.success(
        "**🏆 Mejor servicio**  \n"
        "**E2 Demanda -20%**  \n"
        "100% a tiempo · 0 incumplidos  \n"
        "*Cuando baja la demanda, el sistema cumple perfecto*"
    )
    ch2.error(
        "**🚨 Mayor backlog**  \n"
        "**E8 Combinado crítico**  \n"
        "336 pedidos · 1,005 incumplidos  \n"
        "*Demanda +20% + sin HE = colapso operacional*"
    )
    ch3.warning(
        "**📦 Mayor throughput**  \n"
        "**E1 Demanda +20%**  \n"
        "3,096 uds · 249 backlog máx  \n"
        "*Más volumen pero servicio cae a 88.9%*"
    )

    st.info(
        "**Insight principal para el S&OP:** los escenarios E4 (carga comercial +15%), "
        "E5 (sin horas extra), E6 (lead time +7d), E7 (stock hilo bajo) y E9 (combinado favorable) "
        "tienen prácticamente el mismo throughput y servicio que la base. Esto confirma que "
        "**la palanca crítica del sistema es el manejo de la demanda**, no el inventario ni los "
        "tiempos de reposición. Si la demanda se dispara +20% (E1), el servicio cae 8.4 puntos "
        "porcentuales; si además se combina con otros estresores (E8), el sistema colapsa."
    )

    st.markdown("---")
    st.markdown("### 🔬 Simulación interactiva (tu semilla y parámetros)")
    st.caption(
        f"Los siguientes resultados se recalculan en vivo con **seed = {seed}** y "
        f"**demanda ajustada al multiplicador {demand_multiplier}**. "
        "Pueden diferir de los valores consolidados de arriba."
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
    st.markdown("### 📑 Plan Agregado y Estrategia S&OP")
    with st.expander("Ver propuesta de plan agregado", expanded=True):
        st.markdown("""
**Contexto estratégico (Make-to-Order):**

Dado que Magic Spirit opera bajo Make-to-Order, la estrategia agregada **no puede depender
de inventario de producto terminado** — debe enfocarse en **capacidad efectiva y gestión de flujo**.
Proponemos un enfoque pull controlado donde el número de pedidos procesados diariamente se ajusta
a la capacidad real del sistema, particularmente en la etapa de empaque (cuello de botella).

**Diagnóstico del escenario base (E0):**
- Throughput de **2,725 unidades** en el año con **97.3%** de pedidos a tiempo.
- Backlog máximo de **81 pedidos** — por encima del límite operacional de 30.
- Utilización de empaque promedio de **43.8%** — holgura en meses valle.

**Las 5 políticas del plan S&OP:**

**1. Flujo pull controlado (Producción).** Los pedidos diarios se ajustan a la capacidad real
del sistema, no al volumen nominal de la demanda. Esto evita acumulación en cola y mantiene
cycle times predecibles.

**2. Horas extra estratégicas (Capacidad).** Activar 60 minutos adicionales/día en empaque
durante picos: **febrero, mayo, noviembre y diciembre**. El escenario E5 muestra que desactivarlas
no afecta el throughput anual pero **eleva la utilización al 61.4% vs 43.8% base** — el sistema
queda sin margen. Esto no es contratación permanente, es expansión temporal.

**3. Priorización dual de pedidos (Servicio).** Distinguir entre pedidos **estándar (1–3 días)**
y **premium (same-day)**. La priorización permite mejor asignación de capacidad limitada sin
afectar significativamente el throughput global (E3 muestra impacto +0.1%).

**4. Backorders controlados (Demanda).** Durante saturación, aceptar backorders con plazos
claramente comunicados al cliente en lugar de perder ventas. Es preferible al colapso E8
donde se pierden 1,005 pedidos.

**5. Alineación marketing–operaciones (Integración comercial).** En Magic Spirit, la publicidad
genera demanda directa. Las decisiones comerciales deben coordinarse con capacidad operativa.
El escenario E8 muestra qué pasa si marketing dispara campañas sin esta alineación:
backlog de **336 pedidos** y servicio cayendo a **85.6%**.

**Señal de alarma operacional:** si el throughput semanal excede a la capacidad del plan
durante más de 2 semanas consecutivas, activar protocolos de contención antes de entrar
en zona E8.
        """)


# ============================================================
# TAB 5 - DESAGREGACIÓN POR PRODUCTO
# ============================================================

with tab5:
    st.header("Desagregación Táctica — Por Producto y Mes")
    st.markdown(
        "Hasta aquí vimos resultados **agregados** del sistema. El verdadero valor del gemelo "
        "digital está en bajar al nivel **táctico**: qué producto consume qué capacidad, cuándo, "
        "y dónde se concentra el riesgo. Esto es lo que el equipo de S&OP necesita para "
        "decisiones de ejecución día a día."
    )

    # ── Resumen anual por producto (del notebook, escenario E9) ──
    st.markdown("### 📊 Resumen anual por producto")
    st.caption(
        "Datos del escenario E9 (Combinado favorable) — equivalente al plan S&OP recomendado. "
        "Cálculos sobre 2,722 unidades despachadas en el año."
    )

    df_desagregacion = pd.DataFrame([
        {"Producto": "7 Nudos de Protección",  "Uds producidas": 790, "Hilo (cm)": 63987,  "Horas-hombre": 102.7, "% Volumen": 30.4, "% Capacidad": 17.4},
        {"Producto": "Decenario San Benito",   "Uds producidas": 586, "Hilo (cm)": 31644,  "Horas-hombre": 164.0, "% Volumen": 22.5, "% Capacidad": 27.8},
        {"Producto": "Brazalete Tibetano",     "Uds producidas": 495, "Hilo (cm)": 102492, "Horas-hombre": 183.2, "% Volumen": 19.0, "% Capacidad": 31.0},
        {"Producto": "7 Nudos San Benito",     "Uds producidas": 434, "Hilo (cm)": 36025,  "Horas-hombre":  60.8, "% Volumen": 16.7, "% Capacidad": 10.3},
        {"Producto": "7 Nudos Tibetano",       "Uds producidas": 277, "Hilo (cm)": 26865,  "Horas-hombre":  44.6, "% Volumen": 10.6, "% Capacidad":  7.5},
        {"Producto": "7 Nudos Colores",        "Uds producidas": 140, "Hilo (cm)": 11340,  "Horas-hombre":  23.9, "% Volumen":  5.4, "% Capacidad":  4.0},
    ])

    st.dataframe(
        df_desagregacion.set_index("Producto").style.format({
            "Uds producidas": "{:,.0f}",
            "Hilo (cm)":      "{:,.0f}",
            "Horas-hombre":   "{:.1f}",
            "% Volumen":      "{:.1f}%",
            "% Capacidad":    "{:.1f}%",
        }),
        use_container_width=True,
    )

    # ── Dos gráficas lado a lado: la "inversión" del ranking ──
    st.markdown("### 🔄 La paradoja del volumen vs capacidad")
    st.markdown(
        "Cuando comparamos los mismos productos en dos métricas diferentes, **el ranking se invierte**. "
        "Este es el hallazgo gerencial más importante de la desagregación."
    )

    col_g1, col_g2 = st.columns(2)

    # Gráfica 1: Unidades producidas
    df_sorted_uds = df_desagregacion.sort_values("Uds producidas", ascending=False)
    fig_uds = go.Figure()
    fig_uds.add_bar(
        x=df_sorted_uds["Producto"],
        y=df_sorted_uds["Uds producidas"],
        marker_color=COLORS["primary"],
        text=df_sorted_uds["Uds producidas"].astype(int),
        textposition="outside",
    )
    fig_uds.update_layout(
        title="Ranking por VOLUMEN (unidades/año)",
        yaxis_title="Unidades",
        height=380, template="plotly_white",
        xaxis=dict(tickangle=-25),
        showlegend=False,
    )
    col_g1.plotly_chart(fig_uds, use_container_width=True)

    # Gráfica 2: Horas-hombre
    df_sorted_hh = df_desagregacion.sort_values("Horas-hombre", ascending=False)
    fig_hh = go.Figure()
    fig_hh.add_bar(
        x=df_sorted_hh["Producto"],
        y=df_sorted_hh["Horas-hombre"],
        marker_color=COLORS["secondary"],
        text=df_sorted_hh["Horas-hombre"].round(1),
        textposition="outside",
    )
    fig_hh.update_layout(
        title="Ranking por CAPACIDAD (horas-hombre/año)",
        yaxis_title="Horas-hombre",
        height=380, template="plotly_white",
        xaxis=dict(tickangle=-25),
        showlegend=False,
    )
    col_g2.plotly_chart(fig_hh, use_container_width=True)

    # ── Hallazgos gerenciales ──
    st.markdown("### 💡 Implicaciones gerenciales")

    c1, c2, c3 = st.columns(3)
    c1.info(
        "**🥇 7 Nudos de Protección**  \n"
        "Líder en volumen (30.4%)  \n"
        "pero apenas 17.4% de capacidad  \n"
        "*Producto eficiente: alto volumen, bajo consumo relativo*"
    )
    c2.error(
        "**⚠️ Brazalete Tibetano**  \n"
        "Solo 19% del volumen  \n"
        "pero 31% de la capacidad  \n"
        "*Cada unidad toma 3× más tiempo (19.8 vs 7.8 min). Concentra 38% del hilo crítico.*"
    )
    c3.success(
        "**📈 Palanca de escalamiento**  \n"
        "Mix hacia productos  \n"
        "de menor H-H/unidad  \n"
        "*Triplica throughput con el mismo equipo*"
    )

    # ── Conexión con el plan S&OP ──
    st.markdown("### 🔗 Conexión con el plan S&OP")
    st.markdown("""
Esta desagregación tiene **tres implicaciones concretas** para las políticas del plan agregado:

1. **Stock de hilo rojo por producto.** El Brazalete Tibetano concentra el 38% del consumo de hilo.
   La política de stock mínimo debe calibrarse priorizando este SKU — no es el mismo riesgo tener
   escasez de hilo para un 7 Nudos (80 cm/ud) que para un Brazalete (207 cm/ud).

2. **Priorización en cola de producción.** En diciembre (utilización empaque 94%),
   la cola debe priorizar pedidos que maximicen throughput por H-H — es decir,
   pedidos de productos "eficientes" tipo 7 Nudos de Protección antes que Brazaletes,
   salvo los pedidos premium con plazo crítico.

3. **Decisiones de pricing/mix comercial.** Si marketing quiere escalar ventas,
   la palanca está en el mix: empujar productos de alta eficiencia (7 Nudos)
   permite 3× más throughput con el mismo equipo vs empujar Brazaletes.
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
