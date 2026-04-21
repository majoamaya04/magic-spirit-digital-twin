"""
Gemelo Digital - Magic Spirit
Core del modelo: entidades, parámetros, generador de demanda, sensores,
motor de simulación y escenarios what-if.

Basado en el notebook Proyecto_2_magic_spirit.ipynb del equipo.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ============================================================
# ENUMERACIONES
# ============================================================

class EstadoPedido(Enum):
    EN_COLA_PRODUCCION = "en_cola_produccion"
    EN_PRODUCCION      = "en_produccion"
    EN_COLA_EMPAQUE    = "en_cola_empaque"
    EN_EMPAQUE         = "en_empaque"
    EMPACADO           = "empacado"
    DESPACHADO         = "despachado"
    INCUMPLIDO         = "incumplido"


class TipoPedido(Enum):
    ESTANDAR = "estandar"
    PREMIUM  = "premium"


class Producto(Enum):
    NUDOS_PROTECCION     = "7 Nudos de Protección"
    NUDOS_SAN_BENITO     = "7 Nudos San Benito"
    NUDOS_COLORES        = "7 Nudos Colores"
    NUDOS_TIBETANO       = "7 Nudos Tibetano"
    BRAZALETE_TIBETANO   = "Brazalete Tibetano"
    DECENARIO_SAN_BENITO = "Decenario San Benito"


# ============================================================
# PARÁMETROS POR PRODUCTO
# ============================================================

@dataclass
class ParametrosProducto:
    nombre:                str
    tiempo_produccion_min: float   # min/unidad
    hilo_rojo_cm:          float   # cm de hilo 1mm por unidad
    horas_hombre:          float   # horas-hombre por unidad


PARAMS_PRODUCTO: Dict[Producto, ParametrosProducto] = {
    Producto.NUDOS_PROTECCION:     ParametrosProducto("7 Nudos de Protección",   7.80,  81, 0.13),
    Producto.NUDOS_SAN_BENITO:     ParametrosProducto("7 Nudos San Benito",       8.50,  83, 0.14),
    Producto.NUDOS_COLORES:        ParametrosProducto("7 Nudos Colores",          7.80,  81, 0.13),
    Producto.NUDOS_TIBETANO:       ParametrosProducto("7 Nudos Tibetano",         9.75,  97, 0.16),
    Producto.BRAZALETE_TIBETANO:   ParametrosProducto("Brazalete Tibetano",      19.80, 207, 0.33),
    Producto.DECENARIO_SAN_BENITO: ParametrosProducto("Decenario San Benito",    14.40,  54, 0.24),
}


# ============================================================
# PARÁMETROS GLOBALES DEL SISTEMA
# ============================================================

@dataclass
class ParametrosSistema:
    tiempo_produccion_min_unidad: float = 11.34
    tiempo_empaque_min_unidad:    float = 2.43
    plazo_estandar_dias:          int   = 3
    plazo_premium_dias:           int   = 1
    proporcion_premium:           float = 0.20
    unidades_promedio_pedido:     float = 1.5
    unidades_std_pedido:          float = 0.8
    inventario_producto_terminado: bool = False
    max_backlog_pedidos:          int   = 30

    participacion_productos: dict = field(default_factory=lambda: {
        Producto.NUDOS_PROTECCION:     0.28,
        Producto.DECENARIO_SAN_BENITO: 0.23,
        Producto.BRAZALETE_TIBETANO:   0.18,
        Producto.NUDOS_SAN_BENITO:     0.16,
        Producto.NUDOS_TIBETANO:       0.10,
        Producto.NUDOS_COLORES:        0.06,
    })

    factores_estacionales: dict = field(default_factory=lambda: {
        1: 1.10, 2: 1.35, 3: 1.15, 4: 1.10, 5: 1.25, 6: 1.00,
        7: 0.95, 8: 0.80, 9: 0.90, 10: 0.80, 11: 1.20, 12: 1.40,
    })

    fraccion_comercial_mensual: dict = field(default_factory=lambda: {
        1: 0.25, 2: 0.40, 3: 0.30, 4: 0.28, 5: 0.38, 6: 0.25,
        7: 0.25, 8: 0.20, 9: 0.22, 10: 0.20, 11: 0.35, 12: 0.45,
    })

    meses_horas_extra:          list  = field(default_factory=lambda: [2, 5, 11, 12])
    max_horas_extra_dia_min:    float = 60.0


# Instancia global reutilizable
PARAMS_SISTEMA = ParametrosSistema()


# ============================================================
# PRONÓSTICO BASE HOLT-WINTERS (marzo 2026 - febrero 2027)
# Valores obtenidos del notebook con HW multiplicativo
# ============================================================

FORECAST_FAMILIA: Dict[pd.Timestamp, int] = {
    pd.Timestamp("2026-03-01"): 284,
    pd.Timestamp("2026-04-01"): 251,
    pd.Timestamp("2026-05-01"): 298,
    pd.Timestamp("2026-06-01"): 221,
    pd.Timestamp("2026-07-01"): 198,
    pd.Timestamp("2026-08-01"): 175,
    pd.Timestamp("2026-09-01"): 203,
    pd.Timestamp("2026-10-01"): 181,
    pd.Timestamp("2026-11-01"): 263,
    pd.Timestamp("2026-12-01"): 341,
    pd.Timestamp("2027-01-01"): 247,
    pd.Timestamp("2027-02-01"): 312,
}


# ============================================================
# ENTIDAD: PEDIDO
# ============================================================

@dataclass
class Pedido:
    id_pedido:         int
    fecha_llegada:     pd.Timestamp
    cantidad_unidades: int
    producto:          Producto
    tipo_pedido:       TipoPedido

    consumo_hilo_cm:   float                 = field(init=False)
    tiempo_produccion: float                 = field(init=False)
    tiempo_empaque:    float                 = field(init=False)
    fecha_prometida:   Optional[pd.Timestamp] = field(default=None)

    estado:            EstadoPedido          = field(default=EstadoPedido.EN_COLA_PRODUCCION)
    fecha_inicio_prod: Optional[pd.Timestamp] = field(default=None)
    fecha_fin_prod:    Optional[pd.Timestamp] = field(default=None)
    fecha_inicio_emp:  Optional[pd.Timestamp] = field(default=None)
    fecha_despacho:    Optional[pd.Timestamp] = field(default=None)
    cycle_time_dias:   Optional[float]        = field(default=None)

    def __post_init__(self):
        params = PARAMS_PRODUCTO[self.producto]
        self.consumo_hilo_cm   = params.hilo_rojo_cm * self.cantidad_unidades
        self.tiempo_produccion = params.tiempo_produccion_min * self.cantidad_unidades
        self.tiempo_empaque    = PARAMS_SISTEMA.tiempo_empaque_min_unidad * self.cantidad_unidades

        plazo = (PARAMS_SISTEMA.plazo_premium_dias
                 if self.tipo_pedido == TipoPedido.PREMIUM
                 else PARAMS_SISTEMA.plazo_estandar_dias)
        self.fecha_prometida = self.fecha_llegada + pd.Timedelta(days=plazo)

    def despachar(self, timestamp: pd.Timestamp):
        self.fecha_despacho  = timestamp
        self.estado          = EstadoPedido.DESPACHADO
        self.cycle_time_dias = (timestamp - self.fecha_llegada).total_seconds() / 86400

    @property
    def a_tiempo(self) -> bool:
        if self.fecha_despacho and self.fecha_prometida:
            return self.fecha_despacho <= self.fecha_prometida
        return False


# ============================================================
# ENTIDAD: OPERARIOS E INVENTARIO
# ============================================================

@dataclass
class OperarioProduccion:
    nombre:                        str   = "Operario Producción"
    tiempo_disponible_min:         float = 180.0   # 3 h/día
    tiempo_produccion_general_min: float = 11.34

    tiempo_ocupado_min:  float = 0.0
    apoyando_empaque:    bool  = False
    unidades_producidas: int   = 0

    @property
    def tiempo_libre_min(self) -> float:
        return max(0.0, self.tiempo_disponible_min - self.tiempo_ocupado_min)

    @property
    def utilizacion(self) -> float:
        return min(1.0, self.tiempo_ocupado_min / self.tiempo_disponible_min)

    def resetear_dia(self):
        self.tiempo_ocupado_min  = 0.0
        self.apoyando_empaque    = False
        self.unidades_producidas = 0


@dataclass
class OperarioEmpaque:
    nombre:                     str   = "Operario Empaque/Ventas"
    tiempo_nominal_min:         float = 60.0
    tiempo_empaque_general_min: float = 2.43
    fraccion_comercial:         float = 0.30

    tiempo_ocupado_empaque_min: float = 0.0
    tiempo_comercial_min:       float = 0.0
    unidades_empacadas:         int   = 0
    horas_extra_min:            float = 0.0

    @property
    def tiempo_efectivo_empaque_min(self) -> float:
        return self.tiempo_nominal_min * (1 - self.fraccion_comercial)

    @property
    def tiempo_libre_empaque_min(self) -> float:
        return max(
            0.0,
            self.tiempo_efectivo_empaque_min + self.horas_extra_min
            - self.tiempo_ocupado_empaque_min,
        )

    @property
    def utilizacion(self) -> float:
        total = self.tiempo_efectivo_empaque_min + self.horas_extra_min
        if total == 0:
            return 0.0
        return min(1.0, self.tiempo_ocupado_empaque_min / total)

    def resetear_dia(self):
        self.tiempo_ocupado_empaque_min = 0.0
        self.tiempo_comercial_min       = self.tiempo_nominal_min * self.fraccion_comercial
        self.unidades_empacadas         = 0
        self.horas_extra_min            = 0.0


@dataclass
class InventarioHiloRojo:
    CM_POR_ROLLO:           float = 8229.6
    stock_cm:               float = 8229.6 * 2
    punto_reorden_cm:       float = 8229.6
    cantidad_pedido_rollos: int   = 3
    lead_time_dias:         int   = 30
    lead_time_std_dias:     int   = 7
    costo_rollo_normal:     float = 33000
    costo_rollo_emergencia: float = 49000

    pedidos_en_transito: list = field(default_factory=list)
    compras_emergencia:  int  = 0
    eventos_escasez:     int  = 0

    @property
    def stock_rollos(self) -> float:
        return self.stock_cm / self.CM_POR_ROLLO

    def consumir(self, cm: float) -> bool:
        if self.stock_cm >= cm:
            self.stock_cm -= cm
            return True
        self.eventos_escasez += 1
        return False

    def reabastecer(self, rollos: int, emergencia: bool = False):
        self.stock_cm += rollos * self.CM_POR_ROLLO
        if emergencia:
            self.compras_emergencia += rollos

    def necesita_reorden(self) -> bool:
        return self.stock_cm <= self.punto_reorden_cm


# ============================================================
# GENERADOR DE DEMANDA
# ============================================================

class GeneradorDemanda:
    def __init__(self, forecast_familia: dict, seed: int = 42):
        self.forecast_familia = forecast_familia
        self.rng = np.random.default_rng(seed)

    def unidades_diarias_esperadas(self, fecha: pd.Timestamp) -> float:
        mes_inicio = fecha.replace(day=1)
        if mes_inicio not in self.forecast_familia:
            return 0.0
        unidades_mes = self.forecast_familia[mes_inicio]
        dias_habiles = self._dias_habiles_mes(mes_inicio)
        factor_est   = PARAMS_SISTEMA.factores_estacionales[fecha.month]
        return (unidades_mes / dias_habiles) * factor_est

    def _dias_habiles_mes(self, mes_inicio: pd.Timestamp) -> int:
        rango = pd.date_range(
            mes_inicio, mes_inicio + pd.offsets.MonthEnd(0), freq="B"
        )
        return max(len(rango), 1)

    def generar_pedidos_dia(self, fecha: pd.Timestamp, contador_id: int):
        media_unidades = self.unidades_diarias_esperadas(fecha)
        if media_unidades <= 0:
            return [], contador_id

        n_pedidos = self.rng.poisson(
            lam=max(1, media_unidades / PARAMS_SISTEMA.unidades_promedio_pedido)
        )

        pedidos_dia = []
        for _ in range(n_pedidos):
            unidades = max(1, int(self.rng.normal(
                PARAMS_SISTEMA.unidades_promedio_pedido,
                PARAMS_SISTEMA.unidades_std_pedido
            )))

            productos = list(PARAMS_SISTEMA.participacion_productos.keys())
            pesos     = np.array(list(PARAMS_SISTEMA.participacion_productos.values()),
                                 dtype=float)
            pesos     = pesos / pesos.sum()  # normalizar por si suman ≠ 1
            producto  = self.rng.choice(productos, p=pesos)

            tipo = (TipoPedido.PREMIUM
                    if self.rng.random() < PARAMS_SISTEMA.proporcion_premium
                    else TipoPedido.ESTANDAR)

            pedido = Pedido(
                id_pedido         = contador_id,
                fecha_llegada     = fecha,
                cantidad_unidades = unidades,
                producto          = producto,
                tipo_pedido       = tipo,
            )
            pedidos_dia.append(pedido)
            contador_id += 1

        return pedidos_dia, contador_id

    def generar_horizonte(self) -> pd.DataFrame:
        filas = []
        for mes, unidades in self.forecast_familia.items():
            factor   = PARAMS_SISTEMA.factores_estacionales[mes.month]
            dias_hab = self._dias_habiles_mes(mes)
            upd      = (unidades / dias_hab) * factor
            filas.append({
                "Mes":                  mes.strftime("%b-%Y"),
                "MesTimestamp":         mes,
                "Unidades forecast":    unidades,
                "Factor estacional":    factor,
                "Días hábiles":         dias_hab,
                "Unidades/día (media)": round(upd, 2),
                "Pedidos/día (media)":  round(upd / PARAMS_SISTEMA.unidades_promedio_pedido, 2),
            })
        return pd.DataFrame(filas)


# ============================================================
# SENSORES VIRTUALES
# ============================================================

@dataclass
class LecturaSensor:
    fecha: pd.Timestamp

    cola_produccion: int = 0
    cola_empaque:    int = 0
    wip_total:       int = 0

    unidades_producidas_dia: int = 0
    unidades_empacadas_dia:  int = 0
    pedidos_despachados_dia: int = 0

    cycle_time_promedio_dias: float = 0.0
    espera_produccion_dias:   float = 0.0
    espera_empaque_dias:      float = 0.0

    utilizacion_produccion:   float = 0.0
    utilizacion_empaque:      float = 0.0
    horas_extra_empaque_min:  float = 0.0
    apoyo_produccion_empaque: bool  = False

    backlog_acumulado:       int   = 0
    pedidos_incumplidos_dia: int   = 0
    pct_despacho_a_tiempo:   float = 0.0
    pct_premium_a_tiempo:    float = 0.0

    stock_hilo_cm:           float = 0.0
    stock_hilo_rollos:       float = 0.0
    alerta_reorden:          bool  = False
    compras_emergencia_acum: int   = 0
    eventos_escasez_acum:    int   = 0

    pedidos_llegados_dia:  int = 0
    unidades_llegadas_dia: int = 0


class SistemasSensores:
    def __init__(self):
        self.historial: List[LecturaSensor] = []

    def registrar(self, lectura: LecturaSensor):
        self.historial.append(lectura)

    def a_dataframe(self) -> pd.DataFrame:
        if not self.historial:
            return pd.DataFrame()
        return pd.DataFrame([vars(l) for l in self.historial]).set_index("fecha")

    def verificar_alertas(self, lectura: LecturaSensor) -> List[str]:
        alertas = []
        if lectura.cola_empaque > 10:
            alertas.append(f"⚠️ Cuello de botella empaque — {lectura.cola_empaque} pedidos en cola")
        if lectura.backlog_acumulado > PARAMS_SISTEMA.max_backlog_pedidos:
            alertas.append(f"🔴 Backlog crítico — {lectura.backlog_acumulado} pedidos pendientes")
        if lectura.alerta_reorden:
            alertas.append(f"🧵 Reorden hilo rojo — stock: {lectura.stock_hilo_rollos:.1f} rollos")
        if lectura.utilizacion_empaque > 0.90:
            alertas.append(f"📦 Empaque al límite — utilización: {lectura.utilizacion_empaque * 100:.1f}%")
        if lectura.pct_despacho_a_tiempo < 0.80:
            alertas.append(f"🚨 Servicio bajo — {lectura.pct_despacho_a_tiempo * 100:.1f}% a tiempo")
        return alertas

    def resumen_kpis(self) -> dict:
        df = self.a_dataframe()
        if df.empty:
            return {}
        return {
            "Throughput total (unidades)":       int(df["unidades_empacadas_dia"].sum()),
            "Cycle time promedio (días)":        round(df["cycle_time_promedio_dias"].mean(), 2),
            "WIP promedio":                      round(df["wip_total"].mean(), 2),
            "WIP máximo":                        int(df["wip_total"].max()),
            "Backlog promedio":                  round(df["backlog_acumulado"].mean(), 2),
            "Backlog máximo":                    int(df["backlog_acumulado"].max()),
            "% despacho a tiempo (promedio)":    round(df["pct_despacho_a_tiempo"].mean() * 100, 1),
            "% premium a tiempo (promedio)":     round(df["pct_premium_a_tiempo"].mean() * 100, 1),
            "Utilización producción (promedio)": round(df["utilizacion_produccion"].mean() * 100, 1),
            "Utilización empaque (promedio)":    round(df["utilizacion_empaque"].mean() * 100, 1),
            "Horas extra empaque (min totales)": round(df["horas_extra_empaque_min"].sum(), 1),
            "Días con apoyo prod→empaque":       int(df["apoyo_produccion_empaque"].sum()),
            "Eventos escasez hilo":              int(df["eventos_escasez_acum"].max()),
            "Compras emergencia (rollos)":       int(df["compras_emergencia_acum"].max()),
        }


# ============================================================
# MOTOR DE SIMULACIÓN
# ============================================================

class MotorSimulacion:
    def __init__(self, seed: int = 42, forecast: Optional[dict] = None):
        self.op_produccion = OperarioProduccion()
        self.op_empaque    = OperarioEmpaque()
        self.inventario    = InventarioHiloRojo()
        self.generador     = GeneradorDemanda(forecast or FORECAST_FAMILIA, seed=seed)
        self.sensores      = SistemasSensores()

        self.cola_produccion: List[Pedido] = []
        self.cola_empaque:    List[Pedido] = []

        self.contador_id          = 1
        self.pedidos_despachados: List[Pedido] = []
        self.pedidos_incumplidos: List[Pedido] = []
        self.backlog_acumulado    = 0
        self.alertas_log: List[str] = []

    def _llegada_pedidos(self, fecha: pd.Timestamp):
        nuevos, self.contador_id = self.generador.generar_pedidos_dia(
            fecha, self.contador_id
        )
        for p in nuevos:
            p._uds_prod_restantes = p.cantidad_unidades
            p._uds_emp_restantes  = p.cantidad_unidades

        premium  = [p for p in nuevos if p.tipo_pedido == TipoPedido.PREMIUM]
        estandar = [p for p in nuevos if p.tipo_pedido == TipoPedido.ESTANDAR]
        self.cola_produccion = premium + self.cola_produccion + estandar
        return nuevos

    def _actualizar_fraccion_comercial(self, fecha: pd.Timestamp):
        self.op_empaque.fraccion_comercial = (
            PARAMS_SISTEMA.fraccion_comercial_mensual[fecha.month]
        )

    def _ejecutar_produccion(self, fecha: pd.Timestamp):
        self.op_produccion.resetear_dia()

        while self.cola_produccion:
            pedido       = self.cola_produccion[0]
            t_una_unidad = PARAMS_PRODUCTO[pedido.producto].tiempo_produccion_min

            if self.op_produccion.tiempo_libre_min < t_una_unidad:
                break

            uds_posibles = int(self.op_produccion.tiempo_libre_min / t_una_unidad)
            uds_a_hacer  = min(pedido._uds_prod_restantes, uds_posibles)

            if uds_a_hacer == 0:
                break

            hilo_necesario = PARAMS_PRODUCTO[pedido.producto].hilo_rojo_cm * uds_a_hacer
            if not self.inventario.consumir(hilo_necesario):
                rollos_emerg = max(
                    self.inventario.cantidad_pedido_rollos,
                    int(hilo_necesario / self.inventario.CM_POR_ROLLO) + 1
                )
                self.inventario.reabastecer(rollos_emerg, emergencia=True)
                self.inventario.consumir(hilo_necesario)

            self.op_produccion.tiempo_ocupado_min  += t_una_unidad * uds_a_hacer
            self.op_produccion.unidades_producidas += uds_a_hacer
            pedido.fecha_inicio_prod                = pedido.fecha_inicio_prod or fecha
            pedido._uds_prod_restantes             -= uds_a_hacer

            if pedido._uds_prod_restantes == 0:
                pedido.estado         = EstadoPedido.EN_COLA_EMPAQUE
                pedido.fecha_fin_prod = fecha
                self.cola_produccion.pop(0)
                self.cola_empaque.append(pedido)

    def _ejecutar_empaque(self, fecha: pd.Timestamp, con_horas_extra: bool = True):
        self.op_empaque.resetear_dia()

        if con_horas_extra and fecha.month in PARAMS_SISTEMA.meses_horas_extra:
            if len(self.cola_empaque) > 3:
                self.op_empaque.horas_extra_min = PARAMS_SISTEMA.max_horas_extra_dia_min

        empacados    = []
        t_por_unidad = PARAMS_SISTEMA.tiempo_empaque_min_unidad

        while self.cola_empaque:
            tiempo_disp = self.op_empaque.tiempo_libre_empaque_min
            if tiempo_disp < t_por_unidad:
                break

            pedido       = self.cola_empaque[0]
            uds_posibles = int(tiempo_disp / t_por_unidad)
            uds_empacar  = min(pedido._uds_emp_restantes, uds_posibles)

            if uds_empacar == 0:
                break

            tiempo_usado = t_por_unidad * uds_empacar
            self.op_empaque.tiempo_ocupado_empaque_min += tiempo_usado
            self.op_empaque.unidades_empacadas         += uds_empacar
            pedido.fecha_inicio_emp                     = pedido.fecha_inicio_emp or fecha
            pedido._uds_emp_restantes                  -= uds_empacar

            if pedido._uds_emp_restantes == 0:
                pedido.estado = EstadoPedido.EMPACADO
                self.cola_empaque.pop(0)
                pedido.despachar(fecha)
                self.pedidos_despachados.append(pedido)
                empacados.append(pedido)
            else:
                break

        return empacados

    def _apoyo_produccion_empaque(self, fecha: pd.Timestamp) -> bool:
        if not self.cola_empaque or self.op_produccion.tiempo_libre_min <= 0:
            return False

        self.op_produccion.apoyando_empaque = True
        tiempo_apoyo = self.op_produccion.tiempo_libre_min
        t_por_unidad = PARAMS_SISTEMA.tiempo_empaque_min_unidad

        while self.cola_empaque and tiempo_apoyo >= t_por_unidad:
            pedido       = self.cola_empaque[0]
            uds_posibles = int(tiempo_apoyo / t_por_unidad)
            uds_empacar  = min(pedido._uds_emp_restantes, uds_posibles)

            if uds_empacar == 0:
                break

            tiempo_apoyo                          -= t_por_unidad * uds_empacar
            self.op_produccion.tiempo_ocupado_min += t_por_unidad * uds_empacar
            self.op_empaque.unidades_empacadas    += uds_empacar
            pedido.fecha_inicio_emp                = pedido.fecha_inicio_emp or fecha
            pedido._uds_emp_restantes             -= uds_empacar

            if pedido._uds_emp_restantes == 0:
                pedido.estado = EstadoPedido.EMPACADO
                self.cola_empaque.pop(0)
                pedido.despachar(fecha)
                self.pedidos_despachados.append(pedido)

        return True

    def _gestionar_inventario(self, fecha: pd.Timestamp):
        llegaron = [(f, cm) for f, cm in self.inventario.pedidos_en_transito if f <= fecha]
        for _, cm in llegaron:
            self.inventario.stock_cm += cm
        self.inventario.pedidos_en_transito = [
            (f, cm) for f, cm in self.inventario.pedidos_en_transito if f > fecha
        ]
        if self.inventario.necesita_reorden() and not self.inventario.pedidos_en_transito:
            lead = max(7, int(np.random.normal(
                self.inventario.lead_time_dias, self.inventario.lead_time_std_dias
            )))
            fecha_llegada_hilo = fecha + pd.Timedelta(days=lead)
            cm_pedido = self.inventario.cantidad_pedido_rollos * self.inventario.CM_POR_ROLLO
            self.inventario.pedidos_en_transito.append((fecha_llegada_hilo, cm_pedido))

    def _registrar_sensores(self, fecha, nuevos, empacados, apoyo):
        ct_dia = np.mean([p.cycle_time_dias for p in empacados
                          if p.cycle_time_dias]) if empacados else 0.0
        total_desp = len(self.pedidos_despachados)
        a_tiempo   = sum(1 for p in self.pedidos_despachados if p.a_tiempo)
        pct_at     = a_tiempo / total_desp if total_desp > 0 else 1.0
        prem_desp  = [p for p in self.pedidos_despachados if p.tipo_pedido == TipoPedido.PREMIUM]
        pct_prem_at = sum(1 for p in prem_desp if p.a_tiempo) / len(prem_desp) if prem_desp else 1.0

        ids_incumplidos = {p.id_pedido for p in self.pedidos_incumplidos}
        vencidos_hoy = [
            p for p in self.cola_produccion + self.cola_empaque
            if p.fecha_prometida and p.fecha_prometida < fecha
            and p.id_pedido not in ids_incumplidos
        ]
        for p in vencidos_hoy:
            p.estado = EstadoPedido.INCUMPLIDO
            self.pedidos_incumplidos.append(p)

        self.backlog_acumulado = len(self.cola_produccion) + len(self.cola_empaque)

        lectura = LecturaSensor(
            fecha                    = fecha,
            cola_produccion          = len(self.cola_produccion),
            cola_empaque             = len(self.cola_empaque),
            wip_total                = self.backlog_acumulado,
            unidades_producidas_dia  = self.op_produccion.unidades_producidas,
            unidades_empacadas_dia   = self.op_empaque.unidades_empacadas,
            pedidos_despachados_dia  = len(empacados),
            cycle_time_promedio_dias = round(ct_dia, 3),
            utilizacion_produccion   = round(self.op_produccion.utilizacion, 3),
            utilizacion_empaque      = round(self.op_empaque.utilizacion, 3),
            horas_extra_empaque_min  = self.op_empaque.horas_extra_min,
            apoyo_produccion_empaque = apoyo,
            backlog_acumulado        = self.backlog_acumulado,
            pedidos_incumplidos_dia  = len(vencidos_hoy),
            pct_despacho_a_tiempo    = round(pct_at, 3),
            pct_premium_a_tiempo     = round(pct_prem_at, 3),
            stock_hilo_cm            = round(self.inventario.stock_cm, 1),
            stock_hilo_rollos        = round(self.inventario.stock_rollos, 2),
            alerta_reorden           = self.inventario.necesita_reorden(),
            compras_emergencia_acum  = self.inventario.compras_emergencia,
            eventos_escasez_acum     = self.inventario.eventos_escasez,
            pedidos_llegados_dia     = len(nuevos),
            unidades_llegadas_dia    = sum(p.cantidad_unidades for p in nuevos),
        )
        self.sensores.registrar(lectura)
        for alerta in self.sensores.verificar_alertas(lectura):
            self.alertas_log.append(f"[{fecha.strftime('%d-%b-%Y')}] {alerta}")

    def tick(self, fecha: pd.Timestamp, con_horas_extra: bool = True):
        self._actualizar_fraccion_comercial(fecha)
        nuevos    = self._llegada_pedidos(fecha)
        self._ejecutar_produccion(fecha)
        empacados = self._ejecutar_empaque(fecha, con_horas_extra)
        apoyo     = self._apoyo_produccion_empaque(fecha)
        self._gestionar_inventario(fecha)
        self._registrar_sensores(fecha, nuevos, empacados, apoyo)

    def correr(
        self,
        con_horas_extra: bool = True,
        fecha_inicio: str = "2026-03-01",
        fecha_fin:    str = "2027-02-28",
    ):
        fechas = pd.bdate_range(start=fecha_inicio, end=fecha_fin, freq="B")
        for fecha in fechas:
            self.tick(fecha, con_horas_extra)


# ============================================================
# RUNNER DE ESCENARIOS WHAT-IF
# ============================================================

def correr_escenario(
    nombre:              str,
    forecast_mod:        Optional[dict]  = None,
    seed:                int             = 42,
    con_horas_extra:     bool            = True,
    prop_premium:        Optional[float] = None,
    fracc_comercial_add: float           = 0.0,
    tiempo_prod_add:     float           = 0.0,
    lead_time_add:       int             = 0,
    stock_inicial_rollos: Optional[float] = None,
) -> MotorSimulacion:
    """
    Corre una simulación con parámetros custom y restaura el estado global.
    """
    # Guardar valores originales
    prop_premium_orig    = PARAMS_SISTEMA.proporcion_premium
    fracc_comercial_orig = copy.deepcopy(PARAMS_SISTEMA.fraccion_comercial_mensual)
    t_prod_orig          = PARAMS_SISTEMA.tiempo_produccion_min_unidad

    # Aplicar modificaciones
    if prop_premium is not None:
        PARAMS_SISTEMA.proporcion_premium = prop_premium

    if fracc_comercial_add != 0.0:
        for mes in PARAMS_SISTEMA.fraccion_comercial_mensual:
            PARAMS_SISTEMA.fraccion_comercial_mensual[mes] = min(
                0.95,
                PARAMS_SISTEMA.fraccion_comercial_mensual[mes] + fracc_comercial_add
            )

    if tiempo_prod_add != 0.0:
        PARAMS_SISTEMA.tiempo_produccion_min_unidad += tiempo_prod_add

    forecast = forecast_mod if forecast_mod else FORECAST_FAMILIA
    motor    = MotorSimulacion(seed=seed, forecast=forecast)

    if lead_time_add != 0:
        motor.inventario.lead_time_dias += lead_time_add

    if stock_inicial_rollos is not None:
        motor.inventario.stock_cm = stock_inicial_rollos * motor.inventario.CM_POR_ROLLO

    motor.correr(con_horas_extra=con_horas_extra)

    # Restaurar parámetros globales
    PARAMS_SISTEMA.proporcion_premium               = prop_premium_orig
    PARAMS_SISTEMA.fraccion_comercial_mensual       = fracc_comercial_orig
    PARAMS_SISTEMA.tiempo_produccion_min_unidad     = t_prod_orig

    return motor


def tabla_comparativa(escenarios: Dict[str, MotorSimulacion]) -> pd.DataFrame:
    filas = []
    for nombre, motor in escenarios.items():
        df_s = motor.sensores.a_dataframe()
        kpis = motor.sensores.resumen_kpis()
        filas.append({
            "Escenario":           nombre,
            "Uds empacadas":       kpis.get("Throughput total (unidades)", 0),
            "Ped. despachados":    len(motor.pedidos_despachados),
            "Ped. incumplidos":    len(motor.pedidos_incumplidos),
            "Backlog máx":         int(df_s["backlog_acumulado"].max()),
            "Cycle time prom (d)": kpis.get("Cycle time promedio (días)", 0),
            "Util. prod (%)":      kpis.get("Utilización producción (promedio)", 0),
            "Util. emp (%)":       kpis.get("Utilización empaque (promedio)", 0),
            "% a tiempo":          kpis.get("% despacho a tiempo (promedio)", 0),
            "HExtra emp (min)":    kpis.get("Horas extra empaque (min totales)", 0),
            "Días apoyo":          kpis.get("Días con apoyo prod→empaque", 0),
            "Escasez hilo":        kpis.get("Eventos escasez hilo", 0),
            "Compras emerg.":      kpis.get("Compras emergencia (rollos)", 0),
        })
    return pd.DataFrame(filas).set_index("Escenario")
