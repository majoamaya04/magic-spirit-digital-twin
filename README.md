# 🧵 Magic Spirit — Gemelo Digital Interactivo

**Proyecto C2 · Planeación, Programación y Control de las Operaciones**
Universidad de La Sabana · 2026

Modelo virtual interactivo del sistema de producción y S&OP de la empresa **Magic Spirit** (manufactura artesanal de pulseras y decenarios con hilo rojo). Implementa un gemelo digital con demanda estocástica, motor de simulación diaria, sensores virtuales y análisis de escenarios *what-if*.

---

## 🔗 Contenido del modelo interactivo

La aplicación está organizada en **4 pestañas** que siguen la rúbrica del proyecto:

| Pestaña | Qué contiene |
|---|---|
| 📋 **Estudio Preliminar** | Parámetros por SKU, métricas de eficiencia (WIP, takt time, throughput, cycle time), análisis de capacidad vs demanda, identificación del cuello de botella, factores estacionales. |
| 📈 **Pronóstico de Demanda** | Pronóstico Holt-Winters multiplicativo para Mar 2026 – Feb 2027, desagregación por producto, métricas de desempeño del modelo. |
| 🏭 **Simulación Gemelo Digital** | KPIs principales, dashboard de 6 paneles de sensores virtuales, resumen mensual, log de alertas del sistema SCADA virtual. |
| 🔍 **Escenarios What-If & S&OP** | 10 escenarios (demanda ±20%, premium x2, carga comercial, sin horas extra, lead time, stock hilo, combinados), tabla comparativa, plan S&OP derivado. |

**Controles globales en la barra lateral:** semilla aleatoria, activación de horas extra, multiplicador global de demanda.

---

## 🚀 Opción A — Ejecutar localmente (para presentación en vivo)

### Requisitos
- Python 3.10 o superior
- pip

### Pasos

```bash
# 1) Clonar o descargar este repositorio
cd magic_spirit_app

# 2) (Recomendado) Crear un entorno virtual
python -m venv venv
# En Windows:
venv\Scripts\activate
# En macOS / Linux:
source venv/bin/activate

# 3) Instalar dependencias
pip install -r requirements.txt

# 4) Ejecutar la aplicación
streamlit run app.py
```

La app se abrirá automáticamente en `http://localhost:8501`.

> ⏱️ La primera simulación tarda ~0.5 s. Los 10 escenarios *what-if* tardan ~5-10 s la primera vez. Después queda todo en caché y navegar entre pestañas es instantáneo.

---

## ☁️ Opción B — Desplegar en la nube (Streamlit Community Cloud, gratis)

Esto genera un link público que puedes compartir con el profesor.

### Paso 1: Subir el código a GitHub

1. Crear una cuenta en [github.com](https://github.com) si no tienes una.
2. Crear un nuevo repositorio **público** llamado `magic-spirit-digital-twin` (o el nombre que prefieras).
3. Subir todos los archivos de esta carpeta al repositorio. Desde la terminal:

```bash
cd magic_spirit_app
git init
git add .
git commit -m "Gemelo digital Magic Spirit"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/magic-spirit-digital-twin.git
git push -u origin main
```

O simplemente arrastra los archivos en la interfaz web de GitHub ("Add file → Upload files").

### Paso 2: Desplegar en Streamlit Cloud

1. Ir a [share.streamlit.io](https://share.streamlit.io).
2. Iniciar sesión con la cuenta de GitHub.
3. Clic en **"New app"**.
4. Seleccionar el repositorio `magic-spirit-digital-twin`, la rama `main` y el archivo `app.py`.
5. Clic en **"Deploy"**.

En ~2 minutos tendrás un link tipo `https://magic-spirit-digital-twin.streamlit.app` listo para compartir.

---

## 📁 Estructura del proyecto

```
magic_spirit_app/
├── app.py                  ← Aplicación Streamlit (interfaz)
├── digital_twin.py         ← Motor del gemelo digital (lógica)
├── requirements.txt        ← Dependencias de Python
├── README.md               ← Este archivo
├── .gitignore
└── .streamlit/
    └── config.toml         ← Tema visual (colores Magic Spirit)
```

---

## 🏗️ Arquitectura del gemelo digital

El módulo `digital_twin.py` contiene:

- **Entidades:** `Pedido`, `OperarioProduccion`, `OperarioEmpaque`, `InventarioHiloRojo`.
- **Parámetros:** por producto (6 SKUs) y globales del sistema (plazos, proporción premium, factores estacionales, fracción comercial mensual).
- **Generador de demanda:** convierte el pronóstico mensual HW en llegadas diarias con distribución Poisson sobre el número de pedidos y Normal sobre unidades por pedido.
- **Sistema de sensores virtuales:** registra 22 variables de estado por día hábil (colas, WIP, throughput, utilización, servicio, inventario, alertas).
- **Motor de simulación:** 7 pasos por tick diario — llegada de pedidos, actualización de fracción comercial, producción (priorizada premium), empaque con horas extra si aplica, apoyo cruzado producción→empaque, gestión de inventario con lead time variable, registro de sensores.
- **Runner de escenarios:** función `correr_escenario()` que aplica modificaciones a los parámetros globales, corre la simulación completa y restaura el estado.

---

## 🎓 Rúbrica del proyecto cubierta

| Criterio de la rúbrica | Dónde se evidencia |
|---|---|
| **Estudio Preliminar** | Pestaña 1 — parámetros, métricas de eficiencia, cuello de botella, estacionalidad. |
| **Diseño del Gemelo Digital** | `digital_twin.py` + pestaña 3 — entidades, sensores, motor de simulación con demanda estocástica. |
| **Pronósticos de demanda** | Pestaña 2 — Holt-Winters multiplicativo con estacionalidad anual. |
| **Sensores virtuales** | 22 variables registradas por tick, dashboard interactivo con 6 paneles. |
| **Escenarios What-If** | Pestaña 4 — 10 escenarios con comparativa cuantitativa y evolución temporal. |
| **Sistema de planeación agregada** | Pestaña 4 — plan S&OP derivado con políticas de producción, capacidad, inventario y pricing. |
| **Medidas de desempeño** | Throughput, cycle time, WIP, utilización, % a tiempo, backlog, escasez hilo, compras emergencia. |

---

## 📝 Notas técnicas

- La simulación cubre **~260 días hábiles** (Mar 2026 – Feb 2027) con ~2,000-3,500 pedidos.
- Los resultados se cachean con `@st.cache_data` por combinación de `(seed, horas_extra, forecast)`.
- El pronóstico base (HW multiplicativo) está codificado en `digital_twin.py` como `FORECAST_FAMILIA` — valores obtenidos del notebook original tras comparar con HW aditivo sobre conjunto de validación de 12 meses.

---

*Construido con Streamlit, Plotly y un motor de simulación diaria en Python.*
