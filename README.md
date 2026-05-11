# microAI — Agente de IA para Priorización Geográfica de Microcrédito Digital en Colombia

Proyecto de grado que implementa un agente conversacional de inteligencia artificial para apoyar la toma de decisiones sobre en qué departamentos de Colombia desplegar estratégicamente un producto de microcrédito digital. El sistema combina análisis de datos estructurados, un índice de oportunidad multidimensional, visualización cartográfica interactiva y búsqueda web en tiempo real, todo accesible desde una interfaz conversacional construida con Streamlit.

---

## Características principales

- Agente conversacional con **tool calling real** (el modelo decide qué herramienta invocar)
- **Índice de oportunidad** calculado a partir de 5 variables normalizadas por departamento
- **Mapa coroplético interactivo** de Colombia generado con Folium
- **Simulación de pesos** para explorar escenarios de priorización alternativos
- **Búsqueda web en tiempo real** mediante la API de Tavily
- Interfaz web construida con **Streamlit**
- Modo de respaldo automático si el proveedor de LLM no soporta tool calling

---

## Estructura del proyecto

```
microAI/
├── app.py                          # Interfaz Streamlit (punto de entrada)
├── notebooks/
│   ├── 01_ingesta_datos.ipynb      # Carga y verificación de datos crudos
│   ├── 02_dataset_maestro.ipynb    # Consolidación del dataset maestro
│   └── 03_indice_oportunidad.ipynb # Cálculo del índice y ranking
├── src/
│   ├── agent_tools.py              # Herramientas del agente y loop agéntico
│   ├── tools.py                    # Lógica de cada herramienta
│   ├── llm_client.py               # Cliente para la API del LLM
│   ├── config.py                   # Configuración y variables de entorno
│   ├── data_loader.py              # Carga del dataset procesado
│   ├── prompt_builder.py           # Construcción de prompts para modo texto
│   ├── intent_router.py            # Clasificador de intención del usuario
│   └── chat_agente.py              # Manejo del chat sin LLM externo
├── data/
│   ├── raw/                        # Archivos Excel originales por fuente
│   ├── processed/                  # Dataset maestro y ranking final
│   └── geo/                        # GeoJSON de Colombia
├── reports/
│   └── mapa_oportunidad.html       # Mapa generado (se sobreescribe con cada consulta)
├── docs/
│   └── Diccionario de datos.xlsx   # Descripción de variables y fuentes
├── requirements.txt
└── .env                            # Variables de entorno (no incluido en el repo)
```

---

## Requisitos previos

- Python 3.11 o superior
- Una API key de [Groq](https://console.groq.com) (gratuita, soporta tool calling)
- Una API key de [Tavily](https://tavily.com) (gratuita, para búsqueda web)

---

## Instalación

**1. Clonar el repositorio**
```bash
git clone https://github.com/[tu-usuario]/microAI.git
cd microAI
```

**2. Crear y activar el entorno virtual**
```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate
```

**3. Instalar dependencias**
```bash
pip install -r requirements.txt
```

**4. Configurar las variables de entorno**

Crear un archivo `.env` en la raíz del proyecto con el siguiente contenido:

```
MODEL_API_URL=https://api.groq.com/openai/v1/chat/completions
MODEL_API_KEY=tu_api_key_de_groq
MODEL_NAME=llama-3.3-70b-versatile
TAVILY_API_KEY=tu_api_key_de_tavily
```

---

## Uso

### Paso 1 — Ejecutar los notebooks (una sola vez)

Los notebooks procesan los datos crudos y generan el dataset final. Deben correrse en orden:

```bash
# Abrir Jupyter
jupyter notebook
```

Ejecutar en orden:
1. `notebooks/01_ingesta_datos.ipynb`
2. `notebooks/02_dataset_maestro.ipynb`
3. `notebooks/03_indice_oportunidad.ipynb`

Al finalizar, se habrá generado el archivo `data/processed/ranking_oportunidad.csv`.

### Paso 2 — Lanzar la interfaz del agente

```bash
streamlit run app.py
```

La aplicación quedará disponible en `http://localhost:8501`.

---

## Variables del modelo

Todas las variables están normalizadas en escala 0-1:

| Variable | Descripción | Dirección |
|---|---|---|
| `pobreza_n` | Necesidad económica | 1 = máxima pobreza |
| `microcredito_n` | Brecha de microcrédito | 1 = menor acceso |
| `productos_n` | Brecha de inclusión financiera | 1 = menor inclusión |
| `atm_n` | Carencia de infraestructura ATM | 1 = menos cajeros |
| `internet_n` | Viabilidad digital | 1 = mejor conectividad |

---

## Ejemplos de consultas al agente

- *"¿Cuáles son los 5 departamentos con mayor oportunidad?"*
- *"Explícame el caso de Chocó"*
- *"Genera el mapa"*
- *"Recalcula el ranking dando más peso a pobreza: 0.4 y los demás en 0.15"*
- *"¿Qué departamentos son de nivel Alto?"*
- *"Dame una recomendación de despliegue"*
- *"Busca en internet: iniciativas del gobierno colombiano para bancarizar zonas rurales 2024"*

---

## Tecnologías utilizadas

| Tecnología | Uso |
|---|---|
| Python 3.11 | Lenguaje principal |
| Pandas / NumPy | Procesamiento de datos |
| Folium | Mapas coropléticos interactivos |
| Streamlit | Interfaz web conversacional |
| Groq API + Llama 3.3 | Modelo de lenguaje con tool calling |
| Tavily API | Búsqueda web en tiempo real |
| python-dotenv | Gestión de variables de entorno |

---

## Fuentes de datos

- **Pobreza**: DANE — Encuesta Nacional de Calidad de Vida 2024
- **Microcrédito**: Superintendencia Financiera de Colombia 2024
- **Productos financieros**: Superintendencia Financiera de Colombia 2024
- **Infraestructura ATM**: Superintendencia Financiera de Colombia 2024
- **Conectividad internet**: MinTIC 2024
