# ANÁLISIS EXHAUSTIVO - RUBIDIUM API

**Fecha de análisis:** 30 de julio de 2026  
**Versión del proyecto:** 2.0.0  
**Autores:** Diego Velandia y Daniel Barajas

---

## 1. DESCRIPCIÓN GENERAL

**Rubidium API** es un proyecto de inteligencia artificial que implementa un **Transformer mini-GPT** para generación de texto en español. El sistema combina múltiples enfoques de procesamiento de lenguaje natural:

- **Transformer personalizado** implementado desde cero en NumPy con autodiferenciación
- **Motor de chat** con múltiples fuentes de respuesta (intenciones, código, conocimiento, generación)
- **Aceleración en Rust** via PyO3 para inferencia rápida
- **Entrenamiento en GPU** usando PyTorch en Kaggle
- **Interfaz web** con Gradio y API REST con FastAPI

El proyecto está diseñado como un asistente de programación en español, capaz de responder preguntas sobre conceptos de programación, generar código y mantener conversaciones.

---

## 2. STACK TECNOLÓGICO

### Lenguajes principales
| Lenguaje | Uso |
|----------|-----|
| **Python** | Backend, ML, API, tokenizers |
| **Rust** | Aceleración de inferencia (PyO3) |
| **C/CUDA** | Entrenamiento en GPU (PyTorch) |

### Frameworks y librerías
| Componente | Tecnología |
|------------|------------|
| **API REST** | FastAPI + Uvicorn |
| **Web UI** | Gradio 4.44.0 |
| **ML Framework** | PyTorch (entrenamiento), NumPy (inferencia) |
| **Rust Bindings** | PyO3 0.22 + maturin |
| **Serialización** | pickle (modelos) |
| **Validación** | Pydantic |

### Infraestructura
- **Containerización:** Docker (Python 3.11-slim + Rust toolchain)
- **Despliegue:** Heroku (Procfile), Kaggle Notebooks
- **GPU:** Tesla P100 (Kaggle), CUDA 11.8
- **Almacenamiento:** GitHub, Google Drive (modelos CAD)

---

## 3. ESTRUCTURA DE DIRECTORIOS

```
rubidium-api/
├── app.py                    # Interfaz Gradio (UI web)
├── main.py                   # API REST FastAPI (principal)
├── transformer.py            # Transformer NumPy con autodiferenciación (840 líneas)
├── chat_engine.py            # Motor de chat con múltiples fuentes
├── knowledge_base.py         # Base de conocimiento (búsqueda semántica)
├── neural_code.py            # Base de conocimiento de programación
├── frontal_lobe.py           # Sistema de decisión de respuestas
├── response_source.py        # Tipos de fuentes de respuesta
├── text_similarity.py        # Similitud de texto (Jaccard + Cosine)
│
├── tokenizers/               # Sistema de tokenización modular
│   ├── __init__.py
│   ├── base.py               # Interfaz base + configuración
│   ├── bpe.py                # Byte Pair Encoding
│   ├── unigram.py            # Unigram tokenizer
│   ├── wordpiece.py          # WordPiece tokenizer
│   ├── sentencepiece.py      # SentencePiece wrapper
│   ├── dynamic.py            # Vocabulario dinámico
│   └── factory.py            # Factory pattern
│
├── rubidium-core/            # Extensión Rust para inferencia rápida
│   ├── Cargo.toml            # pyo3, numpy, ndarray, rand
│   ├── Cargo.lock
│   └── src/
│       └── lib.rs            # Forward pass + generación en Rust (384 líneas)
│
├── resources/                # Corpus de entrenamiento
│   ├── chat_01.txt ... chat_39.txt   # Diálogos conversacionales
│   ├── corpus_01.txt                 # General
│   ├── corpus_02_programacion.txt    # Programación
│   ├── corpus_03_web_dev.txt         # Desarrollo web
│   ├── corpus_04_algoritmos.txt      # Algoritmos
│   └── corpus_05_git.txt             # Git
│
├── corpus/                   # Corpus adicionales (CorpusAgent)
│   └── CorpusAgent1.txt ... CorpusAgent15.txt
│
├── agents/                   # Sistema de agentes jerárquicos (85 archivos)
│   ├── cto.md               # Nivel más alto
│   ├── architecture_director.md
│   ├── development_director.md
│   ├── quality_director.md
│   ├── operations_director.md
│   └── ... (85 roles especializados)
│
├── templates/                # Plantillas Markdown
│   ├── agent.md              # Plantilla de agente
│   ├── task.md               # Plantilla de tarea
│   └── task_system.md        # Sistema de tareas
│
├── tasks/                    # Gestión de tareas
│   └── TASK_001.md           # Tarea de optimización Kaggle
│
├── queue/                    # Cola de tareas
│   ├── pending/
│   ├── working/
│   ├── completed/
│   └── review/
│
├── reports/                  # Informes generados
│
├── train_pytorch.py          # Entrenamiento CUDA/PyTorch (10M+ params)
├── train_local.py            # Entrenamiento local NumPy
├── train_full.py             # Entrenamiento completo (8000 steps)
├── train_improved.py         # Entrenamiento mejorado (400K params)
├── train_small.py            # Entrenamiento pequeño
│
├── gen_cell.py               # Generador de celdas Kaggle (compressión)
├── compress_torch.py         # Compresión de train_pytorch.py
├── check_indent.py           # Verificación de indentación
│
├── kaggle_notebook/          # Configuración Kaggle
│   ├── kernel-metadata.json
│   ├── train_pytorch.py
│   ├── train_cuda.py         # Entrenamiento CUDA puro (500M params)
│   └── train.py
│
├── kaggle_output/            # Salidas de Kaggle
├── kaggle_output_cuda/
├── kaggle_output_v3/
├── kaggle_output_v4/
├── kaggle_logs/
├── kaggle_logs2/
├── kaggle_logs3/
│
├── checkpoints/              # Checkpoints de entrenamiento
├── checkpoints_10m/
│
├── model.pkl                 # Modelo NumPy serializado
├── model_10m.pkl             # Modelo 10M params
├── model_10m_final.pkl       # Modelo 10M final
├── model_test.pkl            # Modelo de prueba
│
├── Dockerfile                # Containerización
├── Procfile                  # Despliegue Heroku
├── requirements.txt          # Dependencias Python
├── README.md                 # Documentación
├── Plan.md                   # Plan maestro del proyecto
└── .gitignore
```

---

## 4. FUNCIONALIDADES PRINCIPALES

### 4.1 Transformer personalizado (`transformer.py`)
Implementación completa de un Transformer desde cero:
- **AutogradTensor**: Sistema de autodiferenciación personalizado
- **MultiHeadAttention**: Atención multi-cabeza con máscara causal
- **FeedForward**: Red feedforward con activación ReLU
- **TransformerBlock**: Bloque con residual connections y LayerNorm
- **NumpyTransformer**: Modelo completo con entrenamiento e inferencia
- **Mixed Precision**: Soporte fp16 con loss scaling dinámico
- **Data Prefetching**: Hilo de background para carga de datos
- **Gradient Accumulation**: Acumulación de gradientes
- **Cosine LR con Warmup**: Programación de learning rate
- **Checkpointing**: Guardado periódico de checkpoints

### 4.2 Motor de Chat (`chat_engine.py`)
Sistema de respuestas múltiples:
- **Detección de intenciones**: Saludos, despedidas, agradecimientos
- **NeuralCode**: Base de conocimiento de programación (40+ temas)
- **KnowledgeBase**: Búsqueda semántica en corpus
- **Generación**: Producción de texto vía el Transformer
- **FrontalLobe**: Sistema de decisión con pesos por fuente

### 4.3 Tokenizers (`tokenizers/`)
Sistema modular de tokenización:
- **BPE** (Byte Pair Encoding)
- **Unigram**
- **WordPiece**
- **SentencePiece** (BPE y Unigram)
- **Dynamic Vocabulary**: Crecimiento dinámico del vocabulario
- **Factory Pattern**: Creación flexible de tokenizers

### 4.4 Aceleración Rust (`rubidium-core/`)
Extensión nativa para inferencia rápida:
- Forward pass completo en Rust
- Generación de texto acelerada
- Carga de modelos pickle via PyO3
- Operaciones matriciales con ndarray

### 4.5 Entrenamiento en GPU (`train_pytorch.py`)
Pipeline de entrenamiento en Kaggle:
- Detección automática de GPU
- Instalación de PyTorch compatible (cu118)
- torch.compile para optimización
- Gradient accumulation (GA=16)
- AdamW con weight decay
- Guardado en formato pickle

---

## 5. ENDPOINTS / APIs

### API REST (FastAPI - `main.py`)

| Método | Endpoint | Descripción | Request Body |
|--------|----------|-------------|--------------|
| `GET` | `/` | Estado del servicio | - |
| `GET` | `/state` | Estado del modelo | - |
| `POST` | `/train` | Entrenar modelo | `TrainRequest` |
| `POST` | `/generate` | Generar texto | `GenerateRequest` |
| `POST` | `/save` | Guardar modelo | - |
| `POST` | `/load` | Cargar modelo | - |
| `POST` | `/clear-cache` | Limpiar caché | - |

#### Modelos de Request/Response

**GenerateRequest:**
```json
{
  "seed": "string (default: '')",
  "max_chars": "integer (default: 200)",
  "temperature": "float (default: 0.8)",
  "top_k": "integer (default: 20)"
}
```

**GenerateResponse:**
```json
{
  "text": "string"
}
```

**TrainRequest:**
```json
{
  "corpus": "string (default: '')",
  "block_size": "integer (default: 128)",
  "d_model": "integer (default: 128)",
  "n_head": "integer (default: 4)",
  "n_layer": "integer (default: 4)",
  "d_ff": "integer (default: 512)",
  "max_steps": "integer (default: 1000)",
  "learning_rate": "float (default: 3e-4)",
  "use_resources": "boolean (default: true)"
}
```

**StateResponse:**
```json
{
  "is_trained": "boolean",
  "vocab_size": "integer",
  "model_size": "string",
  "engine": "string (rust|numpy)",
  "cache_size": "integer"
}
```

### Interfaz Web (Gradio - `app.py`)

Puerto: 7860

**Pestañas:**
1. **Chat**: Generación de texto con parámetros ajustables
2. **Entrenar**: Entrenamiento con corpus personalizado
3. **Modelo**: Guardar/cargar modelo

---

## 6. MODELOS DE DATOS / BASE DE DATOS

### 6.1 Estructura del Modelo Transformer

```
NumpyTransformer
├── token_embedding: Embedding(vocab_size, d_model)
├── pos_embedding: AutogradTensor(1, block_size, d_model)
├── layers: List[TransformerBlock]
│   ├── ln1_w, ln1_b: LayerNorm
│   ├── attn: MultiHeadAttention
│   │   ├── wq: Linear(d_model, d_model)
│   │   ├── wk: Linear(d_model, d_model)
│   │   ├── wv: Linear(d_model, d_model)
│   │   └── wo: Linear(d_model, d_model)
│   ├── ln2_w, ln2_b: LayerNorm
│   └── mlp: FeedForward
│       ├── w1: Linear(d_model, d_ff)
│       └── w2: Linear(d_ff, d_model)
├── ln_f_w, ln_f_b: LayerNorm final
└── lm_head: Linear(d_model, vocab_size)
```

### 6.2 Configuraciones de Modelo

| Configuración | Params | d_model | n_head | n_layer | d_ff | block_size |
|--------------|--------|---------|--------|---------|------|------------|
| Local (small) | ~400K | 96 | 4 | 3 | 384 | 96 |
| Local (full) | ~840K | 128 | 4 | 4 | 512 | 128 |
| Kaggle (10M) | 14.4M | 2048 | 32 | 10 | 8192 | 256 |
| Kaggle (500M) | ~500M | 2048 | 32 | 10+ | 8192 | 256 |

### 6.3 Formato de Serialización (Pickle)

```python
{
    "vocab_size": int,
    "block_size": int,
    "d_model": int,
    "n_head": int,
    "n_layer": int,
    "d_ff": int,
    "char_to_id": dict,
    "id_to_char": dict,
    "token_emb": ndarray,      # (V, D)
    "pos_emb": ndarray,        # (1, T, D)
    "ln_f_w": ndarray,         # (D,)
    "ln_f_b": ndarray,         # (D,)
    "lm_w": ndarray,           # (D, V)
    "lm_b": ndarray,           # (V,)
    "layers": [                # List[layer_state]
        {
            "ln1_w": ndarray,
            "ln1_b": ndarray,
            "attn_wq_w": ndarray,
            "attn_wq_b": ndarray,
            # ... (16 tensores por capa)
        }
    ]
}
```

### 6.4 Corpus de Entrenamiento

**Formato de diálogos (`chat_*.txt`):**
```
U: [mensaje del usuario]
B: [respuesta del bot]
```

**Archivos:**
- 39 archivos de diálogos conversacionales
- 5 corpus temáticos (general, programación, web, algoritmos, git)
- 15 corpus adicionales (CorpusAgent)

**Tamaño estimado:** ~2.1M caracteres, 44 archivos

### 6.5 Base de Conocimiento (`neural_code.py`)

40+ preguntas/respuestas predefinidas sobre:
- Variables, funciones, bucles, clases
- Manejo de errores, condicionales
- Listas, arrays, diccionarios
- Recursión, algoritmos, debugging
- Interfaces, patrones de diseño
- Git, testing, TDD
- HTML, CSS, JavaScript
- SQL, bases de datos
- Python (lambda, decoradores, generators)

---

## 7. DEPENDENCIAS PRINCIPALES

### Python (`requirements.txt`)
```
numpy           # Computación numérica
fastapi         # API REST framework
uvicorn         # ASGI server
pydantic        # Validación de datos
maturin         # Build de extensiones Rust
```

### Rust (`rubidium-core/Cargo.toml`)
```toml
pyo3 = "0.22"           # Python bindings
numpy = "0.22"           # NumPy integration
ndarray = "0.16"         # Arrays N-dimensionales
rand = "0.8"             # Generación aleatoria
```

### Dependencias implícitas (no en requirements.txt)
- `gradio` (app.py)
- `torch` (train_pytorch.py)
- `pickle` (stdlib)

---

## 8. HALLAZGOS INTERESANTES

### 8.1 Arquitectura Única
- **Autodiferenciación desde cero**: No usa PyTorch para inferencia, implementa su propio sistema de autodiferenciación con `AutogradTensor`
- **Dual engine**: Puede usar tanto Python (NumPy) como Rust para inferencia
- **Mixed precision manual**: Implementa fp16 con loss scaling dinámico sin usar PyTorch

### 8.2 Sistema de Agentes
- **85 roles especializados** definidos en Markdown
- Jerarquía CTO → Directores → Expertos
- Sistema de tareas con cola (pending, working, completed, review)
- Reports estructurados por nivel

### 8.3 Innovaciones
- **Compresión de código**: Scripts para comprimir código Python en celdas de Kaggle usando base64+zlib
- **Auto-training**: Si no hay modelo guardado, entrena automáticamente desde los recursos
- **Cache de respuestas**: Sistema LRU con 200 entradas máximo
- **Fallback inteligente**: Múltiples fuentes de respuesta con sistema de ponderación

### 8.4 Metodología de Desarrollo
- **Filosofía**: "La simplicidad es la máxima sofisticación"
- **Modularidad**: Una responsabilidad por módulo
- **Documentación**: Templates para agentes y tareas
- **Control de versiones**: Git con .gitignore completo

### 8.5 Datos de Entrenamiento
- Corpus conversacional en español
- Datos de programación en múltiples lenguajes
- Formato U:/B: para diálogos
- Mezcla de conversación general y conocimiento técnico

---

## 9. POSIBLES MEJORAS

### 9.1 Código
1. **Dependencias faltantes**: Agregar `gradio` y `torch` a `requirements.txt`
2. **Tipado**: Agregar type hints completos en todos los archivos
3. **Tests**: No hay tests unitarios; agregar pytest con cobertura
4. **Linting**: Configurar pylint/ruff y black
5. **Docstrings**: Agregar documentación a todas las funciones públicas

### 9.2 Arquitectura
1. **Separar concerns**: `transformer.py` tiene 840 líneas; dividir en módulos
2. **Configuración**: Usar archivos YAML/JSON en lugar de hardcodear valores
3. **Logging**: Reemplazar print() con logging estructurado
4. **Gestión de estado**: Usar un ORM o base de datos para metadatos
5. **API versioning**: Agregar `/v1/` a los endpoints

### 9.3 Seguridad
1. **Rate limiting**: No hay límite de peticiones
2. **Autenticación**: API abierta sin auth
3. **Validación**: Sanitizar inputs del corpus
4. **Secrets**: No hay gestión de secretos (usar env vars)
5. **CORS**: Configurar origins específicos en producción

### 9.4 Rendimiento
1. **Batch processing**: Soporte para generación en lote
2. **Model caching**: Cachear modelos en memoria
3. **Async**: Hacer endpoints async con `await`
4. **Connection pooling**: Para bases de datos futuras
5. **CDN**: Para estáticos de Gradio

### 9.5 DevOps
1. **CI/CD**: No hay pipeline de integración continua
2. **Monitoring**: Agregar métricas (Prometheus/Grafana)
3. **Health checks**: Endpoint de salud más detallado
4. **Backup**: Estrategia de backup para modelos
5. **Scaling**: Configuración para múltiples instancias

### 9.6 Documentación
1. **API docs**: Swagger/ReDoc automático (FastAPI lo soporta)
2. **Guía de inicio**: Paso a paso para nuevos desarrolladores
3. **Arquitectura**: Diagrama de componentes
4. **Changelog**: Historial de cambios
5. **Contributing**: Guía de contribución

### 9.7 Funcionalidad
1. **Multi-idioma**: Soporte para inglés y otros idiomas
2. **Fine-tuning**: Interfaz para ajustar modelos pre-entrenados
3. **Evaluación**: Métricas de calidad de generación (perplejidad, BLEU)
4. **Streaming**: Respuestas streaming en la API
5. **WebSocket**: Para chat en tiempo real

---

## 10. CONCLUSIONES

Rubidium API es un proyecto ambicioso que demuestra:

1. **Dominio técnico**: Implementación completa de un Transformer desde cero
2. **Versatilidad**: Múltiples motores (NumPy, Rust, PyTorch)
3. **Organización**: Sistema de agentes jerárquicos bien estructurado
4. **Aprendizaje**: Proyecto educativo con enfoque en programación

El proyecto tiene una base sólida pero podría beneficiarse de:
- Mejores prácticas de ingeniería (tests, CI/CD, documentación)
- Optimización del código (modularidad, tipado)
- Seguridad y escalabilidad para producción

**Estado actual**: Funcional para desarrollo y demostración  
**Listo para producción**: No (requiere mejoras de seguridad y testing)

---

*Análisis generado automáticamente por opencode*
