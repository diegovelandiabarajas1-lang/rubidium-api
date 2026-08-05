# Servicios, Utilidades y Lógica Core - Rubidium API

## Tabla de Contenidos
1. [Arquitectura General](#arquitectura-general)
2. [Motor de Chat (chat_engine.py)](#motor-de-chat)
3. [Base de Conocimiento (knowledge_base.py)](#base-de-conocimiento)
4. [Código Neural (neural_code.py)](#código-neural)
5. [Lóbulo Frontal (frontal_lobe.py)](#lóbulo-frontal)
6. [Similitud de Texto (text_similarity.py)](#similitud-de-texto)
7. [Fuentes de Respuesta (response_source.py)](#fuentes-de-respuesta)
8. [Transformer (transformer.py)](#transformer)
9. [Sistema de Tokenizers](#sistema-de-tokenizers)
10. [API REST (main.py)](#api-rest)
11. [Interfaz Gradio (app.py)](#interfaz-gradio)
12. [Conexiones e Integraciones](#conexiones-e-integraciones)

---

## Arquitectura General

Rubidium API es un sistema de IA para generación de texto y chat con arquitectura inspirada en el cerebro humano. El sistema integra múltiples componentes especializados:

```
┌─────────────────────────────────────────────────────────────┐
│                    RUBIDIUM API                              │
├─────────────────────────────────────────────────────────────┤
│  API REST (FastAPI)  │  Interfaz Gradio                     │
├─────────────────────────────────────────────────────────────┤
│                    ChatEngine                               │
│  ┌──────────┐ ┌──────────────┐ ┌───────────────┐           │
│  │ FrontalL │ │ NeuralCode   │ │ KnowledgeBase │           │
│  │ (Decide) │ │ (Código)     │ │ (Corpus)      │           │
│  └──────────┘ └──────────────┘ └───────────────┘           │
├─────────────────────────────────────────────────────────────┤
│              TextSimilarity + WordEmbeddings                │
├─────────────────────────────────────────────────────────────┤
│              Transformer (NumpyTransformer)                 │
│              + Tokenizers (BPE, Unigram, etc.)              │
└─────────────────────────────────────────────────────────────┘
```

---

## Motor de Chat

**Archivo:** `chat_engine.py`

### Descripción
El motor principal de conversación que orquesta todas las fuentes de conocimiento para generar respuestas. Funciona como el "cerebro" del sistema.

### Métodos Principales

| Método | Parámetros | Retorno | Descripción |
|--------|------------|---------|-------------|
| `__init__` | `model=None` | - | Inicializa el motor con modelo transformer opcional |
| `load_knowledge` | `corpus_text: str` | - | Carga corpus de conocimiento para responder preguntas |
| `respond` | `user_message: str` | `str` | Genera respuesta integrando todas las fuentes |
| `_generate_reply` | `seed: str` | `str` | Genera texto usando el modelo transformer |
| `_seed_from` | `message: str` | `str` | Extrae semilla del mensaje para generación |
| `_is_coherent` | `text: str` | `bool` | Valida coherencia del texto generado |
| `_match_intent` | `lower: str` | `Optional[str]` | Detecta intenciones (saludos, despedidas, agradecimientos) |
| `_fallback` | - | `str` | Retorna respuesta de reserva aleatoria |

### Flujo de Respuesta
```
Mensaje Usuario
    ↓
┌─────────────────────────┐
│ 1. Detectar Intención   │ → Saludo/Despedida/Agradecimiento
│ 2. NeuralCode.query()   │ → Respuestas de código
│ 3. KnowledgeBase.query()│ → Respuestas del corpus
│ 4. Transformer.generate()│ → Texto generado
│ 5. Fallback             │ → Respuesta de reserva
└─────────────────────────┘
    ↓
FrontalLobe.decide() → Mejor candidato
```

### Constantes
- `GREETINGS`: Lista de saludos en español
- `FAREWELLS`: Lista de despedidas
- `THANKS`: Lista de agradecimientos
- `KNOW_MIN_SCORE = 0.22`: Score mínimo para conocimiento
- `GEN_COHERENT_MIN_WORDS = 4`: Mínimo de palabras para coherencia
- `GEN_UNIQUE_RATIO = 0.6`: Ratio mínimo de palabras únicas
- `FALLBACKS`: Lista de respuestas de reserva

---

## Base de Conocimiento

**Archivo:** `knowledge_base.py`

### Descripción
Almacena y busca información en corpus de texto usando similitud semántica. Divide el corpus en pasajes y extrae las oraciones más relevantes.

### Métodos Principales

| Método | Parámetros | Retorno | Descripción |
|--------|------------|---------|-------------|
| `__init__` | - | - | Inicializa la base de conocimiento |
| `load` | `corpus_text: str` | - | Carga corpus de texto |
| `use_embeddings` | `embeddings: Optional[WordEmbeddings]` | - | Configura embeddings para similitud |
| `answer` | `query_tokens, query_vec` | `Tuple[Optional[str], float]` | Busca mejor respuesta para la consulta |
| `_ensure_built` | - | - | Construye pasajes internos si es necesario |
| `_extract_best_sentences` | `passage, query_tokens, query_vec` | `str` | Extrae oración más relevante del pasaje |

### Funciones Auxiliares
- `_split_paragraphs(text)`: Divide texto en párrafos
- `_split_sentences(paragraph)`: Divide párrafo en oraciones
- `_count_words(s)`: Cuenta palabras en texto

### Constantes
- `MAX_PASSAGES = 600`: Máximo de pasajes indexados
- `MIN_PASSAGE_WORDS = 8`: Mínimo de palabras por pasaje

### Estructura Interna
```python
class _Passage:
    text: str          # Texto completo del pasaje
    tokens: Set[str]   # Tokens de contenido
    vector: np.ndarray # Vector embedding promedio
    sentences: List[str] # Oraciones del pasaje
```

---

## Código Neural

**Archivo:** `neural_code.py`

### Descripción
Base de conocimiento especializada en programación con ~50 respuestas predefinidas sobre conceptos de código. Detecta lenguajes de programación y provee explicaciones.

### Métodos Principales

| Método | Parámetros | Retorno | Descripción |
|--------|------------|---------|-------------|
| `__init__` | - | - | Inicializa con base de conocimiento de código |
| `use_embeddings` | `embeddings: Optional[WordEmbeddings]` | - | Configura embeddings para similitud |
| `looks_like_code` | `tokens: Set[str]` | `bool` | Detecta si el mensaje contiene términos de código |
| `answer` | `user_tokens, user_vec` | `Tuple[Optional[str], float]` | Busca mejor respuesta de código |
| `_detect_language` | `tokens: Set[str]` | `Optional[str]` | Detecta lenguaje: Python, JavaScript, Java, SQL, C#, HTML, CSS |
| `_build_knowledge_base` | - | - | Construye ~50 pares pregunta-respuesta |

### Lenguajes Detectados
- JavaScript / JS
- Python
- Java
- SQL
- C# / CSharp
- HTML
- CSS / Flexbox / Grid

### Categorías de Respuestas
1. **Conceptos básicos**: Variables, funciones, clases, objetos
2. **Estructuras de control**: If/else, for, while
3. **Data structures**: Listas, arrays, diccionarios, pilas, colas
4. **Programación web**: HTML, CSS, Flexbox, Grid, responsive
5. **Python avanzado**: Lambdas, generadores, decoradores, f-strings, venv
6. **JavaScript**: Async/await, funciones flecha
7. **SQL**: SELECT, JOINs
8. **Git**: Commits, ramas, control de versiones
9. **Testing**: Pruebas unitarias, TDD
10. **Arquitectura**: Patrones de diseño, buenas prácticas

---

## Lóbulo Frontal

**Archivo:** `frontal_lobe.py`

### Descripción
Componente de decisión que selecciona la mejor respuesta entre múltiples candidatas usando pesos por fuente y evitando repeticiones.

### Métodos Principales

| Método | Parámetros | Retorno | Descripción |
|--------|------------|---------|-------------|
| `__init__` | - | - | Inicializa con historial de respuestas recientes |
| `decide` | `candidates: List[ResponseCandidate]` | `str` | Selecciona mejor candidata y retorna respuesta |
| `_recently_said` | `text: str` | `bool` | Verifica si la respuesta fue dicha recientemente |
| `_remember` | `text: str` | - | Guarda respuesta en historial |

### Pesos por Fuente
| Fuente | Peso | Descripción |
|--------|------|-------------|
| `Intent` | 1.15 | Saludos, despedidas (mayor prioridad) |
| `Code` | 1.12 | Respuestas de código |
| `Knowledge` | 1.02 | Conocimiento del corpus |
| `Generation` | 0.70 | Texto generado por red neuronal |
| `Fallback` | 0.10 | Respuesta de reserva (menor prioridad) |

### Constantes
- `ACCEPT_THRESHOLD = 0.30`: Score mínimo para aceptar respuesta
- `MAX_RECENT = 4`: Máximo de respuestas en historial

### Propiedades
- `last_source`: Última fuente usada
- `last_source_label`: Nombre legible de la fuente

---

## Similitud de Texto

**Archivo:** `text_similarity.py`

### Descripción
Utilidades para procesamiento de texto y cálculo de similitud entre documentos usando Jaccard y Cosine similarity.

### Funciones Principales

| Función | Parámetros | Retorno | Descripción |
|---------|------------|---------|-------------|
| `content_tokens` | `text: str` | `Set[str]` | Extrae tokens de contenido (sin stop words) |
| `content_sequence` | `text: str` | `List[str]` | Extrae tokens en orden secuencial |
| `sentence_vector` | `tokens, embeddings` | `Optional[np.ndarray]` | Calcula vector promedio de tokens |
| `combined` | `a_tokens, a_vec, b_tokens, b_vec` | `float` | Combina Jaccard (0.55) + Cosine (0.45) |
| `jaccard` | `a: Set, b: Set` | `float` | Similitud de Jaccard |
| `cosine` | `a: np.ndarray, b: np.ndarray` | `float` | Similitud coseno |

### Clase WordEmbeddings
```python
class WordEmbeddings:
    dimension: int
    _vectors: Dict[str, np.ndarray]
    
    def try_embed(word: str) -> Optional[np.ndarray]
    def add_embedding(word: str, vector: np.ndarray)
```

### Stop Words (español + inglés)
Incluye artículos, preposiciones, pronombres y verbos comunes (~40 palabras).

---

## Fuentes de Respuesta

**Archivo:** `response_source.py`

### Descripción
Define las fuentes de respuesta disponibles y la estructura de candidatos.

### Enum ResponseSource
| Valor | Nombre | Descripción |
|-------|--------|-------------|
| 0 | Intent | Intención detectada (saludo, etc.) |
| 1 | Code | Respuesta de código |
| 2 | Knowledge | Conocimiento del corpus |
| 3 | Generation | Texto generado por red |
| 4 | Fallback | Respuesta de reserva |

### NamedTuple ResponseCandidate
```python
class ResponseCandidate(NamedTuple):
    text: str           # Texto de la respuesta
    confidence: float   # Score de confianza (0-1)
    source: ResponseSource  # Fuente de origen
```

---

## Transformer

**Archivo:** `transformer.py`

### Descripción
Implementación completa de un Transformer tipo GPT desde cero usando NumPy. Incluye autograd, multi-head attention, y entrenamiento con mixed precision.

### Clases Principales

#### AutogradTensor
Sistema de diferenciación automática para backpropagation.

| Método | Operación |
|--------|-----------|
| `__add__`, `__sub__`, `__mul__`, `__truediv__` | Operaciones aritméticas |
| `__matmul__` | Multiplicación matricial (@) |
| `sum`, `mean` | Reducciones |
| `exp`, `log`, `relu` | Funciones de activación |
| `reshape`, `transpose` | Manipulación de forma |
| `backward` | Backpropagation |

#### Linear
Capa lineara: `output = x @ weight + bias`

#### Embedding
Capa de embedding con one-hot encoding.

#### MultiHeadAttention
Atención multi-cabeza con mascara causal.

| Método | Descripción |
|--------|-------------|
| `__call__(x)` | Forward pass con autograd |
| `forward_numpy(x)` | Forward pass optimizado NumPy |

#### FeedForward
Red feed-forward: `w2(relu(w1(x)))`

#### TransformerBlock
Bloque transformer: `x + Attn(LN(x))` + `x + FFN(LN(x))`

#### NumpyTransformer
Modelo completo tipo GPT.

| Método | Parámetros | Descripción |
|--------|------------|-------------|
| `__init__` | vocab_size, block_size, d_model, n_head, n_layer, d_ff, max_steps, lr | Constructor |
| `forward` | `x: np.ndarray` | Forward pass con autograd |
| `forward_numpy` | `x: np.ndarray` | Forward pass optimizado |
| `loss` | `logits, targets, label_smoothing` | Pérdida cross-entropy con label smoothing |
| `train` | `text: str` | Agrega texto al corpus |
| `fit` | gradient accumulation, warmup, fp16, checkpointing | Entrenamiento completo |
| `generate` | `seed, max_chars, temperature, top_k` | Generación de texto |
| `save` / `load` | `path: str` | Serialización del modelo |

### Parámetros por Defecto
```python
vocab_size = 256
block_size = 128
d_model = 128
n_head = 4
n_layer = 4
d_ff = 512
max_steps = 1000
learning_rate = 3e-4
```

### Características de Entrenamiento
- Mixed precision (fp16/fp32)
- Data prefetching en background thread
- Gradient accumulation
- Cosine LR con warmup
- Dynamic loss scaling
- Checkpointing automático

---

## Sistema de Tokenizers

**Carpeta:** `tokenizers/`

### Arquitectura
```
tokenizer/
├── __init__.py          # Exportaciones
├── base.py              # Interfaz base y config
├── factory.py           # Factory pattern
├── bpe.py               # Byte Pair Encoding
├── unigram.py           # Unigram (Viterbi)
├── wordpiece.py         # WordPiece (BERT-style)
├── sentencepiece.py     # SentencePiece (BPE/Unigram)
└── dynamic.py           # Vocabulary dinámico
```

### Interfaz Base (ITokenizer)

| Método | Descripción |
|--------|-------------|
| `train(corpus, vocab_size)` | Entrena tokenizador |
| `encode(text) -> List[int]` | Texto a tokens |
| `decode(tokens) -> str` | Tokens a texto |
| `token_to_id(token) -> int` | Token a ID |
| `id_to_token(id) -> str` | ID a token |
| `get_vocab() -> Dict` | Obtiene vocabulario |
| `save(path)` / `load(path)` | Serialización |

### TokenizerConfig
```python
type: TokenizerType      # BPE, Unigram, WordPiece, etc.
vocab_size: int = 1024
add_special_tokens: bool = True
pad_token: str = "<PAD>"
unk_token: str = "<UNK>"
bos_token: str = "<BOS>"
eos_token: str = "<EOS>"
min_frequency: int = 2
pre_tokenize: bool = True
pre_tokenize_regex: str = r"\w+|[^\w\s]"
rust_acceleration: bool = True
```

### BPETokenizer
- Entrenamiento: Frecuencia de pares → merges
- Encode: Aplica merges por prioridad
- Decode: Reconstruye texto con espacios

### UnigramTokenizer
- Entrenamiento: EM algorithm (E-step, M-step)
- Encode: Viterbi decoding (más probable segmentación)
- Usa log-probabilities

### WordPieceTokenizer
- Similar a BPE pero con scoring diferente
- Prefijo `##` para sub-palabras
- Usado en BERT

### SentencePieceTokenizer
- Normalización UTF-8 a bytes
- Soporta BPE y Unigram
- Maneja texto sin pre-tokenización

### DynamicVocabularyTokenizer
- Wrapper que crece vocabulario dinámicamente
- Tracking de Out-Of-Vocabulary tokens
- Crece cada `dynamic_growth_interval` steps
- Threshold para agregar tokens frecuentes

### TokenizerFactory
```python
TokenizerFactory.create(config) -> ITokenizer
```
Crea tokenizador basado en tipo de config.

---

## API REST

**Archivo:** `main.py`

### Descripción
API REST con FastAPI que expone endpoints para entrenamiento, generación y gestión del modelo.

### Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/` | Info del servicio y estado |
| `GET` | `/state` | Estado detallado del modelo |
| `POST` | `/train` | Entrena modelo con corpus |
| `POST` | `/generate` | Genera texto |
| `POST` | `/save` | Guarda modelo a disco |
| `POST` | `/load` | Carga modelo desde disco |
| `POST` | `/clear-cache` | Limpia cache de respuestas |

### Modelos de Request/Response

```python
class GenerateRequest:
    seed: str = ""
    max_chars: int = 200
    temperature: float = 0.8
    top_k: int = 20

class TrainRequest:
    corpus: str = ""
    block_size: int = 128
    d_model: int = 128
    n_head: int = 4
    n_layer: int = 4
    d_ff: int = 512
    max_steps: int = 1000
    learning_rate: float = 3e-4
    use_resources: bool = True
```

### Funcionalidades
- **Cache de respuestas**: Max 200 entradas con LRU eviction
- **Auto-entrenamiento**: Si no hay modelo, entrena desde resources/
- **Aceleración Rust**: Intenta usar `rubidium_core` para inferencia rápida
- **CORS habilitado**: Permite cualquier origen

---

## Interfaz Gradio

**Archivo:** `app.py`

### Descripción
Interfaz web con Gradio para interactuar con el modelo de forma visual.

### Pestañas

1. **Chat**: Generación de texto con parámetros
2. **Entrenar**: Entrenamiento con corpus personalizado
3. **Modelo**: Guardar/cargar modelo

### Parámetros Interactivos
- Prompt (texto semilla)
- Max chars (10-500)
- Temperature (0.1-2.0)
- Top K (1-100)
- Block Size, D Model, N Head, N Layer, D FF
- Max Steps, Learning Rate

---

## Conexiones e Integraciones

### Flujo de Datos Completo

```
                    ┌─────────────────┐
                    │  Usuario (API)  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   FastAPI       │
                    │   /generate     │
                    └────────┬────────┘
                             │
              ┌──────────────▼──────────────┐
              │         ChatEngine          │
              │  (Orquestador Principal)    │
              └──────────────┬──────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    ┌────▼────┐      ┌──────▼──────┐      ┌─────▼─────┐
    │ Intent  │      │ NeuralCode  │      │ Knowledge │
    │ Matcher │      │ (Código)    │      │ Base      │
    └────┬────┘      └──────┬──────┘      └─────┬─────┘
         │                  │                   │
         │         ┌────────▼────────┐          │
         │         │ TextSimilarity  │          │
         │         │ (Jaccard+Cosine)│          │
         │         └────────┬────────┘          │
         │                  │                   │
         │         ┌────────▼────────┐          │
         │         │ WordEmbeddings  │          │
         │         │ (Vectores)      │          │
         │         └─────────────────┘          │
         │                                      │
         └───────────────────┬──────────────────┘
                             │
                    ┌────────▼────────┐
                    │  FrontalLobe    │
                    │  (Decisión)     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Respuesta      │
                    └─────────────────┘
```

### Dependencias entre Módulos

| Módulo | Depende de |
|--------|------------|
| `chat_engine.py` | text_similarity, response_source, neural_code, knowledge_base, frontal_lobe |
| `knowledge_base.py` | text_similarity |
| `neural_code.py` | text_similarity |
| `frontal_lobe.py` | response_source |
| `text_similarity.py` | numpy (standalone) |
| `response_source.py` | (standalone) |
| `transformer.py` | numpy, pickle, rubidium_core (opcional) |
| `main.py` | transformer, rubidium_core (opcional) |
| `app.py` | transformer, tokenizers |

### Integraciones Externas

1. **rubidium_core (Rust)**
   - Aceleración de inferencia
   - Carga desde pickle
   - Fallback a Python si no disponible

2. **Archivos de Recursos**
   - `resources/*.txt`: Corpus para auto-entrenamiento
   - `model.pkl`: Modelo serializado
   - `checkpoints/`: Checkpoints durante entrenamiento

3. **No hay bases de datos externas**
   - Todo es in-memory
   - Persistencia via archivos pickle

### Procesamiento de Datos

1. **Tokenización de entrada**: content_tokens() extrae palabras relevantes
2. **Vectorización**: sentence_vector() promedia embeddings
3. **Matching**: combined() calcula similitud Jaccard + Cosine
4. **Selección**: FrontalLobe.decide() elige mejor candidata
5. **Generación**: Transformer genera texto carácter a carácter

### Almacenamiento

| Archivo | Contenido |
|---------|-----------|
| `model.pkl` | Modelo transformer serializado |
| `checkpoints/model_step_*.pkl` | Checkpoints de entrenamiento |
| `resources/*.txt` | Corpus de texto para entrenamiento |

---

## Resumen de Servicios

| Servicio | Archivo | Función Principal |
|----------|---------|-------------------|
| **ChatEngine** | chat_engine.py | Orquestación de conversación |
| **KnowledgeBase** | knowledge_base.py | Almacenamiento y búsqueda de conocimiento |
| **NeuralCode** | neural_code.py | Respuestas de programación |
| **FrontalLobe** | frontal_lobe.py | Selección de respuestas |
| **TextSimilarity** | text_similarity.py | Cálculo de similitud |
| **ResponseSource** | response_source.py | Definición de fuentes |
| **NumpyTransformer** | transformer.py | Modelo GPT desde cero |
| **Tokenizers** | tokenizers/ | Sistema de tokenización |
| **FastAPI** | main.py | API REST |
| **Gradio** | app.py | Interfaz web |
