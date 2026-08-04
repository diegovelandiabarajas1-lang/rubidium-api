# Análisis del Pipeline de Entrenamiento - Rubidium

## Resumen Ejecutivo

Este documento analiza el pipeline completo de generación de corpus, entrenamiento y API del proyecto Rubidium, un transformer de 505M parámetros diseñado para chatbot en español. El sistema incluye generadores de corpus sintético, entrenamiento CUDA/C++ optimizado, tokenización BPE híbrida y fine-tuning LoRA.

---

## 1. Pipeline de Generación de Corpus

### 1.1 Generadores Disponibles

| Archivo | Propósito | Pares Generados | Método |
|---------|-----------|----------------|--------|
| `generate_corpus.py` | Generador base NeMo Style | 10,000 (10 archivos × 1,000) | Templates + aleatorio |
| `generate_corpus_fast.py` | Generación rápida CPU | 150 (3 temas × 50) | Templates combinatorios |
| `generate_corpus_expanded.py` | Generación masiva | 51,000+ (6 temas × 8,500) | Expansión combinatoria |
| `generate_corpus_llm.py` | Generación con LLM local | 1,000 (20 temas × 50) | Phi-3-mini-4k-instruct |
| `resources/generate_corpus_expanded.py` | Copia del expanded | 51,000+ | Templates combinatorios |

### 1.2 Formato de Salida

**Formato U:/B: (Usuario/Bot):**
```
U: ¿Qué es la inteligencia artificial?
B: La inteligencia artificial es una tecnología que permite crear sistemas capaces de realizar tareas que normalmente requieren inteligencia humana.
```

**Formato JSONL:**
```json
{"user": "¿Qué es la inteligencia artificial?", "bot": "La inteligencia artificial es...", "topic": "tecnologia"}
```

**Formato JSON compacto:**
```json
[{"u": "pregunta", "b": "respuesta"}]
```

### 1.3 Análisis de Generadores

#### generate_corpus.py
- **Funcionalidad:** Genera 10 archivos temáticos (tecnología, ciencia, cotidiano, educación, salud, negocios, cultura, programación, matemáticas, historia)
- **Dependencias:** Solo stdlib (random, json, os)
- **Checkpoint system:** Guarda progreso en `/kaggle/working/checkpoints/`
- **Problema:** Respuestas genéricas y repetitivas basadas en templates fijos

#### generate_corpus_fast.py
- **Funcionalidad:** Generación rápida sin GPU
- **Limitación:** Solo 3 temas (tecnología, ciencia, cotidiano)
- **Output:** `corpus_generated_1k.jsonl` y `corpus_generated_1k.txt`
- **Velocidad:** Segundos para 150 pares

#### generate_corpus_expanded.py
- **Funcionalidad:** Generación masiva con 6 temas
- **Temas:** programación, ciencia, vida cotidiana, tecnología, matemáticas, historia, filosofía, salud
- **Deduplicación:** Elimina pares por pregunta exacta
- **Output:** `corpus_expanded.jsonl`, `corpus_expanded.txt`, `corpus_compact.json`

#### generate_corpus_llm.py
- **Funcionalidad:** Usa Phi-3-mini-4k-instruct (3.8B params) en GPU
- **Requisitos:** GPU P100 16GB, transformers, torch
- **Calidad:** Superior a templates, genera respuestas naturales
- **Limitación:** 1,000 pares máximo por sesión de Kaggle

### 1.4 Corpus Actual (resources/)

**Archivos .txt (39 chat_*.txt + 6 corpus_*.txt):**
- `chat_01.txt` a `chat_39.txt`: Diálogos sobre temas diversos
- `corpus_01.txt`: Saludos y conversación básica
- `corpus_02_programacion.txt`: Programación
- `corpus_03_web_dev.txt`: Desarrollo web
- `corpus_04_algoritmos.txt`: Algoritmos
- `corpus_05_git.txt`: Git
- `corpus_compact.json`: Versión JSON del corpus
- `corpus_expanded.jsonl`: Corpus expandido
- `corpus_expanded.txt`: Versión TXT del expandido

**Ejemplo de contenido (chat_01.txt):**
```
U: ¿Qué es un ambiente laboral tóxico?
B: Es aquel donde predominan el miedo, la desconfianza, los conflictos constantes...
```

**Ejemplo de contenido (corpus_01.txt):**
```
U: hola
B: ¡Hola! ¿Cómo estás hoy?

U: buenos días
B: ¡Buenos días! ¿Qué tal amaneciste?
```

---

## 2. Pipeline de Entrenamiento

### 2.1 Entrenador CUDA/C++ (`train_cuda.py`)

**Configuración del modelo:**
```python
V = 32000    # Tamaño vocabulario
T = 512      # Longitud contexto
D = 1536     # Dimensión modelo
H = 24       # Cabezas atención
L = 10       # Capas transformer
FF = 6144    # Dimensión FFN
```

**Hiperparámetros:**
- Batch size: 2
- Gradient accumulation: 16
- Effective batch: 32
- Learning rate: 3e-4
- Warmup: 6,000 steps
- Max steps: 200,000
- Weight decay: 0.1
- Gradient clipping: 1.0

**Proceso:**
1. Detecta GPU y verifica CUDA
2. Clona `rubidium-train` desde GitHub
3. Compila motor CUDA con cmake/make
4. Genera corpus expandido
5. Copia corpus a directorio de datos
6. Ejecuta entrenamiento (timeout: 10 horas)
7. Guarda `model_final.bin` y checkpoints

### 2.2 Entrenador PyTorch (`train_pytorch.py`)

**Arquitectura:**
```python
class Rubidium(nn.Module):
    # V=256, T=256, D=2048, H=32, L=10, FF=8192
    # ~505M parámetros
```

**Características:**
- Implementación PyTorch estándar
- `torch.compile` para GPU sm_70+
- Guarda en formato pickle (`model_10m_final.pkl`)
- Test rápido con seeds: "Hola", "Buenos dias", "Quien eres", "Que puedes hacer"

### 2.3 Motor CUDA/C++ (`main.cpp`)

**Configuración por defecto:**
```cpp
cfg.init(32000, 512, 1536, 24, 10, 6144);  // 250M params
cfg.use_fp16 = true;
cfg.use_activation_checkpointing = true;
```

**Optimizaciones:**
- FP16 mixed precision
- Activation checkpointing
- cuBLAS para GEMM
- cuDNN para operaciones optimizadas
- Checkpoints cada 5,000 steps

**Proceso de entrenamiento:**
1. Carga corpus desde directorio (glob *.txt)
2. Construye vocabulario byte-level (V=256 por defecto)
3. Codifica texto a tokens
4. Inicializa modelo transformer
5. Loop de entrenamiento con gradient accumulation
6. Guarda checkpoints y modelo final
7. Test de generación con 4 seeds

---

## 3. Tokenización

### 3.1 Tokenizer BPE + Unigram (`tokenizer.h` / `tokenizer.py`)

**Implementación:**
- **BPE (Byte Pair Encoding):** Aprende mergeos de caracteres frecuentes
- **Unigram:** Calcula probabilidades para selección óptima de subpalabras
- **Tokens especiales:** `<pad>`, `<unk>`, `<bos>`, `<eos>`, `<sep>`

**Entrenamiento:**
```python
tokenizer = BPETokenizer(vocab_size=32000)
tokenizer.train_bpe(corpus_text, num_merges=30000)
```

**Uso:**
```python
tokens = tokenizer.encode("Hola, ¿cómo estás?")  # [2, ...ids..., 3]
text = tokenizer.decode(tokens)  # "Hola, ¿cómo estás?"
```

**Estadísticas de compresión:**
- Texto: "Hola, ¿cómo estás?" → ~15 tokens (1.5x compresión)
- Texto: "La inteligencia artificial..." → ~25 tokens (2x compresión)

### 3.2 Tokenización Byte-Level (Entrenador CUDA)

**En `main.cpp`:**
```cpp
// Vocabulario byte-level simple
for (unsigned char c : full_text) {
    if (c2i.find(c) == c2i.end()) {
        int id = c2i.size();
        c2i[c] = id;
        i2c[id] = c;
    }
}
```

**Ventajas:**
- Sin OOV (out-of-vocabulary)
- Simple y rápido
- Funciona con cualquier idioma

**Desventajas:**
- Tokens más largos
- Menor compresión que BPE
- Requiere más contexto para mismo significado

### 3.3 Estado del Tokenizer

| Componente | Estado | Notas |
|------------|--------|-------|
| `tokenizer.h` | Implementado | Parser JSON manual, BPE funcional |
| `tokenizer.py` | Implementado | Entrenamiento y uso completos |
| `tokenizer_cuda.cu` | Implementado | Versión CUDA para inferencia |
| Integración en main.cpp | Parcial | Usa byte-level, no BPE |

**Problema:** El tokenizer BPE no está integrado en el pipeline de entrenamiento principal. El entrenador CUDA usa tokenización byte-level simple.

---

## 4. Fine-Tuning con LoRA

### 4.1 Implementación (`lora.h`)

**Configuración LoRA:**
```cpp
struct LoRAConfig {
    int rank = 16;
    float alpha = 32.0f;
    float dropout = 0.1f;
    float scaling() const { return alpha / rank; }
};
```

**Módulos objetivo:**
- Q projection (atención)
- K projection (atención)
- V projection (atención)
- O projection (atención)
- FFN W1 (feed-forward)
- FFN W2 (feed-forward)

**Fórmula LoRA:**
```
W = W_base + A @ B * scaling
A: [d, r], B: [r, d] -> A@B: [d, d]
```

**Parámetros LoRA estimados:**
- Q,K,V,O: 4 × 2 × d × r = 4 × 2 × 1536 × 16 = 196,608
- W1: 2 × d × r + 2 × r × FF = 2 × 1536 × 16 + 2 × 16 × 6144 = 245,760
- W2: 2 × FF × r + 2 × r × d = 2 × 6144 × 16 + 2 × 16 × 1536 = 245,760
- **Total: ~688,128 parámetros (~0.7M)**

### 4.2 Estado de Implementación

| Función | Estado | Notas |
|---------|--------|-------|
| `init()` | Implementado | Asigna memoria CUDA |
| `init_weights()` | Stub | Solo declaración |
| `forward_q()` | Implementado | Matmul con cuBLAS |
| `backward_q()` | Stub | Solo declaración |
| `optimizer_step()` | Stub | Solo declaración |
| `free()` | Implementado | Liberación de memoria |
| `lora.cu` | Vacío | Archivo vacío (0 bytes) |

**Problema crítico:** Las funciones de backward y optimizer step no están implementadas. `lora.cu` está completamente vacío.

---

## 5. Análisis de Calidad del Corpus

### 5.1 Corpus Actual vs Requerido

| Métrica | Actual | Mínimo | Recomendado | Ideal |
|---------|--------|--------|-------------|-------|
| Tokens | ~984K | 100M | 200M | 2B |
| Pares U:/B: | ~2,500 | 30,000 | 630,000 | 6,000,000 |
| Épocas@200K | 1,665x | 16x | 8x | 0.8x |
| Calidad | POOR | BASIC | GOOD+ | EXCELLENT+ |

### 5.2 Problemas Identificados

1. **Sobreajuste extremo:** 505M parámetros con 984K tokens = ratio 515:1
2. **Respuestas genéricas:** Templates generan texto repetitivo
3. **Falta diversidad:** Pocos temas cubiertos
4. **Calidad inconsistente:** Mezcla de texto literario (Sherlock Holmes) con Q&A sintético
5. **Sin validación:** No hay split train/val

### 5.3 Distribución de Archivos Corpus

```
corpus/
├── CorpusAgent1.txt    119,466 bytes
├── CorpusAgent2.txt    253,555 bytes (mayor)
├── CorpusAgent3.txt    (vacío)
├── CorpusAgent4.txt    (vacío)
├── CorpusAgent5.txt     90,501 bytes
├── CorpusAgent6.txt    217,474 bytes
├── CorpusAgent7.txt     31,321 bytes
├── CorpusAgent8.txt     12,868 bytes
├── CorpusAgent9.txt     23,288 bytes
├── CorpusAgent10.txt    22,514 bytes
├── CorpusAgent11.txt   135,465 bytes
├── CorpusAgent12.txt    60,934 bytes
├── CorpusAgent13.txt    (vacío)
├── CorpusAgent14.txt    52,568 bytes
└── CorpusAgent15.txt    (vacío)
```

**Problemas:**
- 4 archivos vacíos
- Tamaños inconsistentes (12KB a 253KB)
- Sin organización temática clara

---

## 6. Configuraciones de Entrenamiento

### 6.1 CUDA vs CPU

| Aspecto | CUDA (train_cuda.py) | CPU (train_pytorch.py) |
|---------|---------------------|----------------------|
| Modelo | 250M params | 505M params |
| Hardware | P100 16GB | CPU genérico |
| Velocidad | ~0.8 steps/s | ~0.01 steps/s |
| FP16 | Sí | No |
| Activation ckpt | Sí | No |
| torch.compile | No (cmake) | Sí (si sm_70+) |
| Tiempo 200K steps | ~69 horas | ~2,000+ horas |

### 6.2 Configuración Kaggle

**kernel-metadata.json:**
```json
{
  "id": "diegovelandiabarajas/rubidium-corpus-fast",
  "title": "Rubidium Corpus Fast Generator",
  "enable_gpu": false,
  "enable_internet": true
}
```

**Problema:** `enable_gpu: false` para un generador de corpus que puede usar LLM local.

### 6.3 Requisitos de Hardware

| Configuración | VRAM Necesaria | VRAM Disponible | Headroom |
|---------------|----------------|-----------------|----------|
| BS=1, T=256 | ~8.0 GB | 16 GB | 8.0 GB |
| BS=2, T=256 | ~8.1 GB | 16 GB | 7.9 GB |
| BS=8, T=256 | ~8.7 GB | 16 GB | 7.3 GB |
| BS=16, T=256 | ~9.5 GB | 16 GB | 6.5 GB |

---

## 7. Conexiones entre Componentes

### 7.1 Flujo del Pipeline

```
Generación Corpus (Python)
        ↓
    corpus/*.txt / corpus_expanded.jsonl
        ↓
Tokenización (tokenizer.py o byte-level)
        ↓
    tokens.bin / data/*.txt
        ↓
Entrenamiento (train_cuda.py / main.cpp)
        ↓
    model_final.bin / model_10m_final.pkl
        ↓
Inferencia (API / chat_engine.py)
```

### 7.2 Dependencias

```
kaggle_notebook_cuda/
├── generate_corpus*.py → Produce corpus
├── train_cuda.py → Usa corpus, compila rubidium-train
└── kernel-metadata.json → Configuración Kaggle

rubidium-train/src/
├── main.cpp → Entrenador principal
├── model.h/cu → Arquitectura transformer
├── tokenizer.h/py → Tokenización BPE
├── lora.h/cu → Fine-tuning (incompleto)
└── generate_corpus_expanded.py → Generador alternativo

rubidium-api/
├── resources/*.txt → Corpus de entrenamiento
├── train_pytorch.py → Entrenador PyTorch
├── corpus/*.txt → Corpus legacy
└── app.py → API de inferencia
```

---

## 8. Recomendaciones

### 8.1 Corpus (Prioridad ALTA)

1. **Expandir a 200M tokens mínimo**
   - Usar generate_corpus_llm.py con Phi-3 o GPT-4
   - Scraping de foros españoles (Reddit, Discourse)
   - Libros de dominio público (Project Gutenberg)

2. **Mejorar calidad**
   - Filtrar respuestas genéricas
   - Eliminar duplicados fuzzily
   - Validar gramática con LanguageTool
   - Mantener consistencia de persona

3. **Organizar en 10 archivos temáticos**
   ```
   01_daily_conversations.txt    (40M tokens)
   02_knowledge_facts.txt        (30M tokens)
   03_emotional_support.txt      (20M tokens)
   04_creative_writing.txt       (20M tokens)
   05_technology_science.txt     (30M tokens)
   06_health_wellness.txt        (20M tokens)
   07_culture_entertainment.txt  (20M tokens)
   08_education_learning.txt     (10M tokens)
   09_opinions_debate.txt        (10M tokens)
   10_meta_conversation.txt      (10M tokens)
   ```

### 8.2 Tokenización (Prioridad MEDIA)

1. **Integrar tokenizer BPE en entrenamiento CUDA**
   - Entrenar tokenizer en corpus completo
   - Guardar `tokenizer.json`
   - Modificar `main.cpp` para usar BPE

2. **Vocabulario óptimo**
   - Spanish: V=32,000 (actual: 256 byte-level)
   - Mejor compresión, menos tokens por secuencia

### 8.3 Entrenamiento (Prioridad MEDIA)

1. **Agregar validación split (5-10%)**
   - Prevenir sobreajuste
   - Monitorear perplexity

2. **Implementar dropout (p=0.1)**
   - Regularización adicional
   - Mejor generalización

3. **Mixed precision completo**
   - FP16 para forward/backward
   - FP32 para optimizer states

### 8.4 LoRA (Prioridad BAJA)

1. **Completar implementación**
   - Implementar `backward_q()`, `backward_k()`, etc.
   - Implementar `optimizer_step()` con AdamW
   - Llenar `lora.cu` con kernels CUDA

2. **Testing**
   - Verificar convergencia
   - Comparar con fine-tuning completo

### 8.5 Infraestructura (Prioridad BAJA)

1. **Kaggle metadata**
   - Cambiar `enable_gpu: true` para generadores LLM
   - Agregar dataset sources

2. **CI/CD**
   - Tests automáticos
   - Validación de corpus
   - Benchmarks de calidad

---

## 9. Métricas Clave a Monitorear

### Durante Entrenamiento
| Métrica | Objetivo | Señal de Alerta |
|---------|----------|-----------------|
| Training loss | Decreciente estable | Meseta antes de 50K steps |
| Loss variance | Baja varianza | Alta oscilación |
| Perplexity | < 10 al final | > 50 al final |
| Gradient norm | < 1.0 (clipped) | Clipping frecuente |

### Durante Evaluación
| Métrica | Objetivo | Señal de Alerta |
|---------|----------|-----------------|
| Coherencia | Gramaticalmente correcto | Salida incoherente |
| Relevancia temática | Mantiene tema | Cambios aleatorios |
| Precisión factual | Mayormente correcta | Alucinaciones frecuentes |
| Flujo conversacional | Intercambios naturales | Respuestas robóticas |
| Diversidad | Respuestas variadas | Salidas repetitivas |

---

## 10. Conclusión

El pipeline de Rubidium tiene una arquitectura sólida pero presenta carencias críticas:

1. **Corpus insuficiente:** 984K tokens vs 200M recomendados (200x deficit)
2. **Tokenización no integrada:** BPE implementado pero no usado en entrenamiento principal
3. **LoRA incompleto:** Solo forward implementado, backward y optimizer vacíos
4. **Calidad variable:** Mezcla de corpus real y generado sintéticamente

**Prioridad inmediata:** Expandir corpus a 100M+ tokens usando generación con LLM y scraping web.

---

*Documento generado: 2026-08-04*
*Análisis basado en: rubidium-api/kaggle_notebook_cuda/, rubidium-train/src/, rubidium-api/resources/*
