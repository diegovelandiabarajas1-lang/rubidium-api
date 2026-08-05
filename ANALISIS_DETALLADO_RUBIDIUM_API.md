# Análisis Detallado de Rubidium API

## 1. Flujo de Autenticación y Autorización

**Estado actual: NO IMPLEMENTADO**

La API no tiene ningún mecanismo de autenticación o autorización. Todos los endpoints son públicos y accesibles sin credenciales. Esto representa un riesgo significativo de seguridad ya que cualquier usuario puede:
- Entrenar el modelo con datos arbitrarios
- Generar texto sin restricciones
- Modificar el estado del servidor

**Recomendaciones:**
- Implementar autenticación JWT o API keys
- Añadir roles y permisos para operaciones sensibles (entrenamiento, guardado)
- Limitar acceso a endpoints de modificación

## 2. Lógica de Negocio Principal

### 2.1 Arquitectura del Sistema
El sistema implementa un modelo transformer mini-GPT con dos interfaces:
- **Interfaz Gradio** (`app.py`): Interfaz web para entrenamiento y generación
- **API REST** (`main.py`): FastAPI con endpoints para integración programática

### 2.2 Componentes Principales

#### Motor de Transformers (`transformer.py`)
- Implementación completa de autodiferenciación personalizada (`AutogradTensor`)
- Arquitectura transformer con:
  - Capas de embedding (tokens y posiciones)
  - Múltiples bloques transformer (atención multi-cabeza + feed-forward)
  - Normalización de capa (Layer Normalization)
  - Capa de salida lineal
- Optimizador Adam con:
  - Precisión mixta (fp16/fp32)
  - Acumulación de gradientes
  - Programación de learning rate con warmup y cosine decay
  - Escalado dinámico de pérdida
  - Checkpointing automático

#### Sistema de Tokenización (`tokenizers/`)
Múltiples algoritmos de tokenización:
- **BPE** (Byte Pair Encoding): Implementación principal
- **Unigram**: Tokenización basada en probabilidad
- **WordPiece**: Usado en BERT
- **SentencePiece**: Tokenización a nivel de oración
- **DynamicVocabulary**: Vocabulario dinámico con crecimiento automático

#### Motor de Chat (`chat_engine.py`)
Sistema de chat con múltiples fuentes de respuesta:
1. **Intent matching**: Saludos, despedidas, agradecimientos
2. **NeuralCode**: Base de conocimiento específica para programación
3. **KnowledgeBase**: Base de conocimiento general desde corpus
4. **Generación**: Modelo transformer entrenado
5. **Fallback**: Respuestas por defecto

#### Lóbulo Frontal (`frontal_lobe.py`)
Sistema de decisión que selecciona la mejor respuesta entre candidatas basado en:
- Puntuación de confianza
- Peso por fuente de respuesta
- Historial de respuestas recientes (evita repetición)

## 3. Patrones de Diseño Implementados

### 3.1 Factory Pattern
`TokenizerFactory` crea diferentes tipos de tokenizadores basado en configuración.

### 3.2 Strategy Pattern
Múltiples algoritmos de tokenización intercambiables (BPE, Unigram, WordPiece, etc.).

### 3.3 Repository Pattern
`KnowledgeBase` actúa como repositorio de conocimiento con métodos para cargar, indexar y consultar.

### 3.4 Chain of Responsibility
`FrontalLobe.decide()` evalúa múltiples candidatos de respuesta en cadena.

### 3.5 Observer Pattern
El sistema de caché de respuestas observa cambios en el modelo para invalidarse.

### 3.6 Builder Pattern
Construcción progresiva del transformer con parámetros configurables.

## 4. Manejo de Errores

### 4.1 Nivel de API (FastAPI)
```python
# Excepciones HTTP específicas
raise HTTPException(status_code=400, detail="Model not trained")
raise HTTPException(status_code=400, detail="No model to save")
raise HTTPException(status_code=400, detail="No saved model found")
```

### 4.2 Nivel de Transformación
```python
# Manejo de errores en entrenamiento
try:
    transformer.train(line)
except Exception as e:
    return f"Error: {e}"
```

### 4.3 Nivel de Chat
```python
# Fallback en múltiples niveles
- Intent matching falla → continúa con otras fuentes
- Generación falla → usa respuesta de conocimiento
- Todas las fuentes fallan → usa fallback por defecto
```

### 4.4 Patrones de Error
- **Degradación elegante**: Si Rust no está disponible, usa Python
- **Caché como fallback**: Si el modelo falla, usa respuestas cacheadas
- **Validación de entrada**: Pydantic para validación de requests

## 5. Configuración del Servidor

### 5.1 FastAPI (`main.py`)
```python
app = FastAPI(title="Rubidium API - Transformer Generator", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todos los orígenes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servidor Uvicorn
uvicorn.run(app, host="0.0.0.0", port=port)
```

### 5.2 Gradio (`app.py`)
```python
demo.launch(server_name="0.0.0.0", server_port=7860)
```

### 5.3 Docker
```dockerfile
FROM python:3.11-slim
# Instala Rust para extensión nativa
# Construye rubidium-core si existe
EXPOSE 8000
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
```

## 6. Variables de Entorno Requeridas

| Variable | Descripción | Valor por Defecto |
|----------|-------------|-------------------|
| `PORT` | Puerto del servidor FastAPI | `8000` |

**Nota:** El proyecto actualmente solo usa la variable `PORT`. Se recomienda agregar:
- `HOST`: Dirección del servidor (default: `0.0.0.0`)
- `MODEL_PATH`: Ruta del modelo (default: `model.pkl`)
- `LOG_LEVEL`: Nivel de logging
- `CORS_ORIGINS`: Orígenes permitidos (actualmente `*`)
- `MAX_CACHE_SIZE`: Tamaño máximo de caché (default: `200`)

## 7. Integraciones Externas

### 7.1 Bases de Datos
**No implementa bases de datos externas.** Utiliza:
- Archivos pickle para persistencia del modelo (`model.pkl`)
- Sistema de archivos para corpus (`resources/*.txt`)
- Caché en memoria para respuestas

### 7.2 Servicios Cloud
**No tiene integraciones con servicios cloud.** El proyecto es autocontenido.

### 7.3 Dependencias Externas
```txt
# requirements.txt
numpy          # Computación numérica
fastapi        # Framework web
uvicorn        # Servidor ASGI
pydantic       # Validación de datos
maturin        # Build de extensiones Rust
```

### 7.4 Extensión Rust (`rubidium-core`)
- Aceleración de inferencia con Rust via PyO3
- Carga de modelos pickle desde Rust
- Implementación nativa de attention y feed-forward

## 8. Análisis de Seguridad

### 8.1 Vulnerabilidades Críticas

#### 1. Deserialización Insegura (Pickle)
```python
# transformer.py:796-798
def load(self, path: str):
    with open(path, "rb") as f:
        state = pickle.load(f)  # PELIGRO: Ejecuta código arbitrario
```
**Riesgo:** Atacante puede crear modelo malicioso que ejecute código al cargarse.

#### 2. CORS Totalmente Abierto
```python
# main.py:21-27
allow_origins=["*"],  # Permite cualquier origen
```
**Riesgo:** Ataques CSRF desde cualquier sitio web.

#### 3. Sin Autenticación
Todos los endpoints son públicos, incluyendo entrenamiento y modificación del modelo.

#### 4. Exposición de Información
```python
# main.py:120-121
return {"service": "Rubidium API", "version": "2.0", "engine": engine, "status": "running"}
```
Expone información del sistema.

### 8.2 Vulnerabilidades Medias

#### 1. Sin Rate Limiting
No hay límite de peticiones, vulnerable a DoS.

#### 2. Sin Validación de Corpus
El endpoint `/train` acepta cualquier texto sin validación.

#### 3. Caché sin Límite de Memoria
```python
_CACHE_MAX = 200  # Límite suave, pero sin monitoreo de memoria
```

### 8.3 Vulnerabilidades Bajas

#### 1. Logs Sensibles
```python
print(f"Model loaded from {MODEL_PATH} (vocab={transformer.vocab_size})")
```

#### 2. Errores Detallados
Excepciones pueden contener información sensible del sistema.

### 8.4 Recomendaciones de Seguridad

1. **Urgente:** Reemplazar pickle con formato seguro (JSON + pesos separados)
2. **Urgente:** Implementar autenticación
3. **Alto:** Restringir CORS a orígenes específicos
4. **Alto:** Añadir rate limiting
5. **Medio:** Validar y sanitizar corpus de entrenamiento
6. **Medio:** Implementar logging seguro sin datos sensibles
7. **Bajo:** Añadir headers de seguridad (HSTS, CSP, etc.)

## 9. Documentación de Endpoints

### 9.1 GET `/`
**Descripción:** Endpoint raíz con información del servicio.

**Respuesta:**
```json
{
  "service": "Rubidium API",
  "version": "2.0",
  "engine": "rust|numpy",
  "status": "running"
}
```

**Autenticación:** No requerida

---

### 9.2 GET `/state`
**Descripción:** Estado actual del modelo.

**Respuesta:**
```json
{
  "is_trained": true,
  "vocab_size": 256,
  "model_size": "1.2M params",
  "engine": "rust|numpy",
  "cache_size": 42
}
```

**Autenticación:** No requerida

---

### 9.3 POST `/train`
**Descripción:** Entrena el modelo con un corpus.

**Request Body:**
```json
{
  "corpus": "Texto de entrenamiento...",
  "block_size": 128,
  "d_model": 128,
  "n_head": 4,
  "n_layer": 4,
  "d_ff": 512,
  "max_steps": 1000,
  "learning_rate": 0.0003,
  "use_resources": true
}
```

**Parámetros:**
- `corpus`: Texto para entrenamiento (opcional si `use_resources=true`)
- `block_size`: Tamaño del bloque de contexto (default: 128)
- `d_model`: Dimensión del modelo (default: 128)
- `n_head`: Número de cabezas de atención (default: 4)
- `n_layer`: Número de capas transformer (default: 4)
- `d_ff`: Dimensión de feed-forward (default: 512)
- `max_steps`: Pasos máximos de entrenamiento (default: 1000)
- `learning_rate`: Tasa de aprendizaje (default: 0.0003)
- `use_resources`: Usar corpus de `resources/` (default: true)

**Respuesta:**
```json
{
  "status": "success|error",
  "message": "Model trained and saved"
}
```

**Autenticación:** No requerida (debería requerirse)

---

### 9.4 POST `/generate`
**Descripción:** Genera texto usando el modelo entrenado.

**Request Body:**
```json
{
  "seed": "Hola mundo",
  "max_chars": 200,
  "temperature": 0.8,
  "top_k": 20
}
```

**Parámetros:**
- `seed`: Texto semilla para generación (default: "")
- `max_chars`: Caracteres máximos a generar (default: 200)
- `temperature`: Temperatura para muestreo (default: 0.8)
- `top_k`: Top-K sampling (default: 20)

**Respuesta:**
```json
{
  "text": "Texto generado por el modelo..."
}
```

**Autenticación:** No requerida

**Errores:**
- `400`: Modelo no entrenado

---

### 9.5 POST `/save`
**Descripción:** Guarda el modelo actual en disco.

**Respuesta:**
```json
{
  "status": "saved",
  "path": "model.pkl"
}
```

**Autenticación:** No requerida (debería requerirse)

**Errores:**
- `400`: No hay modelo para guardar

---

### 9.6 POST `/load`
**Descripción:** Carga un modelo desde disco.

**Respuesta:**
```json
{
  "status": "loaded",
  "vocab_size": 256
}
```

**Autenticación:** No requerida (debería requerirse)

**Errores:**
- `400`: No hay modelo guardado

---

### 9.7 POST `/clear-cache`
**Descripción:** Limpia la caché de respuestas.

**Respuesta:**
```json
{
  "status": "cleared",
  "entries_removed": 42
}
```

**Autenticación:** No requerida

## 10. Flujo de Datos

### 10.1 Flujo de Entrenamiento
```
1. POST /train con corpus
2. Validación de parámetros (Pydantic)
3. Creación de NumpyTransformer con hiperparámetros
4. Tokenización del corpus
5. Entrenamiento con Adam optimizer
   - Forward pass → Loss → Backward pass → Update weights
6. Guardado del modelo en model.pkl
7. Respuesta de éxito/error
```

### 10.2 Flujo de Generación
```
1. POST /generate con seed y parámetros
2. Verificación de modelo entrenado
3. Verificación de caché (MD5 de parámetros)
4. Si no está en caché:
   - Inferencia con Rust (si disponible) o Python
   - Muestreo con temperature y top-k
5. Almacenamiento en caché (LRU simple)
6. Respuesta con texto generado
```

### 10.3 Flujo de Chat (App Gradio)
```
1. Usuario envía mensaje
2. ChatEngine procesa:
   a. Intent matching (saludos, etc.)
   b. NeuralCode (consulta programación)
   c. KnowledgeBase (consulta general)
   d. Generación transformer (si entrenado)
3. FrontalLobe selecciona mejor respuesta
4. Respuesta al usuario
```

## 11. Métricas y Monitoreo

### 11.1 Métricas Disponibles
- Estado del modelo (`/state`)
- Tamaño de caché
- Número de parámetros del modelo
- Motor activo (Rust/NumPy)

### 11.2 Métricas Recomendadas
- Tiempo de respuesta por endpoint
- Uso de memoria
- Tasa de aciertos de caché
- Distribución de fuentes de respuesta (Intent/Code/Knowledge/Generation)
- Errores por tipo

## 12. Escalabilidad

### 12.1 Limitaciones Actuales
- **Memoria:** Modelo completo en memoria
- **CPU:** Entrenamiento en CPU (sin GPU support)
- **Single-thread:** Sin soporte para concurrencia
- **Sin persistencia:** Datos solo en disco local

### 12.2 Recomendaciones de Escalabilidad
1. Implementar soporte GPU para entrenamiento
2. Añadir balanceo de carga para múltiples instancias
3. Implementar caché distribuido (Redis)
4. Añadir soporte para bases de datos vectoriales
5. Implementar streaming para generación larga

## 13. Conclusión

Rubidium API es un proyecto educativo que implementa un transformer mini-GPT completo desde cero. Aunque funcional, tiene carencias significativas de seguridad que deben abordarse antes de producción:

**Fortalezas:**
- Implementación completa de autodiferenciación personalizada
- Múltiples algoritmos de tokenización
- Sistema de chat con múltiples fuentes de conocimiento
- Aceleración nativa con Rust
- Código bien estructurado y modular

**Debilidades:**
- Sin autenticación ni autorización
- Deserialización insegura (pickle)
- CORS abierto a todos los orígenes
- Sin rate limiting
- Sin soporte para GPU
- Sin persistencia distribuida

**Prioridad de Mejoras:**
1. Seguridad: Reemplazar pickle, implementar auth
2. Funcionalidad: Soporte GPU, streaming
3. Escalabilidad: Caché distribuido, balanceo de carga
4. Monitoreo: Métricas detalladas, logging estructurado