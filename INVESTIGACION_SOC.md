# Investigación: Synthetic Online Conversations (SOC) para Generación de Corpus

## 1. ¿Qué es SOC (Synthetic Online Conversations)?

### Definición
**Synthetic Online Conversations (SOC)** es una metodología para generar datos de conversación sintéticos de alta calidad mediante el uso de modelos de lenguaje grandes (LLMs) que simulan interacciones humanas realistas en plataformas online (foros, chats, redes sociales, etc.).

### Origen y Papers Clave

| Paper / Proyecto | Año | Organización | Contribución |
|------------------|-----|--------------|--------------|
| **Self-Instruct** | 2022 | UW / Allen Institute | Generación automática de instrucciones y respuestas |
| **Evol-Instruct** | 2023 | WizardLM Team | Evolución iterativa de instrucciones para mayor complejidad |
| **UltraChat** | 2023 | Tsinghua / Zhipu AI | 1.5M conversaciones sintéticas via ChatGPT API |
| **ShareGPT** | 2023 | Comunidad | Dataset de conversaciones reales de usuarios |
| **OASST** | 2023 | LAION | OpenAssistant - conversaciones humanas anotadas |
| **Dolly** | 2023 | Databricks | 15K instrucciones escritas por humanos |
| **Alpaca** | 2023 | Stanford | 52K instrucciones generadas por text-davinci-003 |
| **Synthetic Data Generation with LLMs** | 2023 | Microsoft Research | Metodología sistemática para datos sintéticos |
| **Nemotron 3 Ultra** | 2024 | NVIDIA | Generación masiva de datos sintéticos de alta calidad |

### Diferencia con otros métodos

| Método | Fuente | Calidad | Escalabilidad | Coste |
|--------|--------|---------|---------------|-------|
| **Human Annotation** | Humanos | ★★★★★ | Baja | Muy Alto |
| **Web Scraping** | Internet | ★★★☆☆ | Media | Bajo |
| **Self-Instruct** | LLM → LLM | ★★★★☆ | Alta | Medio |
| **Evol-Instruct** | LLM → LLM (iterativo) | ★★★★★ | Alta | Medio |
| **SOC** | LLM simula usuarios | ★★★★★ | Muy Alta | Medio |
| **Distillation** | Teacher → Student | ★★★★☆ | Muy Alta | Bajo |

---

## 2. Cómo funciona SOC - Metodología

### Pipeline Básico SOC

```
┌─────────────────────────────────────────────────────────────┐
│ 1. DEFINIR PERSONAS / ROLES                                 │
│    - Usuario: estudiante, profesional, curioso, experto    │
│    - Bot: asistente útil, experto técnico, tutor, creativo │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. GENERAR SEMILLAS (SEEDS)                                 │
│    - Temas: 50-100 categorías                              │
│    - Intenciones: pregunta, debate, creativo, soporte      │
│    - Contexto: longitud, tono, formalidad                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. SIMULAR CONVERSACIÓN (Multi-turn)                        │
│    Usuario → Bot → Usuario → Bot → ... (3-10 turnos)       │
│    - Cada turno condicionado al historial                  │
│    - Inyección de ruido realista (typos, hesitación)       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. FILTRADO Y VALIDACIÓN                                    │
│    - Coherencia temática                                   │
│    - Calidad lingüística                                   │
│    - Seguridad (no tóxico, no PII)                         │
│    - Diversidad (embedding distance)                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. FORMATO DE SALIDA                                        │
│    - U:/B: para entrenamiento                              │
│    - JSONL para fine-tuning                                │
│    - ChatML para chat models                               │
└─────────────────────────────────────────────────────────────┘
```

### Componentes Clave

#### 2.1 Personas (User Simulators)
```python
USER_PERSONAS = [
    "estudiante_universitario",      # Preguntas académicas, informales
    "profesional_tecnologico",       # Preguntas técnicas, precisas
    "curioso_general",               # Preguntas variadas, exploratorias
    "usuario_frustrado",             # Quejas, problemas técnicos
    "creativo_escritor",             # Prompts creativos, storytelling
    "docente_tutor",                 # Preguntas pedagógicas
    "usuario_casual",                # Conversación cotidiana
    "experto_dominio",               # Preguntas avanzadas, técnicas
]
```

#### 2.2 Intenciones de Conversación
```python
INTENTS = {
    "informational": "¿Qué es X? ¿Cómo funciona Y?",
    "procedural": "¿Cómo hago X? Pasos para Y",
    "troubleshooting": "Error X, no funciona Y",
    "creative": "Escribe un poema sobre X",
    "analytical": "Compara X vs Y, analiza Z",
    "opinion": "Qué opinas de X, mejor Y o Z",
    "roleplay": "Actúa como X, responde como Y",
    "meta": "¿Cómo funcionas? ¿Cuáles son tus límites?",
}
```

#### 2.3 Contexto y Memoria
- **Historial**: Cada turno incluye 3-5 turnos previos
- **Contexto largo**: Simular hilos de 10+ mensajes
- **Referencias cruzadas**: "Como mencionaste antes..."
- **Corrección**: Usuario corrige al bot, bot se disculpa y corrige

---

## 3. Implementación Práctica para Rubidium (Español)

### 3.1 Arquitectura Recomendada

```
┌────────────────────────────────────────────────────────────┐
│                    SOC GENERATOR PIPELINE                   │
├────────────────────────────────────────────────────────────┤
│  1. CONFIGURACIÓN                                           │
│     - Modelo generador: Phi-3-mini / Llama-3-8B / GPT-4o  │
│     - Temas objetivo: 50+ categorías español               │
│     - Target: 500K-1M pares U:/B:                          │
├────────────────────────────────────────────────────────────┤
│  2. GENERACIÓN PARALELA                                     │
│     - Múltiples workers (Kaggle: 4x GPU / CPU workers)     │
│     - Batch de prompts → API / modelo local                │
│     - Checkpoint cada 10K pares                            │
├────────────────────────────────────────────────────────────┤
│  3. POST-PROCESADO                                          │
│     - Deduplicación (exacta + fuzzy >90%)                  │
│     - Filtro calidad (perplexity, longitud, coherencia)    │
│     - Split train/val/test (90/5/5)                        │
├────────────────────────────────────────────────────────────┤
│  4. FORMATOS DE SALIDA                                      │
│     - U:/B: para training CUDA/CPU                         │
│     - ShareGPT JSON para fine-tuning chat                  │
│     - Alpaca JSON para instruction tuning                  │
└────────────────────────────────────────────────────────────┘
```

### 3.2 Código: Generador SOC Base (Python)

```python
#!/usr/bin/env python3
"""
SOC Generator para Rubidium - Generación masiva de conversaciones en español
Usa Phi-3-mini-4k-instruct local o API compatible
"""

import json
import random
import asyncio
import aiohttp
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
from tqdm import tqdm
import hashlib

# ============================================================
# CONFIGURACIÓN
# ============================================================

@dataclass
class SOCConfig:
    generator_model: str = "phi3:mini"  # ollama model name
    api_base: str = "http://localhost:11434"  # ollama default
    target_pairs: int = 500000
    batch_size: int = 32
    max_turns: int = 8
    min_turns: int = 3
    temperature: float = 0.8
    top_p: float = 0.9
    output_dir: str = "soc_corpus"
    checkpoint_every: int = 10000

# ============================================================
# PERSONAS Y TEMPLATES
# ============================================================

USER_PERSONAS = {
    "estudiante": {
        "style": "informal, curioso, a veces impreciso",
        "topics": ["tareas", "examenes", "proyectos", "becas", "universidad"],
        "vocab": ["profe", "parcial", "tp", "grupo", "bibliografia"]
    },
    "profesional_tech": {
        "style": "técnico, preciso, usa terminología",
        "topics": ["arquitectura", "deploy", "debugging", "performance", "ci/cd"],
        "vocab": ["latencia", "throughput", "bottleneck", "refactor", "legacy"]
    },
    "curioso_general": {
        "style": "exploratorio, pregunta seguidas, entusiasta",
        "topics": ["ciencia", "historia", "cultura", "tecnologia", "vida"],
        "vocab": ["interesante", "nunca supe", "cómo es que", "qué tal si"]
    },
    "usuario_frustrado": {
        "style": "directo, problema urgente, poco contexto",
        "topics": ["error", "no funciona", "fallo", "bug", "urgente"],
        "vocab": ["nada funciona", "ya probé", "error 500", "timeout", "crash"]
    },
    "creativo": {
        "style": "imaginativo, pide formatos específicos",
        "topics": ["historias", "poemas", "guiones", "ideas", "worldbuilding"],
        "vocab": ["escribe", "inventa", "crea", "imagina", "personaje"]
    },
}

TOPICS_ES = [
    "programacion_python", "programacion_javascript", "web_dev", "machine_learning",
    "devops_cloud", "bases_datos", "seguridad_informatica", "movil_android",
    "ciencia_datos", "inteligencia_artificial", "matematicas", "fisica",
    "biologia", "quimica", "astronomia", "neurociencia",
    "psicologia", "filosofia", "historia", "economia",
    "finanzas_personales", "emprendimiento", "marketing", "diseño_ux",
    "salud_mental", "ejercicio", "nutricion", "meditacion",
    "relaciones", "comunicacion", "liderazgo", "productividad",
    "cocina", "viajes", "fotografia", "musica", "literatura",
    "videojuegos", "cine_series", "anime_manga", "tecnologia_consumidor",
    "automoviles", "bricolaje", "jardineria", "mascotas",
    "educacion", "idiomas", "certificaciones", "carrera_profesional"
]

# ============================================================
# PROMPTS PARA GENERACIÓN
# ============================================================

SYSTEM_PROMPT = """Eres un generador de conversaciones realistas en español.
Genera diálogos naturales entre un USUARIO y un ASISTENTE.

REGLAS:
1. El USUARIO tiene una persona específica (estudiante, profesional, etc.)
2. El ASISTENTE es útil, honesto, y admite cuando no sabe
3. Conversación de {min_turns}-{max_turns} turnos
4. Incluye hesitación natural, correcciones, referencias previas
5. Temas: {topic}
6. Español neutro/latinoamericano, natural
7. Longitud variable: respuestas cortas (1-2 frases) y largas (párrafos)

FORMATO DE SALIDA (JSON):
{{
  "persona": "nombre_persona",
  "topic": "tema",
  "turns": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}},
    ...
  ]
}}"""

USER_PROMPT_TEMPLATE = """Genera una conversación para:
- Persona: {persona} ({style})
- Tema: {topic}
- Vocabulario típico: {vocab}
- Turnos: {min_turns}-{max_turns}

La conversación debe sentirse REAL, como si la hubieras extraído de un foro/chat real."""

# ============================================================
# GENERADOR PRINCIPAL
# ============================================================

class SOCGenerator:
    def __init__(self, config: SOCConfig):
        self.config = config
        self.session = None
        self.generated = 0
        self.seen_hashes = set()
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
    
    def _hash_conversation(self, turns: List[Dict]) -> str:
        """Hash para deduplicación"""
        text = "".join(t["content"] for t in turns)
        return hashlib.md5(text.encode()).hexdigest()[:16]
    
    def _is_quality(self, turns: List[Dict]) -> bool:
        """Filtros de calidad básicos"""
        if len(turns) < self.config.min_turns * 2:
            return False
        # Longitud mínima por turno
        for t in turns:
            if len(t["content"]) < 10:
                return False
        # No repetitivo
        user_texts = [t["content"] for t in turns if t["role"] == "user"]
        if len(set(user_texts)) != len(user_texts):
            return False
        return True
    
    async def generate_batch(self, batch_size: int) -> List[Dict]:
        """Genera un batch de conversaciones"""
        tasks = []
        for _ in range(batch_size):
            persona_key = random.choice(list(USER_PERSONAS.keys()))
            persona = USER_PERSONAS[persona_key]
            topic = random.choice(TOPICS_ES)
            
            prompt = USER_PROMPT_TEMPLATE.format(
                persona=persona_key,
                style=persona["style"],
                topic=topic,
                vocab=", ".join(persona["vocab"]),
                min_turns=self.config.min_turns,
                max_turns=self.config.max_turns
            )
            
            tasks.append(self._generate_one(persona_key, topic, prompt))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid = []
        for r in results:
            if isinstance(r, dict) and self._is_quality(r["turns"]):
                h = self._hash_conversation(r["turns"])
                if h not in self.seen_hashes:
                    self.seen_hashes.add(h)
                    valid.append(r)
        return valid
    
    async def _generate_one(self, persona: str, topic: str, prompt: str) -> Dict:
        """Genera una conversación vía API"""
        payload = {
            "model": self.config.generator_model,
            "prompt": f"{SYSTEM_PROMPT.format(min_turns=self.config.min_turns, max_turns=self.config.max_turns, topic=topic)}\n\n{prompt}",
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "format": "json",
            "stream": False
        }
        
        async with self.session.post(f"{self.config.api_base}/api/generate", json=payload) as resp:
            data = await resp.json()
            response_text = data.get("response", "{}")
            
        try:
            conv = json.loads(response_text)
            conv["persona"] = persona
            conv["topic"] = topic
            return conv
        except json.JSONDecodeError:
            return {"error": "parse_failed", "raw": response_text[:200]}
    
    def save_checkpoint(self, conversations: List[Dict], path: str):
        """Guarda checkpoint en múltiples formatos"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # JSONL
        with open(f"{path}.jsonl", "w", encoding="utf-8") as f:
            for c in conversations:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        
        # U:/B: format
        with open(f"{path}.txt", "w", encoding="utf-8") as f:
            for c in conversations:
                for turn in c["turns"]:
                    role = "U" if turn["role"] == "user" else "B"
                    f.write(f"{role}: {turn['content']}\n")
                f.write("\n")
        
        # ShareGPT format (para fine-tuning chat)
        sharegpt = []
        for c in conversations:
            conv = []
            for turn in c["turns"]:
                conv.append({
                    "from": "human" if turn["role"] == "user" else "gpt",
                    "value": turn["content"]
                })
            sharegpt.append({"conversations": conv})
        
        with open(f"{path}_sharegpt.json", "w", encoding="utf-8") as f:
            json.dump(sharegpt, f, ensure_ascii=False, indent=2)

# ============================================================
# MAIN
# ============================================================

async def main():
    config = SOCConfig(
        target_pairs=500000,
        batch_size=16,
        checkpoint_every=10000
    )
    
    all_conversations = []
    
    async with SOCGenerator(config) as gen:
        pbar = tqdm(total=config.target_pairs, desc="Generando SOC")
        
        while gen.generated < config.target_pairs:
            batch = await gen.generate_batch(config.batch_size)
            
            if batch:
                all_conversations.extend(batch)
                gen.generated += len(batch)
                pbar.update(len(batch))
                
                if gen.generated % config.checkpoint_every == 0:
                    gen.save_checkpoint(
                        all_conversations[-config.checkpoint_every:],
                        f"{config.output_dir}/checkpoint_{gen.generated}"
                    )
                    print(f"\nCheckpoint: {gen.generated} pares")
        
        # Guardado final
        gen.save_checkpoint(all_conversations, f"{config.output_dir}/soc_final")
        print(f"\n✅ Completado: {gen.generated} pares generados")

if __name__ == "__main__":
    asyncio.run(main())
```

### 3.3 Versión para Kaggle (Sin GPU local)

```python
# soc_kaggle.py - Usa API remota (OpenAI, Together, etc.) o modelo local via transformers
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class LocalSOCGenerator:
    def __init__(self, model_id="microsoft/Phi-3-mini-4k-instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=2048,
            temperature=0.8,
            top_p=0.9,
            do_sample=True
        )
    
    def generate_batch(self, prompts: List[str]) -> List[str]:
        outputs = self.pipe(prompts, batch_size=len(prompts))
        return [o[0]["generated_text"] for o in outputs]
```

---

## 4. Herramientas y Librerías Disponibles

| Herramienta | Tipo | Uso para SOC |
|-------------|------|--------------|
| **Ollama** | Local LLM server | Ejecutar Phi-3, Llama-3 localmente |
| **vLLM** | High-throughput inference | Generación masiva paralela en GPU |
| **Text Generation Inference (TGI)** | HuggingFace | Serving optimizado para producción |
| **LangChain** | Framework | Orquestación de prompts y chains |
| **DSPy** | Programming LM | Optimización de prompts programática |
| **Instructor** | Structured output | Garantizar JSON válido |
| **Guardrails AI** | Validación | Filtros de calidad y seguridad |
| **Argilla** | Anotación | Revisión humana de samples |
| **Prodigy** | Anotación activa | Active learning para mejorar datos |

### Setup recomendado para Kaggle

```bash
# Opción 1: vLLM (más rápido, requiere GPU)
pip install vllm
python -m vllm.entrypoints.openai.api_server --model microsoft/Phi-3-mini-4k-instruct --port 8000

# Opción 2: Ollama (más simple)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull phi3:mini
ollama serve

# Opción 3: Transformers local (si no hay vLLM/Ollama)
pip install transformers accelerate bitsandbytes
```

---

## 5. Métricas de Calidad para SOC

### 5.1 Métricas Automáticas

```python
def evaluate_soc_quality(conversations: List[Dict]) -> Dict:
    metrics = {}
    
    # Diversidad léxica
    all_text = " ".join(t["content"] for c in conversations for t in c["turns"])
    tokens = all_text.split()
    metrics["type_token_ratio"] = len(set(tokens)) / len(tokens)
    
    # Longitud promedio
    metrics["avg_turn_length"] = np.mean([len(t["content"]) for c in conversations for t in c["turns"]])
    metrics["avg_turns_per_conv"] = np.mean([len(c["turns"]) for c in conversations])
    
    # Repetición n-gram
    metrics["ngram_repetition"] = compute_ngram_repetition(conversations, n=3)
    
    # Perplexity (con modelo pequeño)
    metrics["perplexity"] = compute_perplexity(conversations)
    
    # Coherencia temática (embedding similarity entre turnos)
    metrics["topic_coherence"] = compute_topic_coherence(conversations)
    
    # Balance de roles
    user_turns = sum(1 for c in conversations for t in c["turns"] if t["role"] == "user")
    asst_turns = sum(1 for c in conversations for t in c["turns"] if t["role"] == "assistant")
    metrics["role_balance"] = user_turns / (user_turns + asst_turns)
    
    return metrics
```

### 5.2 Umbrales Recomendados

| Métrica | Umbral Mínimo | Objetivo |
|---------|---------------|----------|
| Type-Token Ratio | > 0.15 | > 0.25 |
| Avg Turn Length | > 30 chars | 50-200 chars |
| N-gram Repetition (3-gram) | < 0.1 | < 0.05 |
| Perplexity (GPT-2 small) | < 50 | < 20 |
| Topic Coherence | > 0.6 | > 0.75 |
| Role Balance | 0.4 - 0.6 | ~0.5 |

---

## 6. Pipeline Completo para Rubidium

### 6.1 Flujo de 4 Fases

```
FASE 1: GENERACIÓN MASIVA (1-2 semanas)
├── Configurar vLLM en Kaggle/RunPod/Lambda
├── Generar 500K-1M pares con SOC
├── Checkpoints cada 50K
└── Output: soc_corpus_final.txt + .jsonl + _sharegpt.json

FASE 2: CURACIÓN Y LIMPIEZA (2-3 días)
├── Deduplicación exacta + fuzzy (MinHash LSH)
├── Filtro calidad (perplexity, longitud, coherencia)
├── Eliminación PII (regex + NER)
├── Split train/val/test (90/5/5)
└── Output: rubidium_train.txt, rubidium_val.txt, rubidium_test.txt

FASE 3: ENTRENAMIENTO TOKENIZER (1 día)
├── Entrenar BPE+Unigram en corpus train (V=32000)
├── Guardar tokenizer.json
├── Validar compresión (>2.5x)
└── Integrar en main.cpp / train_pytorch.py

FASE 4: ENTRENAMIENTO MODELO (1-2 semanas)
├── Pre-entrenamiento 250M en CPU/CUDA (200K steps)
├── Validación cada 5K steps
├── LoRA fine-tuning en corpus val (50K steps)
└── Modelo final: model_250M_final.bin + lora_final.bin
```

### 6.2 Estimación de Costes

| Recurso | Tiempo | Coste estimado |
|---------|--------|----------------|
| vLLM en RunPod (A100 80GB) | 48h | ~$60 |
| Kaggle P100 (gratis 30h/sem) | 3-4 semanas | $0 |
| Lambda Cloud (A10G) | 24h | ~$25 |
| **Total generación 500K pares** | | **$0-60** |

---

## 7. Comparación con Métodos Actuales

| Aspecto | Templates Actuales | SOC (Propuesto) |
|---------|-------------------|-----------------|
| **Diversidad** | Baja (150 pares, 3 temas) | Alta (500K+, 50+ temas) |
| **Naturalidad** | Robótica, repetitiva | Humana, con hesitaciones |
| **Multi-turn** | No (1 turno) | Sí (3-10 turnos) |
| **Coherencia** | N/A | Alta (contexto mantenido) |
| **Escalabilidad** | Manual | Automática paralela |
| **Calidad para 250M params** | Insuficiente | Suficiente |
| **Tiempo generación** | Segundos | Horas (paralelizable) |

---

## 8. Próximos Pasos Inmediatos

1. **Probar vLLM + Phi-3-mini en Kaggle** (30 min setup)
2. **Generar 10K pares de prueba** y evaluar métricas
3. **Ajustar prompts/personas** según calidad
4. **Escalar a 500K** en sesiones de 30h
5. **Integrar tokenizer BPE** entrenado en corpus SOC
6. **Lanzar pre-entrenamiento 250M** con corpus SOC

---

## 9. Referencias y Lecturas

1. **Self-Instruct**: Wang et al. (2022) - https://arxiv.org/abs/2212.10560
2. **Evol-Instruct**: Xu et al. (2023) - https://arxiv.org/abs/2304.12244
3. **UltraChat**: Ding et al. (2023) - https://arxiv.org/abs/2305.14233
4. **OASST**: Köpf et al. (2023) - https://arxiv.org/abs/2304.07327
5. **Microsoft Synthetic Data**: https://www.microsoft.com/en-us/research/blog/synthetic-data-generation-with-llms/
6. **Nemotron 3 Ultra**: NVIDIA (2024) - https://github.com/NVIDIA/NeMo-Skills
7. **Phi-3 Technical Report**: Microsoft (2024) - https://arxiv.org/abs/2404.14219
8. **Data Quality for LLM Training**: https://arxiv.org/abs/2312.16862

---

*Documento generado: 2026-08-04*
*Para: Rubidium Project - Generación de corpus español via SOC*