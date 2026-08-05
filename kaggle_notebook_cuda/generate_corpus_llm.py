#!/usr/bin/env python3
"""
RUBIDIUM - Generación de Corpus con LLM Local en Kaggle
Genera 1K pares U:/B: de alta calidad en español
GPU: Tesla P100 16GB
Modelo: Phi-3-mini-4k-instruct (3.8B params, ~8GB VRAM)
"""
import torch
import json
import os
import time
from pathlib import Path
from tqdm import tqdm

print("=" * 60)
print("RUBIDIUM - Generación Corpus con LLM Local")
print("=" * 60)

# 1. Verificar GPU
print(f"\nGPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
print(f"CUDA: {torch.version.cuda}")

# 2. Cargar modelo (Phi-3-mini - 3.8B, cabe en P100)
model_id = "microsoft/Phi-3-mini-4k-instruct"
print(f"\nCargando modelo: {model_id}")

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print(f"Modelo cargado. Memoria: {torch.cuda.memory_allocated()/1e9:.1f} GB")

# 3. Temas para diversidad
TEMAS = [
    # Tecnología
    ("programación", "Python, JavaScript, algoritmos, estructuras de datos"),
    ("inteligencia artificial", "machine learning, deep learning, LLMs, transformers"),
    ("ciberseguridad", "hacking ético, vulnerabilidades, encriptación, firewalls"),
    ("cloud computing", "AWS, Azure, Docker, Kubernetes, serverless"),
    ("bases de datos", "SQL, NoSQL, PostgreSQL, MongoDB, optimización"),
    
    # Ciencia
    ("física", "mecánica cuántica, relatividad, termodinámica, partículas"),
    ("biología", "genética, evolución, células, ecosistemas, ADN"),
    ("química", "reacciones, enlaces, tabla periódica, orgánica, inorgánica"),
    ("astronomía", "planetas, estrellas, galaxias, agujeros negros, cosmología"),
    ("neurociencia", "cerebro, neuronas, memoria, aprendizaje, consciencia"),
    
    # Vida Cotidiana
    ("cocina", "recetas, técnicas, ingredientes, nutrición, dietas"),
    ("ejercicio", "fuerza, cardio, flexibilidad, rutinas, recuperación"),
    ("finanzas", "ahorro, inversión, presupuesto, criptomonedas, impuestos"),
    ("psicología", "ansiedad, depresión, terapia, emociones, mindfulness"),
    ("productividad", "gestión tiempo, hábitos, enfoque, procrastinación, metas"),
    
    # Educación
    ("aprendizaje", "técnicas estudio, memorización, lectura rápida, mapas mentales"),
    ("matemáticas", "álgebra, cálculo, estadística, geometría, lógica"),
    ("idiomas", "inglés, francés, alemán, métodos, inmersión, práctica"),
    ("historia", "antigua, medieval, moderna, guerras, civilizaciones"),
    ("filosofía", "ética, existencialismo, lógica, metafísica, epistemología"),
    
    # Trabajo/Profesional
    ("liderazgo", "gestión equipos, comunicación, toma decisiones, mentoría"),
    ("emprendimiento", "startups, modelo negocio, funding, marketing, ventas"),
    ("marketing", "digital, SEO, redes sociales, content marketing, analytics"),
    ("diseño", "UX/UI, gráfica, tipografía, colores, prototipado"),
    ("recursos humanos", "contratación, retención, cultura, evaluación, diversidad"),
    
    # Salud
    ("nutrición", "macros, micros, dietas, suplementos, ayuno intermitente"),
    ("salud mental", "terapia, medicación, estrés, burnout, autoestima"),
    ("sueño", "ciclos, higiene, insomnio, siestas, cronotipos"),
    ("prevención", "checkups, vacunas, screening, hábitos saludables"),
    ("envejecimiento", "longevidad, sarcopenia, cognitivo, calidad vida"),
]

# 4. Prompt template
SYSTEM_PROMPT = """Eres un asistente experto que genera diálogos naturales en español.
Genera pares de Usuario/Bot (U:/B:) sobre el tema dado.
REGLAS:
1. Usuario hace preguntas naturales, variadas y realistas
2. Bot responde con información precisa, útil y conversacional
3. Longitud: Usuario 10-80 chars, Bot 50-300 chars
4. Tono: profesional pero accesible, en español neutro
5. NO uses formato markdown, SOLO texto plano
6. Cada par en línea separada: U: ... B: ..."""

def generate_pairs_for_topic(topic, subtopics, num_pairs=50):
    """Genera pares para un tema específico"""
    prompt = f"""{SYSTEM_PROMPT}

TEMA: {topic}
SUBTEMAS: {subtopics}
GENERA: {num_pairs} pares U:/B:

EJEMPLO:
U: ¿Qué es la herencia en Python?
B: La herencia permite que una clase hija herede atributos y métodos de una clase padre, facilitando la reutilización de código.

U: ¿Cómo se declara una clase abstracta?
B: En Python usas el módulo abc y el decorador @abstractmethod para definir métodos que deben implementar las subclases.

AHORA GENERA {num_pairs} PARES:"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(
        inputs,
        max_new_tokens=2048,
        temperature=0.8,
        top_p=0.95,
        top_k=50,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
    return response

def parse_response(response, topic):
    """Parsea la respuesta del modelo a pares U:/B:"""
    pairs = []
    lines = response.strip().split('\n')
    
    current_u = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('U:'):
            if current_u:
                # Incomplete pair, skip
                pass
            current_u = line[2:].strip()
        elif line.startswith('B:') and current_u:
            current_b = line[2:].strip()
            pairs.append({
                "user": current_u,
                "bot": current_b,
                "topic": topic
            })
            current_u = None
    
    return pairs

# 5. Generar
print("\n" + "=" * 60)
print("GENERANDO 1,000 PARES")
print("=" * 60)

all_pairs = []
pairs_per_topic = 50  # 20 temas × 50 = 1000

for i, (topic, subtopics) in enumerate(TEMAS):
    print(f"\n[{i+1}/{len(TEMAS)}] {topic}...")
    
    try:
        response = generate_pairs_for_topic(topic, subtopics, pairs_per_topic)
        pairs = parse_response(response, topic)
        
        if pairs:
            all_pairs.extend(pairs)
            print(f"  Generados: {len(pairs)} pares")
        else:
            print(f"  ERROR: No se pudieron parsear pares")
            
    except Exception as e:
        print(f"  ERROR: {e}")
        continue
    
    # Guardar progreso cada 5 temas
    if (i + 1) % 5 == 0:
        with open(f'/kaggle/working/corpus_progress_{i+1}.jsonl', 'w') as f:
            for p in all_pairs:
                f.write(json.dumps(p, ensure_ascii=False) + '\n')

# 6. Guardar final
output_file = '/kaggle/working/corpus_generated_1k.jsonl'
with open(output_file, 'w', encoding='utf-8') as f:
    for p in all_pairs:
        f.write(json.dumps(p, ensure_ascii=False) + '\n')

# 7. Convertir a formato U:/B: txt
txt_file = '/kaggle/working/corpus_generated_1k.txt'
with open(txt_file, 'w', encoding='utf-8') as f:
    for p in all_pairs:
        f.write(f"U: {p['user']}\n")
        f.write(f"B: {p['bot']}\n\n")

print(f"\n{'='*60}")
print(f"COMPLETADO: {len(all_pairs)} pares generados")
print(f"Archivos:")
print(f"  - {output_file} (JSONL)")
print(f"  - {txt_file} (TXT formato U:/B:)")
print(f"{'='*60}")

# 8. Estadísticas
if all_pairs:
    user_lens = [len(p['user']) for p in all_pairs]
    bot_lens = [len(p['bot']) for p in all_pairs]
    topics = set(p['topic'] for p in all_pairs)
    
    print(f"\nEstadísticas:")
    print(f"  Temas únicos: {len(topics)}")
    print(f"  Longitud usuario: {min(user_lens)}-{max(user_lens)} (avg {sum(user_lens)//len(user_lens)})")
    print(f"  Longitud bot: {min(bot_lens)}-{max(bot_lens)} (avg {sum(bot_lens)//len(bot_lens)})")
    print(f"  Total chars: {sum(user_lens)+sum(bot_lens):,}")