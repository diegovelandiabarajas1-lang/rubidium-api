#!/usr/bin/env python3
"""
CAMEL-AI SOC Generator - 500K Pairs
Usa Inception Prompting (User Agent + Assistant Agent) con template expansion
"""
import json, random, os, hashlib, re
from pathlib import Path

random.seed(42)

# ============================================================
# CAMEL-AI ROLES (Inception Prompting)
# ============================================================

USER_PERSONAS = {
    "estudiante": {
        "desc": "estudiante universitario de 20 años, informal, curioso",
        "greetings": ["Hola", "Oye", "Ey", "Qué tal", "Buenas", "Disculpa", "Perdona", "Hey"],
        "questions": [
            "Que es {c}?", "Como funciona {c}?", "Para que sirve {c}?",
            "Como aprender {c}?", "Que necesito para {c}?", "Me enseñas {c}?",
            "Duda sobre {c}", "No entiendo {c}", "Explícame {c}",
            "Ejemplo de {c}?", "Tutorial de {c}?", "Consejos para {c}?",
            "Error en {c}", "No me funciona {c}", "Ayuda con {c}",
            "Diferencia entre {c} y {c2}?", "Cuando usar {c}?",
            "Ventajas de {c}?", "Desventajas de {c}?",
            "Que opinas de {c}?", "Vale la pena aprender {c}?",
            "Proyecto con {c}?", "Donde practicar {c}?",
        ],
    },
    "profesional": {
        "desc": "ingeniero senior de 35 años, técnico, preciso",
        "greetings": ["Necesito", "Busco", "Quiero", "Como puedo", "Cual es la mejor forma de"],
        "questions": [
            "Optimizar {c} en producción", "Arquitectura con {c}",
            "Escalabilidad de {c}", "Seguridad en {c}",
            "Rendimiento de {c}", "Monitoreo de {c}",
            "CI/CD para {c}", "Testing en {c}",
            "Refactoring de {c}", "Migración a {c}",
            "Patrones para {c}", "Anti-patrones en {c}",
            "Benchmark de {c}", "Producción con {c}",
            "Mejores prácticas en {c}", "Code review de {c}",
            "Deploy de {c}", "Debugging en {c}",
            "Performance tuning {c}", "Load balancing {c}",
        ],
    },
    "curioso": {
        "desc": "persona curiosa de 28 años, fascinada por todo",
        "greetings": ["Hola", "Oye", "Qué tal", "Buenas", "Cuéntame"],
        "questions": [
            "Que es {c}?", "Por qué es importante {c}?",
            "Como funciona {c}?", "Historia de {c}?",
            "Aplicaciones de {c}?", "Futuro de {c}?",
            "Curiosidades de {c}?", "Dato interesante sobre {c}?",
            "Impacto de {c}?", "Relación entre {c} y {c2}?",
            "Investigación en {c}?", "Descubrimientos sobre {c}?",
            "Teoría de {c}?", "Evolución de {c}?",
        ],
    },
    "frustrado": {
        "desc": "usuario frustrado con problemas técnicos, directo",
        "greetings": ["Error", "No funciona", "Problema", "Bug", "Crash", "Fallo", "Ayuda"],
        "questions": [
            "Error en {c}", "No carga {c}", "Se cuelga {c}",
            "Timeout en {c}", "Exception en {c}", "Memory leak {c}",
            "CPU al 100% en {c}", "Disco lleno en {c}",
            "No compila {c}", "Syntax error en {c}",
            "Import error en {c}", "Connection refused {c}",
            "Permission denied {c}", "File not found {c}",
            "500 error en {c}", "404 en {c}",
        ],
    },
    "creativo": {
        "desc": "escritor creativo de 30 años, imaginativo",
        "greetings": ["Escribe", "Crea", "Inventa", "Imagina", "Genera", "Diseña"],
        "questions": [
            "Historia sobre {c}", "Poema sobre {c}",
            "Personaje de {c}", "Guion sobre {c}",
            "Idea para {c}", "Concepto de {c}",
            "Mundo de {c}", "Aventura de {c}",
            "Misterio de {c}", "Comedia sobre {c}",
            "Drama de {c}", "Romance de {c}",
        ],
    },
    "docente": {
        "desc": "profesor de 45 años, pedagógico, paciente",
        "greetings": ["Explícame", "Enseña", "Concepto de", "Definición de"],
        "questions": [
            "Qué es {c}?", "Definición de {c}",
            "Ejemplo de {c}?", "Ejercicio de {c}",
            "Paso a paso {c}?", "Tutorial de {c}",
            "Problema de {c}?", "Solución de {c}",
            "Aplicación de {c}?", "Teorema de {c}",
            "Fórmula de {c}?", "Cálculo de {c}",
        ],
    },
    "casual": {
        "desc": "persona casual de 25 años, coloquial",
        "greetings": ["Qué tal", "Cómo estás", "Oye", "Cuéntame", "Qué onda"],
        "questions": [
            "Qué opinas de {c}?", "Recomiendas {c}?",
            "Vale la pena {c}?", "Qué hay de nuevo en {c}?",
            "Cómo está {c}?", "Cuál es tu opinión de {c}?",
            "Qué tal {c}?", "Cuéntame sobre {c}",
            "Hablemos de {c}", "Qué sabes de {c}?",
        ],
    },
    "experto": {
        "desc": "investigador de 40 años, profundo, técnico",
        "greetings": ["Analiza", "Compara", "Evalúa", "Investiga", "Revisa"],
        "questions": [
            "State of the art en {c}", "Benchmark de {c}",
            "Paper sobre {c}", "Teoría de {c}",
            "Optimización de {c}", "Arquitectura de {c}",
            "Implementación de {c}", "Comparativa de {c}",
            "Evaluación de {c}", "Análisis de {c}",
            "Complejidad de {c}", "Rendimiento de {c}",
        ],
    },
}

ASSISTANT_RESPONSES = {
    "tecnico": [
        "{c} es {explicacion}. En la practica, se usa para {caso}.",
        "Para entender {c}, debes saber que {explicacion}. Ejemplo: {ejemplo}.",
        "{c} destaca por {ventaja}. A diferencia de {c2}, ofrece {diferencia}.",
        "Los expertos recomiendan {consejo} al trabajar con {c}. Evita {error}.",
        "El futuro de {c} apunta hacia {tendencia}. {ejemplo}.",
        "{c} se usa ampliamente en {caso}. Su principal ventaja es {ventaja}.",
        "Aprender {c} toma {tiempo}. Lo importante es {importante}.",
        "La diferencia principal entre {c} y {c2} es {diferencia}.",
        "En {caso}, {c} es la mejor opcion porque {razon}.",
        "Un error comun con {c} es {error}. Mejor {mejor}.",
        "La comunidad de {c} es muy activa. {ejemplo}.",
        "Para empezar con {c}, necesitas {necesario}.",
        "Un consejo clave: {consejo}.",
        "Muchos cometen {error}. En su lugar, {mejor}.",
        "El ecosistema de {c} incluye {ventajas}.",
        "Comparado con {c2}, {c} ofrece {ventaja}.",
        "En produccion, {c} requiere {consejo}.",
        "La escalabilidad de {c} es {ventaja}.",
        "Para optimizar {c}, {consejo}.",
        "Testing en {c}: {consejo}.",
    ],
    "general": [
        "{c} es {explicacion}. Se aplica en {caso}.",
        "La importancia de {c} radica en {importancia}.",
        "Un dato interesante: {ejemplo}.",
        "La historia de {c} es fascinante. {historia}.",
        "El impacto de {c} es {impacto}.",
        "{c} estudia {objeto}. Su relevancia es {ventaja}.",
        "Los cientificos descubrieron que {c} permite {accion}.",
        "En {campo}, {c} es fundamental porque {razon}.",
        "La teoria de {c} explica {explicacion_simple}.",
        "Investigacion reciente muestra que {c} tiene implicaciones en {aplicacion}.",
        "El impacto de {c} en la sociedad es {impacto_social}.",
        "Para {c}, la clave es {consejo}.",
        "La idea principal es {concepto}. Ejemplo: {ejemplo}.",
        "Empecemos con lo basico: {concepto}.",
        "Un error comun es {error}. La forma correcta es {correcto}.",
    ],
    "emocional": [
        "Entiendo tu situacion. {c} puede ser dificil, pero {consejo}.",
        "No te preocupes, es comun tener problemas con {c}. {solucion}.",
        "Es normal frustrarse con {c}. La solucion es {solucion}.",
        "Tranquilo, {c} se resuelve con {consejo}.",
        "He visto muchos casos como el tuyo con {c}. {solucion}.",
        "La clave es {consejo}. Con practica, {resultado}.",
        "No rendirse. {c} toma {tiempo}. {importante}.",
        "Poco a poco. {consejo}. El resultado valdra la pena.",
    ],
}

TOPICS = [
    "python", "javascript", "typescript", "rust", "go", "java", "c++",
    "react", "vue", "angular", "svelte", "next.js", "nuxt",
    "django", "flask", "fastapi", "express", "spring boot",
    "docker", "kubernetes", "terraform", "aws", "azure", "gcp",
    "postgresql", "mysql", "mongodb", "redis", "sqlite",
    "graphql", "rest", "grpc", "websockets",
    "git", "ci/cd", "github actions", "jenkins",
    "machine learning", "deep learning", "nlp", "computer vision",
    "tensorflow", "pytorch", "scikit-learn", "pandas",
    "html", "css", "tailwind", "sass",
    "linux", "bash", "networking", "ssh",
    "algoritmos", "estructuras de datos", "patrones de diseño",
    "clean code", "solid", "tdd", "devops",
    "inteligencia artificial", "blockchain", "cloud computing",
    "ciberseguridad", "criptografia", "firewall",
    "realidad virtual", "iot", "5g", "robotica",
    "calculo", "algebra", "estadistica", "probabilidad",
    "fisica", "quimica", "biologia", "astronomia",
    "neurociencia", "genetica", "evolucion", "adn",
    "filosofia", "historia", "psicologia", "economia",
    "salud", "nutricion", "ejercicio", "meditacion",
    "finanzas", "inversiones", "emprendimiento", "marketing",
    "fotografia", "musica", "arte", "literatura",
    "cine", "videojuegos", "anime", "literatura",
    "educacion", "idiomas", "productividad", "liderazgo",
]

# ============================================================
# FILL TEMPLATE
# ============================================================

def fill_template(template, params):
    result = template
    for key, val in params.items():
        result = result.replace("{" + key + "}", str(val))
    result = re.sub(r'\{[a-z_0-9]+\}', 'algo', result)
    return result

# ============================================================
# GENERATE SINGLE CONVERSATION (CAMEL-AI Style)
# ============================================================

def generate_conversation():
    """Genera una conversación multi-turno estilo CAMEL-AI"""
    
    # Seleccionar roles
    user_key = random.choice(list(USER_PERSONAS.keys()))
    user_persona = USER_PERSONAS[user_key]
    asst_key = random.choice(list(ASSISTANT_RESPONSES.keys()))
    asst_responses = ASSISTANT_RESPONSES[asst_key]
    
    topic = random.choice(TOPICS)
    
    # Determinar numero de turnos
    num_turns = random.randint(3, 8)
    
    turns = []
    
    # Turno 1: User pregunta
    greeting = random.choice(user_persona["greetings"])
    q_template = random.choice(user_persona["questions"])
    
    # Parametros para templates
    c1 = random.choice(TOPICS)
    c2 = random.choice([t for t in TOPICS if t != c1] or [c1])
    
    params = {
        "c": c1, "c2": c2,
        "accion": random.choice(["crear apps", "automatizar", "analizar datos", "desarrollar software"]),
        "caso": random.choice(["desarrollo web", "ciencia de datos", "backend", "frontend", "DevOps", "cloud"]),
        "explicacion": random.choice(["es un lenguaje moderno", "es un framework potente", "es una herramienta esencial"]),
        "ejemplo": random.choice(["Netflix lo usa", "Google lo usa", "Amazon lo usa", "Spotify lo usa"]),
        "ventaja": random.choice(["velocidad", "escalabilidad", "simplicidad", "flexibilidad", "seguridad"]),
        "diferencia": random.choice(["mayor rendimiento", "mas simple", "mas seguro", "mejor ecosistema"]),
        "consejo": random.choice(["escribir tests", "documentar codigo", "practicar a diario", "leer docs"]),
        "error": random.choice(["no validar entrada", "hardcodear valores", "ignorar errores", "no testear"]),
        "tendencia": random.choice(["IA generativa", "edge computing", "serverless", "microservicios"]),
        "tiempo": random.choice(["unas semanas", "1-3 meses", "6 meses"]),
        "importante": random.choice(["practicar", "ser constante", "construir proyectos"]),
        "razon": random.choice(["su ecosistema", "su comunidad", "su versatilidad"]),
        "mejor": random.choice(["empezar basico", "practicar diario", "leer documentacion"]),
        "necesario": random.choice(["ganas de aprender", "un computador", "tiempo"]),
        "historia": random.choice(["comenzo en los 90", "fue creado para resolver", "evoluciono con el tiempo"]),
        "impacto": random.choice(["transforma industrias", "cambia vidas", "crea oportunidades"]),
        "objeto": random.choice(["fenomenos naturales", "procesos", "estructuras"]),
        "campo": random.choice(["ciencia", "ingenieria", "medicina", "tecnologia"]),
        "aplicacion": random.choice(["medicina", "energia", "exploracion"]),
        "importancia": random.choice(["entender el mundo", "mejorar vidas", "crear futuro"]),
        "consecuencia": random.choice(["no avanzariamos", "no entenderiamos", "no existiria"]),
        "descubrimiento": random.choice(["investigadores lo hallaron", "cientificos lo confirmaron"]),
        "impacto_social": random.choice(["transforma sociedades", "crea empleos", "cambia culturas"]),
        "concepto": random.choice(["una funcion matematica", "un algoritmo", "una estructura"]),
        "correcto": random.choice(["validar siempre", "testear todo", "documentar"]),
        "solucion": random.choice(["reiniciar", "revisar logs", "actualizar", "reinstalar"]),
        "resultado": random.choice(["mejora gradual", "exito total", "aprendizaje"]),
        "historias": ["la historia es larga", "comenzo simple", "evoluciono mucho"],
        "aplicaciones": ["multiples usos", "industria y academia", "vida cotidiana"],
        "futuros": ["mejoras importantes", "nuevas capacidades", "revolucion"],
    }
    
    user_question = f"{greeting} {fill_template(q_template, params)}"
    turns.append({"role": "user", "content": user_question.strip()})
    
    # Turnos intermedios: Assistant responde + User sigue
    for i in range(num_turns - 1):
        # Assistant responde
        asst_template = random.choice(asst_responses)
        asst_response = fill_template(asst_template, params)
        turns.append({"role": "assistant", "content": asst_response.strip()})
        
        # User sigue (si no es ultimo turno)
        if i < num_turns - 2:
            followup_templates = [
                "Y {c2} como se compara con {c}?",
                "Que opinas de {c2} entonces?",
                "Interesante. Ahora dime de {c2}",
                "Ok. Que hay de {c2}?",
                "Entiendo. Cambiando de tema, {c2}",
                "Bien. Otra pregunta sobre {c2}",
                "Ya. Pero que me dices de {c2}?",
                "Vale. Hablando de {c2}...",
            ]
            followup = fill_template(random.choice(followup_templates), params)
            turns.append({"role": "user", "content": followup.strip()})
    
    return turns

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("CAMEL-AI SOC Generator - 500K Pairs")
    print("=" * 60)
    
    TARGET = 500000
    NUM_FILES = 30
    OUTPUT_DIR = "D:/Inteligente/rubidium-api/resources"
    
    # Generate pairs
    pairs = []
    seen = set()
    
    print(f"Generating {TARGET:,} unique pairs...")
    while len(pairs) < TARGET:
        conv = generate_conversation()
        
        # Dedup by first user message
        key = conv[0]["content"].lower()
        if key not in seen:
            seen.add(key)
            pairs.append(conv)
            
            if len(pairs) % 50000 == 0 and len(pairs) > 0:
                print(f"  {len(pairs):,} / {TARGET:,}")
    
    print(f"\nTotal: {len(pairs):,} unique conversations")
    
    # Shuffle
    random.shuffle(pairs)
    
    # Split into files
    per_file = len(pairs) // NUM_FILES
    remainder = len(pairs) % NUM_FILES
    
    print(f"\nSplitting into {NUM_FILES} files...")
    
    start = 0
    for i in range(NUM_FILES):
        end = start + per_file + (1 if i < remainder else 0)
        batch = pairs[start:end]
        
        filename = f"chat_{6 + i:02d}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            for conv in batch:
                for turn in conv:
                    role = "U" if turn["role"] == "user" else "B"
                    f.write(f"{role}: {turn['content']}\n")
                f.write("\n")
        
        size_kb = os.path.getsize(filepath) // 1024
        print(f"  {filename}: {len(batch):,} convs ({size_kb} KB)")
        start = end
    
    # Save JSONL
    jsonl_path = os.path.join(OUTPUT_DIR, "soc_500k.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for conv in pairs:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    
    # Save ShareGPT
    sharegpt_path = os.path.join(OUTPUT_DIR, "soc_500k_sharegpt.json")
    sharegpt = [{"conversations": [{"from": t["role"], "value": t["content"]} for t in conv]} for conv in pairs]
    with open(sharegpt_path, "w", encoding="utf-8") as f:
        json.dump(sharegpt, f, ensure_ascii=False)
    
    # Stats
    all_user = [t["content"] for conv in pairs for t in conv if t["role"] == "user"]
    all_bot = [t["content"] for conv in pairs for t in conv if t["role"] == "assistant"]
    total_chars = sum(len(x) for x in all_user) + sum(len(x) for x in all_bot)
    
    print(f"\n{'=' * 60}")
    print(f"COMPLETADO: {len(pairs):,} conversaciones CAMEL-AI")
    print(f"Archivos: chat_06.txt a chat_{5 + NUM_FILES:02d}.txt")
    print(f"JSONL: {jsonl_path} ({os.path.getsize(jsonl_path) / 1024 / 1024:.1f} MB)")
    print(f"ShareGPT: {sharegpt_path} ({os.path.getsize(sharegpt_path) / 1024 / 1024:.1f} MB)")
    print(f"Total chars: {total_chars:,}")
    print(f"Est. tokens: ~{total_chars // 4:,}")

if __name__ == "__main__":
    main()
