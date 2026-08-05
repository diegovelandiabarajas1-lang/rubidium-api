#!/usr/bin/env python3
"""CAMEL-AI Quality Generator v2 - Fast version"""
import json, random, os
from pathlib import Path

random.seed(42)

# Topic knowledge base (condensed)
KNOWLEDGE = {
    "python": {"desc": "lenguaje versatil para web, datos, IA", "use": "backend, ciencia de datos, automatizacion",
               "ex": ["Netflix data pipelines", "Instagram backend", "Spotify ML"], "pros": ["simple", "ecosistema grande"],
               "cons": ["lento", "GIL"], "tips": ["virtualenv", "pytest", "PEP8"]},
    "javascript": {"desc": "lenguaje para web, async, frontend y backend", "use": "frontend, Node.js, mobile",
                   "ex": ["Facebook UI", "Twitter web", "Airbnb"], "pros": ["ubiquito", "async nativo"],
                   "cons": ["tipado debil", "fragmentacion"], "tips": ["TypeScript", "async/await", "ESLint"]},
    "rust": {"desc": "lenguaje de sistemas con seguridad de memoria", "use": "sistemas, embedded, webassembly",
             "ex": ["Firefox engine", "Dropbox storage", "Cloudflare"], "pros": ["seguro", "rapido como C++"],
             "cons": ["curva alta", "compilacion lenta"], "tips": ["Rustlings", "ownership primero"]},
    "docker": {"desc": "contenedores para desplegar aplicaciones", "use": "devops, microservicios, CI/CD",
               "ex": ["Netflix microservices", "Spotify deployments"], "pros": ["aislamiento", "portabilidad"],
               "cons": ["overhead", "seguridad"], "tips": ["multi-stage builds", "no root"]},
    "kubernetes": {"desc": "orquestacion de contenedores en produccion", "use": "produccion, escalado automatico",
                   "ex": ["Google Search", "Spotify"], "pros": ["auto-healing", "escalado"],
                   "cons": ["complejidad", "requiere expertise"], "tips": ["minikube", "Helm"]},
    "machine learning": {"desc": "IA que aprende de datos sin programacion explicita", "use": "clasificacion, recomendaciones",
                         "ex": ["Netflix recommendations", "Gmail spam filter"], "pros": ["automatiza", "encuentra patrones"],
                         "cons": ["necesita datos", "caja negra"], "tips": ["scikit-learn", "cross-validation"]},
    "deep learning": {"desc": "redes neuronales profundas para IA avanzada", "use": "vision, NLP, generacion",
                      "ex": ["ChatGPT", "DALL-E", "Tesla Autopilot"], "pros": ["SOTA", "aprende representaciones"],
                      "cons": ["GPU costosa", "caja negra"], "tips": ["transfer learning", "pre-entrenados"]},
    "react": {"desc": "libreria JS para interfaces declarativas", "use": "SPA, mobile apps, dashboards",
              "ex": ["Facebook", "Instagram web", "Netflix UI"], "pros": ["componentes", "virtual DOM"],
              "cons": ["solo UI", "JSX"], "tips": ["hooks", "useState", "no mutar state"]},
    "fisica": {"desc": "ciencia de las leyes fundamentales del universo", "use": "ingenieria, tecnologia",
               "ex": ["GPS con relatividad", "semiconductores con QC"], "pros": ["explica universo", "base tecnologia"],
               "cons": ["abstracto", "matematicas avanzadas"], "tips": ["mecanica clasica primero", "vectores"]},
    "biologia": {"desc": "ciencia de los seres vivos", "use": "medicina, biotecnologia",
                 "ex": ["vacunas mRNA", "CRISPR"], "pros": ["entender vida", "avances medicos"],
                 "cons": ["complejidad", "etica"], "tips": ["celula primero", "ADN"]},
    "matematicas": {"desc": "ciencia de estructuras y patrones abstractos", "use": "ciencia datos, finanzas, crypto",
                    "ex": ["Bitcoin crypto", "Google algebra lineal"], "pros": ["pensamiento logico", "universal"],
                    "cons": ["abstracto", "intimidante"], "tips": ["practicar diario", "conceptos antes de formulas"]},
    "nutricion": {"desc": "ciencia de como los alimentos afectan la salud", "use": "salud, deporte, prevencion",
                  "ex": ["dieta mediterranea", "proteina para musculos"], "pros": ["mejora salud", "previene"],
                  "cons": ["info contradictoria", "cultural"], "tips": ["variedad", "frutas", "evitar ultraprocesados"]},
    "ciberseguridad": {"desc": "proteccion de sistemas y datos", "use": "proteccion datos, cumplimiento",
                       "ex": ["Google 2FA", "Signal encryption"], "pros": ["protege datos", "cumple regulaciones"],
                       "cons": ["cambiante", "costo"], "tips": ["2FA", "actualizar", "passwords fuertes"]},
    "guitarra": {"desc": "instrumento musical de cuerdas", "use": "musica, arte, expresion",
                 "ex": ["rock clasico", "flamenco", "acustica"], "pros": ["portatil", "versatil"],
                 "cons": ["dedos duelenn", "practica constante"], "tips": ["acordes basicos", "practicar diario"]},
    "fotografia": {"desc": "arte de capturar imagenes con luz", "use": "arte, comunicacion, documentacion",
                   "ex": ["retrato profesional", "paisajes", "street photography"], "pros": ["expresion artistica"],
                   "cons": ["equipo costoso", "curva"], "tips": ["composicion", "luz natural"]},
    "meditacion": {"desc": "practica de entrenar atencion y conciencia", "use": "salud mental, relajacion",
                   "ex": ["mindfulness", "meditacion trascendental"], "pros": ["reduce estres", "mejora foco"],
                   "cons": ["requiere constancia", "resultados lentos"], "tips": ["empezar 5 min", "respiracion"]},
    "emprendimiento": {"desc": "crear y gestionar negocio propio", "use": "negocios, innovacion",
                       "ex": ["startup tech", "negocio local"], "pros": ["independencia", "potencial"],
                       "cons": ["riesgo", "inestabilidad"], "tips": ["validar idea", "empezar pequeno"]},
    "liderazgo": {"desc": "habilidad de guiar e inspirar", "use": "management, politica",
                  "ex": ["liderazgo transformacional", "liderazgo situacional"], "pros": ["impacto", "crecimiento"],
                  "cons": ["responsabilidad", "presion"], "tips": ["escuchar", "empatia", "ejemplo"]},
    "diseno_ux": {"desc": "experiencia de usuario en productos digitales", "use": "apps, web, software",
                  "ex": ["Apple design", "Google Material"], "pros": ["usabilidad", "satisfaccion"],
                  "cons": ["subjetivo", "iterativo"], "tips": ["user research", "prototipos"]},
    "ia_generativa": {"desc": "IA que crea contenido nuevo", "use": "creatividad, productividad",
                      "ex": ["ChatGPT", "Midjourney", "Copilot"], "pros": ["creatividad infinita", "productividad"],
                      "cons": ["calidad variable", "etica"], "tips": ["prompts claros", "verificar output"]},
}

# Generic topics
GENERIC = {
    "cocina": "preparacion de alimentos",
    "viajes": "desplazarse a otros lugares",
    "lectura": "interpretar textos escritos",
    "yoga": "practica fisica y mental",
    "senderismo": "caminar por naturaleza",
    "pintura": "arte de aplicar color",
    "baile": "actividad ritmica",
    "natacion": "deporte acuatico",
    "ajedrez": "juego de estrategia",
    "jardineria": "cultivar plantas",
    "blockchain": "registro distribuido",
    "impresion_3d": "fabricacion aditiva",
    "robotica": "construir robots",
    "energia_solar": "convertir luz en electricidad",
    "exploracion_espacial": "viaje fuera de la Tierra",
    "neurociencia": "estudiar el cerebro",
    "sostenibilidad": "preservar recursos",
    "creatividad": "generar ideas originales",
    "pensamiento_critico": "analisis objetivo",
    "inteligencia_emocional": "gestionar emociones",
    "oratoria": "hablar en publico",
    "escritura_creativa": "crear textos literarios",
    "cine": "crear peliculas",
    "videojuegos": "juegos electronicos",
    "historia": "estudio de eventos pasados",
    "filosofia": "reflexion fundamental",
    "psicologia": "ciencia del comportamiento",
    "economia": "produccion y consumo",
    "astronomia": "estudio de cuerpos celestes",
    "quimica": "ciencia de la materia",
    "finanzas_personales": "gestionar dinero",
    "inversiones": "generar rendimiento",
    "marketing_digital": "promocion en digital",
    "comunicacion": "intercambiar informacion",
    "productividad": "eficiencia en tiempo",
    "relaciones": "conexion entre personas",
    "educacion": "proceso de aprender",
    "cultura": "costumbres y conocimiento",
    "arte": "expresion estetica",
    "naturaleza": "medio ambiente",
    "tecnologia": "aplicacion practica del conocimiento",
    "innovacion": "crear algo nuevo",
    "deporte": "actividad fisica competitiva",
    "salud": "bienestar fisico y mental",
}

# Questions
Q_TEMPLATES = [
    "Que es {c}?", "Para que sirve {c}?", "Como funciona {c}?",
    "Cuentame sobre {c}", "Hablame de {c}", "Quiero saber de {c}",
    "Que me dices de {c}?", "Necesito info de {c}", "Como se usa {c}?",
    "Que sabes de {c}?", "Quiero entender {c}", "Explicame {c}",
]

FOLLOWUPS = [
    "Que ventajas tiene?", "Y desventajas?", "Algun consejo?",
    "Como aprendo mas?", "Que errores cometo?", "Cuenta mas",
    "Interesante, que mas?", "Entiendo, y que mas?", "Ok, que mas?",
    "Que futuro tiene?", "Y comparado con otra cosa?", "Ejemplos?",
]

# Responses (topic-aware)
def respond(topic, q_type, kb):
    if q_type == 0:  # que_es
        if "desc" in kb: return f"{topic} es {kb['desc']}."
        return f"{topic} es un tema interesante con muchas aplicaciones."
    elif q_type == 1:  # para_que
        if "use" in kb: return f"Se usa en {kb['use']}."
        return f"Se aplica en multiples areas."
    elif q_type == 2:  # ejemplos
        if "ex" in kb: return f"Un ejemplo: {random.choice(kb['ex'])}."
        return f"Hay muchos ejemplos practicos."
    elif q_type == 3:  # ventajas
        if "pros" in kb: return f"Ventajas: {', '.join(random.sample(kb['pros'], min(2, len(kb['pros']))))}."
        return f"Tiene varias ventajas."
    elif q_type == 4:  # desventajas
        if "cons" in kb: return f"Limitaciones: {', '.join(random.sample(kb['cons'], min(2, len(kb['cons']))))}."
        return f"Tiene algunas limitaciones."
    elif q_type == 5:  # consejos
        if "tips" in kb: return f"Consejo: {random.choice(kb['tips'])}."
        return f"Lo importante es practicar y ser constante."
    elif q_type == 6:  # errores
        return f"Errores comunes: no practicar lo suficiente, no pedir ayuda, rendirse pronto."
    elif q_type == 7:  # futuro
        return f"El futuro de {topic} es prometedor con avances constantes."
    elif q_type == 8:  # historia
        return f"La historia de {topic} es rica y ha evolucionado mucho."
    elif q_type == 9:  # curiosidad
        return f"Dato curioso: {topic} tiene aplicaciones que van mas alla de lo comun."
    else:
        return f"{topic} es un tema que vale la pena explorar."

# Generate conversation
def gen_conv():
    # Pick topic
    if random.random() < 0.6:
        topic = random.choice(list(KNOWLEDGE.keys()))
        kb = KNOWLEDGE[topic]
    else:
        topic = random.choice(list(GENERIC.keys()))
        kb = {"desc": GENERIC[topic], "use": "multiples areas"}
    
    turns = [{"role": "user", "content": random.choice(Q_TEMPLATES).replace("{c}", topic)}]
    
    num_turns = random.randint(3, 6)
    for i in range(num_turns - 1):
        q_type = random.randint(0, 10)
        resp = respond(topic, q_type, kb)
        turns.append({"role": "assistant", "content": resp})
        
        if i < num_turns - 2:
            turns.append({"role": "user", "content": random.choice(FOLLOWUPS)})
    
    return turns

# Main
print("Generating 500K quality conversations...")
pairs = []
for i in range(500000):
    pairs.append(gen_conv())
    if (i+1) % 100000 == 0:
        print(f"  {i+1:,}")

random.shuffle(pairs)

OUTPUT = "D:/Inteligente/rubidium-api/resources"
os.makedirs(OUTPUT, exist_ok=True)

per = len(pairs) // 30
rem = len(pairs) % 30
start = 0
print(f"\nSplitting {len(pairs):,} into 30 files...")

for i in range(30):
    end = start + per + (1 if i < rem else 0)
    batch = pairs[start:end]
    fn = f"chat_{6+i:02d}.txt"
    fp = os.path.join(OUTPUT, fn)
    with open(fp, "w", encoding="utf-8") as f:
        for conv in batch:
            for t in conv:
                r = "U" if t["role"]=="user" else "B"
                f.write(f"{r}: {t['content']}\n")
            f.write("\n")
    sz = os.path.getsize(fp)//1024
    print(f"  {fn}: {len(batch):,} ({sz} KB)")
    start = end

# JSONL
jl = os.path.join(OUTPUT, "soc_500k_v2.jsonl")
with open(jl, "w", encoding="utf-8") as f:
    for c in pairs: f.write(json.dumps(c,ensure_ascii=False)+"\n")

# Stats
all_bot = [t["content"] for c in pairs for t in c if t["role"]=="assistant"]
all_words = " ".join(all_bot).split()
unique = set(w.lower() for w in all_words)
ttr = len(unique)/len(all_words) if all_words else 0

print(f"\nDONE: {len(pairs):,} conversations")
print(f"Vocab TTR: {ttr:.3f}")
print(f"JSONL: {jl} ({os.path.getsize(jl)//1024//1024} MB)")
