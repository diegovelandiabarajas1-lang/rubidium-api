#!/usr/bin/env python3
"""
CAMEL-AI Quality Generator v2
Fixed: es es bug, topic coherence, naturalness, vocabulary richness
"""
import json, random, os, re, hashlib
from pathlib import Path

random.seed(42)

# ============================================================
# KNOWLEDGE BASE - Each topic has domain-specific content
# ============================================================

KNOWLEDGE = {
    # PROGRAMACION
    "python": {
        "desc": "lenguaje de programacion versatil, interpretado, con tipado dinamico",
        "use": "desarrollo web, ciencia de datos, automatizacion, scripts",
        "examples": ["Instagram backend", "Netflix data pipelines", "Spotify ML recommendations"],
        "pros": ["sintaxis clara", "gran ecosistema", "miles de librerias", "comunidad masiva"],
        "cons": ["mas lento que C++", "GIL limita paralelismo", "no es ideal para tiempo real"],
        "tips": ["usar virtualenv", "escribir tests con pytest", "documentar con docstrings", "seguir PEP8"],
        "errors": ["indentation error", "NameError", "TypeError", "modificar mutable durante iteracion"],
        "comparisons": {"javascript": "Python es mas legible, JS es mas web-friendly",
                       "java": "Python es mas conciso, Java es mas estricto",
                       "rust": "Python es mas facil, Rust es mas seguro y rapido"},
        "related": ["numpy", "pandas", "flask", "django", "fastapi", "tensorflow", "pytorch"],
    },
    "javascript": {
        "desc": "lenguaje de programacion para web, async, conPrototype-based inheritance",
        "use": "frontend web, backend con Node.js, mobile con React Native",
        "examples": ["Facebook UI", "Twitter fronted", "Airbnb web app"],
        "pros": ["ubiquito en web", "asincrono nativo", "ecosistema npm enorme"],
        "cons": ["tipado debil", "callback hell", "fragmentacion de frameworks"],
        "tips": ["usar TypeScript para proyectos grandes", "aprender async/await", "usar ESLint"],
        "errors": ["undefined is not a function", "cannot read property", "hoisting issues"],
        "comparisons": {"python": "JS es para web, Python para data science",
                       "typescript": "TS anade tipos estaticos a JS"},
        "related": ["react", "vue", "angular", "node.js", "express", "next.js"],
    },
    "rust": {
        "desc": "lenguaje de sistemas con seguridad de memoria, sin garbage collector",
        "use": "sistemas, embedded, webassembly, CLI tools, blockchains",
        "examples": ["Firefox engine", "Dropbox storage", "Cloudflare edge"],
        "pros": ["seguridad de memoria", "rendimiento como C++", "sin data races", "pattern matching potente"],
        "cons": ["curva de aprendizaje alta", "compilacion lenta", "menos librerias que Python"],
        "tips": ["empezar con Rustlings", "entender ownership antes de avanzar", "usar cargo clippy"],
        "errors": ["borrow checker errors", "lifetime issues", "move occurs because value has type"],
        "comparisons": {"c++": "Rust es mas seguro, C++ es mas flexible",
                       "go": "Rust es mas control, Go es mas simple"},
        "related": ["cargo", "tokio", "serde", "actix", "wasm"],
    },
    "typescript": {
        "desc": "JavaScript con tipos estaticos, mejor para proyectos grandes",
        "use": "frontend/backend web, aplicaciones empresariales",
        "examples": ["Angular apps", "VS Code", "Slack desktop"],
        "pros": ["tipado estatico", "autocomplete mejorado", "refactorization segura"],
        "cons": ["adds complexity", "learning curve", "build step required"],
        "tips": ["usar interfaces estrictas", "evitar 'any'", "usar utility types"],
        "errors": ["type 'X' is not assignable", "object is possibly undefined"],
        "comparisons": {"javascript": "TS es JS con tipos, mejor para equipos grandes"},
        "related": ["angular", "react", "vue", "nest.js"],
    },
    "docker": {
        "desc": "plataforma de contenedores para empaquetar y desplegar aplicaciones",
        "use": "devops, microservicios, CI/CD, ambiental reproducible",
        "examples": ["Netflix microservices", "Spotify deployments", "GitHub Actions runners"],
        "pros": ["aislamiento", "reproducibilidad", "portabilidad", "escalabilidad"],
        "cons": ["overhead de recursos", "curva de aprendizaje", "seguridad de imagenes"],
        "tips": ["usar multi-stage builds", "minimizar capas", "no correr como root", "usar docker-compose"],
        "errors": ["cannot connect to Docker daemon", "image not found", "port already in use"],
        "comparisons": {"kubernetes": "Docker es para contenedores, K8s para orquestacion"},
        "related": ["kubernetes", "docker-compose", "podman", "containerd"],
    },
    "kubernetes": {
        "desc": "sistema de orquestacion de contenedores para escalar aplicaciones",
        "use": "produccion, microservicios, escalado automatico",
        "examples": ["Google Search", "Spotify 100M+ users", "Pokemon GO infrastructure"],
        "pros": ["auto-healing", "escalado automatico", "service discovery", "rolling updates"],
        "cons": ["complejidad alta", "requiere expertise", "overhead para apps simples"],
        "tips": ["empezar con minikube", "usar Helm charts", "monitorar con Prometheus"],
        "errors": ["pod CrashLoopBackOff", "ImagePullBackOff", "service unreachable"],
        "comparisons": {"docker-compose": "K8s es para produccion, compose para desarrollo"},
        "related": ["helm", "istio", "prometheus", "grafana", "argo"],
    },
    "machine learning": {
        "desc": "campo de IA que permite a las maquinas aprender de datos sin programacion explicita",
        "use": "clasificacion, regresion, recomendaciones, prediccion",
        "examples": ["Netflix recommendations", "Spotify Discover Weekly", "Gmail spam filter"],
        "pros": ["automatiza decisiones", "encuentra patrones ocultos", "mejora con mas datos"],
        "cons": ["necesita muchos datos", "caja negra", "overfitting", "requiere GPU"],
        "tips": ["empezar con scikit-learn", "validar con cross-validation", "feature engineering es clave"],
        "errors": ["overfitting", "underfitting", "data leakage", "feature scaling issues"],
        "comparisons": {"deep learning": "ML usa modelos simples, DL usa redes neuronales profundas"},
        "related": ["tensorflow", "pytorch", "scikit-learn", "pandas", "numpy"],
    },
    "deep learning": {
        "desc": "subcampo del ML que usa redes neuronales con multiples capas",
        "use": "vision por computadora, NLP, generacion de texto/imagenes",
        "examples": ["ChatGPT", "DALL-E", "Tesla Autopilot", "Google Translate"],
        "pros": ["aprende representaciones automaticas", "state-of-the-art en muchos problemas"],
        "cons": ["necesita mucha GPU", "caja negra total", "datos masivos requeridos"],
        "tips": ["usar transfer learning", "empezar con modelos pre-entrenados", "monitorar loss"],
        "errors": ["vanishing gradients", "exploding gradients", "mode collapse"],
        "comparisons": {"machine learning": "DL es mas poderoso pero mas costoso"},
        "related": ["transformers", "cnn", "rnn", "gan", "diffusion models"],
    },
    "react": {
        "desc": "libreria de JavaScript para construir interfaces de usuario declarativas",
        "use": "single page applications, mobile apps, dashboards",
        "examples": ["Facebook", "Instagram web", "Netflix UI", "Airbnb"],
        "pros": ["componentes reutilizables", "virtual DOM rapido", "ecosistema enorme"],
        "cons": ["solo es UI (necesita mas libs)", "JSX confunde al inicio", "learning curve"],
        "tips": ["aprender hooks primero", "usar useState y useEffect", "no mutar state directamente"],
        "errors": ["hooks rules violation", "infinite re-render", "stale closure"],
        "comparisons": {"vue": "React es mas flexible, Vue es mas facil de aprender"},
        "related": ["next.js", "redux", "react native", "vue", "svelte"],
    },
    # CIENCIA
    "fisica": {
        "desc": "ciencia que estudia las leyes fundamentales del universo",
        "use": "ingenieria, tecnologia, prediccion de fenomenos naturales",
        "examples": ["GPS funciona con relatividad", "semiconductores usan mecanica cuantica"],
        "pros": ["explica el universo", "base de toda la tecnologia", "mente rigorous"],
        "cons": ["conceptos contraintuitivos", "matematicas avanzadas", "experimentos costosos"],
        "tips": ["empezar con mecanica clasica", "entender vectores primero", "practicar problemas"],
        "errors": ["confundir masa con peso", "olvidar unidades", "ignorar signos"],
        "comparisons": {"quimica": "Fisica es mas fundamental, Quimica es mas aplicada"},
        "related": ["mecanica cuantica", "relatividad", "termodinamica", "electromagnetismo"],
    },
    "biologia": {
        "desc": "ciencia que estudia los seres vivos y sus procesos",
        "use": "medicina, biotecnologia, agricultura, conservacion",
        "examples": ["vacunas mRNA", "CRISPR gene editing", "biofuel production"],
        "pros": ["entender la vida", "avances medicos", "tecnologias emergentes"],
        "cons": ["complejidad enorme", "etica controversial", "experimentos lentos"],
        "tips": ["empezar con biologia celular", "entender ADN primero", "usar simulaciones"],
        "errors": ["confundir celula con organulo", "olvidar niveles de organizacion"],
        "comparisons": {"quimica": "Biologia estudia vida, Quimica estudia materia"},
        "related": ["genetica", "ecologia", "neurociencia", "evolucion"],
    },
    "matematicas": {
        "desc": "ciencia de las estructuras, patrones y cantidades abstractas",
        "use": "ciencia de datos, finanzas, ingenieria, criptografia",
        "examples": ["Bitcoin usa criptografia", "Google Search usa algebra lineal", "Juegos usan geometria"],
        "pros": ["pensamiento logico", "base de todo", "universal"],
        "cons": ["abstracto", "puede ser intimidante", "muchos conceptos"],
        "tips": ["practicar diario", "entender conceptos antes de formulas", "resolver problemas variados"],
        "errors": ["operar sin entender", "memorizar sin comprender", "olvidar dominio"],
        "comparisons": {"programacion": "Matematicas es la base, programacion es la aplicacion"},
        "related": ["calculo", "algebra", "estadistica", "probabilidad", "logica"],
    },
    # SALUD
    "nutricion": {
        "desc": "ciencia que estudia como los alimentos afectan la salud",
        "use": "salud, deporte,-prevencion de enfermedades",
        "examples": ["dieta mediterranea reduce cardiopatias", "proteina para musculos"],
        "pros": ["mejora salud", "previene enfermedades", "aumenta energia"],
        "cons": ["informacion contradictoria", "dietas extremas", "culturalmente variable"],
        "tips": ["comer variedad", "frutas y verduras diarias", "evitar ultraprocesados", "hidratarse"],
        "errors": ["saltar comidas", "exceso de azucar", "dietas restrictivas"],
        "comparisons": {"ejercicio": "Nutricion es 70% del resultado, ejercicio 30%"},
        "related": ["dieta", "proteina", "carbohidratos", "vitaminas", "minerales"],
    },
    # TECNOLOGIA
    "ciberseguridad": {
        "desc": "practica de proteger sistemas y datos de accesos no autorizados",
        "use": "proteccion de datos, cumplimiento normativo, privacidad",
        "examples": ["Google Advanced Protection", "Signal encryption", "YubiKey 2FA"],
        "pros": ["protege datos criticos", "cumple regulaciones", "evita perdidas"],
        "cons": ["siempre cambiante", "requiere actualizarse constantemente", "costo de implementacion"],
        "tips": ["usar 2FA siempre", "actualizar software", "no clickar links sospechosos", "usar passwords fuertes"],
        "errors": ["passwords debiles", "phishing", "no actualizar", "exceso de permisos"],
        "comparisons": {"privacidad": "Seguridad protege datos, Privacidad controla acceso"},
        "related": ["firewall", "encryption", "pentesting", "zero trust"],
    },
}

# Generic topics without specific knowledge
GENERIC_TOPICS = {
    "guitarra": {"desc": "instrumento musical de cuerdas", "use": "musica, arte, expresion"},
    "fotografia": {"desc": "arte de capturar imagenes con luz", "use": "arte, comunicacion, documentacion"},
    "cocina": {"desc": "arte y ciencia de preparar alimentos", "use": "nutricion, arte, negocio"},
    "viajes": {"desc": "actividad de desplazarse a otros lugares", "use": "turismo, cultura, negocios"},
    "lectura": {"desc": "actividad de interpretar textos escritos", "use": "educacion, entretenimiento, crecimiento"},
    "meditacion": {"desc": "practica de entrenar la atencion y conciencia", "use": "salud mental, relajacion, foco"},
    "yoga": {"desc": "practica fisica y mental de origen indio", "use": "salud, flexibilidad, equilibrio"},
    "senderismo": {"desc": "actividad de caminar por naturaleza", "use": "ejercicio, aventura, desconectar"},
    "pintura": {"desc": "arte de aplicar color sobre superficies", "use": "expresion artistica, decoracion"},
    "baile": {"desc": "actividad ritmica del cuerpo", "use": "ejercicio, arte, socializacion"},
    "natacion": {"desc": "deporte de desplazarse en agua", "use": "ejercicio, competicion, salud"},
    "ajedrez": {"desc": "juego de estrategia entre dos jugadores", "use": "pensamiento logico, competicion"},
    "jardineria": {"desc": "cultivo de plantas y cuidado del jardin", "use": "bienestar, alimentacion, decoracion"},
    "fotografia_paisaje": {"desc": "fotografia enfocada en naturaleza", "use": "arte, turismo, documental"},
    "programacion_movil": {"desc": "desarrollo de apps para smartphones", "use": "negocios, entretenimiento, servicios"},
    "realidad_virtual": {"desc": "tecnologia que simula entornos inmersivos", "use": "juegos, entrenamiento, terapia"},
    "energia_solar": {"desc": "tecnologia para convertir luz solar en electricidad", "use": "energia limpia, hogar, industria"},
    "robotica": {"desc": "ciencia de disenar y construir robots", "use": "manufactura, salud, exploracion"},
    "blockchain": {"desc": "tecnologia de registro distribuido", "use": "finanzas, contratos, trazabilidad"},
    "impresion_3d": {"desc": "tecnologia de fabricacion aditiva", "use": "manufactura, medicina, prototipos"},
    "ia_generativa": {"desc": "IA que crea contenido nuevo (texto, imagenes, codigo)", "use": "creatividad, productividad, desarrollo"},
    "edge_computing": {"desc": "procesamiento de datos cerca del origen", "use": "IoT, latencia baja, privacidad"},
    "computacion_cuantica": {"desc": "computacion usando principios cuanticos", "use": "criptografia, optimizacion, simulacion"},
    "energia_eolica": {"desc": "generacion de electricidad con viento", "use": "energia renovable, sostenibilidad"},
    "agricultura_vertical": {"desc": "cultivo en capas apiladas en interiores", "use": "alimentacion urbana, sostenibilidad"},
    "exploracion_espacial": {"desc": "investigacion y viaje fuera de la Tierra", "use": "ciencia, tecnologia, inspiracion"},
    "neurociencia": {"desc": "estudio del sistema nervioso y cerebro", "use": "medicina, IA, psicologia"},
    "sostenibilidad": {"desc": "practicas que preservan recursos para el futuro", "use": "medio ambiente, negocios, politica"},
    "diseno_ux": {"desc": "experiencia de usuario en productos digitales", "use": "apps, web, software"},
    "emprendimiento": {"desc": "crear y gestionar un negocio propio", "use": "negocios, innovacion, economia"},
    "liderazgo": {"desc": "habilidad de guiar e inspirar a otros", "use": "management, politica, vida personal"},
    "comunicacion": {"desc": "proceso de intercambiar informacion", "use": "relaciones, negocios, educacion"},
    "productividad": {"desc": "eficiencia en el uso del tiempo y recursos", "use": "trabajo, estudios, vida personal"},
    "finanzas_personales": {"desc": "gestion del dinero personal", "use": "ahorro, inversiones, deudas"},
    "inversiones": {"desc": "asignacion de capital para generar rendimiento", "use": "bolsa, bienes raices, criptomonedas"},
    "marketing_digital": {"desc": "promocion de productos/servicios en canales digitales", "use": "negocios, emprendimiento, branding"},
    "relaciones": {"desc": "conexion entre personas", "use": "vida personal, trabajo, familia"},
    "creatividad": {"desc": "capacidad de generar ideas originales", "use": "arte, negocios, resolucion de problemas"},
    "pensamiento_critico": {"desc": "analisis objetivo de informacion", "use": "decisiones, academia, negocios"},
    "inteligencia_emocional": {"desc": "capacidad de reconocer y gestionar emociones", "use": "liderazgo, relaciones, salud mental"},
    "aprendizaje_autodidacta": {"desc": "aprender por cuenta propia sin profesor", "use": "educacion, carrera, hobbies"},
    "gestion_del_tiempo": {"desc": "organizar el tiempo de forma eficiente", "use": "trabajo, estudios, vida personal"},
    "oratoria": {"desc": "habilidad de hablar en publico", "use": "negocios, politica, educacion"},
    "escritura_creativa": {"desc": "arte de crear textos literarios", "use": "literatura, guiones, blogs"},
    "fotografia_retrato": {"desc": "fotografia enfocada en personas", "use": "arte, moda, retratos familiares"},
    "musica_produccion": {"desc": "creacion y mezcla de musica digital", "use": "industria musical, streaming, content creation"},
    "cine": {"desc": "arte de crear peliculas", "use": "entretenimiento, arte, comunicacion"},
    "videojuegos": {"desc": "juegos electronicos interactivos", "use": "entretenimiento, educacion, deportes electronico"},
    "anime_manga": {"desc": "arte narrativo japonés", "use": "entretenimiento, arte, cultura"},
    "literatura": {"desc": "arte de la escritura creativa", "use": "cultura, educacion, entretenimiento"},
    "historia": {"desc": "estudio de eventos pasados humanos", "use": "educacion, comprension del presente"},
    "filosofia": {"desc": "reflexion fundamental sobre existencia, conocimiento, moral", "use": "pensamiento, etica, decisiones"},
    "psicologia": {"desc": "ciencia del comportamiento y la mente", "use": "salud mental, relaciones, negocios"},
    "economia": {"desc": "estudio de produccion, distribucion y consumo", "use": "politica, negocios, inversiones"},
    "astronomia": {"desc": "estudio de cuerpos celestes y el universo", "use": "ciencia, exploracion, inspiracion"},
    "quimica": {"desc": "ciencia de la materia y sus transformaciones", "use": "medicina, industria, medio ambiente"},
}

# ============================================================
# CONTEXTUAL RESPONSES - Topic-aware, no generic templates
# ============================================================

def get_response(topic, question_type, kbase):
    """Generate topic-specific response"""
    
    if question_type == "que_es":
        if "desc" in kbase:
            return f"{topic.capitalize()} es {kbase['desc']}."
        return f"{topic.capitalize()} es un campo interesante con muchas aplicaciones practicas."
    
    elif question_type == "para_que_sirve":
        if "use" in kbase:
            return f"Se usa principalmente en {kbase['use']}."
        return f"Se aplica en multiples areas y tiene muchas utilidades practicas."
    
    elif question_type == "ejemplos":
        if "examples" in kbase:
            ej = random.choice(kbase["examples"])
            return f"Un ejemplo real es {ej}."
        return f"Hay muchos ejemplos practicos en la vida cotidiana."
    
    elif question_type == "ventajas":
        if "pros" in kbase:
            ventajas = random.sample(kbase["pros"], min(3, len(kbase["pros"])))
            return f"Algunas ventajas son: {', '.join(ventajas)}."
        return f"Tiene varias ventajas que lo hacen valioso."
    
    elif question_type == "desventajas":
        if "cons" in kbase:
            desventajas = random.sample(kbase["cons"], min(3, len(kbase["cons"])))
            return f"Algunas limitaciones son: {', '.join(desventajas)}."
        return f"Tiene algunas limitaciones a considerar."
    
    elif question_type == "consejos":
        if "tips" in kbase:
            tip = random.choice(kbase["tips"])
            return f"Un consejo importante: {tip}."
        return f"Lo mas importante es practicar y ser constante."
    
    elif question_type == "errores":
        if "errors" in kbase:
            err = random.choice(kbase["errors"])
            return f"Un error comun es {err}. Para evitarlo, revisa la documentacion y practica."
        return f"Los errores son parte del aprendizaje. Lo importante es aprender de ellos."
    
    elif question_type == "comparacion":
        if "comparisons" in kbase and len(kbase["comparisons"]) > 0:
            other = random.choice(list(kbase["comparisons"].keys()))
            comp = kbase["comparisons"][other]
            return f"Comparado con {other}: {comp}."
        return f"Cada herramienta tiene sus ventajas segun el caso de uso."
    
    elif question_type == "aprender":
        return f"Para aprender {topic}, empieza con lo basico y practica a diario. Hay muchos recursos gratuitos en linea."
    
    elif question_type == "futuro":
        return f"El futuro de {topic} es prometedor, con avances constantes y nuevas aplicaciones emergiendo."
    
    elif question_type == "historia":
        return f"La historia de {topic} es rica y ha evolucionado significativamente con el tiempo."
    
    elif question_type == "curiosidad":
        return f"Un dato curioso es que {topic} tiene aplicaciones que van mas alla de lo que imaginamos."
    
    else:
        # Generic but coherent
        aspects = []
        if "desc" in kbase:
            aspects.append(kbase["desc"])
        if "use" in kbase:
            aspects.append(f"Se usa en {kbase['use']}")
        if aspects:
            return f"{topic.capitalize()}: {'. '.join(aspects)}."
        return f"{topic.capitalize()} es un tema interesante que vale la pena explorar."

# ============================================================
# QUESTION GENERATOR - Natural variety
# ============================================================

def get_question_type():
    """Random question type with weights"""
    types = [
        ("que_es", 20), ("para_que_sirve", 15), ("ejemplos", 12),
        ("ventajas", 10), ("desventajas", 8), ("consejos", 12),
        ("errores", 8), ("comparacion", 10), ("aprender", 10),
        ("futuro", 5), ("historia", 5), ("curiosidad", 5),
    ]
    total = sum(w for _, w in types)
    r = random.uniform(0, total)
    cumulative = 0
    for qt, w in types:
        cumulative += w
        if r <= cumulative:
            return qt
    return "que_es"

# ============================================================
# CONVERSATION GENERATOR
# ============================================================

def gen_conversation():
    """Generate one natural multi-turn conversation"""
    
    # Pick topic
    all_topics = list(KNOWLEDGE.keys()) + list(GENERIC_TOPICS.keys())
    topic = random.choice(all_topics)
    kbase = KNOWLEDGE.get(topic, GENERIC_TOPICS.get(topic, {}))
    
    # Pick question type
    q_type = get_question_type()
    
    # Generate first question naturally
    greetings = [
        f"Que es {topic}?", f"Para que sirve {topic}?",
        f"Como funciona {topic}?", f"Quiero saber sobre {topic}",
        f"Hablame de {topic}", f"Que me cuentas de {topic}?",
        f"Necesito info sobre {topic}", f"Quiero aprender {topic}",
        f"Como se usa {topic}?", f"Cuentame sobre {topic}",
    ]
    
    first_q = random.choice(greetings)
    
    turns = [{"role": "user", "content": first_q}]
    
    # Generate 3-6 turns
    num_turns = random.randint(3, 6)
    
    for i in range(num_turns - 1):
        # Assistant responds with topic-specific content
        response = get_response(topic, q_type, kbase)
        turns.append({"role": "assistant", "content": response})
        
        # User follow-up (if not last turn)
        if i < num_turns - 2:
            followup_types = [
                f"Y que mas?", f"Interesante, tell me more",
                f"Que ventajas tiene?", f"Y desventajas?",
                f"Como aprendo mas?", f"Algun consejo?",
                f"Que errores cometo?", f"Y comparado con otra cosa?",
                f"Que futuro tiene?", f"Cuenta mas",
                f"Entiendo, que mas?", f"Ok, y que mas?",
            ]
            fu = random.choice(followup_types)
            turns.append({"role": "user", "content": fu})
            
            # Change question type for variety
            q_type = get_question_type()
    
    return turns, topic

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("CAMEL-AI Quality Generator v2")
    print("=" * 60)
    
    TARGET = 500000
    NUM_FILES = 30
    OUTPUT_DIR = "D:/Inteligente/rubidium-api/resources"
    
    pairs = []
    seen = set()
    
    print(f"Generating {TARGET:,} unique conversations...")
    
    while len(pairs) < TARGET:
        conv, topic = gen_conversation()
        
        # Dedup by first message
        key = conv[0]["content"].lower()
        if key not in seen:
            seen.add(key)
            pairs.append({"turns": conv, "topic": topic})
            
            if len(pairs) % 100000 == 0 and len(pairs) > 0:
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
            for item in batch:
                for turn in item["turns"]:
                    role = "U" if turn["role"] == "user" else "B"
                    f.write(f"{role}: {turn['content']}\n")
                f.write("\n")
        
        size_kb = os.path.getsize(filepath) // 1024
        print(f"  {filename}: {len(batch):,} convs ({size_kb} KB)")
        start = end
    
    # Save JSONL
    jsonl_path = os.path.join(OUTPUT_DIR, "soc_500k_v2.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # Stats
    all_user = [t["content"] for item in pairs for t in item["turns"] if t["role"] == "user"]
    all_bot = [t["content"] for item in pairs for t in item["turns"] if t["role"] == "assistant"]
    total_chars = sum(len(x) for x in all_user) + sum(len(x) for x in all_bot)
    
    # Vocabulary richness
    all_words = " ".join(all_bot).split()
    unique_words = set(w.lower() for w in all_words)
    ttr = len(unique_words) / len(all_words) if all_words else 0
    
    print(f"\n{'=' * 60}")
    print(f"COMPLETADO: {len(pairs):,} conversations")
    print(f"User avg: {sum(len(x) for x in all_user) // len(all_user)} chars")
    print(f"Bot avg: {sum(len(x) for x in all_bot) // len(all_bot)} chars")
    print(f"Vocabulary TTR: {ttr:.3f}")
    print(f"Total chars: {total_chars:,}")
    print(f"Est. tokens: ~{total_chars // 4:,}")
    print(f"JSONL: {jsonl_path}")

if __name__ == "__main__":
    main()
