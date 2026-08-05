#!/usr/bin/env python3
"""CAMEL-AI 500K Generator - Fast version without strict dedup"""
import json, random, os, re
from pathlib import Path

random.seed(42)

TOPICS = [
    "python","javascript","typescript","rust","go","java","c++","react","vue","angular",
    "svelte","next.js","django","flask","fastapi","express","docker","kubernetes","aws",
    "azure","gcp","postgresql","mysql","mongodb","redis","graphql","rest","grpc",
    "git","ci/cd","machine learning","deep learning","nlp","computer vision","tensorflow",
    "pytorch","linux","bash","algoritmos","estructuras de datos","patrones de diseño",
    "clean code","solid","tdd","inteligencia artificial","blockchain","ciberseguridad",
    "calculo","algebra","estadistica","fisica","quimica","biologia","astronomia",
    "neurociencia","genetica","filosofia","historia","psicologia","economia",
    "salud","nutricion","ejercicio","finanzas","emprendimiento","marketing",
    "fotografia","musica","arte","educacion","productividad","liderazgo",
]

PERSONAS = {
    "estudiante": {"g":["Hola","Oye","Ey","Qué tal","Buenas","Perdona","Hey","Disculpa"],
        "q":["Que es {c}?","Como funciona {c}?","Para que sirve {c}?","Como aprender {c}?",
            "Que necesito para {c}?","Me enseñas {c}?","Duda sobre {c}","No entiendo {c}",
            "Explícame {c}","Ejemplo de {c}?","Tutorial de {c}?","Consejos para {c}?",
            "Error en {c}","No me funciona {c}","Ayuda con {c}","Diferencia entre {c} y {c2}?",
            "Cuando usar {c}?","Ventajas de {c}?","Desventajas de {c}?","Que opinas de {c}?",
            "Vale la pena aprender {c}?","Proyecto con {c}?","Donde practicar {c}?"]},
    "profesional": {"g":["Necesito","Busco","Quiero","Como puedo","Cual es la mejor forma de"],
        "q":["Optimizar {c}","Arquitectura con {c}","Escalabilidad de {c}","Seguridad en {c}",
            "Rendimiento de {c}","Monitoreo de {c}","CI/CD para {c}","Testing en {c}",
            "Refactoring de {c}","Migración a {c}","Patrones para {c}","Anti-patrones en {c}",
            "Benchmark de {c}","Producción con {c}","Mejores prácticas en {c}","Code review {c}",
            "Deploy de {c}","Debugging en {c}","Performance tuning {c}","Load balancing {c}"]},
    "curioso": {"g":["Hola","Oye","Qué tal","Buenas","Cuéntame","Hablemos"],
        "q":["Que es {c}?","Por qué es importante {c}?","Como funciona {c}?","Historia de {c}?",
            "Aplicaciones de {c}?","Futuro de {c}?","Curiosidades de {c}?","Dato sobre {c}?",
            "Impacto de {c}?","Relación entre {c} y {c2}?","Investigación en {c}?","Teoría de {c}?"]},
    "frustrado": {"g":["Error","No funciona","Problema","Bug","Crash","Fallo","Ayuda","Urgente"],
        "q":["Error en {c}","No carga {c}","Se cuelga {c}","Timeout en {c}","Exception en {c}",
            "Memory leak {c}","CPU al 100% en {c}","No compila {c}","Syntax error en {c}",
            "Import error en {c}","Connection refused {c}","Permission denied {c}",
            "File not found {c}","500 error en {c}","404 en {c}","Crash en {c}"]},
    "creativo": {"g":["Escribe","Crea","Inventa","Imagina","Genera","Diseña","Compón"],
        "q":["Historia sobre {c}","Poema sobre {c}","Personaje de {c}","Guion sobre {c}",
            "Idea para {c}","Concepto creativo de {c}","Mundo de {c}","Aventura de {c}",
            "Misterio de {c}","Comedia sobre {c}","Drama de {c}","Romance de {c}"]},
    "docente": {"g":["Explícame","Enseña","Concepto de","Definición de","Clase sobre"],
        "q":["Qué es {c}?","Definición de {c}","Ejemplo de {c}?","Ejercicio de {c}",
            "Paso a paso {c}?","Tutorial de {c}?","Problema de {c}?","Solución de {c}",
            "Aplicación de {c}?","Teorema de {c}?","Fórmula de {c}?","Cálculo de {c}?"]},
    "casual": {"g":["Qué tal","Cómo estás","Oye","Cuéntame","Qué onda","Hola"],
        "q":["Qué opinas de {c}?","Recomiendas {c}?","Vale la pena {c}?","Qué hay de nuevo en {c}?",
            "Cómo está {c}?","Cuál es tu opinión de {c}?","Qué tal {c}?","Cuéntame sobre {c}",
            "Hablemos de {c}","Qué sabes de {c}?","Opinión de {c}?","Review de {c}?"]},
    "experto": {"g":["Analiza","Compara","Evalúa","Investiga","Revisa","Examina"],
        "q":["State of the art en {c}","Benchmark de {c}","Paper sobre {c}","Teoría de {c}",
            "Optimización de {c}","Arquitectura de {c}","Implementación de {c}","Comparativa de {c}",
            "Evaluación de {c}","Análisis de {c}","Complejidad de {c}","Rendimiento de {c}"]},
}

RESP_TEMPLATES = [
    "{c} es {exp}. Se usa en {caso}.",
    "Para entender {c}, sabes que {exp}. Ejemplo: {ej}.",
    "{c} destaca por {vent}. A diferencia de {c2}, {dif}.",
    "Los expertos recomiendan {cons} con {c}. Evita {err}.",
    "El futuro de {c} es {tend}. {ej}.",
    "{c} se usa en {caso}. Ventaja: {vent}.",
    "Aprender {c} toma {tiemp}. Importante: {imp}.",
    "La diferencia entre {c} y {c2} es {dif}.",
    "En {caso}, {c} es ideal porque {raz}.",
    "Error comun: {err}. Mejor: {mej}.",
    "La comunidad de {c} es activa. {ej}.",
    "Para empezar {c}, necesitas {nec}.",
    "Consejo: {cons}. Te ahorrara {tiemp}.",
    "Muchos fallan con {err}. Mejor {mej}.",
    "El ecosistema de {c} tiene {vent}.",
    "Comparado con {c2}, {c} es {vent}.",
    "En produccion, {c} necesita {cons}.",
    "Escalabilidad de {c}: {vent}.",
    "Optimizar {c}: {cons}.",
    "Testing en {c}: {cons}.",
    "{c} es {exp}. Aplicacion: {caso}.",
    "Importancia de {c}: {imp}. Sin el, {consq}.",
    "Dato: {ej}. Demuestra {vent}.",
    "Historia de {c}: {hist}.",
    "Impacto de {c}: {imp}.",
    "{c} estudia {obj}. Relevancia: {vent}.",
    "Cientificos hallaron que {c} permite {acc}.",
    "En {camp}, {c} es clave porque {raz}.",
    "Teoria de {c}: {exp_simple}.",
    "Investigacion en {c}: implicaciones en {aplic}.",
    "Impacto social de {c}: {imp_soc}.",
    "Clave de {c}: {cons}.",
    "Idea principal: {concepto}. Ej: {ej}.",
    "Basico de {c}: {concepto}.",
    "Error comun: {err}. Correcto: {corr}.",
    "Entiendo. {c} puede ser dificil. {sol}.",
    "Es normal. {c} se resuelve con {cons}.",
    "Tranquilo. {c} toma {tiemp}. {imp}.",
    "Poco a poco. {cons}. Valdra la pena.",
]

def fill(t, p):
    r = t
    for k,v in p.items(): r = r.replace("{"+k+"}", str(v))
    return re.sub(r'\{[a-z_0-9]+\}', 'algo', r)

def gen_conv():
    uk = random.choice(list(PERSONAS.keys()))
    up = PERSONAS[uk]
    c1 = random.choice(TOPICS)
    c2 = random.choice([t for t in TOPICS if t!=c1] or [c1])
    
    p = {"c":c1,"c2":c2,
        "exp":random.choice(["es moderno","es potente","es esencial","es versatil","es popular"]),
        "caso":random.choice(["web","datos","backend","frontend","cloud","mobile","startup"]),
        "ej":random.choice(["Netflix lo usa","Google lo usa","Amazon lo usa","Spotify lo usa","Meta lo usa"]),
        "vent":random.choice(["velocidad","escalabilidad","simplicidad","flexibilidad","seguridad","rendimiento"]),
        "dif":random.choice(["mayor rendimiento","mas simple","mas seguro","mejor ecosistema"]),
        "cons":random.choice(["tests","documentacion","practica","code review","convenciones"]),
        "err":random.choice(["no validar","hardcodear","ignorar errores","no testear","copiar"]),
        "tend":random.choice(["IA","edge","serverless","microservicios","wasm"]),
        "tiemp":random.choice(["semanas","1-3 meses","6 meses","1 ano"]),
        "imp":random.choice(["practicar","constancia","proyectos","documentacion"]),
        "raz":random.choice(["ecosistema","comunidad","versatilidad","rendimiento"]),
        "mej":random.choice(["basico","diario","docs","proyectos"]),
        "nec":random.choice(["ganas","computador","tiempo","internet"]),
        "hist":random.choice(["años 90","2000s","reciente","evolucion"]),
        "obj":random.choice(["fenomenos","procesos","datos","sistemas"]),
        "camp":random.choice(["ciencia","ingenieria","medicina","tech"]),
        "aplic":random.choice(["medicina","energia","espacio","industria"]),
        "consq":random.choice(["no avance","no entienda","no exista"]),
        "concepto":random.choice(["funcion","algoritmo","estructura","patron"]),
        "corr":random.choice(["validar","testear","documentar"]),
        "sol":random.choice(["reiniciar","logs","actualizar"]),
        "acc":random.choice(["crear","automatizar","analizar","optimizar"]),
        "imp_soc":random.choice(["transforma","crea empleos","cambia vidas"]),
        "exp_simple":random.choice(["como funciona","por que existe","origen"]),
    }
    
    num_turns = random.randint(3, 8)
    turns = []
    
    # User first
    g = random.choice(up["g"])
    q = fill(random.choice(up["q"]), p)
    turns.append({"role":"user","content":f"{g} {q}".strip()})
    
    # Alternar
    for i in range(num_turns - 1):
        # Assistant
        resp = fill(random.choice(RESP_TEMPLATES), p)
        turns.append({"role":"assistant","content":resp.strip()})
        
        # User follow-up
        if i < num_turns - 2:
            fu = random.choice(["Y {c2}?",f"Qué {c2}?",f"Hablando de {c2}",
                f"Ok, pero {c2}?",f"Interesante. {c2}?",f"Entiendo. {c2}?",
                f"Vale. {c2}?",f"Bien. {c2}?"])
            turns.append({"role":"user","content":fill(fu,p).strip()})
    
    return turns

# MAIN
print("Generating 500K CAMEL-AI pairs...")
pairs = []
for i in range(500000):
    pairs.append(gen_conv())
    if (i+1) % 100000 == 0:
        print(f"  {i+1:,} / 500,000")

random.shuffle(pairs)

OUTPUT = "D:/Inteligente/rubidium-api/resources"
os.makedirs(OUTPUT, exist_ok=True)

per = len(pairs) // 30
rem = len(pairs) % 30
start = 0
print(f"\nSplitting {len(pairs):,} pairs into 30 files...")

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
    print(f"  {fn}: {len(batch):,} convs ({sz} KB)")
    start = end

# JSONL
jl = os.path.join(OUTPUT, "soc_500k.jsonl")
with open(jl, "w", encoding="utf-8") as f:
    for c in pairs: f.write(json.dumps(c,ensure_ascii=False)+"\n")

print(f"\nDONE: {len(pairs):,} CAMEL-AI conversations")
print(f"JSONL: {jl} ({os.path.getsize(jl)//1024//1024} MB)")
