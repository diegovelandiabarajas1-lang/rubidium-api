#!/usr/bin/env python3
"""
Rubidium - 500K SOC Corpus Generator
Generates 500,000 U:/B: pairs using template expansion
Splits into 30 files (chat_06.txt to chat_35.txt)
"""
import json
import random
import os
import hashlib
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

TARGET_PAIRS = 500000
NUM_FILES = 30  # chat_06.txt to chat_35.txt
OUTPUT_DIR = "D:/Inteligente/rubidium-api/resources"

# Seed for reproducibility
random.seed(42)

# ============================================================
# EXPANDED TEMPLATES (10x more variations)
# ============================================================

PERSONAS = {
    "estudiante_uni": {
        "questions": [
            "Que es {c}?",
            "Como funciona {c}?",
            "Para que sirve {c}?",
            "Ventajas de {c}?",
            "Como aprender {c}?",
            "Errores comunes en {c}?",
            "Mejores practicas en {c}?",
            "Diferencia entre {c} y {c2}?",
            "Futuro de {c}?",
            "Ejemplos de {c}?",
            "Por que usar {c}?",
            "Cuando usar {c}?",
            "Limitaciones de {c}?",
            "Comparar {c} con {c2}",
            "Tutorial basico de {c}",
            "Recurso para aprender {c}",
            "Proyecto para practicar {c}",
            "Errores de principiante en {c}",
            "Como se usa {c} en la practica?",
            "Que necesito para empezar con {c}?",
        ],
        "concepts": [
            "Python", "JavaScript", "TypeScript", "Rust", "Go", "C++", "Java",
            "React", "Vue", "Angular", "Next.js", "Django", "Flask", "FastAPI",
            "Node.js", "Docker", "Kubernetes", "AWS", "Azure", "GCP",
            "PostgreSQL", "MongoDB", "Redis", "GraphQL", "REST", "Git",
            "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
            "TensorFlow", "PyTorch", "HTML", "CSS", "Tailwind", "Linux",
            "Algoritmos", "Estructuras de datos", "Patrones de diseño",
            "Clean Code", "SOLID", "TDD", "CI/CD", "Docker Compose",
        ],
        "responses": [
            "{c} es una herramienta poderosa que permite {accion}. En la practica, se usa para {caso}.",
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
            "Para empezar con {c}, necesitas {necesario}. El primer paso es {paso_simple}.",
            "Un consejo clave: {consejo}. Esto te ahorrara {tiempo}.",
            "Muchos principiantes cometen {error}. En su lugar, {mejor}.",
            "El ecosistema de {c} incluye {ventajas}. Ideal para {caso}.",
        ],
        "actions": ["crear aplicaciones", "automatizar procesos", "analizar datos",
                    "desarrollar software", "construir APIs", "gestionar infraestructura"],
        "cases": ["desarrollo web", "ciencia de datos", "backend", "frontend",
                  "DevOps", "mobile", "cloud", "startup", "empresa grande"],
        "explanations": ["es un lenguaje de programacion", "es un framework moderno",
                         "es una arquitectura de software", "es un patron de diseno"],
        "examples": ["Netflix lo usa para streaming", "Google lo usa en busqueda",
                     "Spotify lo usa para recomendaciones", "Amazon lo usa en e-commerce"],
        "advantages": ["velocidad", "escalabilidad", "simplicidad", "rendimiento",
                       "mantenibilidad", "flexibilidad", "seguridad"],
        "differences": ["mejor rendimiento", "mas simple", "mas seguro", "mas rapido"],
        "advices": ["usar tipos estaticos", "escribir tests", "documentar codigo",
                    "hacer code review", "seguir convenciones", "practicar a diario"],
        "errors": ["no validar entrada", "hardcodear valores", "ignorar errores",
                   "sobreingenieria", "no testear", "copiar sin entender"],
        "trends": ["IA generativa", "edge computing", "WebAssembly", "Rust en backend",
                   "serverless", "microservicios", "event-driven"],
        "times": ["unas semanas", "1-3 meses", "6 meses para dominar", "1 ano para experto"],
        "importants": ["practicar a diario", "construir proyectos", "leer documentacion"],
        "reasons": ["su ecosistema es enorme", "tiene gran comunidad", "es muy versatil"],
        "necessaries": ["ganas de aprender", "un computador", "tiempo diario"],
        "mejors": ["empezar con lo basico", "practicar a diario", "leer documentacion"],
        "paso_simpls": ["instalar Python", "hacer un proyecto simple", "leer la documentacion"],
    },
    "profesional_tech": {
        "questions": [
            "Como optimizar {c}?",
            "Arquitectura de {c}?",
            "Escalabilidad de {c}?",
            "Seguridad en {c}?",
            "Rendimiento de {c}?",
            "Monitoreo de {c}?",
            "CI/CD para {c}?",
            "Testing en {c}?",
            "Refactoring de {c}?",
            "Migracion a {c}?",
            "Mejores practicas en {c}?",
            "Patrones para {c}?",
            "Anti-patrones en {c}?",
            "Benchmark de {c}?",
            "Produccion con {c}?",
        ],
        "concepts": [
            "microservicios", "event-driven", "CQRS", "Domain-Driven Design",
            "Kubernetes", "Docker", "Terraform", "Ansible", "Prometheus",
            "Grafana", "ELK Stack", "Kafka", "RabbitMQ", "Redis",
            "PostgreSQL", "MongoDB", "Cassandra", "Elasticsearch",
            "Nginx", "HAProxy", "Envoy", "Istio", "Linkerd",
            "GitHub Actions", "GitLab CI", "Jenkins", "ArgoCD",
            "Load balancing", "Circuit breaker", "Rate limiting",
            "API gateway", "Service mesh", "Observability",
        ],
        "responses": [
            "Para {c}, la clave es {consejo}. En produccion, {ejemplo}.",
            "La arquitectura de {c} se basa en {ventaja}. Ejemplo: {ejemplo}.",
            "Escalar {c} requiere {escalabilidad}. {ejemplo}.",
            "En seguridad, {c} necesita {seguridad}. {razon}.",
            "El rendimiento de {c} es {rendimiento}. Comparado con {c2}, {comparacion}.",
            "Monitorear {c} es {importante}. Herramientas: {ejemplo}.",
            "CI/CD para {c} implica {consejo}. {ejemplo}.",
            "Testing en {c}: {consejo}. Evita {error}.",
            "Refactorizar {c} requiere {consejo}. Paso a paso: {paso1}.",
            "Migrar a {c} toma {tiempo}. Considera {consejo}.",
            "En {c}, un anti-patron comun es {error}. Mejor: {mejor}.",
            "El patron {c} resuelve {problema}. Ejemplo: {ejemplo}.",
            "Para {c} en produccion, necesitas {necesario}. {ejemplo}.",
            "La escalabilidad de {c} es {escalabilidad}. Caso real: {ejemplo}.",
            "Observabilidad en {c} incluye {consejo}. {ejemplo}.",
        ],
        "consejos": ["implementar caching", "usar connection pooling", "configurar timeouts",
                     "implementar circuit breakers", "usar rate limiting", "monitorear latencia"],
        "errores": ["ignorar timeouts", "no usar caching", "acoplar demasiado",
                    "ignorar errores", "no monitorear", "hardcodear config"],
        "mejors": ["usar patrones probados", "implementar desde el inicio", "documentar decisiones"],
        "ventajas": ["flexibilidad", "escalabilidad", "resiliencia"],
        "seguridades": ["autenticacion robusta", "encriptacion", "validacion de entrada"],
        "rendimientos": ["optimizado", "escalable", "eficiente"],
        "escalabilidades": ["horizontal", "vertical", "elastica"],
        "tiempos": ["1-2 sprints", "1-3 meses", "6 meses para migracion completa"],
        "problemas": ["consistencia de datos", "latencia", "disponibilidad"],
        "necessaries": ["monitoreo", "logging", "alertas", "documentacion"],
    },
    "curioso_general": {
        "questions": [
            "Que es {c}?",
            "Por que es importante {c}?",
            "Como funciona {c}?",
            "Aplicaciones de {c}?",
            "Historia de {c}?",
            "Diferencia entre {c} y {c2}?",
            "Futuro de {c}?",
            "Impacto de {c} en la sociedad?",
            "Curiosidades de {c}?",
            "Dato interesante sobre {c}?",
        ],
        "concepts": [
            "evolucion", "agujeros negros", "materia oscura", "fotosintesis",
            "mecanica cuantica", "relatividad", "ADN", "vacunas",
            "cambio climatico", "neuronas", "cosmologia", "genetica",
            "nanotecnologia", "biotecnologia", "inteligencia artificial",
            "energia solar", "fusion nuclear", "particulas subatomicas",
            "antimateria", "quimica organica", "neurociencia", "astronomia",
        ],
        "responses": [
            "{c} es un fenomeno fundamental en {campo}. {explicacion}.",
            "La importancia de {c} radica en {importancia}. Sin el, {consecuencia}.",
            "Se descubrio cuando {descubrimiento}. Esto cambio {impacto}.",
            "Aplicaciones incluyen {aplicaciones}. En el futuro, {futuro}.",
            "{c} estudia {objeto}. Su relevancia es {ventaja}.",
            "Un dato curioso: {ejemplo}. Esto demuestra {ventaja}.",
            "La historia de {c} es fascinante. {historia}.",
            "El impacto de {c} en la sociedad es {impacto_social}.",
        ],
        "campos": ["biologia", "fisica", "quimica", "astronomia", "neurociencia"],
        "explicaciones": ["contiene informacion genetica", "explica la diversidad de vida",
                          "es la base de la materia", "gobierna el comportamiento cuantico"],
        "importancias": ["entender la vida", "explicar el universo", "desarrollar medicinas"],
        "consecuencias": ["no habria herencia", "no entenderiamos el cosmos",
                          "no tendriamos tecnologia moderna"],
        "descubrimientos": ["Watson y Crick identificaron su estructura",
                            "Einstein publico su teoria", "Cientificos observaron radiacion"],
        "impactos": ["la biologia moderna", "la fisica teorica", "la medicina"],
        "objetos": ["fenomenos naturales", "procesos biologicos", "estructuras materiales"],
        "aplicaciones": ["medicina personalizada", "energia limpia", "exploracion espacial"],
        "futuros": ["terapias geneticas", "computacion cuantica", "colonizacion espacial"],
        "historias": ["comenzo con observaciones simples", "evoluciono con nuevas teorias"],
        "impactos_social": ["transforma industrias", "cambia formas de vida"],
    },
    "usuario_frustrado": {
        "questions": [
            "Error: {c}",
            "No funciona {c}",
            "Ayuda con {c}",
            "Problema con {c}",
            "Bug en {c}",
            "Crash en {c}",
            "Timeout en {c}",
            "Fallo {c}",
            "No carga {c}",
            "Se cuelga {c}",
            "Error 500 en {c}",
            "Exception en {c}",
            "Memory leak en {c}",
            "CPU al 100% en {c}",
            "Disco lleno en {c}",
        ],
        "concepts": [
            "Python", "JavaScript", "Docker", "Kubernetes", "AWS",
            "nginx", "PostgreSQL", "MongoDB", "Redis", "React",
            "Node.js", "Flask", "Django", "FastAPI", "Spring Boot",
            "MySQL", "Elasticsearch", "Kafka", "RabbitMQ", "Git",
        ],
        "responses": [
            "El error {c} usualmente significa {explicacion}. Solucion: {solucion}.",
            "Para {c}, prueba {solucion}. Si no funciona, {alternativa}.",
            "Es un problema comun con {c}. {solucion}.",
            "Revisa {paso1}. Si persiste, {paso2}.",
            "El error {c} puede ser causado por {causa}. Intenta {solucion}.",
            "Solucion rapida: {solucion}. Para prevenir: {prevencion}.",
            "Primero verifica {paso1}. Despues {paso2}. Finalmente {paso3}.",
            "Este error ocurre cuando {causa}. La solucion es {solucion}.",
        ],
        "soluciones": ["reiniciar el servicio", "verificar logs", "revisar configuracion",
                       "actualizar dependencias", "limpiar cache", "revisar permisos"],
        "causas": ["falta de memoria", "configuracion incorrecta", "dependencia corrupta",
                   "puerto ocupado", "permisos insuficientes", "version incompatible"],
        "pasos": ["verificar logs", "reiniciar servicio", "revisar config",
                  "actualizar", "reinstalar", "contactar soporte"],
        "prevenciones": ["monitorear recursos", "usar Docker", "implementar health checks"],
    },
    "creativo": {
        "questions": [
            "Escribe sobre {c}",
            "Crea un personaje de {c}",
            "Inventa una historia de {c}",
            "Genera ideas sobre {c}",
            "Dibuja {c}",
            "Diseña {c}",
            "Crea un poema sobre {c}",
            "Escribe un guion sobre {c}",
            "Imagina {c}",
            "Crea contenido sobre {c}",
        ],
        "concepts": [
            "ciencia ficcion", "fantasia", "misterio", "romance", "aventura",
            "terror", "comedia", "drama", "historia", "poesia",
            "cine", "musica", "arte", "fotografia", "diseno",
            "videojuegos", "animacion", "teatro", "danza", "literatura",
        ],
        "responses": [
            "Para {c}, la idea central es {idea}. Desarrolla {desarrollo}.",
            "Un concepto unico para {c}: {idea}. Esto crea {efecto}.",
            "En {c}, lo importante es {idea}. Ejemplo: {ejemplo}.",
            "La creatividad en {c} se basa en {idea}. Aplica {consejo}.",
            "Para {c}, sugiero {idea}. Desarrolla con {consejo}.",
            "Una historia de {c} podria empezar con {idea}. Despues {desarrollo}.",
            "El estilo de {c} se define por {idea}. Usa {consejo}.",
        ],
        "ideas": ["un mundo distopico", "un detective brillante", "una historia de amor",
                  "un viaje en el tiempo", "una aventura epica", "un misterio por resolver"],
        "desarrollos": ["personajes complejos", "giros inesperados", "mundos ricos"],
        "efectos": ["tension", "suspense", "emocion", "reflexion"],
        "consejos": ["ser original", "crear conflicto", "desarrollar personajes"],
        "ejemplos": ["como en las mejores peliculas", "con profundidad narrativa"],
    },
    "docente_tutor": {
        "questions": [
            "Explica {c}",
            "Enseña {c}",
            "Como entender {c}",
            "Pasos para {c}",
            "Ejercicio de {c}",
            "Ejemplo de {c}",
            "Concepto de {c}",
            "Duda sobre {c}",
            "Problema de {c}",
            "Solucion de {c}",
        ],
        "concepts": [
            "calculo", "algebra", "fisica", "quimica", "programacion",
            "estadistica", "logica", "matematicas", "geometria", "trigonometria",
            "base de datos", "redes", "sistemas operativos", "compiladores",
            "inteligencia artificial", "machine learning", "analisis de datos",
        ],
        "responses": [
            "Para entender {c}, veamos paso a paso: {paso1}.",
            "El concepto clave de {c} es {concepto}. Ejemplo: {ejemplo}.",
            "{c} se resume en {concepto}. Practica con {ejercicio}.",
            "La idea principal de {c} es {concepto}. Para practicar: {ejercicio}.",
            "Empecemos con lo basico de {c}: {concepto}. Ejemplo: {ejemplo}.",
            "Un error comun es {error}. La forma correcta es {correcto}.",
            "La definicion de {c} es {concepto}. Ejemplo practico: {ejercicio}.",
        ],
        "conceptos": ["una funcion matematica", "un algoritmo", "una estructura de datos",
                      "un patron de diseno", "un concepto abstracto"],
        "ejemplos": ["calcula la derivada de x^2", "resolve este problema logico",
                     "implementa una lista enlazada", "analiza este dataset"],
        "ejercicios": ["resolver 5 problemas", "implementar un algoritmo",
                       "analizar un caso practico", "dibujar un diagrama"],
        "errores": ["no revisar units", "olvidar border cases", "no validar entrada"],
        "correctos": ["siempre validar", "testear edge cases", "documentar codigo"],
        "pasos": ["primero entiende la teoria", "despues practica", "finalmente domina"],
    },
    "usuario_casual": {
        "questions": [
            "Que tal {c}?",
            "Opinion sobre {c}?",
            "Que opinas de {c}?",
            "Recomiendas {c}?",
            "Vale la pena {c}?",
            "Como esta {c}?",
            "Que hay de nuevo en {c}?",
            "Cual es tu opinion sobre {c}?",
            "Cuentame sobre {c}",
            "Hablemos de {c}",
        ],
        "concepts": [
            "Python", "JavaScript", "React", "Vue", "Angular",
            "Docker", "Kubernetes", "AWS", "Linux", "Git",
            "machine learning", "inteligencia artificial", "blockchain",
            "realidad virtual", "5G", "internet de las cosas",
            "coches electricos", "energia solar", "espacio", "ciencia",
        ],
        "responses": [
            "Sobre {c}: {opinion}. Es interesante porque {razon}.",
            "{c} es {opinion}. {ejemplo}.",
            "Mi opinion: {opinion}. {ejemplo}.",
            "{c} es {opinion}. La razon es {razon}.",
            "Interesante pregunta sobre {c}. {opinion}. {razon}.",
            "{c} es {opinion}. Un ejemplo: {ejemplo}.",
            "La verdad, {c} es {opinion}. {razon}.",
        ],
        "opiniones": ["muy interesante", "bastante util", "increible", "sorprendente",
                      "revolucionario", "practico", "fascinante", "util"],
        "razones": ["su comunidad es activa", "tiene muchas aplicaciones",
                    "es el futuro", "es muy versatil", "es facil de aprender"],
        "ejemplos": ["Netflix lo usa", "Google lo usa", "Amazon lo usa", "Tesla lo usa"],
    },
    "experto_dominio": {
        "questions": [
            "Analisis de {c}",
            "State of the art en {c}",
            "Benchmark de {c}",
            "Paper sobre {c}",
            "Teoria de {c}",
            "Optimizacion de {c}",
            "Arquitectura de {c}",
            "Implementacion de {c}",
            "Comparativa de {c}",
            "Evaluacion de {c}",
        ],
        "concepts": [
            "transformers", "attention mechanism", "GAN", "VAE", "diffusion models",
            "reinforcement learning", "graph neural networks", "federated learning",
            "quantum computing", "neuromorphic computing", "optical computing",
            "CRISPR", "gene therapy", "mRNA vaccines", "personalized medicine",
            "fusion energy", "superconductors", "photovoltaics", "batteries",
        ],
        "responses": [
            "Segun el paper de {referencia}, {c} demuestra {resultado}.",
            "El state of the art en {c} es {avance}. Benchmark: {benchmark}.",
            "La arquitectura de {c} se basa en {fundamento}. Ventajas: {ventajas}.",
            "En {c}, el problema principal es {problema}. Soluciones: {soluciones}.",
            "La teoria de {c} establece que {afirmacion}. Implicaciones: {implicaciones}.",
            "Optimizar {c} requiere {enfoque}. Resultados: {resultados}.",
            "Comparativa: {c} vs {c2} en {criterio}. {c} es {ventaja}.",
            "La implementacion de {c} necesita {requisitos}. Caso real: {caso}.",
        ],
        "referencias": ["Vaswani et al.", "Hinton et al.", "LeCun et al.", "Bengio et al."],
        "resultados": ["mejores metricas", "mayor eficiencia", "mejor generalizacion"],
        "avances": ["nuevas arquitecturas", "mejores algoritmos", "eficiencia computacional"],
        "benchmarks": ["95% accuracy", "10x faster", "50% less memory"],
        "fundamentos": ["attention is all you need", "backpropagation", "gradient descent"],
        "problemas": ["escalabilidad", "costo computacional", "generalizacion"],
        "soluciones": ["paralelismo", "quantization", "distillation"],
        "afirmaciones": ["el attention es suficiente", "los datos son la clave"],
        "implicaciones": ["nuevos modelos", "mejor rendimiento", "menor costo"],
        "enfoques": ["arquitecturas eficientes", "entrenamiento distribuido"],
        "resultados": ["SOA en multiple benchmarks", "mejor trade-off accuracy/cost"],
        "criterios": ["precision", "velocidad", "memoria", "costo"],
        "requisitos": ["GPU clusters", "grandes datasets", "expertise"],
        "casos": ["GPT-4", "Claude", "Gemini", "Llama"],
    },
}

# ============================================================
# COMPLEMENTARY TOPICS
# ============================================================

TOPICS = [
    "programacion_python", "programacion_javascript", "programacion_rust",
    "web_dev_backend", "web_dev_frontend", "devops_cloud",
    "machine_learning", "deep_learning", "inteligencia_artificial",
    "bases_datos", "seguridad_informatica", "redes",
    "ciencia_fisica", "ciencia_biologia", "ciencia_quimica",
    "matematicas_avanzadas", "estadistica", "probabilidad",
    "filosofia_moderna", "historia_mundial", "psicologia",
    "salud_mental", "nutricion", "ejercicio_fisico",
    "finanzas_personales", "emprendimiento", "marketing_digital",
    "diseno_ux", "fotografia", "musica", "literatura",
    "cine_series", "videojuegos", "tecnologia_consumidor",
    "educacion_online", "idiomas", "productividad",
    "liderazgo", "comunicacion", "relaciones",
]

# ============================================================
# FILL TEMPLATE
# ============================================================

def fill_template(template, params):
    """Fill template with params, replacing missing placeholders."""
    result = template
    for key, val in params.items():
        result = result.replace("{" + key + "}", str(val))
    # Remove remaining placeholders
    import re
    result = re.sub(r'\{[a-z_0-9]+\}', 'algo', result)
    return result

# ============================================================
# GENERATE PAIRS
# ============================================================

def generate_pair():
    """Generate a single U:/B: pair."""
    # Pick random persona and topic
    persona_key = random.choice(list(PERSONAS.keys()))
    persona = PERSONAS[persona_key]
    topic = random.choice(TOPICS)
    
    # Pick question and concept
    q_template = random.choice(persona["questions"])
    c1 = random.choice(persona["concepts"])
    c2 = random.choice([c for c in persona["concepts"] if c != c1] or [c1])
    
    # Build params
    params = {
        "c": c1, "c2": c2,
        "accion": random.choice(persona.get("actions", ["hacer cosas"])),
        "caso": random.choice(persona.get("cases", ["casos generales"])),
        "explicacion": random.choice(persona.get("explanations", ["es util"])),
        "ejemplo": random.choice(persona.get("examples", ["ejemplo basico"])),
        "ventaja": random.choice(persona.get("advantages", ["ventajas"])),
        "diferencia": random.choice(persona.get("differences", ["diferencia"])),
        "consejo": random.choice(persona.get("advices", persona.get("consejos", ["practicar"]))),
        "error": random.choice(persona.get("errors", persona.get("errores", ["errores"]))),
        "tendencia": random.choice(persona.get("trends", ["tendencias"])),
        "tiempo": random.choice(persona.get("times", ["tiempo"])),
        "importante": random.choice(persona.get("importants", ["importante"])),
        "razon": random.choice(persona.get("reasons", ["razon"])),
        "mejor": random.choice(persona.get("mejors", ["mejor opcion"])),
        "necesario": random.choice(persona.get("necessaries", ["necesario"])),
        "paso_simple": random.choice(persona.get("paso_simpls", ["paso basico"])),
        "solucion": random.choice(persona.get("soluciones", ["solucion generica"])),
        "causa": random.choice(persona.get("causas", ["causa desconocida"])),
        "paso1": random.choice(persona.get("pasos", ["verificar logs"])),
        "paso2": random.choice(persona.get("pasos", ["reiniciar"])),
        "paso3": random.choice(persona.get("pasos", ["contactar soporte"])),
        "prevencion": random.choice(persona.get("prevenciones", ["monitorear"])),
        "idea": random.choice(persona.get("ideas", ["idea creativa"])),
        "desarrollo": random.choice(persona.get("desarrollos", ["desarrollo"])),
        "efecto": random.choice(persona.get("efectos", ["efecto"])),
        "concepto": random.choice(persona.get("conceptos", ["concepto"])),
        "ejercicio": random.choice(persona.get("ejercicios", ["ejercicio"])),
        "correcto": random.choice(persona.get("correctos", ["forma correcta"])),
        "opinion": random.choice(persona.get("opiniones", ["opinion"])),
        "referencia": random.choice(persona.get("referencias", ["investigadores"])),
        "resultado": random.choice(persona.get("resultados", ["resultado"])),
        "avance": random.choice(persona.get("avances", ["avance"])),
        "benchmark": random.choice(persona.get("benchmarks", ["benchmark"])),
        "fundamento": random.choice(persona.get("fundamentos", ["fundamento"])),
        "problema": random.choice(persona.get("problemas", ["problema"])),
        "soluciones": random.choice(persona.get("soluciones", ["solucion"])),
        "afirmacion": random.choice(persona.get("afirmaciones", ["afirmacion"])),
        "implicaciones": random.choice(persona.get("implicaciones", ["implicacion"])),
        "enfoque": random.choice(persona.get("enfoques", ["enfoque"])),
        "requisitos": random.choice(persona.get("requisitos", ["requisito"])),
        "caso_real": random.choice(persona.get("casos", ["caso real"])),
        "campo": random.choice(persona.get("campos", ["ciencia"])),
        "objeto": random.choice(persona.get("objetos", ["objeto"])),
        "aplicaciones": random.choice(persona.get("aplicaciones", ["aplicaciones"])),
        "futuro": random.choice(persona.get("futuros", ["futuro"])),
        "historia": random.choice(persona.get("historias", ["historia"])),
        "impacto_social": random.choice(persona.get("impactos_social", ["impacto"])),
        "consecuencia": random.choice(persona.get("consecuencias", ["consecuencia"])),
        "descubrimiento": random.choice(persona.get("descubrimientos", ["descubrimiento"])),
        "impacto": random.choice(persona.get("impactos", ["impacto"])),
        "importancia": random.choice(persona.get("importances", ["importancia"])),
        "ventajas": random.choice(persona.get("ventajas", ["ventajas"])),
        "seguridad": random.choice(persona.get("seguridades", ["seguridad"])),
        "rendimiento": random.choice(persona.get("rendimientos", ["rendimiento"])),
        "escalabilidad": random.choice(persona.get("escalabilidades", ["escalabilidad"])),
        "comparacion": random.choice(persona.get("comparisons", ["comparacion"])),
        "benchmark": random.choice(persona.get("benchmarks", ["benchmark"])),
    }
    
    user = fill_template(q_template, params)
    bot = fill_template(random.choice(persona["responses"]), params)
    
    return {"user": user.strip(), "bot": bot.strip(), "topic": topic, "persona": persona_key}

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("RUBIDIUM - 500K SOC Corpus Generator")
    print("=" * 60)
    
    # Generate pairs
    pairs = []
    seen = set()
    
    print(f"Generating {TARGET_PAIRS} unique pairs...")
    while len(pairs) < TARGET_PAIRS:
        pair = generate_pair()
        key = pair["user"].lower()
        if key not in seen:
            seen.add(key)
            pairs.append(pair)
            if len(pairs) % 50000 == 0:
                print(f"  {len(pairs):,} / {TARGET_PAIRS:,} pairs")
    
    print(f"\nTotal unique pairs: {len(pairs):,}")
    
    # Shuffle
    random.shuffle(pairs)
    
    # Split into files
    pairs_per_file = len(pairs) // NUM_FILES
    remainder = len(pairs) % NUM_FILES
    
    print(f"\nSplitting into {NUM_FILES} files...")
    
    file_index = 0
    start = 0
    for i in range(NUM_FILES):
        end = start + pairs_per_file + (1 if i < remainder else 0)
        batch = pairs[start:end]
        
        filename = f"chat_{6 + i:02d}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            for pair in batch:
                f.write(f"U: {pair['user']}\n")
                f.write(f"B: {pair['bot']}\n\n")
        
        size_kb = os.path.getsize(filepath) // 1024
        print(f"  {filename}: {len(batch):,} pairs ({size_kb} KB)")
        
        start = end
        file_index += 1
    
    # Save JSONL consolidated
    jsonl_path = os.path.join(OUTPUT_DIR, "soc_500k.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    
    # Save compact JSON
    compact_path = os.path.join(OUTPUT_DIR, "soc_500k_compact.json")
    compact = [{"u": p["user"], "b": p["bot"]} for p in pairs]
    with open(compact_path, "w", encoding="utf-8") as f:
        json.dump(compact, f, ensure_ascii=False)
    
    # Stats
    user_lens = [len(p["user"]) for p in pairs]
    bot_lens = [len(p["bot"]) for p in pairs]
    total_chars = sum(user_lens) + sum(bot_lens)
    
    print(f"\n{'=' * 60}")
    print(f"COMPLETE: {len(pairs):,} pairs generated")
    print(f"Files:")
    print(f"  chat_06.txt to chat_{5 + NUM_FILES:02d}.txt")
    print(f"  soc_500k.jsonl ({os.path.getsize(jsonl_path) / 1024 / 1024:.1f} MB)")
    print(f"  soc_500k_compact.json ({os.path.getsize(compact_path) / 1024 / 1024:.1f} MB)")
    print(f"\nStats:")
    print(f"  User: {min(user_lens)}-{max(user_lens)} chars (avg {sum(user_lens)//len(user_lens)})")
    print(f"  Bot: {min(bot_lens)}-{max(bot_lens)} chars (avg {sum(bot_lens)//len(bot_lens)})")
    print(f"  Total chars: {total_chars:,}")
    print(f"  Est. tokens: ~{total_chars // 4:,}")
    print(f"\nOutput: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
