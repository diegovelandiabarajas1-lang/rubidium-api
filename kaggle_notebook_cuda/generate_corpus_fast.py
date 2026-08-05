#!/usr/bin/env python3
"""
RUBIDIUM - Generación Rápida de Corpus (CPU)
Template Expansion - 1K-100K pares en segundos
Sin dependencias GPU, funciona en cualquier entorno
"""
import json
import random
import os
from pathlib import Path

# ============================================================
# TEMPLATES POR TEMA
# ============================================================
TEMPLATES = {
    "tecnologia": {
        "preguntas": [
            "¿Qué es {concepto}?",
            "¿Cómo funciona {concepto}?",
            "¿Para qué sirve {concepto}?",
            "¿Cuáles son las ventajas de {concepto}?",
            "¿Cómo aprender {concepto}?",
            "¿Qué herramientas uso para {concepto}?",
            "¿Diferencia entre {concepto} y {otro}?",
            "¿Mejores prácticas en {concepto}?",
            "¿Errores comunes en {concepto}?",
            "¿Futuro de {concepto}?"
        ],
        "conceptos": [
            "Python", "JavaScript", "machine learning", "Docker", "Kubernetes",
            "APIs REST", "GraphQL", "bases de datos SQL", "MongoDB", "Redis",
            "ciberseguridad", "blockchain", "cloud computing", "DevOps", "Git",
            "React", "Vue", "Node.js", "TypeScript", "Rust", "Go",
            "microservicios", "serverless", "CI/CD", "testing", "clean code"
        ],
        "respuestas": [
            "{concepto} es una tecnología fundamental que permite {accion}. Se usa ampliamente en {caso_uso}.",
            "Para entender {concepto}, primero debes saber que {explicacion}. En la práctica, {ejemplo}.",
            "{concepto} destaca por {ventaja}. A diferencia de {otro}, ofrece {diferencia}.",
            "Los expertos recomiendan {consejo} al trabajar con {concepto}. Evita {error_comun}.",
            "El futuro de {concepto} apunta hacia {tendencia}. Las empresas ya están adoptando {novedad}."
        ],
        "acciones": ["automatizar tareas", "procesar datos", "crear aplicaciones", "escalar sistemas", "optimizar rendimiento"],
        "casos_uso": ["desarrollo web", "ciencia de datos", "backend", "frontend", "DevOps", "mobile"],
        "explicaciones": ["es un lenguaje de programación", "es un framework", "es una arquitectura", "es un patrón de diseño"],
        "ejemplos": ["Netflix lo usa para streaming", "Google lo usa en búsqueda", "Startups lo usan para MVP"],
        "ventajas": ["velocidad", "escalabilidad", "simplicidad", "rendimiento", "mantenibilidad"],
        "otros": ["alternativas tradicionales", "versiones anteriores", "competidores directos"],
        "diferencias": ["mejor rendimiento", "más simple", "más seguro", "más rápido de desarrollar"],
        "consejos": ["usar tipos estáticos", "escribir tests", "documentar código", "hacer code review"],
        "errores_comunes": ["no validar entrada", "hardcodear valores", "ignorar errores", "sobreingeniería"],
        "tendencias": ["IA generativa", "edge computing", "WebAssembly", "Rust en backend"],
        "novedades": ["nuevas versiones", "estándares actualizados", "herramientas mejoradas"]
    },
    
    "ciencia": {
        "preguntas": [
            "¿Qué es {concepto}?",
            "¿Por qué es importante {concepto}?",
            "¿Cómo se descubrió {concepto}?",
            "¿Aplicaciones de {concepto}?",
            "¿Diferencia entre {concepto} y {otro}?"
        ],
        "conceptos": [
            "ADN", "evolución", "agujeros negros", "materia oscura", "fotosíntesis",
            "mecánica cuántica", "relatividad", "células madre", "vacunas", "cambio climático",
            "neuronas", "fotosíntesis", "tabla periódica", "enlaces químicos", "termodinámica"
        ],
        "respuestas": [
            "{concepto} es un fenómeno fundamental en {campo}. {explicacion}.",
            "La importancia de {concepto} radica en {importancia}. Sin él, {consecuencia}.",
            "Se descubrió cuando {descubrimiento}. Esto cambió {impacto}.",
            "Aplicaciones incluyen {aplicaciones}. En el futuro, {futuro}."
        ],
        "campos": ["biología", "física", "química", "astronomía", "neurociencia"],
        "explicaciones": ["contiene información genética", "explana la diversidad de vida", "son regiones de gravedad extrema"],
        "importancias": ["entender la vida", "explicar el universo", "desarrollar medicinas"],
        "consecuencias": ["no habría herencia", "no entenderíamos el cosmos", "no tendríamos tecnología moderna"],
        "descubrimientos": ["Watson y Crick identificaron su estructura", "Einstein publicó su teoría", "Científicos observaron radiación"],
        "impactos": ["la biología moderna", "la física teórica", "la medicina"],
        "aplicaciones": ["medicina personalizada", "energía limpia", "exploración espacial"],
        "futuros": ["terapias génicas", "computación cuántica", "colonización espacial"]
    },
    
    "cotidiano": {
        "preguntas": [
            "¿Cómo {accion}?",
            "¿Qué necesito para {accion}?",
            "¿Cuánto tiempo toma {accion}?",
            "¿Consejos para {accion}?",
            "¿Errores al {accion}?"
        ],
        "conceptos": [
            "cocinar sano", "hacer ejercicio", "ahorrar dinero", "dormir mejor",
            "reducir estrés", "aprender idioma", "organizar casa", "leer más",
            "meditar", "planificar semana", "beber agua", "caminar diario"
        ],
        "respuestas": [
            "Para {accion}, lo clave es {clave}. Empieza por {paso1}, luego {paso2}.",
            "Necesitas {necesario}. No requiere {no_necesario}. Lo importante es {importante}.",
            "Toma {tiempo} si {condicion}. La consistencia {resultado}.",
            "Mi consejo: {consejo}. Evita {evitar}. La paciencia {resultado_final}.",
            "Error común: {error}. Mejor: {mejor}."
        ],
        "claves": ["constancia", "planificación", "simplicidad", "disfrute", "progreso gradual"],
        "pasos": ["definir objetivo", "crear rutina", "empezar pequeño", "medir progreso", "ajustar"],
        "necesarios": ["motivación", "tiempo", "espacio", "herramientas básicas", "conocimiento básico"],
        "no_necesarios": ["equipo caro", "suscripciones", "perfección", "mucho tiempo libre"],
        "importantes": ["hábito diario", "disfrutar proceso", "no rendirse", "celebrar logros"],
        "tiempos": ["15-30 min", "unas semanas", "21 días para hábito", "3 meses para resultados"],
        "condiciones": ["eres constante", "te organizas", "disfrutas el proceso"],
        "resultados": ["mejora gradual", "hábito sólido", "cambio duradero", "bienestar"],
        "consejos": ["empezar hoy", "no buscar perfección", "disfrutar camino", "ser amable contigo"],
        "evitar": ["compararse", "exigirse demasiado", "rendirse pronto", "buscar atajos"],
        "resultados_finales": ["vale la pena", "transforma vida", "crea libertad"],
        "errores": ["querer todo ya", "no planificar", "compararse", "rendirse"],
        "mejores": ["empezar pequeño", "ser constante", "disfrutar", "aprender de errores"]
    }
}

# Frases de relleno para variar respuestas
CONECTORES = [
    "Además,", "Por otro lado,", "En resumen,", "Cabe mencionar que,",
    "Es importante notar que,", "Por ejemplo,", "En la práctica,",
    "Los expertos coinciden en que,", "La evidencia sugiere que,"
]

def generar_respuesta(tema, concepto, plantilla):
    """Genera respuesta rellenando plantilla"""
    respuesta = random.choice(plantilla["respuestas"])
    
    # Rellenar placeholders
    respuesta = respuesta.replace("{concepto}", concepto)
    respuesta = respuesta.replace("{accion}", concepto)
    respuesta = respuesta.replace("{campo}", random.choice(plantilla.get("campos", ["la ciencia"])))
    respuesta = respuesta.replace("{explicacion}", random.choice(plantilla.get("explicaciones", ["es fundamental"])))
    respuesta = respuesta.replace("{accion}", random.choice(plantilla.get("acciones", ["crear"])))
    respuesta = respuesta.replace("{caso_uso}", random.choice(plantilla.get("casos_uso", ["la tecnología"])))
    respuesta = respuesta.replace("{ventaja}", random.choice(plantilla.get("ventajas", ["eficiencia"])))
    respuesta = respuesta.replace("{otro}", random.choice(plantilla.get("otros", ["alternativas"])))
    respuesta = respuesta.replace("{diferencia}", random.choice(plantilla.get("diferencias", ["mejora significativa"])))
    respuesta = respuesta.replace("{consejo}", random.choice(plantilla.get("consejos", ["practicar"])))
    respuesta = respuesta.replace("{error_comun}", random.choice(plantilla.get("errores_comunes", ["no planificar"])))
    respuesta = respuesta.replace("{tendencia}", random.choice(plantilla.get("tendencias", ["innovación"])))
    respuesta = respuesta.replace("{novedad}", random.choice(plantilla.get("novedades", ["herramientas nuevas"])))
    respuesta = respuesta.replace("{importancia}", random.choice(plantilla.get("importancias", ["el conocimiento"])))
    respuesta = respuesta.replace("{consecuencia}", random.choice(plantilla.get("consecuencias", ["no avanzaríamos"])))
    respuesta = respuesta.replace("{descubrimiento}", random.choice(plantilla.get("descubrimientos", ["investigaciones"])))
    respuesta = respuesta.replace("{impacto}", random.choice(plantilla.get("impactos", ["el campo"])))
    respuesta = respuesta.replace("{aplicaciones}", random.choice(plantilla.get("aplicaciones", ["múltiples usos"])))
    respuesta = respuesta.replace("{futuro}", random.choice(plantilla.get("futuros", ["avances"])))
    respuesta = respuesta.replace("{clave}", random.choice(plantilla.get("claves", ["constancia"])))
    respuesta = respuesta.replace("{paso1}", random.choice(plantilla.get("pasos", ["empezar"])))
    respuesta = respuesta.replace("{paso2}", random.choice(plantilla.get("pasos", ["continuar"])))
    respuesta = respuesta.replace("{necesario}", random.choice(plantilla.get("necesarios", ["poco"])))
    respuesta = respuesta.replace("{no_necesario}", random.choice(plantilla.get("no_necesarios", ["mucho"])))
    respuesta = respuesta.replace("{importante}", random.choice(plantilla.get("importantes", ["constancia"])))
    respuesta = respuesta.replace("{tiempo}", random.choice(plantilla.get("tiempos", ["poco"])))
    respuesta = respuesta.replace("{condicion}", random.choice(plantilla.get("condiciones", ["eres constante"])))
    respuesta = respuesta.replace("{resultado}", random.choice(plantilla.get("resultados", ["mejora"])))
    respuesta = respuesta.replace("{consejo}", random.choice(plantilla.get("consejos", ["empezar hoy"])))
    respuesta = respuesta.replace("{evitar}", random.choice(plantilla.get("evitar", ["perfeccionismo"])))
    respuesta = respuesta.replace("{resultado_final}", random.choice(plantilla.get("resultados_finales", ["vale la pena"])))
    respuesta = respuesta.replace("{error}", random.choice(plantilla.get("errores", ["apresurarse"])))
    respuesta = respuesta.replace("{mejor}", random.choice(plantilla.get("mejores", ["paso a paso"])))
    
    # Agregar conector aleatorio al inicio a veces
    if random.random() < 0.3:
        respuesta = random.choice(CONECTORES) + " " + respuesta[0].lower() + respuesta[1:]
    
    return respuesta

def generar_pares(tema, num_pares):
    """Genera N pares para un tema"""
    plantilla = TEMPLATES[tema]
    conceptos = plantilla["conceptos"]
    preguntas = plantilla["preguntas"]
    
    pares = []
    for _ in range(num_pares):
        concepto = random.choice(conceptos)
        pregunta_template = random.choice(preguntas)
        pregunta = pregunta_template.format(concepto=concepto, otro=random.choice(conceptos), accion=concepto)
        respuesta = generar_respuesta(tema, concepto, plantilla)
        
        pares.append({
            "user": pregunta,
            "bot": respuesta,
            "topic": tema
        })
    
    return pares

def main():
    print("=" * 60)
    print("RUBIDIUM - Generación Rápida Corpus (CPU)")
    print("=" * 60)
    
    # Configuración
    PARES_POR_TEMA = 50  # 3 temas × 50 = 150 pares base
    TEMAS_USAR = list(TEMPLATES.keys())  # tecnologia, ciencia, cotidiano
    
    print(f"Temas: {TEMAS_USAR}")
    print(f"Pares por tema: {PARES_POR_TEMA}")
    print(f"Total estimado: {len(TEMAS_USAR) * PARES_POR_TEMA}")
    
    # Generar
    all_pares = []
    for tema in TEMAS_USAR:
        print(f"\nGenerando {tema}...")
        pares = generar_pares(tema, PARES_POR_TEMA)
        all_pares.extend(pares)
        print(f"  {len(pares)} pares")
    
    # Mezclar
    random.shuffle(all_pares)
    
    # Guardar JSONL
    output_jsonl = "corpus_generated_1k.jsonl"
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for p in all_pares:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    
    # Guardar TXT formato U:/B:
    output_txt = "corpus_generated_1k.txt"
    with open(output_txt, 'w', encoding='utf-8') as f:
        for p in all_pares:
            f.write(f"U: {p['user']}\n")
            f.write(f"B: {p['bot']}\n\n")
    
    # Estadísticas
    print(f"\n{'='*60}")
    print(f"COMPLETADO: {len(all_pares)} pares generados")
    print(f"Archivos:")
    print(f"  - {output_jsonl} ({os.path.getsize(output_jsonl)/1024:.1f} KB)")
    print(f"  - {output_txt} ({os.path.getsize(output_txt)/1024:.1f} KB)")
    
    # Stats
    user_lens = [len(p['user']) for p in all_pares]
    bot_lens = [len(p['bot']) for p in all_pares]
    topics = set(p['topic'] for p in all_pares)
    
    print(f"\nEstadísticas:")
    print(f"  Temas: {len(topics)}")
    print(f"  Usuario: {min(user_lens)}-{max(user_lens)} chars (promedio {sum(user_lens)//len(user_lens)})")
    print(f"  Bot: {min(bot_lens)}-{max(bot_lens)} chars (promedio {sum(bot_lens)//len(bot_lens)})")
    print(f"  Total chars: {sum(user_lens)+sum(bot_lens):,}")

if __name__ == "__main__":
    main()