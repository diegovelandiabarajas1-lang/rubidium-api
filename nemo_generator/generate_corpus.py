#!/usr/bin/env python3
"""
RUBIDIUM - Generador de Corpus de Alta Calidad
NeMo Data Designer Style - 10 archivos x 1000 pares U:/B:
CON CHECKPOINTS Y GUARDADO SEGURO
"""
import json
import os
import sys
import time
import random
from typing import List, Dict, Tuple
from datetime import datetime

# ============================================================
# CONFIGURACIÓN
# ============================================================
TOPICS = [
    {
        "name": "tecnologia",
        "filename": "chat_tecnologia_1k.txt",
        "system_prompt": "Eres un asistente técnico experto en tecnología. Responde de forma clara, concisa y útil.",
        "themes": [
            "inteligencia artificial", "programación", "ciberseguridad", "cloud computing",
            "blockchain", "realidad virtual", "internet de las cosas", "5G",
            "computación cuántica", "robótica", "automatización", "devops",
            "bases de datos", "redes", "sistemas operativos", "móviles",
            "videojuegos", "software libre", "startups", "transformación digital"
        ],
        "question_types": [
            "¿Qué es {theme}?",
            "¿Cómo funciona {theme}?",
            "¿Cuáles son los usos de {theme}?",
            "¿Qué ventajas tiene {theme}?",
            "¿Cómo se aplica {theme} en la industria?",
            "¿Qué carrera se necesita para {theme}?",
            "¿Cuáles son las tendencias de {theme}?",
            "¿Cómo aprender {theme}?",
            "¿Qué empresas lideran {theme}?",
            "¿Cuál es el futuro de {theme}?"
        ]
    },
    {
        "name": "ciencia",
        "filename": "chat_ciencia_1k.txt",
        "system_prompt": "Eres un científico apasionado que explica conceptos científicos de forma accesible.",
        "themes": [
            "física cuántica", "evolución", "astronomía", "genética",
            "neurociencia", "química orgánica", "ecología", "geología",
            "oceanografía", "meteorología", "paleontología", "microbiología",
            "astrofísica", "cosmología", "biología molecular", "nanotecnología",
            "energía nuclear", "cambio climático", "espacio exterior", "materia oscura"
        ],
        "question_types": [
            "¿Qué es {theme}?",
            "¿Por qué es importante {theme}?",
            "¿Cómo se estudia {theme}?",
            "¿Cuáles son los descubrimientos recientes en {theme}?",
            "¿Qué impacto tiene {theme} en nuestra vida?",
            "¿Cómo explicarías {theme} a un niño?",
            "¿Qué científicos destacan en {theme}?",
            "¿Cuáles son los misterios de {theme}?",
            "¿Cómo se relaciona {theme} con otras ciencias?",
            "¿Cuál es el futuro de {theme}?"
        ]
    },
    {
        "name": "cotidiano",
        "filename": "chat_cotidiano_1k.txt",
        "system_prompt": "Eres un amigo cercano que da consejos prácticos para la vida diaria.",
        "themes": [
            "cocina saludable", "organización del hogar", "finanzas personales",
            "bienestar emocional", "ejercicio en casa", "meditación",
            "relaciones sociales", "productividad", "descanso y sueño",
            "alimentación平衡ada", "gestión del tiempo", "comunicación efectiva",
            "resolución de conflictos", "creatividad", "aprendizaje continuo",
            "viajes económicos", "sostenibilidad", "tecnología en el hogar",
            "moda y estilo", "cuidado personal"
        ],
        "question_types": [
            "¿Cómo puedo mejorar mi {theme}?",
            "¿Qué consejos me das sobre {theme}?",
            "¿Cuáles son los mejores trucos para {theme}?",
            "¿Cómo empezar con {theme}?",
            "¿Qué errores cometo con {theme}?",
            "¿Cuáles son los beneficios de {theme}?",
            "¿Cómo mantener la constancia en {theme}?",
            "¿Qué herramientas necesito para {theme}?",
            "¿Cuánto tiempo dedicar a {theme}?",
            "¿Cómo medir el progreso en {theme}?"
        ]
    },
    {
        "name": "educacion",
        "filename": "chat_educacion_1k.txt",
        "system_prompt": "Eres un educador experto que ayuda a aprender de forma efectiva.",
        "themes": [
            "técnicas de estudio", "memorización", "pensamiento crítico",
            "aprendizaje autodidacta", "idiomas", "matemáticas",
            "lectura comprensiva", "escritura creativa", "oratoria",
            "metodologías de enseñanza", "tecnología educativa",
            "educación inclusiva", "aprendizaje cooperativo",
            "evaluación formativa", "diseño curricular",
            "neuroeducación", "gamificación", "aprendizaje móvil",
            "educación emocional", "preparación de exámenes"
        ],
        "question_types": [
            "¿Cómo mejorar en {theme}?",
            "¿Qué técnicas existen para {theme}?",
            "¿Cuáles son los errores comunes en {theme}?",
            "¿Cómo enseñar {theme} de forma efectiva?",
            "¿Qué recursos hay para {theme}?",
            "¿Cómo evaluar el progreso en {theme}?",
            "¿Qué dicen las investigaciones sobre {theme}?",
            "¿Cómo adaptar {theme} a diferentes edades?",
            "¿Cuáles son las tendencias en {theme}?",
            "¿Cómo motivar a otros en {theme}?"
        ]
    },
    {
        "name": "salud",
        "filename": "chat_salud_1k.txt",
        "system_prompt": "Eres un profesional de la salud que da información confiable y práctica.",
        "themes": [
            "nutrición", "ejercicio físico", "salud mental", "prevención de enfermedades",
            "bienestar general", "sueño reparador", "manejo del estrés",
            "salud cardiovascular", "fortalecimiento inmunológico",
            "recuperación muscular", "hidratación", "suplementación",
            "postura corporal", "respiración", "yoga y pilates",
            "alimentación consciente", "ayuno intermitente", "salud digestiva",
            "salud ósea", "envejecimiento saludable"
        ],
        "question_types": [
            "¿Cómo mejorar mi {theme}?",
            "¿Qué hábitos debo adoptar para {theme}?",
            "¿Cuáles son los mitos sobre {theme}?",
            "¿Cómo prevenir problemas de {theme}?",
            "¿Qué alimentos ayudan con {theme}?",
            "¿Cuánto ejercicio necesito para {theme}?",
            "¿Cómo saber si tengo problemas de {theme}?",
            "¿Qué suplementos son útiles para {theme}?",
            "¿Cómo mantener la constancia en {theme}?",
            "¿Cuándo consultar a un profesional sobre {theme}?"
        ]
    },
    {
        "name": "negocios",
        "filename": "chat_negocios_1k.txt",
        "system_prompt": "Eres un consultor de negocios con experiencia en emprendimiento y management.",
        "themes": [
            "emprendimiento", "marketing digital", "ventas", "liderazgo",
            "gestión de equipos", "finanzas corporativas", "inversiones",
            "networking", "comunicación empresarial", "innovación",
            "transformación digital", "recursos humanos", "logística",
            "cadenas de suministro", "comercio exterior", "e-commerce",
            "branding", "customer experience", "agilidad empresarial",
            "sostenibilidad corporativa"
        ],
        "question_types": [
            "¿Cómo empezar un negocio de {theme}?",
            "¿Qué estrategias funcionan para {theme}?",
            "¿Cuáles son los errores comunes en {theme}?",
            "¿Cómo medir el éxito en {theme}?",
            "¿Qué herramientas necesito para {theme}?",
            "¿Cómo superar crisis en {theme}?",
            "¿Cuáles son las tendencias de {theme}?",
            "¿Cómo escalar un negocio de {theme}?",
            "¿Qué métricas debo observar en {theme}?",
            "¿Cómo innovar en {theme}?"
        ]
    },
    {
        "name": "cultura",
        "filename": "chat_cultura_1k.txt",
        "system_prompt": "Eres un apasionado de la cultura que comparte conocimientos sobre arte, historia y tradiciones.",
        "themes": [
            "arte contemporáneo", "literatura clásica", "cine independiente",
            "música del mundo", "teatro", "danza", "fotografía",
            "arquitectura", "diseño gráfico", "cultura pop",
            "mitología", "tradiciones populares", "festivales",
            "cine latinoamericano", "literatura hispanoamericana",
            "arte africano", "cultura asiática", "filosofía occidental",
            "cultura digital", "patrimonio cultural"
        ],
        "question_types": [
            "¿Qué es {theme}?",
            "¿Cuál es la historia de {theme}?",
            "¿Por qué es importante {theme}?",
            "¿Cómo apreciar {theme}?",
            "¿Qué obras destacan en {theme}?",
            "¿Cómo se relaciona {theme} con la sociedad?",
            "¿Cuáles son las corrientes de {theme}?",
            "¿Cómo crear {theme}?",
            "¿Qué artistas son referentes en {theme}?",
            "¿Cuál es el futuro de {theme}?"
        ]
    },
    {
        "name": "programacion",
        "filename": "chat_programacion_1k.txt",
        "system_prompt": "Eres un desarrollador senior que explica conceptos de programación de forma clara.",
        "themes": [
            "Python", "JavaScript", "Java", "C++", "Rust", "Go",
            "estructuras de datos", "algoritmos", "patrones de diseño",
            "bases de datos", "APIs REST", "GraphQL", "Docker",
            "Kubernetes", "CI/CD", "testing", "arquitectura de software",
            "machine learning", "desarrollo web", "desarrollo móvil"
        ],
        "question_types": [
            "¿Cómo aprender {theme}?",
            "¿Qué proyectos puedo hacer con {theme}?",
            "¿Cuáles son las mejores prácticas de {theme}?",
            "¿Cómo resolver problemas comunes en {theme}?",
            "¿Qué librerías son esenciales para {theme}?",
            "¿Cómo optimizar código en {theme}?",
            "¿Cuáles son los errores comunes en {theme}?",
            "¿Cómo preparar entrevistas de {theme}?",
            "¿Qué tendencias hay en {theme}?",
            "¿Cómo integrar {theme} con otros lenguajes?"
        ]
    },
    {
        "name": "matematicas",
        "filename": "chat_matematicas_1k.txt",
        "system_prompt": "Eres un matemático que hace las matemáticas accesibles y interesantes.",
        "themes": [
            "álgebra", "geometría", "cálculo", "estadística", "probabilidad",
            "teoría de números", "combinatoria", "lógica matemática",
            "análisis matemático", "álgebra lineal", "topología",
            "matemáticas discretas", "criptografía", "modelado matemático",
            "optimización", "ecuaciones diferenciales", "geometría analítica",
            "teoría de grafos", "matemáticas financieras", "inteligencia artificial"
        ],
        "question_types": [
            "¿Qué es {theme}?",
            "¿Para qué sirve {theme}?",
            "¿Cómo se resuelven problemas de {theme}?",
            "¿Cuáles son los conceptos clave de {theme}?",
            "¿Cómo se aplica {theme} en la vida real?",
            "¿Qué ejercicios me recomiendas para {theme}?",
            "¿Cuáles son los errores comunes en {theme}?",
            "¿Cómo explicar {theme} de forma simple?",
            "¿Qué herramientas digitales hay para {theme}?",
            "¿Cuál es la belleza de {theme}?"
        ]
    },
    {
        "name": "historia",
        "filename": "chat_historia_1k.txt",
        "system_prompt": "Eres un historiador que narra eventos históricos de forma engaging y educativa.",
        "themes": [
            "antigüedad clásica", "edad media", "renacimiento",
            "revolución industrial", "guerras mundiales", "guerra fría",
            "historia de América", "historia de Europa", "historia de Asia",
            "historia de África", "imperios antiguos", "revoluciones",
            "descubrimientos científicos", "historia del arte",
            "historia de la tecnología", "movimientos sociales",
            "historia económica", "historia militar", "historia cultural",
            "historia contemporánea"
        ],
        "question_types": [
            "¿Qué pasó en {theme}?",
            "¿Por qué es importante {theme}?",
            "¿Quiénes fueron los protagonistas de {theme}?",
            "¿Cuáles son las consecuencias de {theme}?",
            "Cómo se vivía en {theme}?",
            "¿Qué legado dejó {theme}?",
            "¿Cuáles son los mitos sobre {theme}?",
            "¿Cómo se relaciona {theme} con el presente?",
            "¿Qué fuentes hay sobre {theme}?",
            "¿Qué podemos aprender de {theme}?"
        ]
    }
]

# ============================================================
# GENERADOR DE PARES
# ============================================================
class CorpusGenerator:
    def __init__(self, topic_config: Dict):
        self.config = topic_config
        self.pairs = []
        self.used_combinations = set()
    
    def generate_question(self, theme: str, q_type: str) -> str:
        """Genera una pregunta variada"""
        question = q_type.format(theme=theme)
        # Agregar variación
        prefixes = [
            "", "Cuéntame sobre ", "Explícame ", "Háblame de ",
            "Quiero saber sobre ", "Necesito información sobre ",
            "¿Puedes explicarme ", "¿Me puedes decir "
        ]
        suffixes = [
            "", " por favor", " de forma sencilla", " con ejemplos",
            " en detalle", " para principiantes", " de manera práctica",
            " con casos reales"
        ]
        
        if random.random() < 0.3:  # 30% con prefijo
            question = random.choice(prefixes) + question.lower()
        if random.random() < 0.2:  # 20% con sufijo
            question = question.rstrip('?') + random.choice(suffixes) + '?'
        
        return question
    
    def generate_answer(self, theme: str, question: str) -> str:
        """Genera una respuesta de alta calidad"""
        # Plantillas de respuesta variadas
        templates = [
            f"Excelente pregunta sobre {theme}. ",
            f"Me encanta hablar de {theme}. ",
            f"{theme} es un tema fascinante. ",
            f"Permíteme explicarte sobre {theme}. ",
            f"{theme} tiene muchos aspectos interesantes. ",
            f"Para entender {theme}, primero debemos considerar ",
            f"{theme} es fundamental porque ",
            f"La clave de {theme} está en ",
            f"Un aspecto importante de {theme} es ",
            f"En resumen sobre {theme}: "
        ]
        
        body_templates = [
            f"Se trata de un concepto que combina conocimiento práctico y teórico. "
            f"En esencia, {theme} nos permite comprender mejor nuestro entorno y tomar mejores decisiones. "
            f"Los expertos coinciden en que dominar {theme} es valioso tanto personal como profesionalmente.",
            
            f"Es un campo en constante evolución que requiere actualización continua. "
            f"Los principios básicos de {theme} se han mantenido estables, pero su aplicación varía según el contexto. "
            f"Recomiendo empezar con lo fundamental y progresar gradualmente.",
            
            f"Una de las cosas más interesantes de {theme} es su diversidad de aplicaciones. "
            f"Desde lo cotidiano hasta lo especializado, {theme} tiene relevancia en múltiples áreas. "
            f"Lo importante es encontrar el enfoque que mejor se adapte a tus necesidades específicas.",
            
            f"La investigación reciente ha demostrado la importancia de {theme} en diversos campos. "
            f"Los estudios muestran que las personas que invierten tiempo en {theme} obtienen mejores resultados. "
            f"Es un tema que merece atención y práctica constante.",
            
            f"Existen diferentes perspectivas sobre {theme}, pero todas comparten puntos en común. "
            f"Lo fundamental es entender los principios básicos antes de explorar las complejidades. "
            f"Con dedicación y las herramientas adecuadas, cualquiera puede mejorar en {theme}."
        ]
        
        closing_templates = [
            "¿Te gustaría profundizar en algún aspecto específico?",
            "¿Hay algo más que quieras saber sobre este tema?",
            "¿Tienes alguna otra pregunta sobre {theme}?",
            "Espero que esta información te sea útil.",
            "No dudes en preguntar si necesitas más detalles.",
            "¿Qué opinas tú sobre esto?",
            "¿Te ha quedado alguna duda?",
            "Estoy aquí para ayudarte con cualquier otra consulta."
        ]
        
        answer = random.choice(templates) + random.choice(body_templates)
        answer = answer.replace("{theme}", theme)
        answer += random.choice(closing_templates).replace("{theme}", theme)
        
        return answer
    
    def generate_pair(self) -> Tuple[str, str]:
        """Genera un par U:/B: de alta calidad"""
        theme = random.choice(self.config["themes"])
        q_type = random.choice(self.config["question_types"])
        
        question = self.generate_question(theme, q_type)
        answer = self.generate_answer(theme, question)
        
        return question, answer
    
    def generate_pairs(self, count: int = 1000) -> List[Tuple[str, str]]:
        """Genera múltiples pares únicos"""
        self.pairs = []
        self.used_combinations = set()
        
        attempts = 0
        max_attempts = count * 3  # Límite para evitar loops infinitos
        
        while len(self.pairs) < count and attempts < max_attempts:
            attempts += 1
            question, answer = self.generate_pair()
            
            # Crear key para evitar duplicados
            key = question[:50].lower().strip()
            if key not in self.used_combinations:
                self.used_combinations.add(key)
                self.pairs.append((question, answer))
        
        return self.pairs
    
    def save_to_file(self, pairs: List[Tuple[str, str]], filepath: str):
        """Guarda los pares en formato U:/B:"""
        with open(filepath, 'w', encoding='utf-8') as f:
            for question, answer in pairs:
                f.write(f"U: {question}\n")
                f.write(f"B: {answer}\n\n")
        
        print(f"  Guardado: {filepath} ({len(pairs)} pares)")
        return len(pairs)

# ============================================================
# SISTEMA DE CHECKPOINTS
# ============================================================
class CheckpointManager:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(checkpoint_dir, "checkpoint.json")
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, completed_files: List[str], total_pairs: int, 
                       current_file: str = None, current_pairs: int = 0):
        """Guarda checkpoint del progreso"""
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "completed_files": completed_files,
            "total_pairs_generated": total_pairs,
            "current_file": current_file,
            "current_file_pairs": current_pairs,
            "status": "in_progress" if current_file else "completed"
        }
        
        # Guardar con nombre único para historial
        history_file = os.path.join(
            self.checkpoint_dir, 
            f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)
        
        print(f"  Checkpoint guardado: {len(completed_files)} archivos, {total_pairs} pares")
        return checkpoint
    
    def load_checkpoint(self) -> Dict:
        """Carga último checkpoint si existe"""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            print(f"  Checkpoint encontrado: {checkpoint['timestamp']}")
            print(f"  Archivos completados: {len(checkpoint['completed_files'])}")
            print(f"  Pares generados: {checkpoint['total_pairs_generated']}")
            return checkpoint
        return None
    
    def is_complete(self) -> bool:
        """Verifica si el proceso está completo"""
        checkpoint = self.load_checkpoint()
        return checkpoint and checkpoint['status'] == 'completed'

# ============================================================
# MAIN CON CHECKPOINTS
# ============================================================
def main():
    print("=" * 60)
    print("RUBIDIUM - Generador de Corpus de Alta Calidad")
    print("NeMo Data Designer Style - CON CHECKPOINTS")
    print("=" * 60)
    
    # Configurar directorios
    output_dir = "/kaggle/working/corpus"
    checkpoint_dir = "/kaggle/working/checkpoints"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Inicializar checkpoint manager
    cp_manager = CheckpointManager(checkpoint_dir)
    
    # Verificar si hay checkpoint previo
    checkpoint = cp_manager.load_checkpoint()
    completed_files = []
    total_pairs = 0
    start_idx = 0
    
    if checkpoint:
        completed_files = checkpoint['completed_files']
        total_pairs = checkpoint['total_pairs_generated']
        
        # Encontrar dónde continuar
        for i, topic in enumerate(TOPICS):
            if topic['filename'] not in completed_files:
                start_idx = i
                break
        
        print(f"\n  Continuando desde archivo {start_idx + 1}/10")
        print(f"  Archivos restantes: {10 - len(completed_files)}")
    else:
        print("\n  Iniciando nueva generación")
    
    # Confirmación del usuario
    print("\n" + "=" * 60)
    print("CONFIGURACIÓN DE GENERACIÓN")
    print("=" * 60)
    print(f"Archivos a generar: {10 - len(completed_files)}")
    print(f"Pares por archivo: 1000")
    print(f"Total estimado: {(10 - len(completed_files)) * 1000} pares")
    print(f"Tiempo estimado: {((10 - len(completed_files)) * 1.5):.0f} minutos")
    print(f"\nArchivos completados previamente: {len(completed_files)}")
    for f in completed_files:
        print(f"  ✓ {f}")
    
    print("\n" + "=" * 60)
    response = input("¿Continuar? (s/n): ").strip().lower()
    
    if response != 's':
        print("Generación cancelada por el usuario")
        return
    
    # Continuar generación
    start_time = time.time()
    
    for i in range(start_idx, len(TOPICS)):
        topic = TOPICS[i]
        print(f"\n[{i+1}/10] Generando: {topic['filename']}")
        print(f"  Temas: {len(topic['themes'])} dominios")
        
        generator = CorpusGenerator(topic)
        pairs = generator.generate_pairs(1000)
        
        filepath = os.path.join(output_dir, topic['filename'])
        saved = generator.save_to_file(pairs, filepath)
        
        # Actualizar progreso
        completed_files.append(topic['filename'])
        total_pairs += saved
        
        # Guardar checkpoint después de cada archivo
        cp_manager.save_checkpoint(
            completed_files=completed_files,
            total_pairs=total_pairs,
            current_file=topic['filename'],
            current_pairs=saved
        )
        
        # Mostrar progreso
        elapsed = time.time() - start_time
        avg_per_file = elapsed / (i - start_idx + 1)
        eta = avg_per_file * (len(TOPICS) - i - 1)
        
        print(f"  Total hasta ahora: {total_pairs} pares")
        print(f"  Tiempo: {elapsed/60:.1f} min | ETA: {eta/60:.1f} min")
    
    # Marcar como completado
    cp_manager.save_checkpoint(
        completed_files=completed_files,
        total_pairs=total_pairs,
        current_file=None,
        current_pairs=0
    )
    
    # Resumen final
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("GENERACIÓN COMPLETADA")
    print("=" * 60)
    print(f"Archivos generados: {len(completed_files)}")
    print(f"Total de pares: {total_pairs}")
    print(f"Tiempo total: {total_time/60:.1f} minutos")
    print(f"Promedio por archivo: {total_time/len(completed_files)/60:.1f} minutos")
    print(f"\nArchivos en: {output_dir}")
    
    # Crear archivo combinado
    combined_path = os.path.join(output_dir, "corpus_completo_10k.txt")
    print(f"\nCreando corpus combinado...")
    
    with open(combined_path, 'w', encoding='utf-8') as outfile:
        for topic in TOPICS:
            filepath = os.path.join(output_dir, topic['filename'])
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    outfile.write(content)
                    outfile.write("\n")
                print(f"  + {topic['filename']}")
    
    print(f"\nCorpus combinado: {combined_path}")
    print(f"Tamaño: {os.path.getsize(combined_path) / 1024:.1f} KB")
    
    # Crear backup
    backup_dir = "/kaggle/working/backup"
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    import shutil
    backup_path = os.path.join(backup_dir, f"corpus_backup_{timestamp}")
    shutil.copytree(output_dir, backup_path)
    print(f"\nBackup creado: {backup_path}")
    
    print("\n" + "=" * 60)
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print("=" * 60)

if __name__ == "__main__":
    main()
