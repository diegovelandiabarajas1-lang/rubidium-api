from typing import List, Optional, Set, Tuple, NamedTuple
import numpy as np
from text_similarity import content_tokens, sentence_vector, combined, WordEmbeddings


CODE_KEYWORDS: Set[str] = {
    "código", "codigo", "programar", "programación", "programacion", "función", "funcion",
    "método", "metodo", "clase", "objeto", "variable", "bucle", "for", "while", "if", "else",
    "algoritmo", "recursión", "recursion", "array", "arreglo", "lista", "diccionario",
    "compilar", "compilador", "sintaxis", "error", "bug", "excepción", "excepcion", "depurar",
    "debug", "herencia", "interfaz", "patrón", "patron", "c#", "csharp", "python", "java",
    "script", "string", "int", "booleano", "null", "return", "import", "namespace", "git",
    "api", "sql", "json", "framework", "librería", "libreria", "refactor",
    "javascript", "js", "node", "select", "join", "query", "consulta", "commit", "branch",
    "rama", "merge", "push", "pull", "repositorio", "pila", "stack", "cola", "queue", "hash",
    "árbol", "arbol", "grafo", "testing", "prueba", "pruebas", "test", "tdd",
    "unitaria", "asíncrono", "asincrono", "async", "await", "promesa",
    "html", "css", "etiqueta", "selector", "flexbox", "grid", "responsive", "div", "formulario",
    "estilo", "estilos", "web", "página", "pagina", "enlace", "lambda", "generador", "yield",
    "decorador", "comprensión", "comprension", "venv", "fstring", "f-string", "padding", "margin"
}


class _Entry(NamedTuple):
    question: str
    answer: str
    tokens: Set[str]
    vector: Optional[np.ndarray]


class NeuralCode:
    def __init__(self):
        self._kb: List[_Entry] = []
        self._embeddings: Optional[WordEmbeddings] = None
        self._vectors_built: bool = False
        self._build_knowledge_base()

    def use_embeddings(self, embeddings: Optional[WordEmbeddings]):
        self._embeddings = embeddings
        self._vectors_built = False

    def looks_like_code(self, tokens: Set[str]) -> bool:
        return any(t in CODE_KEYWORDS for t in tokens)

    def answer(self, user_tokens: Set[str], user_vec: Optional[np.ndarray]) -> Tuple[Optional[str], float]:
        self._ensure_vectors()
        lang = self._detect_language(user_tokens)

        best: Optional[str] = None
        best_score = 0.0
        for e in self._kb:
            score = combined(user_tokens, user_vec, e.tokens, e.vector)
            if lang and lang in e.answer:
                score += 0.15
            if score > best_score:
                best_score = score
                best = e.answer

        return best, best_score

    @staticmethod
    def _detect_language(tokens: Set[str]) -> Optional[str]:
        if "javascript" in tokens or "js" in tokens:
            return "JavaScript"
        if "python" in tokens:
            return "Python"
        if "java" in tokens:
            return "Java"
        if "sql" in tokens:
            return "SQL"
        if "csharp" in tokens or "c#" in tokens:
            return "C#"
        if "html" in tokens:
            return "HTML"
        if "css" in tokens or "flexbox" in tokens or "grid" in tokens:
            return "CSS"
        return None

    def _ensure_vectors(self):
        if self._vectors_built:
            return
        new_kb = []
        for e in self._kb:
            vec = sentence_vector(e.tokens, self._embeddings)
            new_kb.append(_Entry(e.question, e.answer, e.tokens, vec))
        self._kb = new_kb
        self._vectors_built = True

    def _add(self, question: str, answer: str):
        self._kb.append(_Entry(question, answer, content_tokens(question), None))

    def _build_knowledge_base(self):
        self._add("qué es una variable",
            "Una variable es un espacio con nombre donde se guarda un dato que puede cambiar. En C#: int edad = 30; En Python: edad = 30. El nombre te permite leer y modificar ese valor después.")
        self._add("qué es una función o método",
            "Una función agrupa instrucciones reutilizables que hacen una tarea y suelen devolver un resultado. En C#: int Sumar(int a, int b) { return a + b; } Se llama con Sumar(2, 3).")
        self._add("cómo hacer un bucle for",
            "Un bucle for repite código un número de veces. En C#: for (int i = 0; i < 5; i++) { Console.WriteLine(i); } En Python: for i in range(5): print(i). La variable i avanza en cada vuelta.")
        self._add("qué es una clase y un objeto",
            "Una clase es el molde que describe datos y comportamiento; un objeto es una instancia concreta de esa clase. En C#: class Persona { public string Nombre; } y luego var p = new Persona();")
        self._add("cómo manejar errores o excepciones",
            "Usa try y catch: el código de riesgo va en try y, si lanza una excepción, catch la captura. Ejemplo en C#: try { int x = int.Parse(texto); } catch (FormatException) { Console.WriteLine(\"No es un número\"); }")
        self._add("qué es un condicional if else",
            "Un condicional ejecuta código según una condición. En C#: if (numero % 2 == 0) Console.WriteLine(\"par\"); else Console.WriteLine(\"impar\"); El bloque de if corre si la condición es verdadera.")
        self._add("qué es una lista o array",
            "Una lista guarda varios elementos y se recorre con un bucle. En C#: var frutas = new List<string> { \"pera\", \"uva\" }; foreach (var f in frutas) Console.WriteLine(f); Puedes añadir con frutas.Add(\"melón\").")
        self._add("diferencia entre C# y Python",
            "C# es de tipado estático y compilado, ideal para aplicaciones Windows con .NET; usa llaves y punto y coma. Python es de tipado dinámico e interpretado, muy legible, y usa la indentación para los bloques. Ambos sirven para casi todo.")
        self._add("qué es la recursión",
            "La recursión es cuando una función se llama a sí misma para resolver un problema más pequeño, y siempre necesita un caso base que la detenga. Ejemplo: el factorial de n es n por el factorial de n menos uno, y el factorial de cero es uno.")
        self._add("qué es un algoritmo",
            "Un algoritmo es una secuencia finita de pasos que resuelve un problema. Un buen algoritmo es correcto y eficiente. Ejemplos comunes son la búsqueda binaria y el ordenamiento por mezcla.")
        self._add("cómo encontrar un error o hacer debugging",
            "Para depurar: reproduce el error de forma fiable, lee el mensaje y la traza de la pila, aísla la parte que falla, imprime o inspecciona el valor de las variables clave y corrige la causa, no el síntoma. Luego añade una prueba para ese caso.")
        self._add("qué es una interfaz",
            "Una interfaz define un contrato: una lista de métodos que una clase se compromete a implementar, sin decir cómo. Programar contra interfaces permite cambiar la implementación sin tocar el resto del código.")
        self._add("qué es un patrón de diseño",
            "Un patrón de diseño es una solución probada a un problema común de organización del código. Ejemplos: Singleton (una única instancia), Factory (crear objetos en un solo sitio), Observer (avisar de cambios) y MVVM (separar interfaz y lógica).")
        self._add("cómo escribir buen código o buenas prácticas",
            "Usa nombres descriptivos, escribe funciones pequeñas con una sola responsabilidad, evita repetir código, mantén un formato consistente, comenta el porqué y prueba tu código. La solución más simple que funciona suele ser la mejor.")
        self._add("cómo estructurar un proyecto o arquitectura",
            "Separa responsabilidades en capas: presentación, lógica y datos. Busca bajo acoplamiento y alta cohesión, programa contra interfaces y refleja la arquitectura en la estructura de carpetas. Empieza simple y añade estructura solo cuando haga falta.")
        self._add("cómo empezar a programar o crear un programa",
            "Primero define el objetivo, divídelo en tareas pequeñas, escribe cada tarea como una función, prueba cada parte por separado y únelas al final. Practicar con programas pequeños y no temer a los errores es la mejor forma de aprender.")
        self._add("qué es una API",
            "Una API es un conjunto de funciones que un programa ofrece para que otros lo usen, sin conocer sus detalles internos. Gracias a las API, distintos sistemas se comunican y comparten datos.")
        self._add("qué es una base de datos o SQL",
            "Una base de datos guarda información de forma organizada. Las relacionales la ordenan en tablas con filas y columnas y se consultan con SQL, un lenguaje para pedir, insertar, actualizar y borrar datos.")
        self._add("qué es JavaScript",
            "JavaScript es el lenguaje de la web: se ejecuta en el navegador y también en el servidor con Node.js. Se imprime con console.log(\"Hola\") y se declaran variables con let y const.")
        self._add("qué es una función flecha en JavaScript",
            "Una función flecha es una forma corta de escribir funciones en JavaScript. En vez de function sumar(a, b) { return a + b; } se escribe const sumar = (a, b) => a + b; Es muy usada como callback en métodos como map y filter.")
        self._add("qué es async await o programación asíncrona",
            "async/await permite escribir código asíncrono de forma legible. Una función async puede usar await para esperar a que termine una promesa sin bloquear el resto: const datos = await fetch(url).then(r => r.json());")
        self._add("cómo hacer una consulta SELECT en SQL",
            "Para leer datos en SQL se usa SELECT: SELECT nombre, edad FROM usuarios WHERE edad > 18 ORDER BY edad DESC; WHERE filtra filas y ORDER BY las ordena.")
        self._add("qué es un JOIN en SQL",
            "Un JOIN combina filas de dos tablas relacionadas según una columna en común: SELECT u.nombre, p.total FROM usuarios u JOIN pedidos p ON u.id = p.usuario_id;")
        self._add("qué es una pila stack",
            "Una pila (stack) sigue el principio último en entrar, primero en salir (LIFO). Se añade con push y se saca con pop, siempre por el mismo extremo. Se usa para deshacer acciones o para las llamadas a funciones.")
        self._add("qué es una cola queue",
            "Una cola (queue) sigue el principio primero en entrar, primero en salir (FIFO): se añade por un extremo y se saca por el otro. Es útil para procesar tareas en el orden en que llegan.")
        self._add("qué es una tabla hash o diccionario",
            "Una tabla hash (diccionario) asocia claves con valores y permite buscar por clave de forma casi inmediata. En C# es Dictionary y en Python es dict. Ideal cuando necesitas encontrar datos rápido por un identificador.")
        self._add("qué es un árbol o un grafo",
            "Un árbol organiza datos de forma jerárquica con un nodo raíz y ramas; un árbol binario de búsqueda mantiene los valores ordenados. Un grafo representa nodos y las relaciones (aristas) entre ellos, como una red social o un mapa.")
        self._add("qué es Git o control de versiones",
            "Git es un sistema de control de versiones: guarda el historial de cambios, permite volver atrás y facilita el trabajo en equipo. Se empieza con git init y se revisa el estado con git status.")
        self._add("cómo hacer un commit en Git",
            "Para guardar cambios en Git: primero git add archivo para prepararlos, y luego git commit -m \"Describe el cambio\" para confirmarlos. Con git log ves el historial y con git push subes los cambios al remoto.")
        self._add("qué es una rama branch en Git",
            "Una rama permite trabajar en una función sin tocar la versión principal: git checkout -b nueva-funcion. Cuando está lista, se integra con git merge. Usar ramas por tarea mantiene el proyecto ordenado.")
        self._add("qué es una prueba unitaria o testing",
            "Una prueba unitaria comprueba de forma automática una función aislada, siguiendo el patrón preparar, actuar y comprobar. En C# con xUnit: [Fact] public void Test() { Assert.Equal(5, Sumar(2,3)); } En Python con pytest: assert sumar(2,3) == 5.")
        self._add("qué es TDD desarrollo guiado por pruebas",
            "El desarrollo guiado por pruebas (TDD) consiste en escribir primero la prueba que falla, luego el código mínimo para que pase y por último mejorar el diseño. Ayuda a pensar qué debe hacer el código antes de escribirlo.")
        self._add("qué es HTML o cómo estructurar una página web",
            "HTML es el lenguaje de marcado que estructura una página web con etiquetas. La base: <!DOCTYPE html><html><head><title>...</title></head><body><h1>Hola</h1></body></html>. head lleva la info y body el contenido visible.")
        self._add("cómo hacer un enlace o una imagen en HTML",
            "Un enlace: <a href=\"https://ejemplo.com\">Ir</a>. Una imagen: <img src=\"foto.jpg\" alt=\"descripción\">. El atributo href es el destino y alt describe la imagen para accesibilidad.")
        self._add("cómo hacer un formulario en HTML",
            "Un formulario recoge datos: <form action=\"/enviar\" method=\"post\"><input type=\"text\" name=\"nombre\"><button type=\"submit\">Enviar</button></form>. El type del input puede ser text, email, password, number o checkbox.")
        self._add("qué son las etiquetas semánticas en HTML",
            "Las etiquetas semánticas describen el significado de cada parte: header (cabecera), nav (navegación), main (contenido), section, article y footer. Mejoran la accesibilidad y el posicionamiento en buscadores.")
        self._add("qué es CSS o cómo dar estilo a una página",
            "CSS da estilo a la web: colores, tipografías y disposición. Una regla tiene un selector y declaraciones: p { color: navy; font-size: 16px; }. Se selecciona por clase con punto (.tarjeta) o por id con almohadilla (#cabecera).")
        self._add("qué es el modelo de caja en CSS",
            "Cada elemento es una caja con contenido, padding (relleno interior), border (borde) y margin (margen exterior). Ejemplo: .tarjeta { padding: 16px; border: 1px solid gray; margin: 8px; border-radius: 8px; }")
        self._add("qué es Flexbox en CSS",
            "Flexbox coloca elementos de forma flexible en fila o columna: .contenedor { display: flex; justify-content: space-between; align-items: center; gap: 12px; } justify-content alinea en el eje principal y align-items en el perpendicular.")
        self._add("qué es CSS Grid",
            "Grid organiza el contenido en una cuadrícula: .galeria { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; } Aquí se crean tres columnas iguales.")
        self._add("cómo hacer una página responsive con media queries",
            "El diseño responsive adapta la página al tamaño de pantalla con media queries: @media (max-width: 600px) { .galeria { grid-template-columns: 1fr; } } En pantallas estrechas la galería pasa a una columna.")
        self._add("qué es una comprensión de listas en Python",
            "Una comprensión de listas crea listas de forma compacta: cuadrados = [n*n for n in range(10)] o con filtro pares = [n for n in numeros if n % 2 == 0]. También hay comprensiones de diccionario y de conjunto.")
        self._add("qué es una función lambda en Python",
            "Una lambda es una función anónima de una sola expresión: doblar = lambda x: x * 2. Se usa mucho como clave en sorted, map y filter, por ejemplo sorted(datos, key=lambda p: p.edad).")
        self._add("qué es un generador o yield en Python",
            "Un generador produce valores de uno en uno con yield, sin construir toda la lista en memoria: def contar(n): i=0; while i<n: yield i; i+=1. Es eficiente para secuencias grandes o infinitas.")
        self._add("qué es un decorador en Python",
            "Un decorador envuelve una función para añadirle comportamiento sin modificarla. Se aplica con @nombre encima de la función. Se usan para medir tiempos, registrar llamadas o controlar el acceso.")
        self._add("qué es una f-string en Python",
            "Una f-string permite insertar variables dentro de una cadena poniendo una f delante: nombre = \"Ana\"; print(f\"Hola, {nombre}\"). Es la forma moderna y legible de formatear texto en Python.")
        self._add("qué es un entorno virtual venv en Python",
            "Un entorno virtual aísla las dependencias de un proyecto. Se crea con python -m venv env y se activa antes de instalar paquetes con pip install, así cada proyecto tiene sus propias versiones.")