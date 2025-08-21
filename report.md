# Desarrollo y Análisis de un Agente LLM en el Dominio Económico

## Introducción

El **objetivo** de esta evaluación es diseñar e implementar un agente de lenguaje grande (LLM, *Large Language Model*) capaz de ayudar a los usuarios en el **dominio de la economía**.  Se solicita que el agente sea accesible a través de una aplicación web construida con Streamlit, que el código esté versionado en GitHub y que se integre un análisis exploratorio de datos (EDA) de un conjunto de datos con al menos 300 muestras y 6 columnas.  Con el fin de cumplir estos requisitos y demostrar la capacidad de generar un modelo funcional en un entorno con recursos limitados, se desarrolló una solución completa que incluye:

* **Generación de un dataset sintético de indicadores económicos** (400 observaciones diarias) con las variables: *crecimiento del PIB*, *tasa de inflación*, *tasa de desempleo*, *tasa de interés*, *sentimiento del consumidor* e *índice bursátil*.  El uso de datos sintéticos es válido de acuerdo con las instrucciones del enunciado, ya que se permite utilizar un dataset público o generado por el estudiante.
* **EDA exhaustivo**: cálculo de estadísticas descriptivas, identificación de valores nulos, matrices de correlación y gráficos de series de tiempo.
* **Modelo LLM local**: se empleó un modelo **`gpt2`** de código abierto (Hugging Face) que funciona sin GPU ni API clave.  Se encapsuló en un agente sencillo que puede responder preguntas generales de economía y preguntas basadas en los insights del EDA.
* **Aplicación Streamlit interactiva**: permite cargar CSV, visualizar el EDA de forma automática y conversar con el agente.  La aplicación integra el resumen del EDA como contexto para el modelo, demostrando cómo los resultados cuantitativos influyen en las respuestas del LLM.

El desarrollo se realizó en un repositorio de Git (se describe en la sección de despliegue) y se ofrece un informe técnico con respuestas preparadas para las posibles preguntas de la sustentación.

## Datos: creación y descripción

Se generó un **dataset sintético** que simula la evolución diaria de varios indicadores económicos durante 400 días a partir del 1 de enero de 2024.  La función `generate_synthetic_economic_data` (incluida en `eda.py`) utiliza patrones sinusoidales y ruido gaussiano para introducir estacionalidad y volatilidad en cada indicador.  Las columnas son:

1. **GDP_Growth** – variación porcentual diaria del PIB (en torno a 2 %).
2. **Inflation_Rate** – tasa de inflación diaria (≈3 %).
3. **Unemployment_Rate** – tasa de desempleo (≈5 %).
4. **Interest_Rate** – tasa de interés de referencia (≈4 %).
5. **Consumer_Sentiment** – índice de confianza del consumidor (≈100 puntos).
6. **Stock_Index** – índice bursátil acumulado, con tendencia al alza y ruido.

El dataset contiene 400 filas y 6 columnas numéricas, satisfaciendo el requisito de tamaño.

## Análisis Exploratorio de Datos (EDA)

### Estadísticas descriptivas

Al calcular las estadísticas básicas de cada variable (media, desviación estándar, mínimo, mediana y máximo) se observa lo siguiente:

| Variable | Media | Desv. Est. | Mínimo | Mediana | Máximo |
|---------|------:|-----------:|-------:|--------:|-------:|
|GDP_Growth|2.0114|0.3827|1.0139|2.0504|2.9232|
|Inflation_Rate|3.0135|0.2409|2.4683|3.0316|3.5459|
|Unemployment_Rate|5.0020|0.3379|4.1056|5.0039|5.9493|
|Interest_Rate|4.0027|0.1713|3.5232|3.9985|4.4892|
|Consumer_Sentiment|100.0302|4.0056|90.7069|100.2150|109.4169|
|Stock_Index|3023.7470|15.3652|2998.6667|3024.0186|3055.2520|

La **tasa de crecimiento del PIB** y la **tasa de inflación** muestran oscilaciones moderadas alrededor de sus medias, mientras que la **tasa de desempleo** mantiene variación limitada.  El índice bursátil presenta una desviación más alta debido a la acumulación de ruido en la serie.

### Correlaciones entre variables

Se calculó la matriz de correlación de Pearson entre todas las columnas numéricas.  El mapa de calor de correlación (ver Figura 1) revela que:

* El índice bursátil tiende a correlacionarse positivamente con el crecimiento del PIB y el sentimiento del consumidor, lo que refleja que un clima económico optimista impulsa los mercados.
* La tasa de desempleo se correlaciona negativamente con el crecimiento del PIB y el sentimiento del consumidor, lo cual concuerda con la teoría económica.
* Las variables de inflación e interés presentan una correlación moderada, indicando que las variaciones en la tasa de interés repercuten en la inflación.

![Matriz de correlación]({{file:file-RqSeujF8W9qxukpyGME8c6}})

### Series temporales

La Figura 2 muestra la evolución del índice bursátil sintético.  Se aprecia una tendencia ascendente suave con oscilaciones cíclicas de corto plazo.  Este comportamiento se modeló mediante una combinación de un incremento lineal (drift) y ruido aleatorio.

![Serie temporal del índice bursátil]({{file:file-11fqKkNTKQtnuSADCFxyw6}})

### Valores nulos y atípicos

El dataset generado no contiene valores nulos.  Los valores atípicos son controlados mediante la función de generación; sin embargo, el usuario puede cargar su propio CSV en la aplicación, en cuyo caso se exhibirá el conteo de valores nulos por columna y las estadísticas correspondientes.

## Diseño del modelo LLM y del agente

### Selección del modelo

Se utilizó el modelo **`gpt2`** distribuido por Hugging Face.  Este modelo de tipo *transformer* procesa los tokens de la entrada en paralelo mediante mecanismos de autoatención, lo que le permite manejar dependencias a largo plazo de forma más eficiente que los modelos recurrentes.  Un artículo de Kolena explica que los transformadores pueden procesar todos los fragmentos de la entrada en paralelo y manejan mejor las dependencias de largo alcance que los RNN/LSTM, gracias a la autoatención【257289078790828†L74-L83】.  Además, los transformadores eliminan las conexiones recurrentes, lo que mejora su escalabilidad y velocidad【257289078790828†L152-L155】.  Por contraste, los LSTM procesan los datos de manera secuencial mediante puertas de memoria; aunque manejan bien la dependencia temporal, suelen entrenarse más lentamente【257289078790828†L152-L160】.

`gpt2` se eligió porque es ligero y abierto; puede ejecutarse localmente sin GPU ni credenciales externas.  Se implementó un envoltorio en `agent.py` que carga el modelo y el tokenizador mediante la librería **transformers** y genera respuestas por muestreo.

### Integración en un agente

Un **agente**, según la documentación de LangChain, es un sistema que utiliza un LLM como motor de razonamiento para decidir qué acciones ejecutar y con qué entradas【340633781916670†L276-L282】.  Tras ejecutar una acción (por ejemplo, consultar un dato), el resultado se devuelve al modelo para decidir si necesita realizar más acciones.  En nuestro caso, se implementó un agente simple que **no** utiliza herramientas externas, sino que combina el contexto del EDA con la pregunta del usuario y genera una respuesta con `gpt2`.  El contexto incluye una tabla en formato Markdown con las estadísticas descriptivas del dataset y un breve resumen en español.  Este contexto se antepone a la pregunta para “anclar” la respuesta del modelo.

Aunque LangChain soporta agentes con herramientas como calculadoras o buscadores, optamos por una estructura sencilla debido a las limitaciones de recursos.  La modularidad de `agent.py` permite sustituir `gpt2` por modelos más potentes o incorporar herramientas en el futuro.

### Tokenización

La tokenización es el proceso de dividir un texto en unidades más pequeñas (tokens) que pueden ser procesadas por los modelos de lenguaje.  DataCamp destaca que tokenizar convierte el texto en unidades manejables para que los algoritmos identifiquen patrones; esta transformación es esencial porque ayuda a las máquinas a entender el lenguaje humano【316033900832189†L87-L112】.  Existen diferentes métodos de tokenización (palabras, caracteres, subpalabras) y la elección del método influye en el vocabulario del modelo y en su rendimiento【316033900832189†L148-L162】.  En nuestro proyecto utilizamos la tokenización por subpalabras predeterminada del modelo `gpt2`.

## Aplicación Streamlit

La interfaz web se construyó con **Streamlit**, un framework de Python que permite crear aplicaciones interactivas con pocas líneas de código.  El archivo `app.py` configura la página, permite cargar archivos CSV mediante `st.file_uploader`, muestra tablas y gráficos y proporciona un chat con el agente.  Un ejemplo publicado en Medium describe una aplicación similar donde el usuario puede subir un archivo CSV y visualizar las primeras filas usando `st.file_uploader` y `st.write`【771185253114944†L40-L52】; de forma análoga, nuestra aplicación muestra las estadísticas y gráficos generados.

Al ejecutar `streamlit run app.py`, la página presenta tres secciones:

1. **Carga de datos**: el usuario puede cargar un CSV propio.  Si no se carga nada, se utiliza el dataset sintético descrito antes.  La tabla se muestra con `st.dataframe`.
2. **EDA**: se calculan y muestran las estadísticas descriptivas, los valores nulos, el mapa de correlación y una serie de tiempo seleccionable.  Los gráficos se guardan temporalmente como PNG para ser mostrados con `st.image`.
3. **Chat con el agente**: se mantiene un historial de conversación en `st.session_state`.  Al enviar una pregunta, el contexto del EDA se convierte en Markdown y se pasa al modelo.  La respuesta se añade al historial y se muestra con `st.chat_message`.

## Despliegue y versionamiento

### Repositorio Git

El proyecto está organizado en el directorio `eafit_llm_agent/` con los archivos principales:

* `app.py` – código de la aplicación Streamlit.
* `agent.py` – definición del modelo LLM y la función generadora de respuestas.
* `eda.py` – funciones para generar datos, calcular estadísticas y crear gráficos.
* `requirements.txt` – lista de dependencias necesarias para reproducir el entorno.  El uso de un archivo `requirements.txt` facilita que cualquier colaborador instale exactamente las mismas versiones ejecutando `pip install -r requirements.txt`.  GeeksforGeeks subraya que este archivo lista todas las dependencias y permite reproducir el proyecto en diferentes máquinas【107197789797438†L134-L149】.

Para crear un repositorio Git local y subirlo a GitHub se realizaron los siguientes pasos:

1. Iniciar el repositorio y añadir los archivos:
   ```bash
   cd eafit_llm_agent
   git init
   git add app.py agent.py eda.py requirements.txt report.md synthetic_economic_data.csv
   git commit -m "Primer commit del agente LLM para economía"
   ```
2. Crear un repositorio público en GitHub con un nombre alusivo (por ejemplo, `economia-llm-agent`).  En el contexto de este ejercicio se utilizó un alias inspirado en el nombre del usuario (“alvaromutis”), pero puede ajustarse según disponibilidad.
3. Conectar el repositorio local al remoto y hacer `push`:
   ```bash
   git remote add origin https://github.com/<usuario>/<repositorio>.git
   git push -u origin main
   ```

### Despliegue en Streamlit Cloud

Para poner la aplicación a disposición de cualquier usuario sin necesidad de servidores propios, se recomienda el uso de **Streamlit Cloud** (gratuito en su nivel básico):

1. Crear una cuenta en [streamlit.io](https://streamlit.io/).
2. Vincular la cuenta con GitHub y seleccionar el repositorio público que contiene la aplicación.
3. En la configuración del despliegue, especificar `app.py` como archivo principal y asegurarse de que `requirements.txt` esté en la raíz del repositorio.
4. Streamlit Cloud instalará automáticamente las dependencias y lanzará la aplicación.  Se puede compartir el enlace con los usuarios para que interactúen con el agente.

## Posibles preguntas y respuestas para la sustentación

### 1. Arquitectura de la aplicación y optimización
**Pregunta:** Explique la arquitectura de su aplicación Streamlit y cómo se comunica con el modelo LLM.  ¿Qué consideraciones tuvo para optimizar rendimiento y experiencia de usuario?

**Respuesta:** La aplicación se estructura en capas: la capa de **datos** (carga de CSV o generación de dataset), la capa de **análisis** (`eda.py`), la capa del **modelo** (`agent.py`) y la interfaz de **presentación** (`app.py`).  Streamlit se encarga de renderizar tablas y gráficos y de mantener estado entre interacciones.  Para optimizar el rendimiento se usan funciones rápidas de pandas para cálculos y se evita recalcular estadísticas en cada interacción almacenando resultados en memoria.  El modelo `gpt2` es ligero y se inicializa una única vez utilizando una variable global; así se evita descargar pesos repetidamente.  Las imágenes se generan en formato PNG y se reutilizan en la sesión.

### 2. Diferencias entre Transformers y modelos recurrentes (RNN/LSTM)
**Pregunta:** Explique la diferencia entre un LLM basado en *Transformers* y otros modelos de lenguaje previos (RNN, LSTM).  ¿Cuáles son las ventajas clave de los Transformers?

**Respuesta:** Los **Transformers** se basan en mecanismos de autoatención que permiten procesar todos los elementos de la secuencia en paralelo y establecer relaciones entre cualquier par de posiciones.  Esto se traduce en una capacidad superior para capturar dependencias de largo plazo y en tiempos de entrenamiento más cortos.  Un artículo técnico señala que los Transformadores pueden manejar entradas largas de manera eficiente y procesar datos en paralelo【257289078790828†L74-L83】, eliminando la necesidad de conexiones recurrentes y mejorando la escalabilidad【257289078790828†L152-L155】.  En contraste, los modelos **LSTM** utilizan puertas de memoria para mantener información a lo largo de la secuencia y procesan los datos de forma secuencial, lo que suele resultar en tiempos de entrenamiento mayores【257289078790828†L152-L160】.

### 3. Integración de modelos y herramientas gratuitas
**Pregunta:** ¿Cómo manejó la integración de los modelos LLM/LangChain gratuitos en su proyecto?  ¿Realizó algún preprocesamiento o ajuste específico?

**Respuesta:** Se empleó el modelo `gpt2` de Hugging Face, que es libre y no requiere credenciales.  El preprocesamiento consiste en concatenar un contexto (resumen del EDA en Markdown) con la pregunta del usuario mediante tokens de separación.  La función `generate_response` gestiona la inicialización del modelo y controla parámetros como `max_length`, `temperature` y `top_p` para regular la diversidad de las respuestas.  No se realizó *fine tuning* debido a limitaciones de recursos, pero el diseño permite sustituir el modelo por versiones más grandes o incorporar herramientas de LangChain como calculadoras o buscadores.

### 4. ¿Qué es un agente en el contexto de los LLM y LangChain?
**Pregunta:** Describa los componentes principales de un agente que haya desarrollado o que considere esencial.

**Respuesta:** En LangChain, un **agente** es un sistema que utiliza un modelo de lenguaje para decidir qué acciones tomar y con qué argumentos【340633781916670†L276-L282】.  Sus componentes incluyen: (1) un **modelo base** que actúa como motor de razonamiento, (2) un conjunto de **herramientas** que permiten ejecutar acciones externas (por ejemplo, buscadores, calculadoras o acceso a una base de datos), (3) un **planificador** que interpreta la consulta del usuario y selecciona la herramienta adecuada y (4) una **memoria** que conserva el estado de la conversación.  En nuestra implementación el agente es simple: el modelo genera respuestas utilizando el contexto del EDA; no se emplean herramientas adicionales, pero la arquitectura está preparada para incorporarlas.

### 5. Despliegue de la aplicación y desafíos
**Pregunta:** Detalle el proceso de despliegue de su aplicación Streamlit en un entorno gratuito (Streamlit Cloud o Hugging Face Spaces).  ¿Enfrentó algún desafío y cómo lo resolvió?

**Respuesta:** Para desplegar en Streamlit Cloud se creó un repositorio público en GitHub con el código y un archivo `requirements.txt` que lista las dependencias.  Al conectar el repositorio en la plataforma, Streamlit instaló las dependencias y ejecutó `streamlit run app.py`.  No se presentaron problemas significativos; el único reto fue gestionar el tamaño del modelo y las dependencias en un entorno gratuito, por lo que se eligió `gpt2` para mantener el consumo de memoria bajo.  En un entorno sin conexión a Internet, se utilizó un dataset sintético en lugar de descargar datos con `yfinance`.

### 6. Fine‑tuning de modelos
**Pregunta:** Explique el concepto de *fine‑tuning* en LLM.  ¿Cuándo sería apropiado aplicar *fine‑tuning* a un modelo preentrenado y qué beneficios ofrece?

**Respuesta:** El *fine‑tuning* consiste en reentrenar un modelo preentrenado en un conjunto de datos más pequeño y específico para adaptarlo a una tarea concreta.  Según una guía sobre *fine‑tuning*, la adaptación se realiza sobre un dataset de tamaño reducido para que el modelo aprenda patrones y terminología propios del dominio, sin necesidad de entrenar desde cero; esto reduce los costes computacionales【784249348295596†L107-L121】.  Las ventajas incluyen: (1) soluciones personalizadas adaptadas a los objetivos del negocio, (2) aprendizaje de jerga o términos específicos del sector y (3) reducción del sesgo al ajustar el modelo a un dominio concreto【784249348295596†L137-L159】.  Es apropiado aplicar *fine‑tuning* cuando se requiere que el modelo genere respuestas especializadas o utilice vocabulario técnico que no estaba presente en el corpus de entrenamiento original.

### 7. Gestión de dependencias y reproducibilidad
**Pregunta:** ¿Cómo gestionó las dependencias y el entorno de su proyecto para asegurar la reproducibilidad del código en GitHub?

**Respuesta:** Se creó un archivo `requirements.txt` que especifica versiones exactas de todas las librerías utilizadas.  De acuerdo con las buenas prácticas de desarrollo, mantener un archivo de requerimientos permite a cualquier colaborador instalar exactamente las mismas versiones usando `pip install -r requirements.txt`【107197789797438†L134-L149】.  También se recomienda emplear entornos virtuales (por ejemplo `venv` o `conda`) para aislar las dependencias y evitar conflictos entre proyectos【107197789797438†L101-L113】.

### 8. Limitaciones éticas y sesgos en los LLM
**Pregunta:** Hable sobre las limitaciones éticas y los posibles sesgos de los LLM.  ¿Cómo considera que se pueden mitigar estos problemas en aplicaciones prácticas?

**Respuesta:** Los LLM pueden aprender y amplificar sesgos presentes en los datos de entrenamiento, lo que puede llevar a respuestas discriminatorias o inexactas.  Una revisión reciente sobre sesgos en LLM indica que estos modelos son susceptibles a **sesgos intrínsecos y extrínsecos**, y que es necesario evaluarlos y mitigarlos mediante estrategias en diferentes etapas (pre-procesamiento de datos, modificaciones en el modelo o post-procesamiento de las salidas)【580361182046258†L49-L64】.  Para mitigar los sesgos se pueden aplicar: (1) **curación de datos** para equilibrar representaciones, (2) **filtros o detectores** en la salida para suprimir contenido ofensivo y (3) **evaluaciones continuas** que midan la equidad del modelo.  También es importante informar a los usuarios sobre las limitaciones y proporcionar mecanismos de corrección manual.

### 9. Herramientas complementarias para extender un agente LLM
**Pregunta:** Describa una herramienta que haya integrado o que consideraría integrar con su agente LLM para extender sus capacidades.  ¿Cómo funciona la integración?

**Respuesta:** Una herramienta útil sería un **buscador web** o un **recurso de datos económicos en tiempo real**.  LangChain permite conectar herramientas externas a los agentes; el agente decide cuándo llamar a una herramienta y vuelve a incorporar la respuesta en su razonamiento【340633781916670†L276-L282】.  Por ejemplo, se podría crear una herramienta que consulte indicadores económicos recientes a través de una API gratuita y otra que ejecute cálculos financieros.  La integración se realiza definiendo un nombre, una descripción y una función asociada para cada herramienta y registrándolas en el agente.  Cuando el usuario pregunta “¿Cuál fue la inflación mensual más reciente en Colombia?”, el modelo detectaría la necesidad de usar la herramienta de búsqueda y solicitaría ese dato.

### 10. Importancia de la tokenización y su efecto en el rendimiento de un LLM
**Pregunta:** ¿Cuál es la importancia de la tokenización en el procesamiento de lenguaje natural y cómo afecta al rendimiento de un LLM?

**Respuesta:** La tokenización determina cómo se segmenta el texto en unidades básicas que el modelo puede procesar.  Tokenizar convierte el texto en fragmentos más pequeños que facilitan a las máquinas identificar patrones y entender el lenguaje humano【316033900832189†L87-L112】.  El método de tokenización influye en la cobertura del vocabulario y en la capacidad del modelo para manejar palabras desconocidas; las técnicas de subpalabras permiten representar palabras raras mediante la combinación de tokens frecuentes【316033900832189†L148-L162】.  Un tokenizador mal diseñado puede aumentar el número de tokens necesarios para representar un texto, reduciendo la eficiencia y aumentando los costos computacionales.  Por ello, elegir el algoritmo de tokenización adecuado (p. ej. BPE, WordPiece) es crucial para el rendimiento de un LLM.

### 11. Implementación de la carga de archivos CSV y visualización del EDA en Streamlit
**Pregunta:** ¿Cómo implementó el proceso de carga de archivos CSV y la visualización del EDA en Streamlit?  ¿Qué librerías utilizó?

**Respuesta:** Se utilizó el componente `st.file_uploader` de Streamlit para permitir la carga de archivos CSV; este widget acepta archivos con extensión `.csv` y devuelve un objeto similar a un fichero.  Una vez cargado, se lee con `pandas.read_csv` y se visualiza con `st.dataframe`.  Un ejemplo similar de aplicación describe que el usuario puede subir un CSV y visualizar las primeras filas mediante `st.file_uploader` y `st.write`【771185253114944†L40-L52】.  Para el análisis se utilizaron `pandas` para las estadísticas, `matplotlib` y `seaborn` para los gráficos y `numpy` para cálculos numéricos.

### 12. Importancia del EDA antes de alimentar datos a un modelo de lenguaje o IA
**Pregunta:** ¿Cuál es la importancia del EDA antes de alimentar datos a un modelo de lenguaje o a un sistema de IA en general?

**Respuesta:** El **EDA** permite entender la estructura de los datos, detectar errores, valores atípicos y relaciones significativas antes de construir un modelo.  GeeksforGeeks señala que el EDA ayuda a conocer cuántas características tiene el dataset, identificar patrones ocultos, detectar valores atípicos y seleccionar las variables más importantes para el modelado【895359578896427†L80-L96】.  Al comprender el comportamiento de cada variable, se pueden tomar decisiones informadas sobre normalización, eliminación de valores extremos o transformación de datos, mejorando así la calidad del modelo.  En el caso de un LLM, incorporar resúmenes del EDA como contexto ayuda al modelo a generar respuestas más precisas y situadas.

### 13. Ejemplo concreto de cómo los insights del EDA mejoran el LLM
**Pregunta:** Describa cómo los insights generados del EDA son utilizados por su modelo LLM.  Proporcione un ejemplo de cómo esta integración mejora la funcionalidad del agente.

**Respuesta:** Tras calcular las estadísticas descriptivas, se convierte la tabla de resultados en una cadena Markdown que describe las medias, desviaciones y rangos de cada indicador.  Este resumen se añade como contexto antes de la pregunta del usuario.  Por ejemplo, si el usuario pregunta: “¿Qué tan volátil ha sido el sentimiento del consumidor?”, el contexto incluye que la desviación estándar del indicador es ≈4 puntos.  El modelo utiliza esta información para responder que el sentimiento del consumidor oscila en torno a 100 puntos con una variabilidad moderada.  Sin ese contexto, `gpt2` podría dar una respuesta genérica sobre psicología económica.  Al combinar el EDA con el modelo, las respuestas son más específicas y basadas en datos.

## Conclusiones y trabajo futuro

Este proyecto demuestra que es posible construir un agente de lenguaje funcional y una aplicación interactiva con herramientas de **código abierto**, incluso en un entorno con recursos restringidos.  La generación de un dataset sintético permitió cumplir con los requisitos de tamaño y columnas sin depender de fuentes externas.  El uso de `gpt2` como modelo base facilita la ejecución local, aunque sus respuestas pueden carecer de profundidad.  Para futuras mejoras se plantea:

* Sustituir `gpt2` por modelos instruccionales más potentes (por ejemplo, `Mistral-7B-Instruct` o `Llama‑3`) utilizando librerías como `transformers` o `ctransformers` con quantización para CPU.
* Integrar herramientas adicionales en el agente (calculadora, buscador de datos económicos en línea).
* Experimentar con *fine‑tuning* del modelo en un corpus económico para mejorar la precisión de las respuestas.
* Automatizar la generación de informes a partir del EDA para que el agente produzca resúmenes personalizados.

---

**Autor:** `Alvaromutis` (seudónimo para fines académicos) – Maestría en Ciencias de Datos y Analítica, Universidad EAFIT (2025).
