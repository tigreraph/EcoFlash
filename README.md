# EcoFlash 🌱♻️
**Plataforma web para la clasificación inteligente de residuos mediante Inteligencia Artificial**

EcoFlash es una plataforma web que utiliza **Deep Learning y visión por computador** para identificar residuos sólidos a partir de imágenes y recomendar su **correcta disposición según normativa ambiental**.

El sistema integra **modelos de aprendizaje profundo, bases de datos híbridas y visualización web**, con el objetivo de apoyar la **educación ambiental y la correcta clasificación de residuos urbanos**.

---

# Objetivo del proyecto

Diseñar e implementar una **plataforma web inteligente** capaz de:

- Clasificar residuos a partir de imágenes usando **redes neuronales convolucionales (CNN)**.
- Recomendar la **correcta disposición del residuo según normativa ambiental**.
- Integrar **bases de datos relacionales y no relacionales** para el almacenamiento de información.
- Proporcionar una **interfaz accesible para usuarios** mediante una aplicación web interactiva.

---

# Problema abordado

La incorrecta clasificación de residuos sólidos urbanos genera:

- Contaminación ambiental
- Pérdida de materiales reciclables
- Ineficiencia en los sistemas de recolección

EcoFlash busca apoyar este problema mediante **herramientas tecnológicas accesibles** que ayuden a los ciudadanos a identificar residuos correctamente.

---

# Arquitectura del sistema

El sistema EcoFlash está compuesto por los siguientes componentes:

**Usuario**
→ carga una imagen del residuo

**Frontend**
→ Aplicación web desarrollada con **Streamlit**

**Modelo de Inteligencia Artificial**
→ Clasificador de imágenes basado en **ResNet50 (CNN)**

**Bases de datos**
- **PostgreSQL** → almacenamiento estructurado
- **MongoDB** → almacenamiento de imágenes y metadatos

**Infraestructura**
- **Render** → hosting de base de datos
- **Streamlit Cloud** → despliegue de la aplicación

---

# Tecnologías utilizadas

### Lenguaje
- Python

### Inteligencia Artificial
- PyTorch
- ResNet50
- Transfer Learning

### Backend / Datos
- PostgreSQL
- MongoDB

### Web
- Streamlit

### Infraestructura
- Render
- Streamlit Cloud
- Hugging Face

---

# Metodología

El desarrollo del proyecto siguió el marco **CRISP-DM**, el cual incluye:

1. Comprensión del problema
2. Comprensión de los datos
3. Preparación de datos
4. Modelado
5. Evaluación
6. Despliegue

Este enfoque permitió estructurar el desarrollo del sistema basado en datos reales.

---

# Dataset utilizado

Para el entrenamiento del modelo se utilizó el dataset:

**TACO (Trash Annotations in Context)**

Contiene imágenes reales de residuos en distintos contextos urbanos.

Clases utilizadas en el proyecto:

- Papel
- Cartón
- Plástico
- Vidrio
- Metal
- Basura

---

# Resultados del modelo

El modelo de clasificación alcanzó aproximadamente:

- **Accuracy:** 93 %
- **F1 Score:** 0.93

Las métricas indican un **buen desempeño general**, aunque existen algunas confusiones entre materiales visualmente similares.

---

# Funcionalidades principales

EcoFlash permite:

- Cargar una imagen de un residuo
- Clasificar automáticamente el material
- Recomendar el **color de funda correspondiente**
- Visualizar resultados en una interfaz web
- Registrar información en bases de datos

---

# Estructura del proyecto

```
EcoFlash/
│
├── data/
│   └── dataset_residuos
│
├── model/
│   └── modelo_resnet50.pth
│
├── app/
│   └── app_streamlit.py
│
├── database/
│   ├── postgres_schema.sql
│   └── mongodb_config.py
│
├── notebooks/
│   └── entrenamiento_modelo.ipynb
│
└── README.md
```

---

# Cómo ejecutar el proyecto

### 1. Instalar dependencias

```
pip install -r requirements.txt
```

### 2. Ejecutar la aplicación

```
streamlit run app_streamlit.py
```

### 3. Abrir en el navegador

```
http://localhost:8501
```

---

# Limitaciones del proyecto

- Dataset limitado
- Número reducido de categorías
- Falta de evaluación con usuarios finales

---

# Trabajo futuro

- Ampliar el dataset
- Integrar más tipos de residuos
- Implementar versión móvil
- Mejorar precisión del modelo
- Integración con sistemas de gestión municipal

---

# Contribuidores

Proyecto desarrollado por:

- Investigador principal – Desarrollo de Inteligencia Artificial
- Asistente de investigación – Desarrollo web y documentación
- Analista de datos – Preparación de datos y evaluación del modelo

---

# Impacto del proyecto

EcoFlash demuestra la viabilidad de integrar **Inteligencia Artificial, bases de datos y aplicaciones web** para apoyar la **gestión ambiental urbana**, contribuyendo a la educación y concienciación sobre el reciclaje.
