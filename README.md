EcoFlash 🌱♻️

Plataforma web para la clasificación inteligente de residuos mediante Inteligencia Artificial

EcoFlash es una plataforma web que utiliza Deep Learning y visión por computador para identificar residuos sólidos a partir de imágenes y recomendar su correcta disposición según normativa ambiental.

El sistema integra modelos de aprendizaje profundo, bases de datos híbridas y visualización web, con el objetivo de apoyar la educación ambiental y la correcta clasificación de residuos urbanos.

Objetivo del proyecto

Diseñar e implementar una plataforma web inteligente capaz de:

Clasificar residuos a partir de imágenes usando redes neuronales convolucionales (CNN).

Recomendar la correcta disposición del residuo según normativa ambiental.

Integrar bases de datos relacionales y no relacionales para el almacenamiento de información.

Proporcionar una interfaz accesible para usuarios mediante una aplicación web interactiva.

Problema abordado

La incorrecta clasificación de residuos sólidos urbanos genera:

Contaminación ambiental

Pérdida de materiales reciclables

Ineficiencia en los sistemas de recolección

EcoFlash busca apoyar este problema mediante herramientas tecnológicas accesibles que ayuden a los ciudadanos a identificar residuos correctamente.

Arquitectura del sistema

El sistema EcoFlash está compuesto por los siguientes componentes:

Usuario
→ carga una imagen del residuo

Frontend
→ Aplicación web desarrollada con Streamlit

Modelo de Inteligencia Artificial
→ Clasificador de imágenes basado en ResNet50 (CNN)

Bases de datos

PostgreSQL → almacenamiento estructurado

MongoDB → almacenamiento de imágenes y metadatos

Infraestructura

Render → hosting de base de datos

Streamlit Cloud → despliegue de la aplicación

Tecnologías utilizadas
Lenguaje

Python

Inteligencia Artificial

PyTorch

ResNet50

Transfer Learning

Backend / Datos

PostgreSQL

MongoDB

Web

Streamlit

Infraestructura

Render

Streamlit Cloud

Hugging Face (almacenamiento de dataset)

Metodología

El desarrollo del proyecto siguió el marco CRISP-DM, el cual incluye:

Comprensión del problema

Comprensión de los datos

Preparación de datos

Modelado

Evaluación

Despliegue

Este enfoque permitió estructurar el desarrollo del sistema basado en datos reales.

Dataset utilizado

Para el entrenamiento del modelo se utilizó:

TACO Dataset (Trash Annotations in Context)
Un conjunto de imágenes reales de residuos en distintos contextos urbanos.

Las clases consideradas fueron:

Papel

Cartón

Plástico

Vidrio

Metal

Basura

Resultados del modelo

El modelo de clasificación alcanzó aproximadamente:

Accuracy: 93 %

F1 Score: 0.93

Las métricas indican un buen desempeño general del modelo, aunque existen algunas confusiones entre materiales visualmente similares.

Funcionalidades principales

EcoFlash permite:

Cargar una imagen de un residuo

Clasificar automáticamente el material

Recomendar el color de funda correspondiente

Visualizar resultados en una interfaz web

Registrar información en bases de datos
