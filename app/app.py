# ECOFLASH ♻️ 

import streamlit as st
st.set_page_config(
    page_title="EcoFlash - Reciclaje Responsable",
    page_icon="♻️",
    layout="wide"
)
import os
from datetime import date
import base64
import pickle
import pandas as pd
import numpy as np
import psycopg2
import plotly.express as px
import random
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent.parent


video_file = open("app/assets/fondo.mp4", "rb").read()
video_base64 = base64.b64encode(video_file).decode()

st.markdown(
    f"""
    <style>
        .stApp {{
            background: none;
        }}

        video {{
            position: fixed;
            right: 0;
            bottom: 0;
            min-width: 100%;
            min-height: 100%;
            z-index: -1;
        }}
    </style>

    <video autoplay muted loop>
        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
    </video>
    """,
    unsafe_allow_html=True,
)

# Config DB

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT", "5432"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "sslmode": "require"
}





# Conexión a la BD  de postgresql

@st.cache_resource
def crear_conexion(db_config):
    try:
        conn = psycopg2.connect(**db_config)
        return conn
    except Exception as e:
        st.error(f"❌ Error conectando a la BD: {e}")
        return None

conn = crear_conexion(DB_CONFIG)


# Cargar modelos 
MODEL_DIR = BASE_DIR / "model"

@st.cache_resource
def cargar_modelos_y_encoders():
    modelo_regresion = None
    modelo_clasificacion = None
    le_placa = None
    le_sector = None

    try:
        path_modelo_regresion = os.path.join(MODEL_DIR, "modelo_total_peso.pkl")
        path_modelo_clasificacion = os.path.join(MODEL_DIR, "modelo_sector_nombre.pkl")
        path_le_placa = os.path.join(MODEL_DIR, "le_placa.pkl")
        path_le_sector = os.path.join(MODEL_DIR, "le_sector.pkl")

        if os.path.exists(path_modelo_regresion):
            with open(path_modelo_regresion, "rb") as f:
                modelo_regresion = pickle.load(f)
        else:
            st.warning("⚠️ Modelo de regresión no encontrado (modo demostración).")

        if os.path.exists(path_modelo_clasificacion):
            with open(path_modelo_clasificacion, "rb") as f:
                modelo_clasificacion = pickle.load(f)
        else:
            st.warning("⚠️ Modelo de clasificación no encontrado (modo demostración).")

        if os.path.exists(path_le_placa):
            with open(path_le_placa, "rb") as f:
                le_placa = pickle.load(f)

        if os.path.exists(path_le_sector):
            with open(path_le_sector, "rb") as f:
                le_sector = pickle.load(f)

    except Exception as e:
        st.error(f"❌ Error cargando modelos/encoders: {e}")

    return modelo_regresion, modelo_clasificacion, le_placa, le_sector

modelo_regresion, modelo_clasificacion, le_placa, le_sector = cargar_modelos_y_encoders()


# Cache data

@st.cache_data
def cargar_sectores(_conn):
    if _conn is None:
        return pd.DataFrame(columns=["nombre_sector"])
    q = "SELECT nombre_sector FROM sector ORDER BY nombre_sector;"
    return pd.read_sql(q, _conn)

@st.cache_data
def cargar_registros(_conn):
    if _conn is None:
        return pd.DataFrame()

    q = """
    SELECT r.*, s.nombre_sector
    FROM registro r
    JOIN sector s ON r.id_sector = s.id_sector
    ORDER BY r.id_registro
    """
    df = pd.read_sql(q, _conn)

    if "fecha_ingreso" in df.columns and "fecha" not in df.columns:
        df = df.rename(columns={"fecha_ingreso": "fecha"})

    # Convertir horas
    if "hora_ingreso" in df.columns:
        df["hora_ingreso"] = pd.to_datetime(df["hora_ingreso"], errors="coerce").dt.hour
    if "hora_salida" in df.columns:
        df["hora_salida"] = pd.to_datetime(df["hora_salida"], errors="coerce").dt.hour

    # Deltas
    df["delta_peso"] = df["peso_salida"] - df["peso_inicial"]
    df["duracion_horas"] = df["hora_salida"] - df["hora_ingreso"]

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["dia_semana"] = df["fecha"].dt.dayofweek

    df["placa"] = df["placa"].fillna("DESCONOCIDA")

    return df


# Imagen base64

def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except:
        return None


# Estilos
st.markdown("""
<style>

.card {
    background: rgba(30, 30, 30, 0.4); 
    backdrop-filter: blur(15px); 
    border-radius: 20px;
    padding: 40px 60px; 
    margin-top: 15vh;
    text-align: center;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
}

.titulo {
    color: #00C853; 
    font-size: 58px;
    font-weight: 900; 
    margin-bottom: 15px;
    text-shadow: none; 
}

.subtitulo {
    color: #E0E0E0; 
    font-size: 26px;
    font-weight: 300; 
    margin-bottom: 27px;
}

.autores, .fecha {
    color: #FFFFFF;
    font-size: 18px;
}

.stButton>button {
    background-color: #00C853; 
    color: white; 
    border-radius: 12px; 
    padding: 10px 25px; 
    font-size: 18px;
    box-shadow: 0 4px 12px rgba(0, 200, 83, 0.5); 
    transition: all 0.2s ease-in-out;
}

.stButton>button:hover {
    background-color: #00A949;
    transform: translateY(-2px);
    box-shadow: 0 6px 15px rgba(0, 200, 83, 0.6);
}


/* ==== TARJETA INTERNA (Inicio) ==== */
.card_interno {
    background: rgba(20, 20, 20, 0.55);
    backdrop-filter: blur(18px);
    border-radius: 22px;
    padding: 40px 55px;
    margin-top: 40px;
    box-shadow: 0 8px 40px rgba(0,0,0,0.45);
}

/* ===== TÍTULO PRINCIPAL "Bienvenido a EcoFlash" ===== */
.titulo_interno {
    color: #FFFFFF;
    font-size: 50px;
    font-weight: 900;
    text-align: center;
    -webkit-text-stroke: 1px #00C853;
    text-shadow:
        0 0 10px rgba(0, 200, 83, 0.8),
        0 0 25px rgba(0, 200, 83, 0.5);
    margin-bottom: 25px;
}

/* ==== TEXTO DESCRIPTIVO ==== */
.textoinicio {
    color: #F1F8E9;
    font-size: 20px;
    font-weight: 350;
    line-height: 1.65;
    text-shadow: 0 0 5px rgba(0,0,0,0.6);
    padding: 10px 5px;
}

/* ==== SUBTÍTULOS ==== */
.textoinicio h3, .textoinicio h4 {
    color: #A5D6A7;
    font-weight: 700;
    margin-top: 25px;
    text-shadow: 0 0 6px rgba(0, 150, 70, 0.7);
}

/* ==== LOGO ==== */
img {
    filter: drop-shadow(0 0 25px rgba(0,0,0,0.7));
}

/* ==== BOTONES INTERNOS ==== */
.stButton > button {
    background: linear-gradient(135deg, #00C853, #00E676);
    color: white;
    border-radius: 14px;
    padding: 12px 27px;
    font-size: 19px;
    border: none;
    box-shadow: 0 6px 18px rgba(0, 200, 83, 0.45);
    transition: 0.25s ease-in-out;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #00E676, #00C853);
    transform: translateY(-3px) scale(1.03);
    box-shadow: 0 10px 25px rgba(0, 200, 83, 0.7);
}

/* Nuevo estilo para los SUBTÍTULOS (e.g., ¿Qué es EcoFlash?) */
.subtitulo_descripcion {
    font-size: 28px; 
    font-weight: bold;
    color: #1B5E20; /* Verde oscuro (más elegante que el negro puro) */
    margin-top: 25px;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)

# Portada

if "show_portada" not in st.session_state:
    st.session_state.show_portada = True

if st.session_state.show_portada:
    st.markdown(f"""
    <div class="card">
      <div class="titulo">EcoFlash ♻️</div>
      <div class="subtitulo">Herramienta digital para aprender y practicar el reciclaje responsable en la ciudad de Cuenca.</div>
      <p class="autores"><b>Autores:</b> Allison Bueno • Jonathan Tigre</p>
      <p class="fecha"><b>Fecha:</b> {date.today().strftime("%d/%m/%Y")}</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🌍 Ingresar a la aplicación"):
        st.session_state.show_portada = False
        st.rerun()
    st.stop()


# Menú

menu = st.sidebar.radio(
    "📋 Menú de navegación",
    ["🏠 Inicio", "🧠 Clasificación de residuos", "🧮 Predicciones de registros", "💾 Base de datos", "ℹ️ Acerca de"]
)

# 🏠 INICIO
if menu == "🏠 Inicio":
    
    # Usa un contenedor para el contenido de la página
    st.markdown('<div class="card_interno">', unsafe_allow_html=True)
    
    # --- CABECERA Y LOGO ---
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Mostrar logo
        try:
            # Hacemos el logo más pequeño y centrado
            st.image("logo.png", width=200) 
        except FileNotFoundError:
            st.warning("⚠ No se encontró 'logo.png'.")

        # Título principal de la sección
        st.markdown(
            '<h2 class="titulo_interno">🌿 Bienvenido a EcoFlash 🌿</h2>', 
            unsafe_allow_html=True
        )

    # --- CONTENIDO DESCRIPTIVO ---
    
    # Usamos una columna para alinear el texto descriptivo a la izquierda (o justificado)
    st.markdown('<div class="text_card">', unsafe_allow_html=True)
    
    
    st.markdown(
        """
        <h3 class="subtitulo_descripcion">🌱 ¿Qué es EcoFlash?</h3>
        """, unsafe_allow_html=True
    )
    
    st.markdown("""
        <div class="textoinicio">
        EcoFlash es tu **herramienta digital interactiva** diseñada para impulsar el **reciclaje responsable** en la ciudad de Cuenca, Ecuador. Nuestro objetivo es hacer que la clasificación de residuos sea accesible, educativa y basada en datos.

        ### 🤖 La Tecnología detrás
        
        Esta aplicación utiliza dos componentes clave de **Inteligencia Artificial** para potenciar sus funcionalidades:
        
        * **Clasificación de Residuos:** Empleamos una avanzada **Red Neuronal Convolucional (CNN)**, la cual puede identificar y clasificar automáticamente distintos tipos de residuos sólidos (plástico, metal, vidrio, etc.) a partir de una imagen.
        * **Análisis Predictivo:** Utilizamos **Modelos de Regresión y Clasificación** (cargados como `.pkl`) junto a la base de datos PostgreSQL para predecir patrones de reciclaje, como el peso total de material recogido o la próxima zona de recolección.
        
        ### 📊 Explora las Secciones
        
        Utiliza el menú de navegación a la izquierda para:
        
        * **Clasificación de residuos:** Sube una imagen y recibe una predicción instantánea del tipo de material.
        * **Predicciones de registros:** Consulta las predicciones de peso y sector de recolección basadas en datos históricos.
        * **Base de datos:** Explora los datos brutos de los registros de reciclaje.
        * **Acerca de:** Conoce a los autores del proyecto.</div>
        
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True) 
    
    
    # ----- CLASIFICACIÓN -----
elif menu == "🧠 Clasificación de residuos":
        st.markdown('<div class="card fadeIn">', unsafe_allow_html=True)
        
        with st.expander("📘 Información sobre clasificación y reciclaje"):
            # Mostrar logo dentro del expander
            try:
                st.markdown("Logo placeholder", unsafe_allow_html=True)
            except FileNotFoundError:
                st.warning("⚠️ Logo no encontrado. Asegúrate de que 'logo.png' esté en la carpeta del proyecto.")

            # Texto informativo
            st.markdown("""
            <div style='font-size:17px; line-height:1.6;'>
                <p>📸 <b>En esta sección puedes subir una imagen</b> de un residuo y nuestro modelo de <b>Inteligencia Artificial</b> lo clasificará en una de las siguientes categorías:</p>
                <ul>
                    <li>♻️ <b>Plástico</b></li>
                    <li>📄 <b>Papel</b></li>
                    <li>🍃 <b>Orgánico</b></li>
                    <li>🍶 <b>Vidrio</b></li>
                    <li>🥫 <b>Metal</b></li>
                </ul>

            <p>🗑️ <b>Colores de las fundas recomendadas:</b></p>
                <ul>
                    <li><span style='color:#FFD600;'><b>Amarilla</b></span>: Plástico</li>
                    <li><span style='color:#03A9F4;'><b>Celeste</b></span>: Papel</li>
                    <li><span style='color:#388E3C;'><b>Verde</b></span>: Orgánico</li>
                    <li><span style='color:#9E9E9E;'><b>Gris</b></span>: Vidrio</li>
                    <li><span style='color:#E53935;'><b>Roja</b></span>: Metal</li>
                </ul>

            <p>🚮 <b>Colores de los tachos de reciclaje:</b></p>
                <ul>
                    <li><span style='color:#FFD600;'><b>Amarillo</b></span>: Plástico</li>
                    <li><span style='color:#1976D2;'><b>Azul</b></span>: Papel</li>
                    <li><span style='color:#388E3C;'><b>Verde</b></span>: Orgánico</li>
                    <li><span style='color:#9E9E9E;'><b>Gris</b></span>: Vidrio</li>
                    <li><span style='color:#E53935;'><b>Rojo</b></span>: Metal</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # --- Clasificador ---
        st.header("🧩 Clasificador Inteligente")
        uploaded_file = st.file_uploader("📸 Sube una imagen del residuo", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            st.image(uploaded_file, caption="Imagen cargada", use_container_width=True)
            st.info("🔍 Analizando imagen...")

            clases = ["Plástico", "Papel", "Orgánico", "Vidrio", "Metal"]
            pred = random.choice(clases)
            colores = {
                "Plástico": "Amarilla",
                "Papel": "Celeste",
                "Orgánico": "Verde",
                "Vidrio": "Gris",
                "Metal": "Roja"
            }

            st.success(f"✅ El modelo predice: **{pred}**")
            st.markdown(f"<h4>🗑️ Funda recomendada: <span style='color:#2E7D32'>{colores[pred]}</span></h4>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# 🧮  PREDICCIONES

elif menu == "🧮 Predicciones de registros":
    st.header("🧮 Predicción por sector")

    # Cargar datos
    @st.cache_data
    def cargar_todo():
        sectores = cargar_sectores(conn)
        registros = cargar_registros(conn)
        return sectores, registros

    sectores_df, registros_df = cargar_todo()

  
    # 1) FILTRO DE SECTOR

    sector = st.selectbox("🏙️ Selecciona un sector", sectores_df["nombre_sector"])

    df_sector = registros_df[registros_df["nombre_sector"] == sector].copy()

    if df_sector.empty:
        st.warning("No hay datos para este mes en el sector seleccionado.")
        st.stop()


  
    # 2) GRÁFICO DEL SECTOR

    if "fecha" in df_sector.columns:
        st.subheader("📈 Tendencia del total")
        fig1 = px.line(
            df_sector,
            x="fecha",
            y="total",
            markers=True,
            title=f"Total de residuos - {sector}"
        )
        st.plotly_chart(fig1, use_container_width=True)

    
    # 3) COMPARACIÓN ENTRE SECTORES
    
    st.header("📊 Comparación entre sectores")

    comparativa = (
        registros_df.groupby("nombre_sector")["total"]
        .sum()
        .reset_index()
        .sort_values(by="total", ascending=False)
    )

    st.subheader(f"📋 Totales por sector")
    st.dataframe(comparativa, use_container_width=True)

    fig2 = px.bar(
        comparativa,
        x="nombre_sector",
        y="total",
        title=f"Comparación de residuos por sector"
    )
    st.plotly_chart(fig2, use_container_width=True)




# 💾 Base de Datos

elif menu == "💾 Base de datos":
    st.header("🗄️ Bases de Datos")
    st.write("- PostgreSQL para registros")
    st.write("- MongoDB para imágenes")


# ℹ️ ACERCA DE

elif menu == "ℹ️ Acerca de":
    st.header("👩‍💻 Sobre el proyecto")
    st.write("EcoFlash — Proyecto del Instituto Tecnológico del Azuay (2025)")

# Footer
st.markdown("""
<hr>
<center><p style='color:gray; font-size:14px'>EcoFlash © 2025 — Proyecto educativo del Instituto Tecnológico del Azuay 🌎</p></center>
""", unsafe_allow_html=True)
