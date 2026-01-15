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
from dotenv import load_dotenv
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import os
import pickle

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent.parent
def get_base64_image(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode()
fondo_base64 = get_base64_image("app/assets/fondo.jpeg")


st.markdown(
    f"""
    <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{fondo_base64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* Capa oscura SIN BLUR */
        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.35);
            z-index: -1;
        }}
    </style>
    """,
    unsafe_allow_html=True
)
# Configuración de la base de datos
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT", "5432"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "sslmode": "require"
}
## | conexión a la base de datos
@st.cache_resource
def crear_conexion(db_config):
    try:
        return psycopg2.connect(**db_config)
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
        # Usar rutas absolutas para asegurar que los archivos estén accesibles
        path_modelo_regresion = os.path.join(str(MODEL_DIR), "modelo_total_peso.pkl")
        path_modelo_clasificacion = os.path.join(str(MODEL_DIR), "modelo_sector_nombre.pkl")
        path_le_placa = os.path.join(str(MODEL_DIR), "le_placa.pkl")
        path_le_sector = os.path.join(str(MODEL_DIR), "le_sector.pkl")

        # Verificar si los archivos existen y cargarlos con pickle
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

## cache data
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

    # Limpieza
    if "fecha_ingreso" in df.columns:
        df.rename(columns={"fecha_ingreso": "fecha"}, inplace=True)

    df["hora_ingreso"] = pd.to_datetime(df["hora_ingreso"], errors="coerce").dt.hour
    df["hora_salida"] = pd.to_datetime(df["hora_salida"], errors="coerce").dt.hour
    df["delta_peso"] = df["peso_salida"] - df["peso_inicial"]
    df["duracion_horas"] = df["hora_salida"] - df["hora_ingreso"]
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["dia_semana"] = df["fecha"].dt.dayofweek

    df["placa"] = df["placa"].fillna("DESCONOCIDA")

    return df
## clasificacion
# Cargar el modelo entrenado
MODEL_PATH = 'model/resnet50_hf_final.pt'  # Ajusta la ruta si es necesario
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(weights=None)
model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(512, 6)  # Asegúrate de que el número de clases sea correcto
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# Transformación para la inferencia
infer_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# Función de predicción
def predict_and_show(image, model, transform, class_names):
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)

    probs = probs.squeeze().cpu().numpy() * 100
    predicted = np.argmax(probs)
    predicted_class = class_names[predicted]

    return predicted_class, probs
# Asignación de colores de fundas
def get_bag_color(material):
    bag_colors = {
        "cartón": "marrón",      # Cartón va en funda marrón
        "vidrio": "verde",       # Vidrio va en funda verde
        "metal": "celeste",      # Metal va en funda celeste
        "papel": "celeste",      # Papel va en funda celeste
        "plástico": "celeste",   # Plástico va en funda celeste
        "basura": "negro"        # Basura va en funda negra
    }
    return bag_colors.get(material, "negro")  # Por defecto se asigna "negro" si no coincide
## funcion para guardar clasificacion
def guardar_clasificacion(conn, material, color_funda):
    if conn is None:
        return

    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO historial_clasificacion (material, color_funda)
                VALUES (%s, %s)
            """, (material, color_funda))
            conn.commit()
    except Exception as e:
        conn.rollback()
        st.error(f"❌ Error guardando clasificación: {e}")

st.markdown("""
<style>

/* ===== TEXTO GENERAL ===== */
.stApp, 
.stApp * {
    color: #FFFFFF !important;
}

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background: rgba(15, 15, 15, 0.45) !important;
    backdrop-filter: blur(10px);
}

/* ===== CARDS ===== */
.card {
    background: rgba(30, 30, 30, 0.40);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 40px 60px;
    margin-top: 15vh;
    text-align: center;
}

.card_interno {
    background: rgba(25, 25, 25, 0.45);
    border-radius: 20px;
    padding: 25px 35px;
}

/* ===== TÍTULOS ===== */
.titulo {
    font-size: 58px;
    font-weight: 900;
    color: #00C853;
}

.subtitulo {
    font-size: 26px;
    color: #E0E0E0;
}

.titulo_interno {
    font-size: 50px;
    font-weight: 900;
    text-align: center;
    -webkit-text-stroke: 1px #00C853;
}

.subtitulo_descripcion {
    color: #00FF88;
    font-size: 30px;
    font-weight: 900;
    text-align: center;
    margin-bottom: 26px;
    text-shadow:
        0px 2px 6px rgba(0,0,0,0.9),
        0px 0px 10px rgba(0,255,136,0.4);
}

/* ===== TEXTO INICIO ===== */
.textoinicio {
    background: rgba(0, 0, 0, 0.70);
    padding: 32px 38px;
    border-radius: 20px;
    max-width: 900px;
    margin: 0 auto;
}

.textoinicio p {
    font-size: 21px;
    line-height: 1.8;
    text-align: justify;
    text-shadow: 0px 2px 6px rgba(0,0,0,0.85);
}

.textoinicio strong {
    color: #00FF88;
    font-weight: 800;
}

/* ===== BOTONES ===== */
.stButton > button {
    background: linear-gradient(135deg, #00C853, #00E676);
    color: #FFFFFF;
    border-radius: 14px;
    padding: 12px 27px;
    font-size: 19px;
    border: none;
    transition: 0.25s;
}

/* ===== FILE UPLOADER ===== */
[data-testid="stFileUploader"] {
    background: rgba(15, 15, 15, 0.9) !important;
    border-radius: 15px;
    padding: 20px;
    border: 2px dashed #00FF88;
}

[data-testid="stFileUploader"] section {
    background: transparent !important;
}

[data-testid="stFileUploader"] label {
    font-size: 18px;
    font-weight: 700;
}

[data-testid="stFileUploader"] small {
    color: #CCCCCC !important;
}

[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #00C853, #00E676) !important;
    color: #000000 !important;
    border-radius: 10px !important;
    padding: 10px 18px !important;
    font-weight: 700 !important;
}

/* ===== SELECTBOX FIX ABSOLUTO ===== */

/* Caja del select */
[data-testid="stSelectbox"] div[aria-haspopup="listbox"] {
    background-color: #0B0B0B !important;
    border: 2px solid #00FF88 !important;
    border-radius: 14px !important;
}

/* TEXTO SELECCIONADO (ESTE ES EL BUENO) */
[data-testid="stSelectbox"] div[aria-haspopup="listbox"] span {
    color: #FFFFFF !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    background: transparent !important;
}

/* Flecha */
[data-testid="stSelectbox"] svg {
    fill: #00FF88 !important;
}

/* Dropdown */
div[role="listbox"] {
    background-color: #0B0B0B !important;
    border: 1px solid #00FF88 !important;
    border-radius: 12px !important;
}

/* Opciones */
div[role="option"] {
    color: #FFFFFF !important;
    font-size: 16px !important;
}

/* Hover opción */
div[role="option"]:hover {
    background-color: rgba(0, 255, 136, 0.25) !important;
}

</style>
""", unsafe_allow_html=True)


if "show_portada" not in st.session_state:
    st.session_state.show_portada = True

if st.session_state.show_portada:
    st.markdown(f"""
    <div class="card">
      <div class="titulo">EcoFlash ♻️</div>
      <div class="subtitulo">Herramienta digital para aprender y practicar el reciclaje responsable en Cuenca.</div>
      <p><b>Autores:</b> Allison Bueno • Jonathan Tigre • Justin Escalante</p>
      <p><b>Fecha:</b> {date.today().strftime("%d/%m/%Y")}</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🌍 Ingresar a la aplicación"):
        st.session_state.show_portada = False
        st.rerun()
    st.stop()


menu = st.sidebar.radio(
    "📋 Menú de navegación",
    ["🏠 Inicio", "🧠 Clasificación de residuos", "📊 Análisis de registros", "🗺️ Mapeo de cantones", "🏭 Empresas que más residuos generan", "ℹ️ Acerca de"]
)
if "ultima_prediccion_guardada" not in st.session_state:
    st.session_state.ultima_prediccion_guardada = None
if menu == "🏠 Inicio":

    st.markdown('<div class="card_interno">', unsafe_allow_html=True)
    st.markdown(
        '<h2 class="titulo_interno">🌿 Bienvenido a EcoFlash 🌿</h2>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="subtitulo_descripcion">🌱 ¿Qué es EcoFlash?</div>

    <div class="textoinicio">
        <p>
            EcoFlash es una herramienta digital interactiva diseñada para impulsar el
            <strong>reciclaje responsable</strong> en la ciudad de Cuenca.
            Utiliza una <strong>Red Neuronal Convolucional (CNN)</strong> para clasificar imágenes de residuos
            y recomendar el tipo correcto de funda o tacho según la normativa municipal.
            Permite visualizar información, registrar datos y analizar estadísticas ambientales.
            Con EcoFlash, reciclar es más fácil, rápido e inteligente ♻️✨
        </p>
    </div>
    """, unsafe_allow_html=True)
elif menu == "🧠 Clasificación de residuos":
    st.markdown('<div class="card_interno">', unsafe_allow_html=True)

    with st.expander("📘 Información sobre clasificación y reciclaje"):
        st.markdown("""
        <div style='font-size:17px; line-height:1.6;'>
            <p>📸 Sube una imagen y EcoFlash la clasificará automáticamente usando IA.</p>
            <ul>
                <li>♻️ Plástico</li>
                <li>📄 Papel</li>
                <li>🍶 Vidrio</li>
                <li>🥫 Metal</li>
                <li>🛑 Basura</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.header("🧩 Clasificador Inteligente")

    uploaded_file = st.file_uploader(
        "📸 Sube una imagen del residuo",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns([2, 3])

        with col1:
            st.image(image, caption="Imagen cargada", use_container_width=True)

        class_names = ['cartón', 'vidrio', 'metal', 'papel', 'plástico', 'basura']

        # 🔮 PREDICCIÓN
        prediccion, probabilidades = predict_and_show(
            image, model, infer_transforms, class_names
        )

        # 🎨 FUNDA
        bag_color = get_bag_color(prediccion)

        # 💾 GUARDAR SOLO UNA VEZ
        if st.session_state.ultima_prediccion_guardada != prediccion:
            guardar_clasificacion(conn, prediccion, bag_color)
            st.session_state.ultima_prediccion_guardada = prediccion

        with col2:
            st.markdown(f"""
            <div style="background-color: black; padding: 20px; border-radius: 10px; color: white;">
                <h3>🔮 Predicción: {prediccion}</h3>
                <ul>
                    <li>cartón: {probabilidades[0]:.2f}%</li>
                    <li>vidrio: {probabilidades[1]:.2f}%</li>
                    <li>metal: {probabilidades[2]:.2f}%</li>
                    <li>papel: {probabilidades[3]:.2f}%</li>
                    <li>plástico: {probabilidades[4]:.2f}%</li>
                    <li>basura: {probabilidades[5]:.2f}%</li>
                </ul>
                <hr>
                <p>🗑️ <b>Funda recomendada:</b> {bag_color}</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown(f"<div style='background-color: #f0ad4e; padding: 10px; border-radius: 5px; color: black;'><strong>💡 El material debe ser guardado en una funda de color: {bag_color}</strong></div>", unsafe_allow_html=True) # Añadir un diseño más limpio y evitar la duplicación de la imagen st.markdown("</div>", unsafe_allow_html=True)

#    # 📜 HISTORIAL
 #   st.subheader("📜 Historial de clasificaciones")

#  @st.cache_data(ttl=5)
#    def cargar_historial():
#        return pd.read_sql("""
#            SELECT material, color_funda, fecha
#            FROM historial_clasificacion
#            ORDER BY fecha DESC
#            LIMIT 10
#        """, conn)

#    df_historial = cargar_historial()
#    st.dataframe(df_historial, use_container_width=True)



elif menu == "📊 Análisis de registros":
    st.markdown('<div class="card_interno">', unsafe_allow_html=True)
    st.header("📊 Análisis de residuos – Año 2024")

    # =========================
    # CARGA OPTIMIZADA DE DATOS
    # =========================
    @st.cache_data(show_spinner=False)
    def cargar_registros_cached():
        return cargar_registros(conn)

    registros_df = cargar_registros_cached()

    if registros_df.empty:
        st.warning("No existen registros disponibles.")
        st.stop()

    # =========================
    # PREPROCESAMIENTO
    # =========================
    if "fecha" in registros_df.columns:
        registros_df["fecha"] = pd.to_datetime(registros_df["fecha"])

    # =========================
    # EVOLUCIÓN DIARIA (TODOS LOS SECTORES)
    # =========================
    residuos_diarios = (
        registros_df
        .groupby(pd.Grouper(key="fecha", freq="D"))["total"]
        .sum()
        .reset_index()
    )

    fig1 = px.line(
        residuos_diarios,
        x="fecha",
        y="total",
        markers=True,
        title="📈 Evolión diaria del total de residuos – 2024"
    )

    fig1.update_layout(
        height=420,
        xaxis_title="Fecha",
        yaxis_title="Total de residuos"
    )

    st.plotly_chart(fig1, use_container_width=True)


    # =========================
    # TOTAL POR SECTOR (ANUAL)
    # =========================
    st.subheader("📊 Total de residuos por sector (2024)")

    total_por_sector = (
        registros_df
        .groupby("nombre_sector", as_index=False)["total"]
        .sum()
        .sort_values(by="total", ascending=False)
    )

    fig2 = px.bar(
        total_por_sector,
        x="nombre_sector",
        y="total",
        title="♻️ Total anual de residuos por sector – 2024",
        text_auto=True,
        color="nombre_sector"
    )

    fig2.update_layout(height=450)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # =========================
    # CARGA DE DATOS
    # =========================
    @st.cache_data(show_spinner=False)
    def cargar_registros_cached():
        return cargar_registros(conn)

    registros_df = cargar_registros_cached()

    if registros_df.empty:
        st.warning("No existen registros para realizar la predicción.")
        st.stop()

    # =========================
    # PREPROCESAMIENTO
    # =========================
    registros_df["fecha"] = pd.to_datetime(registros_df["fecha"])

    residuos_diarios = (
        registros_df
        .groupby(pd.Grouper(key="fecha", freq="D"))["total"]
        .sum()
        .reset_index()
        .sort_values("fecha")
    )

    st.info("Presiona el botón para generar la predicción de residuos.")

    
    if st.button("🔮 Generar predicción"):
        
        # Promedio móvil como predicción simple
        residuos_diarios["predicción"] = (
            residuos_diarios["total"]
            .rolling(window=7)
            .mean()
        )

        fig = px.line(
            residuos_diarios,
            x="fecha",
            y=["total", "predicción"],
            title="📈 Residuos reales vs predicción (tendencia)",
            labels={"value": "Total de residuos", "variable": "Serie"}
        )

        fig.update_layout(
            height=450,
            xaxis_title="Fecha",
            yaxis_title="Residuos"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.success("Predicción generada correctamente.")

    st.markdown("</div>", unsafe_allow_html=True)


elif menu == "🗺️ Mapeo de cantones":

    st.markdown('<div class="card_interno">', unsafe_allow_html=True)
    st.header("🗺️ Mapeo de Cantones – Residuos 2024")

    data_cantones = {
        "canton": [
            "Cañar", "Chordeleg", "Deleg", "El Pan", "Guachapala",
            "Gualaceo", "Saraguro", "Sevilla de Oro", "Sigsig"
        ],
        "total_ton": [
            279.35, 1381.22, 610.55, 203.04, 361.82,
            5948.98, 1832.75, 371.57, 2249.54
        ],
        "porcentaje": [
            2.11, 10.43, 4.61, 1.53, 2.73,
            44.94, 13.84, 2.81, 16.99
        ],
        "lat": [
            -2.560, -2.922, -2.734, -2.720, -2.783,
            -2.892, -3.606, -2.744, -2.910
        ],
        "lon": [
            -78.940, -78.780, -78.840, -78.850, -78.820,
            -78.780, -79.210, -78.760, -78.780
        ]
    }

    df_map = pd.DataFrame(data_cantones)

    st.subheader("📍 Selecciona un cantón")
    canton_sel = st.selectbox("Cantón", df_map["canton"])

    df_sel = df_map[df_map["canton"] == canton_sel]

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "♻️ Total de residuos (TON)",
            f"{df_sel['total_ton'].values[0]:,.2f}"
        )

    with col2:
        st.metric(
            "📊 Porcentaje del total",
            f"{df_sel['porcentaje'].values[0]} %"
        )

    st.markdown("---")


    st.subheader("🌍 Distribución geográfica de residuos")

    fig = px.scatter_mapbox(
        df_map,
        lat="lat",
        lon="lon",
        size="total_ton",
        color="total_ton",
        hover_name="canton",
        hover_data={
            "total_ton": True,
            "porcentaje": True,
            "lat": False,
            "lon": False
        },
        zoom=7,
        height=520,
        size_max=45,
        color_continuous_scale="greens"
    )

    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Comparación entre cantones")

    st.dataframe(
        df_map.sort_values("total_ton", ascending=False),
        use_container_width=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

elif menu == "🏭 Empresas que más residuos generan":

    st.markdown("## 🏭 Empresas que más residuos generan")
    st.markdown(
        "Visualización de los principales generadores de residuos según el total registrado."
    )

    data_generadores = {
        "Generador": [
            "ETAPA",
            "Cartopel",
            "Curtiembre Renaciente",
            "Plásticos Rival",
            "Almacenes Juan Eljuri",
            "Gerardo Ortiz e Hijos",
            "Centro Sur",
            "La Europea",
            "Embuandes",
            "Termovent",
            "Italimentos",
            "Piggis",
            "Projasa",
            "Vías del Austro (Grupo Graiman)",
            "Corporación Aeroportuaria de Cuenca (CORPAC)",
            "Austro Gas"
        ],
        "Total_Toneladas": [
            3963.05,
            275.13,
            101.65,
            87.45,
            41.44,
            48.20,
            26.36,
            28.71,
            18.57,
            18.95,
            15.06,
            14.40,
            2.72,
            1.10,
            0.33,
            0.05
        ]
    }

    df_generadores = pd.DataFrame(data_generadores)

    # =========================
    # TOP 10 FIJO
    # =========================
    top_n = 10

    df_top = (
        df_generadores
        .sort_values("Total_Toneladas", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    df_top.index += 1

    col1, col2 = st.columns(2)

    col1.metric(
        "🏭 Generador líder",
        df_top.loc[1, "Generador"]
    )

    col2.metric(
        "♻️ Toneladas del líder",
        f"{df_top.loc[1, 'Total_Toneladas']} ton"
    )

    st.divider()

    fig_bar = px.bar(
        df_top,
        x="Total_Toneladas",
        y="Generador",
        orientation="h",
        text="Total_Toneladas",
        title="📊 Top 10 Generadores Especiales",
        color="Generador",              # 🔥 cada empresa un color
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    fig_bar.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis_title="Toneladas",
        yaxis_title="Generador"
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### 📋 Detalle del ranking")
    st.dataframe(
        df_top.rename_axis("Ranking"),
        use_container_width=True
    )

    st.info(
        "📌 Este ranking permite identificar a los principales generadores de residuos, "
        "sirviendo como base para estrategias de control, reciclaje y gestión ambiental."
    )


elif menu == "ℹ️ Acerca de":

    st.header("👩‍💻 Sobre el proyecto")
    st.write("EcoFlash — Proyecto académico del Instituto Tecnológico del Azuay (2025)")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("app/assets/allison.png", width=120)
        st.markdown("**Allison Bueno**")
        st.caption("allison.bueno@tecazuay.edu.ec")

    with col2:
        st.image("app/assets/jonathan.png", width=120)
        st.markdown("**Jonathan Tigre**")
        st.caption("jonathan.tigre@tecazuay.edu.ec")

    with col3:
        st.image("app/assets/justin.png", width=120)
        st.markdown("**Justin Escalante**")
        st.caption("justin.escalante@tecazuay.edu.ec")

    st.markdown("""
    <hr>
    <center><p style='color:gray; font-size:14px'>EcoFlash © 2025 — Proyecto educativo del Instituto Tecnológico del Azuay 🌎</p></center>
    """, unsafe_allow_html=True)
