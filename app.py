"""
Streamlit application for the EAFIT LLM/Agent evaluation.

This app demonstrates how a lightweight language model can be
combined with an exploratory data analysis to build an interactive
assistant focused on economic data.  Users can load their own CSV
files or explore a default dataset downloaded from Yahoo! Finance.
The app performs a simple EDA, displays tables and plots, and
provides a chat interface to ask questions about the data or general
economic concepts.  Responses are generated locally using a small
causal language model from HuggingFace, avoiding the need for API
keys or external services.

Running the app:

    streamlit run app.py

Dependencies are declared in ``requirements.txt``.  When deploying
to Streamlit Cloud or Hugging Face Spaces, ensure that this file
exists in the root of your repository.

"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

import pandas as pd

from eda import (
    load_stock_data,
    compute_descriptive_statistics,
    identify_missing_values,
    create_correlation_heatmap,
    create_time_series_plot,
    dataframe_to_markdown,
)

from agent import generate_response


def main() -> None:
    st.set_page_config(page_title="Agente LLM para Economía", layout="wide")
    st.title("Evaluación 02 – Agente LLM/EDA para Economía")
    st.markdown(
        """
        Este prototipo demuestra cómo un **agente de lenguaje** puede integrarse con un
        análisis exploratorio de datos (EDA) para proporcionar información
        relevante sobre indicadores económicos.  Puedes cargar tu propio
        archivo CSV con al menos 300 filas y 6 columnas o utilizar el dataset
        de ejemplo basado en datos bursátiles de Yahoo! Finance.
        """
    )

    # Sidebar for dataset selection
    st.sidebar.header("Carga de datos")
    uploaded_file = st.sidebar.file_uploader(
        "Sube un archivo CSV", type=["csv"], help="Debe tener al menos 300 filas y 6 columnas."
    )
    default_symbol = st.sidebar.text_input(
        "Símbolo de la acción para el dataset de ejemplo", value="GOOG", max_chars=10
    )
    period = st.sidebar.selectbox(
        "Periodo del dataset de ejemplo",
        options=["1y", "2y", "5y", "max"],
        index=1,
        help="Cuántos años de historial descargar para el dataset de ejemplo."
    )

    # Load the data
    if uploaded_file is not None:
        # Read uploaded CSV
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")
            return
        data_source = "Datos cargados por el usuario"
    else:
        # Download from yfinance
        with st.spinner("Descargando datos de ejemplo..."):
            df = load_stock_data(default_symbol, period)
        data_source = f"Datos de {default_symbol} ({period}) descargados de Yahoo! Finance"
        if df.empty:
            st.error("No se pudieron descargar los datos. Verifica el símbolo o tu conexión a Internet.")
            return

    st.subheader("Visualización de los datos")
    st.markdown(f"**Fuente:** {data_source}")
    st.dataframe(df, use_container_width=True)

    # EDA section
    st.subheader("Análisis Exploratorio de Datos (EDA)")

    # Descriptive statistics
    stats_df = compute_descriptive_statistics(df)
    st.markdown("**Estadísticas descriptivas:**")
    st.dataframe(stats_df.transpose(), use_container_width=True)

    # Missing values
    missing = identify_missing_values(df)
    st.markdown("**Valores nulos por columna:**")
    st.write(missing)

    # Correlation heatmap
    st.markdown("**Mapa de correlaciones:**")
    corr_path = create_correlation_heatmap(df)
    st.image(str(corr_path), caption="Matriz de correlación", use_column_width=True)

    # Time series plot of a selected numeric column
    numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if numeric_columns:
        column_choice = st.selectbox("Selecciona una columna para visualizar como serie de tiempo", numeric_columns)
        ts_path = create_time_series_plot(df, column_choice)
        st.image(str(ts_path), caption=f"Serie de tiempo de {column_choice}", use_column_width=True)

    st.divider()

    # Chat interface
    st.subheader("Chat con el agente de economía")
    st.markdown(
        """
        Puedes preguntarle al modelo sobre tendencias económicas generales,
        conceptos financieros o sobre los datos cargados.  Por ejemplo:

        • *¿Cuál es la media del precio de cierre y cómo puedo interpretarla?*
        • *Explica la diferencia entre inflación y deflación.*
        • *¿Qué muestra la correlación entre el volumen y el precio de cierre?*
        """
    )

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # List of (role, message)

    # Display existing chat messages
    for role, message in st.session_state.chat_history:
        if role == "user":
            st.chat_message("user").markdown(message)
        else:
            st.chat_message("assistant").markdown(message)

    # Text input for new question
    user_input = st.chat_input(placeholder="Escribe tu pregunta aquí...")
    if user_input:
        # Append user's message to history
        st.session_state.chat_history.append(("user", user_input))

        # Build context from EDA summarised statistics
        summary_markdown = dataframe_to_markdown(stats_df.transpose(), max_rows=stats_df.shape[0])
        context = (
            "Resumen del EDA:\n" + summary_markdown +
            "\n\nDescripción: La tabla anterior muestra estadísticas resumidas (media, desviación estándar, mínimo, mediana y máximo) "
            "para cada columna numérica del dataset seleccionado."
        )

        # Generate response using the agent
        with st.spinner("Generando respuesta..."):
            answer = generate_response(question=user_input, context=context)
        # Append model's answer to history
        st.session_state.chat_history.append(("assistant", answer))

        # Render the latest interaction
        st.chat_message("assistant").markdown(answer)


if __name__ == "__main__":
    main()