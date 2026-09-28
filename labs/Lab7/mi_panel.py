"""Andamiaje del panel de la Parte 2. Renómbrenlo si quieren.

MATERIAL PROVISTO. Lo que ya está escrito acá —los imports, la configuración
de la página y el cargador de datos— es infraestructura y no se evalúa: son
las mismas líneas para cualquier panel sobre este dataset. Lo que sí se evalúa
son las cuatro secciones de más abajo.

Las secciones están en un orden que funciona, pero no es obligatorio:
reordenarlas o renombrarlas no descuenta. Lo que se corrige es que las cuatro
cosas estén y que cada gráfico explique por qué está ahí.

Para trabajar:

    uv run streamlit run mi_panel.py

Streamlit reejecuta el archivo completo cada vez que alguien mueve un control,
así que el navegador se actualiza solo al guardar.
"""

from pathlib import Path

import plotly.express as px
import polars as pl
import streamlit as st

RUTA_DATOS = Path(__file__).parent / "data" / "raw" / "penguins.csv"

st.set_page_config(
    page_title="Pingüinos de Palmer", page_icon="🐧", layout="wide"
)
st.title("🐧 Pingüinos del archipiélago de Palmer")


@st.cache_data
def cargar_datos() -> pl.DataFrame:
    """Lee el CSV sin modificarlo.

    `null_values=["NA"]` es necesario: el archivo viene de R, donde `NA` marca
    los faltantes. Sin ese argumento, polars lee las columnas numéricas como
    texto. Con él, los nulos quedan adentro — que es lo que queremos, porque
    este laboratorio no limpia nada.
    """
    return pl.read_csv(RUTA_DATOS, null_values=["NA"])


df = cargar_datos()

# raise NotImplementedError(
#     "Completen las cuatro secciones de este archivo y borren esta línea "
#     "antes de ejecutar el programa."
# )

# --- 1) La tabla interactiva -----------------------------------------------
#
# Una tabla con el dataset que el lector pueda ordenar por cualquier columna y
# filtrar con al menos dos controles: uno categórico y uno de rango numérico.
# Esos mismos filtros deben afectar también a los cuatro gráficos. Los
# registros sin valor en la columna del filtro numérico quedan fuera de la
# selección filtrada. Si ningún registro cumple los filtros, muestren un aviso.
#
# Ordenar y buscar los trae `st.dataframe` de fábrica, sin programar nada.
# Filtrar no: los controles devuelven la selección y ustedes filtran el
# DataFrame antes de pasárselo a la tabla. Denle formato a las columnas, que
# `flipper_length_mm` no es un encabezado para mostrarle a un cliente.
#
#   https://docs.streamlit.io/develop/api-reference/data/st.dataframe
#   https://docs.streamlit.io/develop/api-reference/data/st.column_config
#   https://docs.streamlit.io/develop/api-reference/widgets/st.multiselect
#   https://docs.streamlit.io/develop/api-reference/widgets/st.slider

# Su código aquí
species_box = st.selectbox("Especie", options=df["species"].unique().to_list())
body_mass_g_slider = st.slider(
    label="Masa corporal (g)",
    min_value=df["body_mass_g"].min(),
    max_value=df["body_mass_g"].max(),
    value=(df["body_mass_g"].min(), df["body_mass_g"].max()),
)

st.header("Tabla Interactiva General")
columns_config = {
    "species": st.column_config.TextColumn(
        "Especie",
        help="Especies de los pingüinos registrados",
        max_chars=100,
        width="medium",
    ),
    "island": st.column_config.TextColumn(
        "Isla",
        help="Isla de origen de los pingüinos registrados",
        max_chars=100,
        width="medium",
    ),
    "culmen_length_mm": st.column_config.NumberColumn(
        "Largo de pico",
        help="Largo del pico de los pingüinos registrados en unidad de milímetros (mm)",
        format="%.2f mm",
        min_value=df["culmen_length_mm"].min(),
        max_value=df["culmen_length_mm"].max(),
    ),
    "culmen_depth_mm": st.column_config.NumberColumn(
        "Alto de pico",
        help="Alto del pico de los pingüinos registrados en unidad de milímetros (mm)",
        format="%.2f mm",
        min_value=df["culmen_depth_mm"].min(),
        max_value=df["culmen_depth_mm"].max(),
    ),
    "flipper_length_mm": st.column_config.NumberColumn(
        "Largo de aleta",
        help="Largo de la aleta de los pingüinos registrados en unidad de milímetros (mm)",
        format="%.2f mm",
        min_value=df["flipper_length_mm"].min(),
        max_value=df["flipper_length_mm"].max(),
    ),
    "body_mass_g": st.column_config.NumberColumn(
        "Masa corporal",
        help="Masa corporal de los pingüinos registrados en unidad de gramos (g)",
        format="%.2f g",
        min_value=df["body_mass_g"].min(),
        max_value=df["body_mass_g"].max(),
    ),
    "sex": st.column_config.TextColumn(
        "Sexo",
        help="Sexo de los pingüinos registrados",
        max_chars=100,
        width="medium",
    ),
}
df_filtrado = df.filter(
    (pl.col("species") == species_box)
    & pl.col("body_mass_g").is_between(
        body_mass_g_slider[0], body_mass_g_slider[1]
    )
)
if df_filtrado.height > 0:
    st.dataframe(df_filtrado, column_config=columns_config)
else:
    st.warning("No hay registros que cumplan con los filtros seleccionados.")


# --- 2) La calidad de los datos --------------------------------------------
#
# Un informe visible en la página, calculado sobre el CSV completo aunque se
# apliquen filtros: qué columnas tienen nulos y cuántos, cuál es el valor
# inesperado de `sex` y cómo pueden afectar esos problemas los recuentos,
# filtros o gráficos del panel.
#
# Las cifras se calculan desde `df`, no se escriben a mano: si el
# archivo cambiara, un número escrito a mano queda mintiendo.
#
#   https://docs.streamlit.io/develop/api-reference/status/st.warning


# Su código aquí
st.header("Calidad de los datos")
df_null = df.null_count()
st.subheader("Conteo de nulos por columna")
st.dataframe(
    df_null,
    column_config={
        "species": st.column_config.TextColumn(
            "Especie", help="Especies de los pingüinos registrados"
        ),
        "island": st.column_config.TextColumn(
            "Isla", help="Isla de origen de los pingüinos registrados"
        ),
        "culmen_length_mm": st.column_config.NumberColumn(
            "Largo de pico",
            help="Largo del pico de los pingüinos registrados en unidad de milímetros (mm)",
        ),
        "culmen_depth_mm": st.column_config.NumberColumn(
            "Alto de pico",
            help="Alto del pico de los pingüinos registrados en unidad de milímetros (mm)",
        ),
        "flipper_length_mm": st.column_config.NumberColumn(
            "Largo de aleta",
            help="Largo de la aleta de los pingüinos registrados en unidad de milímetros (mm)",
        ),
        "body_mass_g": st.column_config.NumberColumn(
            "Masa corporal",
            help="Masa corporal de los pingüinos registrados en unidad de gramos (g)",
        ),
        "sex": st.column_config.TextColumn(
            "Sexo", help="Sexo de los pingüinos registrados"
        ),
    },
)
st.subheader("Cantidad de pingüinos por cada sexo registrado")
st.dataframe(
    df["sex"].value_counts().sort("count", descending=True),
    column_config={
        "sex": st.column_config.TextColumn(
            "Sexo", help="Sexo de los pingüinos registrados"
        ),
        "count": st.column_config.TextColumn(
            "Conteo",
            help="Cantidad acumulada de los pingüinos registrados asociados",
        ),
    },
)

null_messages = []

for column in df.columns:
    null_count = df_null[column][0]

    if null_count > 0:
        null_messages.append(
            f"La columna `{column}` contiene {null_count} valores nulos."
        )
null_report = "\n".join(null_messages)


valid_sex = ["MALE", "FEMALE"]

unexpected_messages = []

unexpected_df = df.filter(
    pl.col("sex").is_not_null() & ~pl.col("sex").is_in(valid_sex)
)

for value in unexpected_df["sex"].to_list():
    unexpected_messages.append(f"Valor inesperado encontrado: '{value}'")

unexpected_report = "\n".join(unexpected_messages)


st.warning(
    f"{null_report}\n\n"
    f"{unexpected_report}\n\n"
    "Para los atributos numéricos, los valores nulos implican que estos individuos no serán considerados en los recuentos, filtros y gráficos que utilicen dichas variables.\n\n"
    "Para los atributos categóricos, los valores inesperados pueden generar una interpretación incorrecta o no esperada de los datos."
)


# --- 3) Los cuatro gráficos ------------------------------------------------
#
# Cuatro gráficos a elección, de al menos dos tipos distintos. Pueden reusar
# los de la Parte 1 o construir otros. Van a necesitar `plotly.express`:
# impórtenlo arriba, con el resto.
#
# Cada gráfico lleva, JUNTO A ÉL Y VISIBLE EN LA PÁGINA, por qué esa
# información es útil y por qué eligieron esa visualización. Un comentario en
# el código no cuenta: quien abre el panel no lee el código.
#
#   https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart
#   https://docs.streamlit.io/develop/api-reference/text/st.caption
#   https://docs.streamlit.io/develop/api-reference/layout/st.columns

# Su código aquí


fig_explicativo = px.scatter(
    df_filtrado,
    x="flipper_length_mm",
    y="body_mass_g",
    title=(
        "Los pingüinos con aletas más largas tienden a tener mayor masa corporal; "
        "La especie Gentoo queda arriba y separado de Adelie y Chinstrap"
    ),
    labels={
        "flipper_length_mm": "Largo de aleta (mm)",
        "body_mass_g": "Masa corporal (g)",
    },
)

col_grafico, col_texto = st.columns([3, 1])

col_grafico.plotly_chart(fig_explicativo, use_container_width=True)

col_texto.subheader("Comentario:")
col_texto.write(
    "Se utiliza un gráfico de dispersión porque permite visualizar "
    "la relación entre las variables numéricas: el largo de las aletas y la masa corporal de los pingüinos, bajo los filtros aplicados."
)

fig_escala = px.scatter(
    df_filtrado,
    x="flipper_length_mm",
    y="culmen_length_mm",
    color="body_mass_g",
    color_continuous_scale="Viridis",
    labels={
        "flipper_length_mm": "Largo de aleta (mm)",
        "culmen_length_mm": "Largo del culmen (mm)",
        "body_mass_g": "Masa corporal (g)",
    },
)

col_grafico, col_texto = st.columns([3, 1])

col_grafico.plotly_chart(fig_escala, use_container_width=True)

col_texto.subheader("Comentario:")
col_texto.write(
    "Se utiliza un gráfico de dispersión para observar "
    "la relación entre dos variables numéricas: el largo de las aletas "
    "y el largo del pico, bajo los filtros aplicados."
)

fig_masa_hist = px.histogram(
    df_filtrado,
    x="body_mass_g",
    color="sex",
    barmode="overlay",
    opacity=0.9,
    marginal="violin",
    labels={
        "body_mass_g": "Masa corporal (g)",
        "count": "Cantidad de pingüinos",
        "sex": "Sexo",
    },
)
col_grafico, col_texto = st.columns([3, 1])

col_grafico.plotly_chart(fig_masa_hist, use_container_width=True)

col_texto.subheader("Comentario:")
col_texto.write(
    "Se utiliza un histograma junto con gráfico de violín para visualizar la distribución de la masa corporal de los pingüinos "
    "y comparar esta distribución según el sexo, bajo los filtros aplicados."
)

fig_aleta_caja = px.box(
    df_filtrado,
    x="island",
    y="flipper_length_mm",
    labels={
        "island": "Isla de Origen",
        "flipper_length_mm": "Largo de aleta (mm)",
    },
)

col_grafico, col_texto = st.columns([3, 1])

col_grafico.plotly_chart(fig_aleta_caja, use_container_width=True)

col_texto.subheader("Comentario:")
col_texto.write(
    "Se utiliza un gráfico de caja para comparar la distribución del largo de las aletas "
    "entre las distintas islas de origen, bajo los filtros aplicados."
)


# --- 4) El tema --------------------------------------------------------------
#
# Este no se programa acá: vive en `.streamlit/config.toml`, al lado de este
# archivo. Ya existe, con las claves comentadas — descoméntenlas y decidan sus
# colores.
#
# Para comprobar que el suyo está haciendo algo: renombren el archivo,
# reinicien el panel y vean si cambia. Si no cambia, no lo configuraron.
#
#   https://docs.streamlit.io/develop/concepts/configuration/theming
