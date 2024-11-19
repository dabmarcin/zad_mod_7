import json
import streamlit as st
import pandas as pd  # type: ignore
from pycaret.clustering import load_model, predict_model  # type: ignore
import plotly.express as px  # type: ignore

MODEL_NAME = 'welcome_survey_clustering_pipeline_v2'
DATA = 'welcome_survey_simple_v2.csv'
CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v2.json'


@st.cache_data
def get_model():
    return load_model(MODEL_NAME)


@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.loads(f.read())


@st.cache_data
def get_all_participants():
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(model, data=all_df)
    return df_with_clusters


# Sidebar - Użytkownik wprowadza dane
with st.sidebar:
    st.header("Powiedz nam coś o sobie")
    st.markdown("Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")
    age = st.selectbox("Wiek", ['<18', '25-34', '45-54', '35-44', '18-24', '>=65', '55-64', 'unknown'])
    edu_level = st.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'])

    person_df = pd.DataFrame([{
        'age': age,
        'edu_level': edu_level,
        'fav_animals': fav_animals,
        'fav_place': fav_place,
        'gender': gender,
    }])

model = get_model()
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()

# Mapowanie identyfikatorów klastrów na nazwy
cluster_names = {k: v["name"] for k, v in cluster_names_and_descriptions.items()}

# Predykcja grupy użytkownika
predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]

st.header(f"Najbliżej Ci do grupy {predicted_cluster_data['name']}")
st.markdown(predicted_cluster_data['description'])
same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
st.metric("Liczba twoich znajomych", len(same_cluster_df))

# Histogramy dla grupy użytkownika
st.header("Osoby z grupy")
fig = px.histogram(same_cluster_df.sort_values("age"), x="age")
fig.update_layout(
    title="Rozkład wieku w grupie",
    xaxis_title="Wiek",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="edu_level")
fig.update_layout(
    title="Rozkład wykształcenia w grupie",
    xaxis_title="Wykształcenie",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_animals")
fig.update_layout(
    title="Rozkład ulubionych zwierząt w grupie",
    xaxis_title="Ulubione zwierzęta",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_place")
fig.update_layout(
    title="Rozkład ulubionych miejsc w grupie",
    xaxis_title="Ulubione miejsce",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="gender")
fig.update_layout(
    title="Rozkład płci w grupie",
    xaxis_title="Płeć",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

# Dodanie porównania z innymi grupami
st.header("Porównanie z innymi grupami")
with st.sidebar:
    compare_with_cluster_name = st.selectbox(
        "Porównaj swoją grupę z:",
        options=[cluster_names[key] for key in cluster_names.keys()],
    )

# Znajdź odpowiedni identyfikator klastra na podstawie wybranej nazwy
compare_with_cluster_id = [
    key for key, value in cluster_names.items() if value == compare_with_cluster_name
][0]

compare_df = all_df[all_df["Cluster"] == compare_with_cluster_id]

st.subheader(f"Porównanie z grupą {compare_with_cluster_name}")
fig = px.histogram(compare_df, x="age", color_discrete_sequence=["#FF5733"], title="Porównanie wieku")
fig.add_trace(px.histogram(same_cluster_df, x="age", color_discrete_sequence=["#33FF57"]).data[0])
fig.update_layout(barmode="group", xaxis_title="Wiek", yaxis_title="Liczba osób")
st.plotly_chart(fig)

fig = px.histogram(compare_df, x="fav_animals", color_discrete_sequence=["#FF5733"], title="Porównanie ulubionych zwierząt")
fig.add_trace(px.histogram(same_cluster_df, x="fav_animals", color_discrete_sequence=["#33FF57"]).data[0])
fig.update_layout(
    barmode="group",  # Słupki obok siebie
    xaxis_title="Wiek",
    yaxis_title="Liczba osób",
    title="Porównanie wieku",
    showlegend=True  # Dodanie legendy
    )
st.plotly_chart(fig)

# Rekomendacje znajomych
st.header("Propozycje znajomych")
recommended_friends = same_cluster_df[
    (same_cluster_df["fav_place"] == person_df.iloc[0]["fav_place"]) &
    (same_cluster_df["gender"] == person_df.iloc[0]["gender"])
]

st.write("Osoby o podobnych zainteresowaniach:")
st.dataframe(recommended_friends[["age", "edu_level", "fav_animals", "fav_place", "gender"]].head(5))
