"""
Streamlit — Groupe d'une graine de blé (clustering DBSCAN, wheat_seeds_dataset)
Reprend l'application de clustering du prof : on recharge les points cœur de DBSCAN
sauvegardés dans le notebook et on applique sa règle à une nouvelle graine.

En local :  streamlit run app.py
"""

import numpy as np
import pandas as pd
import joblib as jb
import streamlit as st
from sklearn.preprocessing import normalize

st.set_page_config(page_title="Graines de blé", page_icon="🌾", layout="centered")


# ---------- Objets issus du notebook (chargés une seule fois) ----------
@st.cache_resource
def charger_objets():
    return jb.load("modele_dbscan.joblib")


artefacts = charger_objets()
points_coeur = artefacts["points_coeur"]
labels_coeur = artefacts["labels_coeur"]
eps = artefacts["eps"]
colonnes = artefacts["colonnes"]
valeurs_defaut = artefacts["valeurs_defaut"]
exemples = artefacts["exemples"]
noms_classes = artefacts["noms_classes"]

LIBELLES = {
    "area A": "Surface", "perimeter": "Périmètre", "compactness": "Compacité",
    "length of kernel": "Longueur du grain", "width of kernel": "Largeur du grain",
    "asymmetry coefficient": "Coefficient d'asymétrie", "length of kernel groove": "Longueur du sillon",
}


# ---------- Règle de DBSCAN pour une nouvelle graine ----------
def predire_classe(nouvelle_graine):
    graine = normalize(np.array(nouvelle_graine, dtype=float).reshape(1, -1))
    distances = np.linalg.norm(points_coeur - graine, axis=1)
    plus_proche = distances.argmin()
    if distances[plus_proche] <= eps:
        return int(labels_coeur[plus_proche])
    return -1


def libelle(classe):
    return "Graine intermédiaire (anomalie)" if classe == -1 else noms_classes[classe]


def predire_fichier(fichier):
    tableau = pd.read_csv(fichier)
    tableau["Groupe"] = [libelle(predire_classe(ligne[:len(colonnes)])) for ligne in tableau.values]
    return tableau


st.title("🌾 Groupe d'une graine de blé")
st.caption("Grosse ou petite graine ? Modèle DBSCAN entraîné sur 199 graines décrites par 7 mesures.")
onglet_un, onglet_csv = st.tabs(["Une graine", "Fichier CSV"])

with onglet_un:
    depart = st.selectbox("Pré-remplir avec", ["Valeurs médianes", "Graine 1", "Graine 2", "Graine 3"])
    defauts = valeurs_defaut if depart == "Valeurs médianes" else exemples[int(depart[-1]) - 1]
    valeurs = []
    gauche, droite = st.columns(2)
    for i, (nom, val) in enumerate(zip(colonnes, defauts)):
        zone = gauche if i % 2 == 0 else droite
        valeurs.append(zone.number_input(LIBELLES.get(nom, nom), value=float(val), format="%.4f", key=f"{nom}_{depart}"))

    if st.button("Trouver le groupe", type="primary", use_container_width=True):
        classe = predire_classe(valeurs)
        if classe == -1:
            st.warning("**Graine intermédiaire (anomalie)** : elle ne ressemble nettement à aucun des deux groupes.")
        else:
            st.success(f"**Groupe :** {noms_classes[classe]}")

with onglet_csv:
    st.info("Le fichier doit contenir les 7 mesures du jeu de données, dans le même ordre.")
    fichier = st.file_uploader("Choisir un fichier CSV", type="csv")
    if fichier is not None:
        try:
            tableau = predire_fichier(fichier)
            st.dataframe(tableau, use_container_width=True)
            st.download_button("Télécharger les résultats", tableau.to_csv(index=False).encode("utf-8"),
                               "groupes_graines.csv", "text/csv")
        except Exception as erreur:
            st.error(f"Fichier non traité : {erreur}")
