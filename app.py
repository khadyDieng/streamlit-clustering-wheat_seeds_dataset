"""
Streamlit — Groupe d'une graine de blé (clustering DBSCAN, wheat_seeds_dataset)

En local :  streamlit run app.py
"""

import numpy as np
import pandas as pd
import joblib as jb
import streamlit as st
from sklearn.preprocessing import normalize

st.set_page_config(page_title="Graines de blé", page_icon="🌾", layout="wide")


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

# libellé, minimum, maximum et pas de chaque curseur (bornes = plage observée dans les données)
CURSEURS = {
    "area A": ("Surface", 10.0, 22.0, 0.01),
    "perimeter": ("Périmètre", 12.0, 18.0, 0.01),
    "compactness": ("Compacité", 0.80, 0.93, 0.001),
    "length of kernel": ("Longueur du grain", 4.8, 6.8, 0.001),
    "width of kernel": ("Largeur du grain", 2.6, 4.1, 0.001),
    "asymmetry coefficient": ("Coefficient d'asymétrie", 0.5, 8.5, 0.01),
    "length of kernel groove": ("Longueur du sillon", 4.4, 6.7, 0.001),
}
# profils moyens des deux groupes (tableau « profil » du notebook)
PROFILS = pd.DataFrame(
    [[16.55, 15.33, 0.88, 5.89, 3.48, 3.07, 5.62], [11.85, 13.25, 0.85, 5.23, 2.84, 4.63, 5.11]],
    index=[noms_classes[0], noms_classes[1]], columns=[CURSEURS[c][0] for c in colonnes])


def predire_classe(nouvelle_graine):
    graine = normalize(np.array(nouvelle_graine, dtype=float).reshape(1, -1))
    distances = np.linalg.norm(points_coeur - graine, axis=1)
    plus_proche = distances.argmin()
    if distances[plus_proche] <= eps:
        return int(labels_coeur[plus_proche])
    return -1


def libelle(classe):
    return "Graine intermédiaire" if classe == -1 else noms_classes[classe]


# ---------- Panneau latéral : description de la graine ----------
with st.sidebar:
    st.header("🌾 La graine")
    depart = st.radio("Partir de", ["Graine médiane", "Graine 1", "Graine 2", "Graine 3"])
    defauts = valeurs_defaut if depart == "Graine médiane" else exemples[int(depart[-1]) - 1]
    valeurs = []
    for nom, val in zip(colonnes, defauts):
        texte, mini, maxi, pas = CURSEURS[nom]
        valeurs.append(st.slider(texte, mini, maxi, float(np.clip(val, mini, maxi)), pas, key=f"{nom}_{depart}"))

# ---------- Page principale : résultat en direct ----------
st.title("Grosse ou petite graine ?")
st.write("Réglez les mesures dans le panneau de gauche : le groupe se met à jour immédiatement.")

classe = predire_classe(valeurs)
carte, explication = st.columns([1, 2])
with carte:
    st.metric("Groupe estimé", libelle(classe))
with explication:
    if classe == -1:
        st.warning("Cette graine est trop éloignée des zones denses des deux groupes : DBSCAN la considère comme "
                   "une anomalie. Dans les données, ce sont surtout des graines de taille intermédiaire.")
    elif classe == 0:
        st.success("Graine de grande taille : surface et largeur élevées, faible asymétrie.")
    else:
        st.info("Graine de petite taille : surface et largeur faibles, asymétrie plus marquée.")

st.subheader("Comparaison avec les profils moyens")
comparaison = pd.concat([pd.DataFrame([valeurs], index=["Cette graine"], columns=PROFILS.columns), PROFILS])
st.dataframe(comparaison.round(2), width="stretch")

with st.expander("Classer un fichier de graines (CSV)"):
    st.caption("Le fichier doit contenir les 7 mesures, dans l'ordre du jeu de données.")
    fichier = st.file_uploader("Fichier CSV", type="csv")
    if fichier is not None:
        try:
            tableau = pd.read_csv(fichier)
            tableau["Groupe"] = [libelle(predire_classe(l[:len(colonnes)])) for l in tableau.values]
            st.dataframe(tableau, width="stretch")
            st.download_button("Télécharger", tableau.to_csv(index=False).encode("utf-8"), "groupes_graines.csv", "text/csv")
        except Exception as erreur:
            st.error(f"Fichier non traité : {erreur}")
