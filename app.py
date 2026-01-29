import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Configuration de la page
st.set_page_config(page_title="DiagCardio", page_icon="❤️")

# Charger le modèle
# Assure-toi que le fichier est bien dans le même dossier que ce script
with open('modele_cardiaque.pkl', 'rb') as f:
    model = pickle.load(f)

# --- EN-TÊTE ET DESCRIPTION ---
st.title("❤️ Assistant de Diagnostic Cardiaque")
st.markdown("""
Cette application utilise un modèle de **Machine Learning (Régression Logistique)** entraîné sur des données cliniques pour évaluer la probabilité qu'un patient souffre d'une maladie cardiaque.
*Veuillez remplir les informations médicales ci-dessous pour obtenir une analyse instantanée.*
""")

st.info("**ATTENTION ⚠️:** Cet outil est à but éducatif uniquement et ne remplace en aucun cas un avis médical professionnel. En cas de soucis, contactez un médecin.")
st.divider()

# --- FORMULAIRE DE SAISIE ---

age = st.slider("Âge", 1, 100, 50)
sex = st.selectbox("Sexe", options=[0, 1], format_func=lambda x: "Homme" if x == 1 else "Femme")
cp = st.selectbox("Type de douleur thoracique", options=[1, 2, 3, 4])
blood_press = st.number_input("Pression Artérielle (mmHg)", value=120)
chol = st.number_input("Cholestérol (mg/dL)", value=200)
fbs = st.selectbox("Glycémie > 120 mg/dL", options=[0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
restecg = st.selectbox("Résultats EKG", options=[0, 1, 2]) # Corrigé : 'options' au lieu de 'option'
thalach = st.number_input("Rythme Cardiaque Max", value=150)
exang = st.selectbox("Angine d'exercice", options=[0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
oldpeak = st.number_input("Dépression ST", value=0.2, format="%.1f")
slope = st.selectbox("Pente ST (Slope)", options=[1, 2, 3])
ca = st.selectbox("Nb vaisseaux (Fluro)", options=[0, 1, 2, 3])
thal = st.selectbox("Thallium", options=[1, 2, 3, 4, 5, 6, 7])

# --- PRÉDICTION ---
if st.button("Lancer la prédiction"):
    # Préparation des données (Ordre respecté)
    donnees_patient = np.array([[
        age, sex, cp, blood_press, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    ]])

    # Calcul des prédictions
    prediction = model.predict(donnees_patient)
    probabilite = model.predict_proba(donnees_patient)[0][1]

    # Affichage des résultats
    if prediction[0] == 1:
        st.error(f"Risque de maladie cardiaque détecté ({probabilite:.2%})")
    else:
        st.success(f"Faible risque détecté ({probabilite:.2%})")

# --- PIED DE PAGE (FOOTER) ---
st.markdown("---")
footer = """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: black;
        text-align: center;
        padding: 12px;
        font-size: 19px;
    }
    </style>
    <div class="footer">
        <p>© 2026 - Tous droits réservés - Développé par @Thegretia sur github</p>
    </div>
"""
st.markdown(footer, unsafe_allow_html=True)