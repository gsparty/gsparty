import streamlit as st
from transformers import pipeline

import asyncio

# Ensure the event loop is running
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Initialisierung des Modells für Sentiment-Analyse
@st.cache_resource
def load_model():
    sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
    return sentiment_analyzer

# Lade das Modell
model = load_model()

# Titel der App
st.title("Sentiment-Analyse App")

# Eingabefeld für den Benutzertex
user_input = st.text_area("Gib hier deinen Text ein:")

# Wenn der Benutzer Text eingibt, analysiere den Sentiment
if user_input:
    result = model(user_input)
    sentiment = result[0]['label']
    score = result[0]['score']
    
    # Zeige das Ergebnis
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Vertrauenswahrscheinlichkeit: {score:.2f}")

    # Visualisierung der Ergebnisse
    if sentiment == 'POSITIVE':
        st.markdown("<h3 style='color:green;'>Positive Stimmung</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color:red;'>Negative Stimmung</h3>", unsafe_allow_html=True)