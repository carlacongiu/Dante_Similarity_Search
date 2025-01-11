import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import weaviate
from weaviate.classes.init import Auth
import spacy

# Titolo dell'app
st.title("Voci dall'Inferno - Analisi di Similitudine")

# Configurazione di base per Weaviate
WEAVIATE_URL = "https://rabbfo5drdqk3qa0lpjpvq.c0.europe-west3.gcp.weaviate.cloud"
WEAVIATE_API_KEY = "KIqaLabClZWRjtyT9G10ki3FhzqETU6x81yv"
COLLECTION_NAME = "Voci_dall_Inferno"

# Connessione a Weaviate
client = weaviate.Client(
    url=WEAVIATE_URL,
    auth_client_secret=Auth.api_key(WEAVIATE_API_KEY),
)

# Funzione per creare il modello
@st.cache_resource

def load_model():
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

model = load_model()

# Upload del file CSV
uploaded_file = st.file_uploader("Carica il file delle terzine (CSV):", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file, delimiter=';')
    st.write("Anteprima dei dati caricati:")
    st.dataframe(df.head())

    # Calcolo degli embeddings
    if st.button("Calcola gli embeddings"):
        with st.spinner("Calcolo degli embeddings..."):
            embeddings = model.encode(df.terzina.tolist(), show_progress_bar=True)
            df["embedding"] = list(embeddings)
            st.success("Embeddings calcolati con successo!")
            st.dataframe(df)

# Caricamento su Weaviate
    if st.button("Carica i dati su Weaviate"):
        with st.spinner("Caricamento dei dati su Weaviate..."):
            if client.schema.exists(COLLECTION_NAME):
                client.schema.delete(COLLECTION_NAME)
            
            client.schema.create(
                {
                    "classes": [
                        {
                            "class": COLLECTION_NAME,
                            "properties": [
                                {"name": "cantica", "dataType": ["string"]},
                                {"name": "canto", "dataType": ["string"]},
                                {"name": "range_versi", "dataType": ["string"]},
                                {"name": "terzina", "dataType": ["string"]}
                            ]
                        }
                    ]
                }
            )

            with client.batch as batch:
                for _, row in df.iterrows():
                    batch.add_data_object(
                        {
                            "cantica": row['cantica'],
                            "canto": row['canto'],
                            "range_versi": row['range_versi'],
                            "terzina": row['terzina'],
                        },
                        COLLECTION_NAME,
                        vector=row['embedding']
                    )

            st.success("Dati caricati con successo su Weaviate!")

# Query per similitudini
query = st.text_input("Inserisci un testo per cercare similitudini:")
if query:
    with st.spinner("Cercando similitudini..."):
        query_vector = model.encode([query])[0]
        response = client.query.get(COLLECTION_NAME, ["cantica", "canto", "range_versi", "terzina"]).with_near_vector({"vector": query_vector}).with_limit(10).do()

        if "data" in response and response["data"]["Get"][COLLECTION_NAME]:
            results = response["data"]["Get"][COLLECTION_NAME]
            st.write("Risultati trovati:")
            for result in results:
                st.write(f"**Cantica**: {result['cantica']}, **Canto**: {result['canto']}, **Versi**: {result['range_versi']}")
                st.write(f"**Testo**: {result['terzina']}")
                st.write("---")
        else:
            st.write("Nessun risultato trovato.")
