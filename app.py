import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import os

# OpenAI API Key
os.environ["OPENAI_API_KEY"] = "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"

# App Title
st.title("Dokument-Hilfe: Dein persönlicher Frage-Assistent")

# Schritt 1: Dokument hochladen
st.header("1. Lade dein Dokument hoch")
uploaded_file = st.file_uploader("Ziehe hier eine PDF-Datei hinein oder wähle sie aus", type=["pdf"])

if uploaded_file:
    st.success(f"Datei {uploaded_file.name} erfolgreich hochgeladen!")

    # Speichere die Datei
    with open("temp_uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Schritt 2: Automatische Verarbeitung
    st.header("2. Dokument wird verarbeitet")
    st.info("Bitte warten... Wir erstellen den Frage-Assistenten basierend auf deinem Dokument.")

    try:
        loader = PyPDFLoader("temp_uploaded_file.pdf")
        documents = loader.load()

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents, embeddings)

        st.success("Der Frage-Assistent ist bereit!")

        # Schritt 3: Fragen stellen
        st.header("3. Stelle Fragen zu deinem Dokument")
        frage = st.text_input("Gib deine Frage ein:")

        if frage:
            llm = ChatOpenAI(temperature=0)
            docs = vectorstore.similarity_search(frage)
            antwort = llm.run(docs)
            st.write("Antwort:", antwort)

    except Exception as e:
        st.error(f"Ein Fehler ist aufgetreten: {e}")
