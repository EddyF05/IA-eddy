# rag_pipeline.py
import re
import spacy
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

# =============================
# 1. LIMPIEZA DE TEXTO
# =============================
nlp = spacy.load("es_core_news_sm")

def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"\n", " ", texto)
    texto = re.sub(r"[^\w\s]", "", texto)
    doc = nlp(texto)
    return " ".join([t.lemma_ for t in doc if not t.is_stop])

# =============================
# 2. CARGAR CORPUS
# =============================
with open("data/corpus.txt", "r", encoding="utf-8") as f:
    textos = f.readlines()

textos_limpios = [limpiar_texto(t) for t in textos]

# =============================
# 3. EMBEDDINGS (TF-IDF)
# =============================
vectorizer = TfidfVectorizer(max_features=300)
embeddings = vectorizer.fit_transform(textos_limpios).toarray()

# =============================
# 4. VECTOR STORE (FAISS)
# =============================
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# =============================
# 5. RETRIEVAL
# =============================
def recuperar_contexto(pregunta, k=3):
    pregunta_limpia = limpiar_texto(pregunta)
    q_embedding = vectorizer.transform([pregunta_limpia]).toarray()
    _, indices = index.search(q_embedding, k)
    return [textos[i] for i in indices[0]]
