from rag_pipeline import recuperar_contexto

preguntas = [
    "¿La Generación Z vive una crisis de sentido?",
    "¿Los algoritmos influyen en la identidad?",
    "¿Existe burnout juvenil por presión digital?"
]

for p in preguntas:
    print("\nPREGUNTA:", p)
    for c in recuperar_contexto(p):
        print("-", c.strip())
