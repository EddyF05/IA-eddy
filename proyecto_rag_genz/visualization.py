import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from rag_pipeline import embeddings

pca = PCA(n_components=2)
coords = pca.fit_transform(embeddings)

plt.scatter(coords[:,0], coords[:,1])
plt.title("Mapa semántico – Crisis de sentido Gen Z")
plt.show()
