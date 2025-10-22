# ================== PREGUNTA 1 — TF-IDF + KMEANS ==================
# Reqs: pandas, scikit-learn, numpy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer   # Ítem 1: usar TfidfVectorizer en vez de CountVectorizer
from sklearn.cluster import KMeans
import numpy as np

# ---- Lectura de datos (solo columnas requeridas) ----
datos = pd.read_csv('smogon.csv')
datos = datos[['Pokemon', 'moves']].dropna().reset_index(drop=True)
datos['moves'] = datos['moves'].astype(str).str.lower()

# ---- Ítem 1: generar matriz TF-IDF con n-gramas elegidos (aquí trigramas y tetragramas, como en clase) ----
vectorizer = TfidfVectorizer(ngram_range=(3,4))
X = vectorizer.fit_transform(datos['moves'])

# ---- Ítem 2: mostrar número total de columnas (tamaño del vocabulario) ----
tokens = vectorizer.get_feature_names_out()
print("\n[Ítem 2] Nº columnas (vocabulario):", len(tokens))

# ---- Ítem 3: imprimir TODOS los tokens (vocabulario) ----
print("[Ítem 3] Tokens (vocabulario):")
print(list(tokens))  # puede ser muy largo

# ---- Ítem 4: DataFrame con la matriz TF-IDF e imprimirla (muestra para no colgar) ----
matriz = pd.DataFrame(X[:30].toarray(), columns=tokens)  # muestra 30 filas; si exigen todo, usa X.toarray()
matriz.insert(0, 'Pokemon', datos['Pokemon'].iloc[:30].values)
print("\n[Ítem 4] Matriz TF-IDF (primeras 30 filas):")
print(matriz)

# ---- Ítem 5: agrupar filas (KMeans). Tú eliges k; aquí k=18 como en clase ----
k = 18
km = KMeans(n_clusters=k, n_init=500, random_state=42)
labels = km.fit_predict(X)

# ---- Ítem 6: CSV con Pokemon y cluster ----
out_p1 = pd.DataFrame({'Pokemon': datos['Pokemon'], 'cluster': labels})
out_p1 = out_p1.sort_values('cluster').reset_index(drop=True)
out_p1.to_csv('Pokemons_agrupados_P1.csv', index=False)
print("\n[Ítem 6] CSV generado: Pokemons_agrupados_P1.csv (Pokemon, cluster)")
print(out_p1.head(30))

# ---- Ítem 7: interpretar clusters (tokens representativos por cluster) ----
centros = km.cluster_centers_
for c in range(k):
    top_idx = centros[c].argsort()[::-1][:15]
    print(f"\n[Ítem 7] Cluster {c} → tokens representativos:")
    print([tokens[i] for i in top_idx])
