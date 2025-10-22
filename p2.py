# ================== PREGUNTA 2 — VOCABULARIO CONTROLADO (17 tipos) ==================
# Requisitos: pip install pandas scikit-learn numpy
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# -------- Config de impresión --------
IMPRIMIR_MATRIZ_COMPLETA = False
N_MUESTRA_MATRIZ = 25

# ---------------- Fase 1) Lectura y minúsculas ----------------
datos = pd.read_csv('smogon.csv')
datos = datos[['Pokemon', 'moves']].dropna().reset_index(drop=True)
datos['moves'] = datos['moves'].astype(str).str.lower()

# ---------------- Fase 2) Acolchonar 17 tipos (sin 'normal') ----------------
TIPOS = [
    'bug','dark','dragon','electric','fairy','fighting','fire','flying','ghost',
    'grass','ground','ice','poison','psychic','rock','steel','water'
]

def pad_tipos(text: str) -> str:
    t = f" {text} "
    for tp in TIPOS:            # sin 'normal' (recomendación del enunciado)
        t = re.sub(tp, f" {tp} ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

datos['moves_pad'] = datos['moves'].apply(pad_tipos)

# ---------------- Fase 3) TF-IDF SOLO con esos 17 tipos ----------------
vectorizer = TfidfVectorizer(vocabulary=TIPOS, ngram_range=(1,1))
X = vectorizer.fit_transform(datos['moves_pad'])
tokens = vectorizer.get_feature_names_out()
print("\n[P2] Número de columnas (debería ser 17):", len(tokens))
print("[P2] Tokens (tipos):", list(tokens))

# ---------------- Fase 4) Matriz TF-IDF (mostrar) ----------------
if IMPRIMIR_MATRIZ_COMPLETA:
    matriz = pd.DataFrame(X.toarray(), columns=tokens)
    matriz.insert(0, 'Pokemon', datos['Pokemon'])
    print("\n[P2] MATRIZ TF-IDF (tipos) COMPLETA:")
    print(matriz)
else:
    matriz_head = pd.DataFrame(X[:N_MUESTRA_MATRIZ].toarray(), columns=tokens)
    matriz_head.insert(0, 'Pokemon', datos['Pokemon'].iloc[:N_MUESTRA_MATRIZ].values)
    print(f"\n[P2] MATRIZ TF-IDF (tipos) — primeras {N_MUESTRA_MATRIZ} filas:")
    print(matriz_head)

# ---------------- Fase 5) Clustering (mismos parámetros para comparar) ----------------
k = 18
km = KMeans(n_clusters=k, n_init=500, random_state=42)
labels = km.fit_predict(X)

# ---------------- Fase 6) CSV Pokemon/cluster P2 ----------------
out = pd.DataFrame({'Pokemon': datos['Pokemon'], 'cluster': labels})
out = out.sort_values('cluster').reset_index(drop=True)
out.to_csv('Pokemons_agrupados_P2.csv', index=False)
print("\n[P2] CSV generado: Pokemons_agrupados_P2.csv (Pokemon, cluster)")
print(out.head(30))

# ---------------- Fase 7) Tipos representativos por cluster ----------------
centros = km.cluster_centers_
TOP_N = 5
print("\n[P2] Tipos representativos por cluster:")
for c in range(k):
    idx = centros[c].argsort()[::-1][:TOP_N]
    print(f"Cluster {c}: {[tokens[i] for i in idx]}")

# (Opcional) Guardar interpretación
pd.DataFrame({f'cluster_{c}': [tokens[i] for i in centros[c].argsort()[::-1][:TOP_N]] for c in range(k)}).to_csv('P2_cluster_top_tipos.csv', index=False)
print("\n(Extra) Interpretación guardada en P2_cluster_top_tipos.csv")
