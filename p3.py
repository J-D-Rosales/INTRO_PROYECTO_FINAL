# ================== PREGUNTA 3 — VERIFICACIÓN POR TIPOS ==================
# Requisitos: pip install pandas numpy
import re
import pandas as pd
import numpy as np

# ---------------- Fase 1) Lectura base y minúsculas ----------------
datos = pd.read_csv('smogon.csv')
datos = datos[['Pokemon', 'moves']].dropna().reset_index(drop=True)
datos['moves'] = datos['moves'].astype(str).str.lower()

# ---------------- Fase 2) Acolchonar 17 tipos (igual que en P2) ----------------
TIPOS = [
    'bug','dark','dragon','electric','fairy','fighting','fire','flying','ghost',
    'grass','ground','ice','poison','psychic','rock','steel','water'
]

def pad_tipos(text: str) -> str:
    t = f" {text} "
    for tp in TIPOS:
        t = re.sub(tp, f" {tp} ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

datos['moves_pad'] = datos['moves'].apply(pad_tipos)

# ---------------- Fase 3) Conteo de tipos por Pokémon ----------------
def contar_tipos(text: str) -> dict:
    base = f" {text} "
    res = {}
    for tp in TIPOS:
        res[tp] = len(list(re.finditer(rf" {tp} ", base)))
    return res

conteos = datos['moves_pad'].apply(contar_tipos).apply(pd.Series)
conteos.insert(0, 'Pokemon', datos['Pokemon'])

# ---------------- Fase 4) Top1 y Top2 tipo por Pokémon ----------------
def top2(row: pd.Series) -> pd.Series:
    vals = row[TIPOS]
    orden = vals.values.argsort()[::-1]
    t1, n1 = vals.index[orden[0]], int(vals.values[orden[0]])
    if len(orden) > 1:
        t2, n2 = vals.index[orden[1]], int(vals.values[orden[1]])
    else:
        t2, n2 = "", 0
    return pd.Series({'tipo_top': t1, 'tipo_top_count': n1, 'tipo_2': t2, 'tipo_2_count': n2})

verif = pd.concat([conteos[['Pokemon']], conteos.apply(top2, axis=1)], axis=1)
print("\n[P3] Verificación — primeras 20 filas:")
print(verif.head(20))

# ---------------- Fase 5) Unir con P1 y P2 ----------------
# Asegúrate de haber corrido p1_tfidf_clusters.py y p2_vocab_controlado.py antes:
p1 = pd.read_csv('Pokemons_agrupados_P1.csv').rename(columns={'cluster': 'cluster_p1'})
p2 = pd.read_csv('Pokemons_agrupados_P2.csv').rename(columns={'cluster': 'cluster_p2'})

verif_p1 = p1.merge(verif, on='Pokemon', how='left')
verif_p2 = p2.merge(verif, on='Pokemon', how='left')

# ---------------- Fase 6) Distribución por cluster (para interpretar) ----------------
dist_p1 = verif_p1.groupby(['cluster_p1', 'tipo_top']).size().reset_index(name='n')\
                  .sort_values(['cluster_p1', 'n'], ascending=[True, False])
dist_p2 = verif_p2.groupby(['cluster_p2', 'tipo_top']).size().reset_index(name='n')\
                  .sort_values(['cluster_p2', 'n'], ascending=[True, False])

print("\n[P3] Distribución por cluster (P1) — primeras 30 filas:")
print(dist_p1.head(30))
print("\n[P3] Distribución por cluster (P2) — primeras 30 filas:")
print(dist_p2.head(30))

# ---------------- Fase 7) Guardar salidas de verificación ----------------
verif_p1.to_csv('P1_clusters_con_tipo.csv', index=False)
verif_p2.to_csv('P2_clusters_con_tipo.csv', index=False)
dist_p1.to_csv('P1_cluster_tipo_distribution.csv', index=False)
dist_p2.to_csv('P2_cluster_tipo_distribution.csv', index=False)

print("\n[P3] Archivos generados:")
print(" - P1_clusters_con_tipo.csv")
print(" - P2_clusters_con_tipo.csv")
print(" - P1_cluster_tipo_distribution.csv")
print(" - P2_cluster_tipo_distribution.csv")
