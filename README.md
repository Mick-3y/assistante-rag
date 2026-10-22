# Assistante RAG

Un assistant qui répond à des questions en s'appuyant sur mes documents
personnels (`.md`, `.txt`, `.pdf`, `.docx`), avec une **vraie couche de retrieval
vectoriel**, des **evals de retrieval**, et le **coût par requête** affiché.

---

## Ce que ça fait

- Lit récursivement un dossier (`.md`, `.txt`, `.pdf`, `.docx`), découpe en *chunks*.
- Encode chaque chunk en vecteur (embeddings E5) et cherche par **sens**, pas par mots-clés.
- Construit un prompt *ancré* (le modèle répond **uniquement** à partir des passages,
  sinon « Je ne sais pas ») et génère la réponse via l'API Claude.
- **Affiche le coût** (tokens in/out × prix) de chaque requête.
- Livré avec un **jeu d'évaluation du retrieval** (Recall@5, MRR) et une baseline comparée.

---

## Architecture (le pipeline)

```
 Dossier de docs
      │
      ▼
 1. INGESTION        parcourir_dossier / extraire_texte
      │              (pdf: pypdf · docx: python-docx · txt/md: read_text)
      ▼              → liste de (chemin, texte)
 2. CHUNKING         chunker()  — découpe par paragraphes (\n\n),
      │              fenêtre glissante 1000 car. + overlap 150 sur les blocs longs,
      ▼              rejette les fragments < 50 car.
 3. EMBEDDINGS       intfloat/multilingual-e5-small — 384 dims, normalisés
      │              chunks encodés avec le préfixe "passage: "
      ▼              → une matrice (N chunks × 384), construite une fois au démarrage
 4. RECHERCHE        rechercher()  — question préfixée "query: ",
      │              produit scalaire (= cosinus, vecteurs normalisés),
      ▼              top-k par RANG (k=5)
 5. PROMPT           construire_prompt()  — instruction d'ancrage en tête,
      │              passages numérotés, question en dernier
      ▼
 6. GÉNÉRATION       generer()  — API Claude (claude-haiku-4-5),
      │              + calcul du coût (input/output tokens × tarif)
      ▼
 7. ASSEMBLAGE       repondre()  — enchaîne 4→6 ; boucle interactive
      │
      ▼
 Réponse ancrée + coût
```

---

## Lancer

1. **Dépendances :** `pip install pypdf python-docx sentence-transformers numpy anthropic`
2. **Clé API Anthropic** (voir plus bas), dans une variable d'environnement :
   ```powershell
   $env:ANTHROPIC_API_KEY = "sk-ant-..."
   ```
3. **Lancer :** `python assistante.py <dossier_a_indexer>`
   - `<dossier_a_indexer>` : chemin vers le dossier contenant tes documents (`.txt`,
     `.md`, `.docx`, `.pdf`), parcouru récursivement.
   - Au premier lancement, le modèle E5 se télécharge (~470 Mo).
   - Pose tes questions, `quit` pour sortir.

**Obtenir une clé :** https://console.anthropic.com/ → *API Keys* → *Create Key*
(quelques euros de crédit suffisent largement). La clé est un secret :
**jamais dans le code, jamais sur GitHub** — uniquement via la variable d'env.

---

## Stack

`sentence-transformers` (E5 embeddings) · `numpy` (recherche vectorielle) ·
`pypdf` / `python-docx` (extraction) · API Claude `anthropic` (génération).
