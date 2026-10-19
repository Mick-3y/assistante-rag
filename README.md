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


### Résultat clé : l'effet des préfixes E5

E5 est un modèle **asymétrique** : entraîné avec `query:` sur les questions et
`passage:` sur les documents. Ces préfixes apprennent au modèle à rapprocher une
question de sa réponse *même quand les mots diffèrent*. Mesure de leur effet :

| Config | Recall@5 | MRR@5 |
|---|---|---|
| **Avec** préfixes (usage correct d'E5) | **0.83** | **0.62** |
| **Sans** préfixes (usage naïf, symétrique) | 0.83 | 0.36 |
| écart | **0 pt** | **−26 pts** |

> L'ablation est **symétrique** : « sans préfixe » retire le préfixe des deux côtés
> (requête *et* passages), pour isoler l'effet propre du préfixe sans introduire de
> désaccord d'espace. (Retirer le préfixe d'un seul côté — un oubli fréquent en prod —
> casse en plus le *rappel*, mais c'est un autre bug.)

Deux enseignements :

1. **Le Recall ne bouge pas ; seul le MRR chute.** Les préfixes ne changent pas
   *quels* chunks remontent dans le top-5 (0.83 → 0.83) — ils changent *comment ils
   sont classés* (MRR −26 pts). C'est un levier de **ranking**, pas de rappel. Un bon
   chunk qui glisse du rang 1 au rang 3 reste dans le top-5 (Recall intact) mais voit
   son reciprocal s'effondrer (MRR frappé).
2. **C'est un échec silencieux.** Sans préfixes, rien ne plante : le code tourne,
   remonte 5 chunks, génère des réponses plausibles. La dégradation n'est
   **visible que par l'eval**. Illustration directe de « un système LLM se dégrade
   en silence ».

> **Honnêteté (n=6) :** la *direction* de l'écart est robuste (cohérente avec le
> design documenté d'E5) ; la *magnitude exacte* est bruitée avec 6 questions et
> bougerait sur un jeu de 50. Le jeu est en outre **synthétique et mono-passage**
> (voir limites) → scores optimistes.

---

## Limites assumées

- **Corpus vs. complexité.** Retrieval vectoriel = démonstration de compétence, pas
  nécessité (corpus petit). Assumé.
- **Chunking par caractères sur les blocs sans séparateur.** Un bloc de plus de
  1000 caractères sans `\n\n` est coupé **au caractère près** (fenêtre glissante) —
  ça peut trancher au milieu d'une phrase. Un chunker sémantique (par tokens/phrases)
  serait meilleur ; non fait par choix de simplicité.
- **Petits fragments jetés, pas fusionnés.** Les chunks < 50 caractères sont
  **supprimés**, pas rattachés au voisin. Conséquence concrète : un fichier d'une
  seule ligne (ex. le README d'un sous-projet) est sous-représenté, voire absent —
  d'où des « Je ne sais pas » sur du contenu qui *existe* mais est trop court pour
  survivre au filtre.
- **Scores cosinus compressés.** Avec E5, les scores se tassent dans une bande
  haute et étroite (~0.84–0.87 observés). **On sélectionne donc par RANG (top-k),
  jamais par seuil** — un seuil serait ingérable.
- **Coût dominé par le contexte, pas la réponse.** Contexte long + réponse courte →
  l'**input** pèse plus que l'output *malgré* un output tarifé 5× plus cher au token.
  Implication : ne pas empiler 20 passages quand 5 suffisent.
- **Ancrage imparfait.** Faute de passage net, le modèle peut « étirer » un passage
  proche (ex. répondre sur la période du programme via la fenêtre de candidature P3).
- **Eval synthétique.** Gold écrit à la main, mono-passage, n=6, fort recouvrement
  lexical → optimiste. Bon premier signal, pas un benchmark.

---

## Compétences défendables démontrées

1. **Context engineering** — stratégie de chunking, préfixes E5 asymétriques,
   prompt ancré anti-hallucination. *Exemple : j'ai mesuré que les préfixes E5 valent
   +26 pts de MRR — le Recall, lui, ne bouge pas : les préfixes améliorent le
   classement, pas la présence du bon chunk.*
2. **Evals** — harnais Recall@5 / MRR maison, baseline comparée (avec/sans préfixes),
   biais du jeu de test documentés. *Les evals remplacent les tests unitaires sur un
   système LLM.*
3. **Observabilité & coût** — coût par requête calculé (tokens × tarif), arbitrage
   modèle/coût/qualité (Haiku vs Opus = ×5), détection d'un échec **silencieux** via
   l'eval.
4. **Architecture RAG** — pipeline de bout en bout, décision *rang plutôt que seuil*,
   discipline « simple d'abord, complexité seulement sur une limite réelle ».

---

## Stack

`sentence-transformers` (E5 embeddings) · `numpy` (recherche vectorielle) ·
`pypdf` / `python-docx` (extraction) · API Claude `anthropic` (génération).
