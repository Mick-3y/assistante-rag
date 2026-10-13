import argparse
import docx
import numpy as np
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic

extensions = {".txt", ".md", ".docx", ".pdf"}
excluded_dirs = {"venv", "__pycache__", ".git", "node_modules", "build", "dist"}
modele = SentenceTransformer('intfloat/multilingual-e5-small')

parser = argparse.ArgumentParser(description="Assistant RAG sur des documents personnels.")
parser.add_argument("dossier_docs", type=Path, help="Dossier à indexer (.txt, .md, .docx, .pdf)")
args = parser.parse_args()
DOSSIER_DOCS = args.dossier_docs

prix_input = 1
prix_output = 5

##Evals
gold = [
    ("Quel est le projet phare (flagship) du parcours et son différenciateur technique ?", "SHAP"),
    ("Quelle est la date limite dure pour finir le RAG production-grade ?", "31 juillet"),
    ("Comment un utilisateur se connecte-t-il à l'app Quest IA pro ?", "magic link"),
    ("Quelle métrique privilégier pour le modèle de fraude, et laquelle éviter ?", "precision-recall"),
    ("Avec quelles technologies le front de Quest IA pro est-il codé ?", "Vite"),
    ("Qu'est-ce qui protège les données de chaque utilisateur en base dans Quest IA pro ?", "RLS"),
]

def extraire_texte(fichier: Path):
    content = ""
    if fichier.suffix.lower() == ".pdf":
        reader = PdfReader(fichier)
        for page in reader.pages:
            content += page.extract_text() or ""
        
    elif fichier.suffix.lower() == ".docx":
        doc = docx.Document(fichier)
        content = "\n".join([paragraph.text for paragraph in doc.paragraphs])

    else:
        content = fichier.read_text(encoding="utf-8", errors="ignore")

    return content

def parcourir_dossier(dossier: Path):
    paires = []
    for fichier in dossier.rglob("*"):
        if fichier.is_file() and fichier.suffix.lower() in extensions and not any(part in excluded_dirs for part in fichier.parts):
            texte = extraire_texte(fichier)
            paires.append((str(fichier.relative_to(dossier)), texte))
    return paires

def chunker(texte, taille_max=1000, taille_min=50, overlap=150):
    paragraphs = texte.split("\n\n")
    chunks = []
    for paragraph in paragraphs:
        if len(paragraph) <= taille_max:
            chunks.append(paragraph)
        else:         
            start = 0
            while start < len(paragraph):
                end = min(start + taille_max, len(paragraph))
                chunks.append(paragraph[start:end])
                if end == len(paragraph):
                    break
                start = end - overlap
    chunks = [chunk for chunk in chunks if len(chunk.strip()) >= taille_min]
    return chunks

# ---- Indexation du corpus (exécutée une fois, au niveau module, au démarrage) ----
paires = parcourir_dossier(DOSSIER_DOCS)

list_textes_to_embed = []
chunks_bruts = []
for chemin, texte in paires:
    chunks = chunker(texte)
    for chunk in chunks:
        list_textes_to_embed.append("passage: " + chunk)
        chunks_bruts.append(chunk)
embeddings = modele.encode(list_textes_to_embed, normalize_embeddings=True)
embeddings_sans_prefix = modele.encode(chunks_bruts, normalize_embeddings=True)

def rechercher(question, embeddings, k=5, prefix="query: "):
    query = prefix + question
    query_embedding = modele.encode([query], normalize_embeddings=True)
    scores = np.dot(embeddings, query_embedding[0])
    top_k_indices = np.argsort(scores)[-k:][::-1]
    k_results = [(chunks_bruts[i], scores[i]) for i in top_k_indices]
    return k_results

def construire_prompt(question, k_results):
    prompt = "Réponds uniquement en t'appuyant sur les passages ci-dessous. Si l'information ne s'y trouve pas, réponds 'Je ne sais pas'.\nVoici les passages les plus pertinents extraits de la documentation:\n"
    for i, (chunk, score) in enumerate(k_results):
        prompt += f"\nPassage {i+1}:\n{chunk}\n"
    prompt += f"\nQuestion: {question}\n"
    return prompt

def generer_reponse(prompt, modele_claude="claude-haiku-4-5", max_tokens=1000):
    client = Anthropic()   # lit ANTHROPIC_API_KEY dans l'environnement
    reponse = client.messages.create(
        model=modele_claude,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    reponse_text = ""
    for bloc in reponse.content:
        reponse_text += bloc.text if bloc.type == "text" else ""
    count = (reponse.usage.input_tokens*prix_input + reponse.usage.output_tokens*prix_output)/1_000_000
    return reponse_text, count

def repondre(question, embeddings, k=5):
    k_results = rechercher(question, embeddings, k)
    prompt = construire_prompt(question, k_results)
    reponse, count = generer_reponse(prompt)
    return reponse, count


def recall_at_k(gold, k=5, prefix="query: "):
    # ablation cohérente : sans préfixe requête → passages aussi sans préfixe
    emb = embeddings if prefix else embeddings_sans_prefix
    correct_count = 0
    for question, expected_answer in gold:
        k_results = rechercher(question, emb, k, prefix)
        found = any(expected_answer.lower() in chunk.lower() for chunk, score in k_results)
        if found:
            correct_count += 1
    recall = correct_count / len(gold)
    return recall

def mrr(gold, k=5, prefix="query: "):
    # ablation cohérente : sans préfixe requête → passages aussi sans préfixe
    emb = embeddings if prefix else embeddings_sans_prefix
    reciprocal_ranks = []
    for question, expected_answer in gold:
        k_results = rechercher(question, emb, k, prefix)
        rank = next((i + 1 for i, (chunk, score) in enumerate(k_results) if expected_answer.lower() in chunk.lower()), None)
        if rank is not None:
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)
    mrr_value = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return mrr_value





def main():
    print(f"AVEC PREFIXE - Recall@5: {recall_at_k(gold, k=5, prefix='query: '):.2f} ; MRR@5: {mrr(gold, k=5, prefix='query: '):.2f}")
    print(f"SANS PREFIXE - Recall@5: {recall_at_k(gold, k=5, prefix=''):.2f} ; MRR@5: {mrr(gold, k=5, prefix=''):.2f}")
    while True:
        question = input("\nPosez votre question (ou tapez 'quit' pour quitter): ")
        if question.lower() == 'quit':
            break
        reponse, count = repondre(question, embeddings, k=5)
        print(f"Réponse: {reponse}")
        print(f"Coût: {count:.6f} $")

if __name__ == "__main__":
    main()





