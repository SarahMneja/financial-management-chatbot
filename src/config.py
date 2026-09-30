"""
Configuration centrale du projet RAG.
Tous les reglages importants sont regroupes ici.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


# Racine du projet
RACINE = Path(__file__).resolve().parent.parent

# Dossiers et fichiers de donnees
DATA = RACINE / "data"

CHUNKS = DATA / "chunks.jsonl"
INDEX = DATA / "index.faiss"
METADONNEES = DATA / "chunks.json"


# ============================================================
# MODELES
# ============================================================

MODELE_EMBEDDING = "paraphrase-multilingual-MiniLM-L12-v2"

LONGUEUR_MAX = 512

MODELE_RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

MODELE_LLM = "openai/gpt-oss-120b"


# ============================================================
# RECHERCHE
# ============================================================

N_CANDIDATS = 20
N_PASSAGES = 5

UTILISER_RERANKING = True

SEUIL_FAISS = 0.15


# ============================================================
# GENERATION
# ============================================================

TEMPERATURE = 0.1

MAX_TOKENS_REPONSE = 1500


# ============================================================
# CLE API GROQ
# ============================================================

load_dotenv(RACINE / ".env")

CLE_GROQ = os.getenv("GROQ_API_KEY", "")


# ============================================================
# VERIFICATION
# ============================================================

if __name__ == "__main__":
    print(f"Racine du projet : {RACINE}")
    print(f"Chunks           : {CHUNKS.name:<15} "
          f"{'OK' if CHUNKS.exists() else 'ABSENT'}")
    print(f"Index FAISS      : {INDEX.name:<15} "
          f"{'OK' if INDEX.exists() else 'ABSENT (lancez vectoriser.py)'}")
    print(f"Metadonnees      : {METADONNEES.name:<15} "
          f"{'OK' if METADONNEES.exists() else 'ABSENT (lancez vectoriser.py)'}")
    print()

    print(f"Embedding : {MODELE_EMBEDDING}")
    print(f"Reranker  : {MODELE_RERANKER if UTILISER_RERANKING else '(desactive)'}")
    print(f"LLM       : {MODELE_LLM}")
    print()

    print(f"Cle Groq  : {'OK' if CLE_GROQ else 'ABSENTE, creez un fichier .env'}")