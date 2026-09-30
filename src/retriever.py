"""
=============================================================================
 LE "R" DE RAG : RETRIEVAL
=============================================================================

 question -> vecteur -> FAISS -> candidats
                                  |
                             cross-encodeur
                                  |
                              passages
=============================================================================
"""

import json
import sys

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

import config


class Retriever:
    """Charge l'index une fois, puis permet de rechercher plusieurs questions."""

    def __init__(self, verbeux=True):

        self._verifier_fichiers()

        # ---------------------------------------------------------------
        # 1. Charger le modele d'embedding
        # ---------------------------------------------------------------
        if verbeux:
            print("Chargement du modele d'embedding...")

        self.encodeur = SentenceTransformer(config.MODELE_EMBEDDING)
        self.encodeur.max_seq_length = config.LONGUEUR_MAX

        # ---------------------------------------------------------------
        # 2. Charger FAISS
        # ---------------------------------------------------------------
        if verbeux:
            print("Chargement de l'index FAISS...")

        self.index = faiss.read_index(str(config.INDEX))

        # ---------------------------------------------------------------
        # 3. Charger les chunks/metadonnees
        # ---------------------------------------------------------------
        with open(config.METADONNEES, encoding="utf-8") as fichier:
            self.chunks = json.load(fichier)

        # Verifier que l'index et les chunks correspondent
        if self.index.ntotal != len(self.chunks):
            print(
                f"ERREUR : {self.index.ntotal} vecteurs mais "
                f"{len(self.chunks)} chunks."
            )
            print("Relancez : python src/vectoriser.py")
            sys.exit(1)

        # ---------------------------------------------------------------
        # 4. Charger le cross-encodeur si active
        # ---------------------------------------------------------------
        self.reranker = None

        if config.UTILISER_RERANKING:

            if verbeux:
                print("Chargement du cross-encodeur...")

            self.reranker = CrossEncoder(config.MODELE_RERANKER)

        if verbeux:
            print(
                f"Pret : {self.index.ntotal} vecteurs de dimension "
                f"{self.index.d}\n"
            )

    @staticmethod
    def _verifier_fichiers():
        """Verifie que l'index FAISS existe."""

        if not config.INDEX.exists() or not config.METADONNEES.exists():
            print("ERREUR : aucun index trouve.")
            print("Lancez d'abord : python src/vectoriser.py")
            sys.exit(1)

    # -------------------------------------------------------------------
    # ETAPE 1 : recherche vectorielle avec FAISS
    # -------------------------------------------------------------------

    def _chercher_candidats(self, question, n):
        """Retourne les n chunks les plus proches de la question."""

        # La question utilise exactement le meme modele
        # que celui utilise pour les chunks.
        vecteur = self.encodeur.encode(
            [question],
            normalize_embeddings=True
        )

        vecteur = np.asarray(vecteur, dtype="float32")

        # FAISS retourne :
        # - les scores
        # - les positions des vecteurs dans l'index
        scores, positions = self.index.search(vecteur, n)

        candidats = []

        for score, position in zip(scores[0], positions[0]):

            if position < 0:
                continue

            chunk = self.chunks[int(position)]

            candidats.append({
                "id": chunk["id"],
                "texte": chunk["texte"],
                "metadata": chunk["metadata"],
                "score_faiss": float(score),
            })

        return candidats

    # -------------------------------------------------------------------
    # ETAPE 2 : reranking
    # -------------------------------------------------------------------

    def _reclasser(self, question, candidats):
        """Reclasse les candidats avec le cross-encodeur."""

        if not self.reranker or not candidats:
            return candidats

        # Le cross-encodeur reçoit des paires :
        # (question, passage)
        paires = [
            (question, candidat["texte"])
            for candidat in candidats
        ]

        scores = self.reranker.predict(paires)

        for candidat, score in zip(candidats, scores):
            candidat["score_rerank"] = float(score)

        # Meilleur score en premier
        candidats.sort(
            key=lambda candidat: candidat["score_rerank"],
            reverse=True
        )

        return candidats

    # -------------------------------------------------------------------
    # METHODE PRINCIPALE
    # -------------------------------------------------------------------

    def chercher(self, question, n_passages=None, n_candidats=None):
        """Retourne les passages les plus pertinents."""

        n_passages = n_passages or config.N_PASSAGES
        n_candidats = n_candidats or config.N_CANDIDATS

        candidats = self._chercher_candidats(
            question,
            n_candidats
        )

        candidats = self._reclasser(
            question,
            candidats
        )

        return candidats[:n_passages]


# =============================================================================
# AFFICHAGE POUR TESTER LE RETRIEVER
# =============================================================================

def titre_fiche(metadata):
    """Retrouve le titre dans le chemin des titres."""

    chemin = metadata.get("chemin", [])

    if isinstance(chemin, list):

        for element in chemin:
            if str(element).upper().startswith("FICHE"):
                return str(element)

        return " > ".join(str(e) for e in chemin)

    return str(chemin)


def afficher(passages, question):
    """Affiche les passages trouves."""

    print("=" * 76)
    print(f"QUESTION : {question}")
    print("=" * 76)

    if not passages:
        print("Aucun resultat.")
        return

    for rang, passage in enumerate(passages, start=1):

        meta = passage["metadata"]

        scores = f"faiss {passage['score_faiss']:.3f}"

        if "score_rerank" in passage:
            scores += (
                f"   rerank {passage['score_rerank']:+.2f}"
            )

        print(
            f"\n[{rang}]  {scores}   "
            f"page {meta.get('page', '?')}"
        )

        print(
            f"     {titre_fiche(meta)[:64]}"
        )

        print("     " + "-" * 66)

        corps = passage["texte"].split("\n\n", 1)[-1]

        print(f"     {corps[:300]}...")


# =============================================================================
# TEST DIRECT
# =============================================================================

if __name__ == "__main__":

    question = (
        " ".join(sys.argv[1:])
        or "What is financial management?"
    )

    retriever = Retriever()

    passages = retriever.chercher(question)

    afficher(passages, question)