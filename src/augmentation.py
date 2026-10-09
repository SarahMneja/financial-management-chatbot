"""
=============================================================================
 LE "A" DE RAG : AUGMENTATION
=============================================================================

 Construire le prompt envoye au modele de langage.

 Le modele ne recoit pas directement tout le PDF.
 Il recoit uniquement les passages retrouves par le retriever.

 Un prompt augmente a trois parties :

     SYSTEME    les regles du chatbot
     CONTEXTE   les passages retrouves, avec leurs sources
     QUESTION   la question de l'utilisateur
=============================================================================
"""


# =============================================================================
# LE PROMPT SYSTEME
# =============================================================================

PROMPT_SYSTEME = """Tu es un assistant specialise dans le guide de la
gestion financiere de Fundamentals of Financial Management.

Regles a respecter :

1. Reponds UNIQUEMENT a partir des passages fournis dans le contexte.
   N'utilise aucune connaissance exterieure, meme si tu penses la connaitre.

2. Cite systematiquement tes sources sous la forme [Source N] apres chaque
   affirmation, en reprenant les numeros donnes dans le contexte.
   Utilise des crochets droits [ ], jamais d'autres caracteres.

3. Si les passages ne contiennent pas la reponse, dis-le clairement :
   "Le document ne traite pas ce point." N'invente jamais.

4. Reponds en anglais, de maniere concise et structuree.
   Va droit au but : trois a cinq phrases suffisent le plus souvent."""


# =============================================================================
# LA MISE EN FORME DU CONTEXTE
# =============================================================================

def formater_source(metadata):
    """Fabrique une reference lisible a partir des metadonnees d'un chunk."""

    chemin = metadata.get("chemin", [])

    fiche = ""

    if isinstance(chemin, list):

        for element in chemin:
            if str(element).upper().startswith("FICHE"):
                fiche = str(element)
                break

        if not fiche and chemin:
            fiche = str(chemin[-1])

    else:
        fiche = str(chemin)

    page = metadata.get("page", "?")

    return f"{fiche}, page {page}" if fiche else f"page {page}"


def construire_contexte(passages):
    """Assemble les passages en un bloc de texte numerote."""

    blocs = []

    for numero, passage in enumerate(passages, start=1):

        source = formater_source(passage["metadata"])

        # Le chemin des titres est deja prefixe au texte depuis la Session 1.
        # On le retire ici car la source contient deja cette information.
        corps = passage["texte"].split("\n\n", 1)[-1]

        blocs.append(
            f"[Source {numero}] {source}\n{corps}"
        )

    return "\n\n".join(blocs)


# =============================================================================
# LE PROMPT UTILISATEUR
# =============================================================================

def construire_prompt(question, passages):
    """Assemble le contexte et la question."""

    contexte = construire_contexte(passages)

    return f"""Voici des extraits du document Fundamentals of Financial Management :

{contexte}

---

Question : {question}

Reponds en te fondant uniquement sur ces extraits, et cite tes sources."""


# =============================================================================
# AFFICHAGE POUR LE TEST
# =============================================================================

def afficher_prompt(question, passages):
    """Affiche le prompt complet qui sera envoye au modele."""

    prompt = construire_prompt(question, passages)

    print("=" * 76)
    print("PROMPT SYSTEME")
    print("=" * 76)
    print(PROMPT_SYSTEME)

    print()
    print("=" * 76)
    print("PROMPT UTILISATEUR")
    print("=" * 76)
    print(prompt)

    print("=" * 76)

    total = len(PROMPT_SYSTEME) + len(prompt)

    print(
        f"\nTaille totale : {total} caracteres, "
        f"soit environ {round(total / 3.8)} tokens."
    )


# =============================================================================
# TEST DIRECT
# =============================================================================

if __name__ == "__main__":

    import sys

    from retriever import Retriever

    question = (
        " ".join(sys.argv[1:])
        or "What is financial management?"
    )

    retriever = Retriever()

    passages = retriever.chercher(question)

    afficher_prompt(question, passages)