"""
Generation du chatbot RAG.

Cette etape prend :
    - la question de l'utilisateur
    - les passages recuperes
    - le prompt construit par augmentation.py

Puis elle appelle le modele Groq.
"""

import sys

from groq import Groq

import config
from augmentation import PROMPT_SYSTEME, construire_prompt


class Generateur:
    """Enveloppe l'appel au modele de langage."""

    def __init__(self):
        if not config.CLE_GROQ:
            print("ERREUR : aucune cle API trouvee.")
            print()
            print("1. Creez un compte sur https://console.groq.com")
            print("2. Generez une cle API")
            print("3. Creez un fichier .env a la racine du projet :")
            print()
            print("   GROQ_API_KEY=votre_cle_ici")
            print()
            sys.exit(1)

        self.client = Groq(api_key=config.CLE_GROQ)
        self.derniere_consommation = None

    def generer(self, question, passages):
        """Construit le prompt, appelle le modele, renvoie la reponse."""

        prompt = construire_prompt(question, passages)

        try:
            reponse = self.client.chat.completions.create(
                model=config.MODELE_LLM,

                messages=[
                    {
                        "role": "system",
                        "content": PROMPT_SYSTEME,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],

                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS_REPONSE,
            )

        except Exception as erreur:
            return self._message_erreur(erreur)

        usage = reponse.usage

        self.derniere_consommation = {
            "prompt": usage.prompt_tokens,
            "reponse": usage.completion_tokens,
            "total": usage.total_tokens,
        }

        return self._normaliser(
            reponse.choices[0].message.content
        )

    @staticmethod
    def _normaliser(texte):
        """Remet les citations dans des crochets droits."""

        for ouvrant in ("【", "［"):
            texte = texte.replace(ouvrant, "[")

        for fermant in ("】", "］"):
            texte = texte.replace(fermant, "]")

        return texte.strip()

    @staticmethod
    def _message_erreur(erreur):
        """Transforme une exception en message comprehensible."""

        texte = str(erreur).lower()

        if "authentication" in texte or "api key" in texte or "401" in texte:
            return (
                "ERREUR : cle API refusee. Verifiez le contenu "
                "de votre fichier .env."
            )

        if "rate" in texte or "429" in texte:
            return (
                "ERREUR : trop de requetes. Le quota gratuit est "
                "atteint, attendez une minute."
            )

        if (
            "does not exist" in texte
            or "not found" in texte
            or "decommission" in texte
            or "404" in texte
        ):
            return (
                f"ERREUR : le modele {config.MODELE_LLM} n'est pas "
                f"disponible pour votre cle.\n\n"
                f"Pour voir la liste de vos modeles :\n"
                f"    python src/generateur.py --modeles\n\n"
                f"Puis changez MODELE_LLM dans config.py."
            )

        return f"ERREUR lors de l'appel au modele : {erreur}"

    def lister_modeles(self):
        """Affiche les modeles auxquels cette cle donne acces."""

        modeles = self.client.models.list()

        for modele in modeles.data:
            print(modele.id)


if __name__ == "__main__":

    if "--modeles" in sys.argv:
        generateur = Generateur()
        generateur.lister_modeles()
        sys.exit(0)

    question = " ".join(
        arg for arg in sys.argv[1:]
        if arg != "--modeles"
    )

    if not question:
        question = "What is financial management?"

    from retriever import Retriever

    retriever = Retriever()
    passages = retriever.chercher(question)

    generateur = Generateur()
    reponse = generateur.generer(question, passages)

    print("=" * 76)
    print("QUESTION")
    print("=" * 76)
    print(question)

    print()
    print("=" * 76)
    print("REPONSE")
    print("=" * 76)
    print(reponse)

    if generateur.derniere_consommation:
        consommation = generateur.derniere_consommation

        print()
        print(
            f"Tokens : {consommation['prompt']} envoyes, "
            f"{consommation['reponse']} recus, "
            f"{consommation['total']} au total."
        )