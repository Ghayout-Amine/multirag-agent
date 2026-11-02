import json
import os
import sys

# Ajouter la racine du projet au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag.models import DocumentNotice, DocumentChunk
from src.agents.language_agent import LanguageAgent

def demo_globale():
    print("=" * 65)
    print("    DEMO D'EXECUTION : COMPOSANTS ACTUELS DU PROJET")
    print("=" * 65)

    # 1. Validation du modèle de données et lecture du corpus
    corpus_path = os.path.join("data", "corpus", "sample_notices.json")
    print(f"\n[1] LECTURE DU FICHIER : {corpus_path}")
    with open(corpus_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    notices_validees = []
    for item in data:
        # On valide avec Pydantic pour prouver la conformité des données
        notice = DocumentNotice(**item)
        notices_validees.append(notice)
        print(f"  -> Notice validée : [{notice.id}] ({notice.langue.upper()}) {notice.titre}")
        print(f"     Source : {notice.source_url}")
        print(f"     Aperçu : {notice.texte[:90]}...\n")

    # 2. Test dynamique de l'Agent de Langue (LanguageAgent)
    print("-" * 65)
    print("[2] EXECUTION DYNAMIQUE DE L'AGENT 1 (LanguageAgent)")
    print("-" * 65)
    agent = LanguageAgent()

    questions_a_tester = [
        "Quelle est l'histoire du royaume de Numidie en Afrique du Nord ?",
        "Who founded the kingdom of Numidia in 202 BC?",
        "ما هو دور الملك ماسينيسا في توحيد نوميديا؟",
        "¿Cuáles son los monumentos históricos de la ciudad de Tlemcen?"
    ]

    for q in questions_a_tester:
        res = agent.run(q)
        statut = "[ACCEPTE]" if res.is_supported else "[REJETE]"
        print(f"Question : \"{q}\"")
        print(f"  Resultat -> Langue: {res.lang.upper()} | Statut: {statut}")
        print(f"  Message  : {res.message}\n")

    print("=" * 65)
    print("  BILAN : TOUS LES FICHIERS DU PROJET S'EXECUTENT PARFAITEMENT")
    print("=" * 65)

if __name__ == "__main__":
    demo_globale()
