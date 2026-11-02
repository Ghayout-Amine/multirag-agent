import json
import os
import sys

# Ajouter la racine du projet au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.rag.models import DocumentNotice
from src.rag.vector_store import VectorStoreManager

def main():
    print("=" * 65)
    print("    INDEXATION DU CORPUS ET TEST VECTORIEL (CHROMADB)")
    print("=" * 65)

    # 1. Initialiser le gestionnaire ChromaDB
    print("\n[1] Initialisation de ChromaDB et chargement du modèle MPNet...")
    manager = VectorStoreManager(db_path="data/chroma_db", collection_name="multirag_corpus")
    print("    ChromaDB prêt !")

    # 2. Charger les notices du corpus
    corpus_file = os.path.join("data", "corpus", "sample_notices.json")
    print(f"\n[2] Lecture du fichier corpus : {corpus_file}")
    with open(corpus_file, "r", encoding="utf-8") as f:
        notices_data = json.load(f)

    total_chunks = 0
    for item in notices_data:
        notice = DocumentNotice(**item)
        nb_chunks = manager.add_notice(notice, chunk_size=200, overlap=30)
        total_chunks += nb_chunks
        print(f"  -> Indexé : [{notice.id}] ({notice.langue.upper()}) '{notice.titre}' -> {nb_chunks} chunk(s)")

    print(f"\nTotal des fragments enregistrés dans ChromaDB : {manager.count()} passage(s)")

    # 3. Test de Recherche Sémantique (Retrieval)
    print("-" * 65)
    print("[3] TESTS DE RECHERCHE SEMANTIQUE (RAG RETRIEVAL)")
    print("-" * 65)

    tests_queries = [
        ("FR", "Quel roi a unifié la Numidie ?"),
        ("EN", "Who was King Masinissa and what was his role?"),
        ("AR", "من هو مؤسس مملكة نوميديا القديمة؟"),
        ("HORS_SUJET", "Quelle est la recette traditionnelle du couscous aux légumes ?")
    ]

    for lang_code, query in tests_queries:
        print(f"\nRequete [{lang_code}] : \"{query}\"")
        chunks = manager.search(query, top_k=2, similarity_threshold=0.35)
        
        if not chunks:
            print("  -> [AUCUN PASSAGE RETENU] : La similarite est inferieure au seuil (Refus RAG legitime)")
        else:
            for idx, c in enumerate(chunks, 1):
                print(f"  Passage [{idx}] (Score: {c.score_similarite}) | Titre: {c.titre} | Langue: {c.langue.upper()}")
                print(f"  Extrait : \"{c.texte[:110]}...\"")
                print(f"  Source  : {c.source_url}")

    print("\n" + "=" * 65)
    print("  FIN DU TEST DE L'ETAPE 2 : BASE VECTORIELLE VALIDE ! [OK]")
    print("=" * 65)

if __name__ == "__main__":
    main()
