"""
Tests unitaires pour RetrieverAgent (Agent 2).

Ces tests vérifient le comportement de l'agent dans 4 scénarios :
  1. Question FR pertinente → résultats trouvés
  2. Question EN pertinente → résultats trouvés (cross-lingual)
  3. Question hors corpus → refus légitime
  4. Question vide → refus immédiat

Note : Ces tests utilisent la vraie base ChromaDB (data/chroma_db/).
Assurez-vous que index_corpus.py a été exécuté au moins une fois avant.
"""

import sys
import os

# Ajouter la racine du projet au path Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.retriever_agent import RetrieverAgent, RetrieverResult
from src.rag.models import DocumentChunk


def print_separator(title: str):
    print(f"\n{'=' * 60}")
    print(f"  TEST : {title}")
    print('=' * 60)


def print_result(result: RetrieverResult):
    print(f"  → has_results  : {result.has_results}")
    print(f"  → message      : {result.message}")
    print(f"  → nb chunks    : {len(result.chunks)}")
    for i, chunk in enumerate(result.chunks, 1):
        print(f"     [{i}] {chunk.titre} ({chunk.langue}) — score={chunk.score_similarite}")
        print(f"         {chunk.texte[:80]}...")


def test_query_fr_pertinente(agent: RetrieverAgent):
    """Test 1 : Question française dans le corpus → doit trouver des résultats."""
    print_separator("Question FR pertinente")

    result = agent.run(
        query="Quel roi a unifié la Numidie ?",
        lang="fr"
    )
    print_result(result)

    assert isinstance(result, RetrieverResult), "Le retour doit être un RetrieverResult"
    assert result.has_results is True, "Des résultats pertinents doivent être trouvés"
    assert len(result.chunks) > 0, "La liste de chunks ne doit pas être vide"
    assert all(isinstance(c, DocumentChunk) for c in result.chunks), "Chaque chunk doit être un DocumentChunk"
    assert result.lang == "fr", "La langue doit être transmise correctement"

    print("\n  ✅ TEST 1 PASSÉ")


def test_query_en_cross_lingual(agent: RetrieverAgent):
    """Test 2 : Question anglaise → doit retrouver des docs cross-lingual."""
    print_separator("Question EN cross-lingual")

    result = agent.run(
        query="Who was King Masinissa and what did he accomplish ?",
        lang="en"
    )
    print_result(result)

    assert isinstance(result, RetrieverResult), "Le retour doit être un RetrieverResult"
    assert result.has_results is True, "Des résultats doivent être trouvés même cross-lingual"
    assert result.lang == "en", "La langue doit être transmise correctement"

    print("\n  ✅ TEST 2 PASSÉ")


def test_query_hors_corpus(agent: RetrieverAgent):
    """Test 3 : Question hors corpus → refus légitime (no hallucination)."""
    print_separator("Question hors corpus → refus")

    result = agent.run(
        query="Quelle est la recette du couscous marocain traditionnel ?",
        lang="fr"
    )
    print_result(result)

    assert isinstance(result, RetrieverResult), "Le retour doit être un RetrieverResult"
    assert result.has_results is False, "Aucun résultat ne doit être retourné pour une question hors corpus"
    assert len(result.chunks) == 0, "La liste de chunks doit être vide"
    assert "refuse" in result.message.lower() or "aucun" in result.message.lower(), \
        "Le message doit indiquer un refus ou l'absence de résultats"

    print("\n  ✅ TEST 3 PASSÉ")


def test_query_vide(agent: RetrieverAgent):
    """Test 4 : Question vide → refus immédiat sans requête ChromaDB."""
    print_separator("Question vide → refus immédiat")

    result = agent.run(
        query="   ",
        lang="fr"
    )
    print_result(result)

    assert isinstance(result, RetrieverResult), "Le retour doit être un RetrieverResult"
    assert result.has_results is False, "Une question vide ne doit jamais retourner de résultats"
    assert len(result.chunks) == 0, "Aucun chunk pour une question vide"

    print("\n  ✅ TEST 4 PASSÉ")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  TESTS UNITAIRES — RetrieverAgent (Agent 2)")
    print("=" * 60)
    print("\n[INIT] Connexion à ChromaDB et chargement du modèle MPNet...")

    agent = RetrieverAgent(
        db_path="data/chroma_db",
        collection_name="multirag_corpus",
        similarity_threshold=0.40,
        top_k=3
    )

    print(f"[OK] Base vectorielle chargée — {agent.vector_store.count()} passages indexés")

    tests = [
        test_query_fr_pertinente,
        test_query_en_cross_lingual,
        test_query_hors_corpus,
        test_query_vide,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn(agent)
            passed += 1
        except AssertionError as e:
            print(f"\n  ❌ ÉCHEC : {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  RÉSULTATS : {passed} passés / {failed} échoués / {len(tests)} total")
    print(f"{'=' * 60}\n")
