"""
Tests unitaires et d'intégration pour ResponseAgent (Agent 3).

Vérifie :
  1. Génération factuelle en français avec citation [1]
  2. Question hors contexte documentaire -> refus explicite (zéro hallucination)
  3. Liste de passages vide -> refus immédiat
"""

import sys
import os

# Ajouter la racine du projet au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.response_agent import ResponseAgent, ResponseResult
from src.rag.models import DocumentChunk


def create_sample_chunk() -> DocumentChunk:
    return DocumentChunk(
        chunk_id="notice_fr_001_chunk_0",
        notice_id="notice_fr_001",
        titre="Royaume de Numidie",
        langue="fr",
        source_url="https://fr.wikipedia.org/wiki/Numidie",
        texte=(
            "La Numidie est un ancien royaume berbère qui a existé en Afrique du Nord "
            "de 202 av. J.-C. à 46 av. J.-C. Il a été unifié sous le règne du roi Masinissa, "
            "qui a établi Cirta (actuelle Constantine) comme sa capitale."
        ),
        score_similarite=0.88
    )


def print_separator(title: str):
    print(f"\n{'=' * 60}")
    print(f"  TEST : {title}")
    print('=' * 60)


def print_result(result: ResponseResult):
    print(f"  -> has_answer  : {result.has_answer}")
    print(f"  -> model_used  : {result.model_used}")
    print(f"  -> nb citations: {len(result.citations)}")
    print(f"  -> reponse     :\n{result.answer}")


def test_generation_fr_avec_citation(agent: ResponseAgent):
    """Test 1 : Réponse factuelle en français avec citation [1]."""
    print_separator("Generation FR avec citations")

    chunk = create_sample_chunk()
    result = agent.run(
        query="Quelle etait la capitale de la Numidie et qui l'a unifiee ?",
        chunks=[chunk],
        lang="fr"
    )
    print_result(result)

    assert isinstance(result, ResponseResult)
    assert result.has_answer is True, "Une reponse doit etre formulee"
    assert "Cirta" in result.answer or "Constantine" in result.answer, "La capitale doit figurer"
    assert "Masinissa" in result.answer, "Le roi Masinissa doit figurer"
    assert "[1]" in result.answer, "La reponse doit comporter la citation formelle [1]"
    assert len(result.citations) == 1, "Une citation doit etre attachee"

    print("\n  [OK] TEST 1 PASSE")


def test_refus_hors_contexte(agent: ResponseAgent):
    """Test 2 : Question hors contexte -> refus motivé sans hallucination."""
    print_separator("Question hors contexte -> Refus motive")

    chunk = create_sample_chunk()
    result = agent.run(
        query="Combien d'avions de chasse possedait l'armee numide ?",
        chunks=[chunk],
        lang="fr"
    )
    print_result(result)

    assert isinstance(result, ResponseResult)
    # Le modèle ne doit pas inventer d'avions de chasse
    assert "avion" not in result.answer.lower() or "non trouvée" in result.answer.lower() or "non trouvee" in result.answer.lower()
    print("\n  [OK] TEST 2 PASSE")


def test_chunks_vides(agent: ResponseAgent):
    """Test 3 : Absence de passages -> refus immédiat."""
    print_separator("Absence de passages -> Refus immediat")

    result = agent.run(
        query="Qui a fonde la Numidie ?",
        chunks=[],
        lang="fr"
    )
    print_result(result)

    assert isinstance(result, ResponseResult)
    assert result.has_answer is False, "has_answer doit etre False quand aucun chunk n'est fourni"
    assert len(result.citations) == 0, "Aucune citation quand aucun chunk n'est fourni"

    print("\n  [OK] TEST 3 PASSE")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  TESTS D'INTEGRATION - ResponseAgent (Agent 3)")
    print("=" * 60)

    agent = ResponseAgent()
    print(f"[OK] Agent initialise avec le modele : {agent.model}")

    tests = [
        test_generation_fr_avec_citation,
        test_refus_hors_contexte,
        test_chunks_vides
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn(agent)
            passed += 1
        except AssertionError as e:
            print(f"\n  [ECHEC] : {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  RESULTATS : {passed} passes / {failed} echoues / {len(tests)} total")
    print(f"{'=' * 60}\n")
